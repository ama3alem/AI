"""Tests for the personal baseline engine (Phase 6).

Covers the phase exit criteria directly: the maturity ladder and its transitions, cold-start
fallback to a labelled population prior, robust statistics as the documented default, and
the guarantee that an outlier cannot inflate the baseline's own tolerance and therefore
mask a real change.

The outlier tests compare against the non-robust estimator on identical input. A claim that
an estimator is robust only means something relative to an alternative, so the alternative
is present in the suite rather than described in prose.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from focus_engine.baseline import (
    BASELINE_DIMENSIONS,
    BaselineDimension,
    BaselineEngine,
    BaselineMaturity,
    BaselineProfile,
    BaselineReference,
    BaselineUpdateError,
    PopulationPrior,
    PopulationPriorSet,
    ReferenceSource,
    StatisticsMethod,
    basis_for,
    decimate,
    maturity_for,
    maturity_rank,
    robust_dispersion,
    sample_dispersion,
)
from focus_engine.configuration.thresholds import BaselineSettings
from focus_engine.features import (
    FEATURE_SET_V1,
    INSUFFICIENT_DATA_MARKER,
    FeatureAvailability,
    FeatureName,
    FeatureValue,
    TypedFeatureValue,
)
from focus_engine.schemas.primitives import DataOrigin, InferenceBasis
from focus_engine.utils.determinism import DIGEST_SIZE_BYTES

pytestmark = pytest.mark.unit

BASE_TIME = datetime(2026, 9, 27, 9, 0, 0, tzinfo=UTC)
LEARNER_ID = "learner-0001"
OTHER_LEARNER_ID = "learner-0002"
SESSION_ID = "session-0001"
OTHER_SESSION_ID = "session-0002"

ACCURACY = FeatureName.PERFORMANCE_RECENT_ACCURACY
RESPONSE = FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS

PLAIN = BaselineSettings()
FAST_LADDER = BaselineSettings(
    min_samples_new=0,
    min_samples_early=3,
    min_samples_developing=6,
    min_samples_established=10,
)
NON_ROBUST = BaselineSettings(use_robust_statistics=False)

_PRIOR_DEFAULTS: dict[BaselineDimension, tuple[float, float]] = {
    BaselineDimension.ACCURACY: (0.75, 0.15),
    BaselineDimension.RESPONSE_SECONDS: (22.0, 8.0),
    BaselineDimension.SESSION_SECONDS: (1800.0, 600.0),
    BaselineDimension.CONTENT_DIFFICULTY: (0.5, 0.2),
}


def _prior(*dimensions: BaselineDimension) -> PopulationPriorSet:
    """Build a caller-supplied prior set covering the given dimensions.

    The values are arbitrary but plausible reference statistics. The point of building them
    here is that the engine ships no priors of its own, so a cold start has to be handed one.

    Args:
        *dimensions: The dimensions to cover.

    Returns:
        A synthetic-origin prior set attributed to a named cohort.
    """
    return PopulationPriorSet(
        source="nominal-cohort-v1",
        version="COHORT_V1",
        n_reference=120,
        data_origin=DataOrigin.SYNTHETIC,
        priors=tuple(
            PopulationPrior(
                dimension=dimension,
                centre=_PRIOR_DEFAULTS[dimension][0],
                spread=_PRIOR_DEFAULTS[dimension][1],
                method=StatisticsMethod.ROBUST_WINSORISED_MAD,
            )
            for dimension in dimensions
        ),
    )


def _vector(
    values: Mapping[FeatureName, float | None],
    at: datetime,
    *,
    learner_id: str | None = LEARNER_ID,
    session_id: str = SESSION_ID,
    origins: frozenset[DataOrigin] = frozenset({DataOrigin.REAL}),
) -> FeatureValue:
    """Build a feature vector from a name-to-value mapping.

    A ``None`` becomes an explicit ``INSUFFICIENT_DATA`` entry carrying a reason, which is
    how the feature engine reports absence and therefore how the baseline must be fed.

    Args:
        values: Features present, with ``None`` marking one that could not be computed.
        at: The reference time of the vector.
        learner_id: The learner it belongs to, or ``None`` for an unattributed vector.
        session_id: The session it belongs to.
        origins: The contributing data origins.

    Returns:
        A frozen feature vector.
    """
    entries: list[TypedFeatureValue] = []
    for name, value in values.items():
        if value is None:
            entries.append(
                TypedFeatureValue(
                    name=name,
                    value=INSUFFICIENT_DATA_MARKER,
                    availability=FeatureAvailability.INSUFFICIENT_DATA,
                    reason="the test scenario supplied no evidence for this feature",
                    computed_at=at,
                    feature_set_version=FEATURE_SET_V1,
                    spec_version="1.0.0",
                )
            )
        else:
            entries.append(
                TypedFeatureValue(
                    name=name,
                    value=value,
                    availability=(
                        FeatureAvailability.SYNTHETIC_ONLY
                        if origins == frozenset({DataOrigin.SYNTHETIC})
                        else FeatureAvailability.AVAILABLE
                    ),
                    computed_at=at,
                    feature_set_version=FEATURE_SET_V1,
                    spec_version="1.0.0",
                )
            )
    return FeatureValue(
        feature_set_version=FEATURE_SET_V1,
        computed_at=at,
        values=tuple(entries),
        learner_id=learner_id,
        session_id=session_id,
        origins=origins,
    )


def _fold(
    engine: BaselineEngine,
    profile: BaselineProfile,
    values: Sequence[float],
    source: FeatureName,
    *,
    first: int = 1,
) -> BaselineProfile:
    """Fold a sequence of values for one source feature into a profile.

    Each value gets its own strictly increasing reference time, so a caller cannot
    accidentally exercise the ordering guard while building a scenario.

    Args:
        engine: The engine to fold with.
        profile: The starting profile.
        values: The values to apply, in order.
        source: Which feature supplies them.
        first: Offset of the first observation, in seconds.

    Returns:
        The advanced profile.
    """
    current = profile
    for index, value in enumerate(values):
        current = engine.update(
            current,
            _vector({source: value}, BASE_TIME + timedelta(seconds=first + index)),
        )
    return current


def _profile(engine: BaselineEngine, learner_id: str = LEARNER_ID) -> BaselineProfile:
    """Create an empty profile for a learner.

    Args:
        engine: The engine that will fill it.
        learner_id: The learner to create it for.

    Returns:
        A profile with every dimension present and no observations.
    """
    return engine.create_profile(learner_id, BASE_TIME)


def _response_engine(settings: BaselineSettings = PLAIN) -> BaselineEngine:
    """An engine for response-time scenarios, with no population prior.

    Args:
        settings: Baseline configuration.

    Returns:
        The engine.
    """
    return BaselineEngine(settings)


def _cycling_response(count: int) -> list[float]:
    """A repeating 19/20/21 second pattern, so the baseline has genuine spread.

    Args:
        count: How many observations to produce.

    Returns:
        The response times, in seconds.
    """
    return [19.0 + float(index % 3) for index in range(count)]


# Maturity ladder
# --------------------------------------------------------------------------------------


def test_maturity_follows_the_configured_ladder() -> None:
    assert maturity_for(0, PLAIN) is BaselineMaturity.NEW
    assert maturity_for(19, PLAIN) is BaselineMaturity.NEW
    assert maturity_for(20, PLAIN) is BaselineMaturity.EARLY
    assert maturity_for(99, PLAIN) is BaselineMaturity.EARLY
    assert maturity_for(100, PLAIN) is BaselineMaturity.DEVELOPING
    assert maturity_for(499, PLAIN) is BaselineMaturity.DEVELOPING
    assert maturity_for(500, PLAIN) is BaselineMaturity.ESTABLISHED


def test_maturity_boundaries_move_with_the_settings() -> None:
    assert maturity_for(2, FAST_LADDER) is BaselineMaturity.NEW
    assert maturity_for(3, FAST_LADDER) is BaselineMaturity.EARLY
    assert maturity_for(6, FAST_LADDER) is BaselineMaturity.DEVELOPING
    assert maturity_for(10, FAST_LADDER) is BaselineMaturity.ESTABLISHED


def test_maturity_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        maturity_for(-1, PLAIN)


def test_maturity_rank_orders_the_ladder() -> None:
    ranks = [maturity_rank(level) for level in BaselineMaturity]
    assert ranks == sorted(ranks)
    assert maturity_rank(BaselineMaturity.NEW) < maturity_rank(BaselineMaturity.ESTABLISHED)


def test_basis_is_fixed_by_maturity_so_population_cannot_pose_as_personal() -> None:
    assert basis_for(BaselineMaturity.NEW) is InferenceBasis.POPULATION_PRIOR
    assert basis_for(BaselineMaturity.EARLY) is InferenceBasis.PARTIAL_PERSONAL
    assert basis_for(BaselineMaturity.DEVELOPING) is InferenceBasis.PERSONAL
    assert basis_for(BaselineMaturity.ESTABLISHED) is InferenceBasis.PERSONAL


def test_a_reference_cannot_pair_maturity_with_the_wrong_basis() -> None:
    with pytest.raises(ValidationError, match="does not admit basis"):
        BaselineReference(
            dimension=BaselineDimension.ACCURACY,
            centre=0.8,
            method=StatisticsMethod.ROBUST_WINSORISED_MAD,
            maturity=BaselineMaturity.NEW,
            basis=InferenceBasis.PERSONAL,
            source=ReferenceSource.PERSONAL_HISTORY,
            computed_at=BASE_TIME,
        )


def test_a_learner_walks_the_whole_ladder_in_order() -> None:
    engine = BaselineEngine(FAST_LADDER, _prior(BaselineDimension.RESPONSE_SECONDS))
    profile = _profile(engine)
    dimension = BaselineDimension.RESPONSE_SECONDS

    seen = [engine.reference(profile, dimension).maturity.value]
    for count in range(1, 11):
        profile = _fold(engine, profile, [20.0], RESPONSE, first=count)
        seen.append(engine.reference(profile, dimension).maturity.value)

    assert seen == [
        "new",
        "new",
        "new",
        "early",
        "early",
        "early",
        "developing",
        "developing",
        "developing",
        "developing",
        "established",
    ]


def test_maturity_is_resolved_per_dimension_not_for_the_whole_profile() -> None:
    engine = _response_engine(FAST_LADDER)
    profile = _fold(engine, _profile(engine), [0.80, 0.90, 0.80], ACCURACY)
    accuracies = engine.reference(profile, BaselineDimension.ACCURACY)
    responses = engine.reference(profile, BaselineDimension.RESPONSE_SECONDS)
    assert accuracies.maturity is BaselineMaturity.EARLY
    assert responses.maturity is BaselineMaturity.NEW
    assert profile.observation_count == 3


def test_every_dimension_is_present_in_a_fresh_profile() -> None:
    engine = _response_engine()
    profile = _profile(engine)
    assert tuple(item.dimension for item in profile.statistics) == BASELINE_DIMENSIONS
    assert profile.observation_count == 0
    assert profile.updated_at is None


# Cold start
# --------------------------------------------------------------------------------------


def test_cold_start_returns_a_labelled_population_prior() -> None:
    engine = BaselineEngine(PLAIN, _prior(BaselineDimension.ACCURACY))
    reference = engine.reference(_profile(engine), BaselineDimension.ACCURACY)
    assert reference.centre == pytest.approx(0.75)
    assert reference.spread == pytest.approx(0.15)
    assert reference.maturity is BaselineMaturity.NEW
    assert reference.basis is InferenceBasis.POPULATION_PRIOR
    assert reference.source is ReferenceSource.POPULATION_PRIOR
    assert reference.prior_source == "nominal-cohort-v1"
    assert reference.is_available


def test_cold_start_without_a_prior_is_unavailable_not_zero() -> None:
    engine = _response_engine()
    reference = engine.reference(_profile(engine), BaselineDimension.ACCURACY)
    assert reference.centre is None
    assert reference.source is ReferenceSource.UNAVAILABLE
    assert "no population prior" in (reference.reason or "")
    assert not reference.is_available


def test_cold_start_is_unavailable_when_the_prior_misses_the_dimension() -> None:
    engine = BaselineEngine(PLAIN, _prior(BaselineDimension.ACCURACY))
    reference = engine.reference(_profile(engine), BaselineDimension.RESPONSE_SECONDS)
    assert reference.source is ReferenceSource.UNAVAILABLE
    assert "no population prior" in (reference.reason or "")


def test_a_thin_personal_record_does_not_preempt_the_population_prior() -> None:
    """Fifteen observations is not EARLY. The personal centre is recorded but not used."""
    engine = BaselineEngine(PLAIN, _prior(BaselineDimension.RESPONSE_SECONDS))
    profile = _fold(engine, _profile(engine), _cycling_response(15), RESPONSE)
    item = profile.statistics_for(BaselineDimension.RESPONSE_SECONDS)
    assert item.observations == 15
    assert item.centre is not None

    reference = engine.reference(profile, BaselineDimension.RESPONSE_SECONDS)
    assert reference.maturity is BaselineMaturity.NEW
    assert reference.source is ReferenceSource.POPULATION_PRIOR
    assert reference.centre == pytest.approx(22.0)
    assert reference.centre != pytest.approx(item.centre)


def test_early_maturity_uses_personal_statistics_with_a_partial_basis() -> None:
    engine = _response_engine()
    dimension = BaselineDimension.RESPONSE_SECONDS
    profile = _fold(engine, _profile(engine), _cycling_response(20), RESPONSE)
    reference = engine.reference(profile, dimension)
    assert reference.maturity is BaselineMaturity.EARLY
    assert reference.basis is InferenceBasis.PARTIAL_PERSONAL
    assert reference.source is ReferenceSource.PERSONAL_HISTORY
    assert reference.centre == profile.statistics_for(dimension).centre


def test_developing_and_established_both_use_a_personal_basis() -> None:
    engine = _response_engine(FAST_LADDER)
    profile = _profile(engine)
    expected = {
        3: InferenceBasis.PARTIAL_PERSONAL,
        6: InferenceBasis.PERSONAL,
        10: InferenceBasis.PERSONAL,
    }
    for count in range(1, 11):
        profile = engine.update(
            profile, _vector({RESPONSE: 20.0}, BASE_TIME + timedelta(seconds=count))
        )
        if count not in expected:
            continue
        reference = engine.reference(profile, BaselineDimension.RESPONSE_SECONDS)
        assert reference.basis is expected[count]
        assert reference.source is ReferenceSource.PERSONAL_HISTORY
        assert reference.maturity is maturity_for(count, FAST_LADDER)


# Robust statistics
# --------------------------------------------------------------------------------------


def test_robust_statistics_are_the_default() -> None:
    assert _response_engine().method is StatisticsMethod.ROBUST_WINSORISED_MAD
    assert NON_ROBUST.use_robust_statistics is False
    assert _response_engine(NON_ROBUST).method is StatisticsMethod.MEAN_STDDEV


def test_a_single_observation_has_no_dispersion() -> None:
    assert robust_dispersion(20.0, [20.0]) is None
    assert sample_dispersion(20.0, [20.0]) is None


def test_dispersion_is_unavailable_while_only_one_value_is_retained() -> None:
    engine = _response_engine(BaselineSettings(max_observations_retained=1))
    profile = _fold(engine, _profile(engine), _cycling_response(21), RESPONSE)
    dimension = BaselineDimension.RESPONSE_SECONDS
    reference = engine.reference(profile, dimension)
    assert reference.maturity is BaselineMaturity.EARLY
    assert reference.spread is None

    result = engine.deviation(profile, dimension, 40.0)
    assert result.standardised is None
    assert "fewer than two" in (result.reason or "")


def test_robust_dispersion_is_measured_about_the_supplied_centre() -> None:
    assert robust_dispersion(0.0, [10.0, 10.0, 10.0]) == pytest.approx(10.0 * 1.4826)


def test_robust_dispersion_of_an_identical_window_is_zero() -> None:
    assert robust_dispersion(5.0, [5.0, 5.0, 5.0]) == 0.0


def test_sample_dispersion_uses_a_bessel_corrected_divisor() -> None:
    assert sample_dispersion(0.0, [2.0, -2.0]) == pytest.approx(8**0.5)


def test_an_identical_window_produces_no_standardised_deviation() -> None:
    engine = _response_engine()
    dimension = BaselineDimension.RESPONSE_SECONDS
    profile = _fold(engine, _profile(engine), [20.0] * 20, RESPONSE)
    result = engine.deviation(profile, dimension, 30.0)
    assert result.spread == 0.0
    assert result.standardised is None
    assert "identical" in (result.reason or "")


# Retention
# --------------------------------------------------------------------------------------


def test_decimate_keeps_a_window_that_already_fits() -> None:
    assert decimate((1.0, 2.0, 3.0), 5) == (1.0, 2.0, 3.0)
    assert decimate((1.0, 2.0, 3.0), 3) == (1.0, 2.0, 3.0)
    assert decimate((), 4) == ()


def test_decimate_reduces_an_over_long_window_to_the_limit() -> None:
    window = tuple(float(index) for index in range(20))
    for limit in (1, 2, 3, 4, 5, 8, 19):
        reduced = decimate(window, limit)
        assert len(reduced) <= limit, f"limit {limit} was exceeded"
        assert reduced[-1] == window[-1], "the newest value was dropped"


def test_decimate_rejects_a_limit_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        decimate((1.0,), 0)


def test_the_retained_window_is_bounded_by_the_settings() -> None:
    engine = _response_engine(BaselineSettings(max_observations_retained=8))
    profile = _fold(engine, _profile(engine), _cycling_response(30), RESPONSE)
    item = profile.statistics_for(BaselineDimension.RESPONSE_SECONDS)
    assert len(item.retained) <= 8
    assert item.observations == 30
    assert item.retained[-1] == 21.0


# Outliers
# --------------------------------------------------------------------------------------


def _settled(engine: BaselineEngine) -> BaselineProfile:
    """A profile with 21 ordinary response times, past the EARLY threshold.

    Args:
        engine: The engine to fold with.

    Returns:
        A profile whose spread reflects ordinary variation only.
    """
    return _fold(engine, _profile(engine), _cycling_response(21), RESPONSE)


def test_an_outlier_does_not_inflate_the_robust_tolerance() -> None:
    engine = _response_engine()
    dimension = BaselineDimension.RESPONSE_SECONDS
    before = _settled(engine).statistics_for(dimension)
    assert before.spread is not None
    assert before.spread < 3.0

    after = engine.update(
        _settled(engine),
        _vector({RESPONSE: 3600.0}, BASE_TIME + timedelta(seconds=100)),
    ).statistics_for(dimension)
    assert after.spread is not None
    assert after.spread < 3.0, "one 3600-second value inflated the tolerance"
    assert after.spread < 3.0 * before.spread


def test_an_outlier_cannot_redefine_the_baseline_centre() -> None:
    engine = _response_engine()
    dimension = BaselineDimension.RESPONSE_SECONDS
    before = _settled(engine).statistics_for(dimension).centre
    assert before is not None

    after = (
        engine.update(
            _settled(engine),
            _vector({RESPONSE: 3600.0}, BASE_TIME + timedelta(seconds=100)),
        )
        .statistics_for(dimension)
        .centre
    )
    assert after is not None
    assert abs(after - before) < 1.0, "a single extreme value relocated the centre"


def test_the_non_robust_estimator_fails_the_same_outlier_test() -> None:
    """The comparison that gives the robustness claim its meaning."""
    engine = _response_engine(NON_ROBUST)
    dimension = BaselineDimension.RESPONSE_SECONDS
    before = _settled(engine).statistics_for(dimension)
    assert before.centre is not None
    assert before.spread is not None

    after = engine.update(
        _settled(engine),
        _vector({RESPONSE: 3600.0}, BASE_TIME + timedelta(seconds=100)),
    ).statistics_for(dimension)
    assert after.centre is not None
    assert after.spread is not None
    assert abs(after.centre - before.centre) > 100.0
    assert after.spread > 50.0 * before.spread


def test_a_real_change_after_an_outlier_is_visible_and_not_masked() -> None:
    """A sustained change is large in dispersion units when robust, tiny when not."""

    def _then_change(engine: BaselineEngine) -> tuple[float, float]:
        profile = engine.update(
            _settled(engine),
            _vector({RESPONSE: 3600.0}, BASE_TIME + timedelta(seconds=100)),
        )
        profile = _fold(engine, profile, [45.0, 45.0, 45.0], RESPONSE, first=200)
        result = engine.deviation(profile, BaselineDimension.RESPONSE_SECONDS, 45.0)
        assert result.standardised is not None
        assert result.absolute is not None
        return result.standardised, result.absolute

    robust_z, robust_absolute = _then_change(_response_engine())
    plain_z, plain_absolute = _then_change(_response_engine(NON_ROBUST))

    assert abs(robust_z) > 10.0, "a sustained 25-second shift was not visible"
    assert abs(robust_z) > 10.0 * abs(plain_z), "the robust estimate masked nothing more"
    assert abs(robust_absolute) < abs(plain_absolute), (
        "the non-robust centre should have been dragged further by the outlier, so its "
        "absolute deviation from 45.0 seconds is the larger of the two"
    )


def test_a_genuine_sustained_shift_still_moves_the_centre() -> None:
    """Bounding the step must not freeze the baseline against a real change."""
    engine = _response_engine()
    dimension = BaselineDimension.RESPONSE_SECONDS
    before = _settled(engine).statistics_for(dimension).centre
    assert before is not None

    after = (
        _fold(engine, _settled(engine), [60.0] * 20, RESPONSE, first=200)
        .statistics_for(dimension)
        .centre
    )
    assert after is not None
    assert after > 30.0, "a sustained 40-second shift did not move the centre"
    assert after - before > 5.0


def test_the_winsorisation_limit_is_configurable_and_validated() -> None:
    assert BaselineSettings(winsorisation_limit_spreads=1.0).winsorisation_limit_spreads == 1.0
    with pytest.raises(ValueError, match="must be positive"):
        BaselineSettings(winsorisation_limit_spreads=0.0)


# Deviation reporting
# --------------------------------------------------------------------------------------


def test_a_deviation_carries_the_reference_and_its_standing() -> None:
    engine = _response_engine()
    profile = _settled(engine)
    result = engine.deviation(profile, BaselineDimension.RESPONSE_SECONDS, 40.0)
    assert result.absolute is not None
    assert result.standardised is not None
    assert result.maturity is BaselineMaturity.EARLY
    assert result.basis is InferenceBasis.PARTIAL_PERSONAL
    assert result.source is ReferenceSource.PERSONAL_HISTORY
    assert result.method is StatisticsMethod.ROBUST_WINSORISED_MAD
    assert result.computed_at == profile.reference_time
    assert result.reason is None


def test_a_deviation_reports_distance_without_classifying_it() -> None:
    """Classification belongs to the temporal engine, not to this layer."""
    engine = _response_engine()
    result = engine.deviation(_settled(engine), BaselineDimension.RESPONSE_SECONDS, 400.0)
    assert not hasattr(result, "state")
    assert not hasattr(result, "severity")
    summary = result.to_summary_dict()
    assert "state" not in summary
    assert "severity" not in summary
    assert "standardised" in summary


def test_a_deviation_against_an_unavailable_reference_carries_a_reason() -> None:
    engine = _response_engine()
    result = engine.deviation(_profile(engine), BaselineDimension.RESPONSE_SECONDS, 40.0)
    assert result.standardised is None
    assert result.absolute is None
    assert result.source is ReferenceSource.UNAVAILABLE
    assert result.reason is not None


def test_a_deviation_refuses_an_out_of_range_measurement() -> None:
    engine = _response_engine()
    profile = _settled(engine)
    with pytest.raises(ValueError, match="not been clamped"):
        engine.deviation(profile, BaselineDimension.ACCURACY, 1.5)
    with pytest.raises(ValueError, match="not been clamped"):
        engine.deviation(profile, BaselineDimension.RESPONSE_SECONDS, -1.0)


# Update contract
# --------------------------------------------------------------------------------------


def test_an_update_does_not_mutate_the_input_profile() -> None:
    engine = _response_engine()
    original = _profile(engine)
    snapshot = original.model_dump()
    _fold(engine, original, [20.0, 20.0], RESPONSE)
    assert original.model_dump() == snapshot


def test_update_refuses_a_vector_with_no_learner() -> None:
    engine = _response_engine()
    with pytest.raises(BaselineUpdateError, match="no learner_id"):
        engine.update(
            _profile(engine),
            _vector({RESPONSE: 20.0}, BASE_TIME + timedelta(seconds=1), learner_id=None),
        )


def test_update_refuses_a_vector_for_another_learner() -> None:
    engine = _response_engine()
    with pytest.raises(BaselineUpdateError, match="never populated from"):
        engine.update(
            _profile(engine),
            _vector(
                {RESPONSE: 20.0},
                BASE_TIME + timedelta(seconds=1),
                learner_id=OTHER_LEARNER_ID,
                session_id=OTHER_SESSION_ID,
            ),
        )


def test_update_refuses_an_observation_that_is_not_newer() -> None:
    engine = _response_engine()
    profile = _fold(engine, _profile(engine), [20.0, 20.0], RESPONSE)
    for offset in (1, 2):
        with pytest.raises(BaselineUpdateError, match="not later than"):
            engine.update(
                profile,
                _vector({RESPONSE: 20.0}, BASE_TIME + timedelta(seconds=offset)),
            )


def test_an_insufficient_feature_is_absent_rather_than_a_rejection() -> None:
    engine = _response_engine()
    updated = engine.update(
        _profile(engine), _vector({RESPONSE: None}, BASE_TIME + timedelta(seconds=1))
    )
    item = updated.statistics_for(BaselineDimension.RESPONSE_SECONDS)
    assert item.observations == 0
    assert item.rejected == 0
    assert updated.updated_at == BASE_TIME + timedelta(seconds=1)


def test_an_out_of_range_observation_is_rejected_not_clamped() -> None:
    engine = _response_engine()
    updated = engine.update(
        _profile(engine), _vector({RESPONSE: -5.0}, BASE_TIME + timedelta(seconds=1))
    )
    item = updated.statistics_for(BaselineDimension.RESPONSE_SECONDS)
    assert item.observations == 0
    assert item.rejected == 1
    assert "not clamped" in (item.last_rejection_reason or "")
    assert item.centre is None


def test_update_many_folds_in_order() -> None:
    engine = _response_engine()
    vectors = [
        _vector({RESPONSE: 20.0}, BASE_TIME + timedelta(seconds=index + 1)) for index in range(3)
    ]
    profile = engine.update_many(_profile(engine), vectors)
    assert profile.statistics_for(BaselineDimension.RESPONSE_SECONDS).observations == 3
    assert profile.updated_at == BASE_TIME + timedelta(seconds=3)


# Provenance, version, and determinism
# --------------------------------------------------------------------------------------


def test_a_profile_records_the_settings_fingerprint_that_shaped_it() -> None:
    engine = _response_engine()
    profile = _profile(engine)
    assert profile.settings_fingerprint == engine.settings_fingerprint()
    assert len(profile.settings_fingerprint) == DIGEST_SIZE_BYTES * 2
    int(profile.settings_fingerprint, 16)


def test_the_settings_fingerprint_distinguishes_different_settings() -> None:
    assert (
        _response_engine().settings_fingerprint()
        != _response_engine(NON_ROBUST).settings_fingerprint()
    )


def test_synthetic_origins_are_recorded_on_the_profile() -> None:
    engine = _response_engine()
    profile = _profile(engine)
    assert not profile.is_synthetic_only
    updated = engine.update(
        profile,
        _vector(
            {RESPONSE: 20.0},
            BASE_TIME + timedelta(seconds=1),
            origins=frozenset({DataOrigin.SYNTHETIC}),
        ),
    )
    assert updated.origins == frozenset({DataOrigin.SYNTHETIC})
    assert updated.is_synthetic_only


def test_the_baseline_version_is_recorded_on_the_profile() -> None:
    assert _profile(_response_engine()).baseline_version == "PERSONAL_BASELINE_V1"


def test_replaying_the_same_observations_gives_an_identical_profile() -> None:
    engine = _response_engine()

    def _run() -> BaselineProfile:
        return _fold(engine, _profile(engine), _cycling_response(25) + [33.0, 19.0], RESPONSE)

    assert _run().model_dump() == _run().model_dump()


def test_the_same_input_gives_the_same_deviation() -> None:
    engine = _response_engine()
    profile = _fold(engine, _profile(engine), _cycling_response(25), RESPONSE)
    first = engine.deviation(profile, BaselineDimension.RESPONSE_SECONDS, 27.5)
    second = engine.deviation(profile, BaselineDimension.RESPONSE_SECONDS, 27.5)
    assert first == second


# Model invariants
# --------------------------------------------------------------------------------------


def test_a_profile_must_cover_every_dimension_in_canonical_order() -> None:
    engine = _response_engine()
    dumped = _profile(engine).model_dump()
    with pytest.raises(ValidationError, match="canonical order"):
        BaselineProfile.model_validate({**dumped, "statistics": dumped["statistics"][:2]})


def test_a_population_prior_must_have_a_positive_spread() -> None:
    with pytest.raises(ValidationError, match="strictly"):
        PopulationPrior(
            dimension=BaselineDimension.ACCURACY,
            centre=0.7,
            spread=0.0,
            method=StatisticsMethod.ROBUST_WINSORISED_MAD,
        )


def test_a_population_prior_must_have_an_in_range_centre() -> None:
    with pytest.raises(ValidationError, match="outside"):
        PopulationPrior(
            dimension=BaselineDimension.ACCURACY,
            centre=1.4,
            spread=0.1,
            method=StatisticsMethod.ROBUST_WINSORISED_MAD,
        )


def test_a_prior_set_must_not_repeat_a_dimension() -> None:
    prior = PopulationPrior(
        dimension=BaselineDimension.ACCURACY,
        centre=0.7,
        spread=0.1,
        method=StatisticsMethod.ROBUST_WINSORISED_MAD,
    )
    with pytest.raises(ValidationError, match="repeats dimensions"):
        PopulationPriorSet(
            source="cohort",
            version="COHORT_V1",
            n_reference=10,
            data_origin=DataOrigin.REAL,
            priors=(prior, prior),
        )


def test_an_unavailable_reference_must_state_why() -> None:
    with pytest.raises(ValidationError, match="must state why"):
        BaselineReference(
            dimension=BaselineDimension.ACCURACY,
            maturity=BaselineMaturity.NEW,
            basis=InferenceBasis.POPULATION_PRIOR,
            source=ReferenceSource.UNAVAILABLE,
            computed_at=BASE_TIME,
        )


def test_an_available_reference_must_carry_a_centre() -> None:
    with pytest.raises(ValidationError, match="must carry"):
        BaselineReference(
            dimension=BaselineDimension.ACCURACY,
            maturity=BaselineMaturity.EARLY,
            basis=InferenceBasis.PARTIAL_PERSONAL,
            source=ReferenceSource.PERSONAL_HISTORY,
            computed_at=BASE_TIME,
        )


# Dependency direction
# --------------------------------------------------------------------------------------


def test_the_baseline_layer_imports_no_later_layer() -> None:
    """A static check, because the boundary is easier to break than to notice."""
    allowed = {
        "focus_engine.baseline",
        "focus_engine.configuration",
        "focus_engine.features",
        "focus_engine.schemas",
        "focus_engine.utils",
    }
    import focus_engine.baseline as package

    root = package.__file__
    assert root is not None
    for source in sorted(Path(root).parent.glob("*.py")):
        for match in re.finditer(
            r"^from (focus_engine[\w.]*)", source.read_text(encoding="utf-8"), re.MULTILINE
        ):
            module = match.group(1)
            assert any(module == base or module.startswith(f"{base}.") for base in allowed), (
                f"{source.name} imports from {module}, which is not a permitted dependency"
            )
