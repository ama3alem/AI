"""End-to-end test: simulated events to a labelled personal baseline.

The unit suite in ``tests/unit/test_baseline.py`` feeds the engine hand-built feature
vectors, which proves the baseline's arithmetic but not that it accepts what the feature
engine actually produces. The two are separate contracts: the baseline reads a named
feature out of a vector, and nothing guarantees that name stays present, stays numeric, or
arrives attributed to a learner.

This test closes that gap with the real pipeline — simulator to context to features to
baseline — and checks the properties that only appear when the layers are composed:

* the feature names the baseline reads are real feature-engine outputs;
* the vector's ``computed_at`` is strictly increasing, because the baseline refuses
  anything else and would reject the whole run;
* provenance survives the seam, so a baseline built from simulated events is marked
  synthetic rather than looking observed;
* maturity flows back into the context, which is what finally makes
  ``baseline_maturity_encoded`` a measurement rather than a declared absence.

Two timing facts make this slower than it looks, and both are worth stating because they
are easy to get wrong when reading a baseline count. The feature engine emits **one
vector per session**, not per question, so N sessions is N observations per dimension
rather than N times the question count. And the default ladder needs 20 observations
before a personal value is admissible at all, which is 20 sessions. The maturity tests
therefore use a deliberately fast ladder, and the default ladder's own conservatism is
asserted separately rather than assumed.

Everything here is synthetic. A baseline fitted to a simulator is fitted to that
simulator's assumptions and establishes nothing about a real learner.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from focus_engine.baseline import (
    BaselineDimension,
    BaselineEngine,
    BaselineMaturity,
    PopulationPrior,
    PopulationPriorSet,
    ReferenceSource,
    StatisticsMethod,
    maturity_for,
)
from focus_engine.configuration.thresholds import BaselineSettings
from focus_engine.context import ContextEngine, ContextKey, ContextModel
from focus_engine.events import EventEnvelope
from focus_engine.features import (
    FeatureAvailability,
    FeatureName,
    FeatureValue,
    compute_features,
)
from focus_engine.schemas.primitives import DataOrigin, InferenceBasis
from focus_engine.simulator import (
    SYNTHETIC_WARNING,
    LearnerArchetype,
    SimulationConfig,
    generate_session,
)
from focus_engine.utils.clock import FixedClock

pytestmark = pytest.mark.integration

LEARNER_ID = "learner-integration-0001"
SESSIONS = 4
QUESTIONS_PER_SESSION = 12

#: A ladder that reaches ``ESTABLISHED`` within the session count above. The thresholds
#: are not the subject of these tests; the wiring is, and a real ladder would need twenty
#: simulated sessions to show anything.
FAST_LADDER = BaselineSettings(
    min_samples_new=0,
    min_samples_early=2,
    min_samples_developing=3,
    min_samples_established=4,
)

_START = datetime(2026, 9, 27, 9, 0, 0, tzinfo=UTC)
_STRIDE = timedelta(hours=1)

#: The dimensions the feature engine actually supplies, and the only ones a baseline
#: dimension may be fed from.
_SOURCE_DIMENSIONS = (
    (BaselineDimension.ACCURACY, FeatureName.PERFORMANCE_RECENT_ACCURACY),
    (BaselineDimension.RESPONSE_SECONDS, FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS),
)


def _session_events(archetype: LearnerArchetype, index: int) -> tuple[EventEnvelope, ...]:
    """Generate one synthetic session.

    Args:
        archetype: Behavioural pattern to simulate.
        index: Zero-based session index, used to separate seeds and clock positions.

    Returns:
        The session's events in temporal order.
    """
    result = generate_session(
        SimulationConfig(
            archetype=archetype,
            learner_id=LEARNER_ID,
            session_id=f"session-{archetype.value}-{index:04d}",
            base_seed=20260927 + index,
            question_count=QUESTIONS_PER_SESSION,
            start_time=_START + index * _STRIDE,
        )
    )
    assert SYNTHETIC_WARNING in result.provenance.warning
    return result.events


def _context_for(events: tuple[EventEnvelope, ...]) -> ContextModel:
    """Derive context from a session, with the clock pinned to its last event.

    Args:
        events: The session's events.

    Returns:
        The derived context.
    """
    return ContextEngine(clock=FixedClock(events[-1].timestamp)).process(events)


def _vector_stream(archetype: LearnerArchetype, sessions: int) -> tuple[FeatureValue, ...]:
    """Run the pipeline over several sessions and collect the feature vectors.

    A separate context engine is constructed per session because each simulated session
    has its own identifier, while the baseline is deliberately carried across them: a
    learner's normal behaviour is defined over many sessions, not one.

    Args:
        archetype: Behavioural pattern to simulate.
        sessions: How many sessions to run.

    Returns:
        One feature vector per session, in temporal order.
    """
    return tuple(
        compute_features(events, _context_for(events))
        for events in (_session_events(archetype, index) for index in range(sessions))
    )


def _prior() -> PopulationPriorSet:
    """A synthetic-labelled prior set, so a cold start is exercised honestly.

    Returns:
        A prior set covering the accuracy and response-time dimensions.
    """
    return PopulationPriorSet(
        source="simulator-cohort-v1",
        version="COHORT_V1",
        n_reference=50,
        data_origin=DataOrigin.SYNTHETIC,
        priors=(
            PopulationPrior(
                dimension=BaselineDimension.ACCURACY,
                centre=0.7,
                spread=0.15,
                method=StatisticsMethod.ROBUST_WINSORISED_MAD,
            ),
            PopulationPrior(
                dimension=BaselineDimension.RESPONSE_SECONDS,
                centre=20.0,
                spread=6.0,
                method=StatisticsMethod.ROBUST_WINSORISED_MAD,
            ),
        ),
    )


def test_a_simulated_stream_builds_a_labelled_personal_baseline() -> None:
    """The whole pipeline, and the provenance claim it has to keep making."""
    engine = BaselineEngine(FAST_LADDER, _prior())
    profile = engine.create_profile(LEARNER_ID, _START)

    for vector in _vector_stream(LearnerArchetype.STABLE, SESSIONS):
        profile = engine.update(profile, vector)

    assert profile.observation_count > 0
    assert profile.is_synthetic_only, (
        "a baseline built only from simulated events must be marked synthetic, or a "
        "simulator-derived number becomes indistinguishable from an observed one"
    )
    assert profile.settings_fingerprint == engine.settings_fingerprint()
    assert profile.prior_source == "simulator-cohort-v1"
    assert profile.baseline_version == "PERSONAL_BASELINE_V1"

    for dimension, _source in _SOURCE_DIMENSIONS:
        reference = engine.reference(profile, dimension)
        expected = maturity_for(profile.statistics_for(dimension).observations, FAST_LADDER)
        assert reference.maturity is expected
        assert reference.centre is not None
        assert reference.spread is not None
        assert reference.basis is not InferenceBasis.POPULATION_PRIOR, (
            f"{dimension.value} has {expected.value} observations and must no longer be "
            "answered from the population prior"
        )
        assert reference.source is ReferenceSource.PERSONAL_HISTORY


def test_the_default_ladder_stays_at_cold_start_for_a_short_run() -> None:
    """The shipped thresholds, not just the fast test ladder.

    Four sessions is four observations, and the default ladder needs twenty before a
    personal value is admissible. Asserting it keeps a future edit to the thresholds from
    silently turning a short run into a personal inference.
    """
    engine = BaselineEngine(prior=_prior())
    profile = engine.create_profile(LEARNER_ID, _START)
    for vector in _vector_stream(LearnerArchetype.STABLE, SESSIONS):
        profile = engine.update(profile, vector)

    assert profile.statistics_for(BaselineDimension.ACCURACY).observations == SESSIONS
    reference = engine.reference(profile, BaselineDimension.ACCURACY)
    assert reference.maturity is BaselineMaturity.NEW
    assert reference.basis is InferenceBasis.POPULATION_PRIOR
    assert reference.source is ReferenceSource.POPULATION_PRIOR
    assert reference.centre == pytest.approx(0.7)
    assert reference.observations == SESSIONS


def test_the_feature_engine_actually_supplies_the_baseline_source_features() -> None:
    """The seam the unit suite cannot check: are the source features ever computed?"""
    engine = BaselineEngine(FAST_LADDER, _prior())
    profile = engine.create_profile(LEARNER_ID, _START)
    vector = _vector_stream(LearnerArchetype.STABLE, 1)[0]

    names = {entry.name for entry in vector.values}
    for _dimension, source in _SOURCE_DIMENSIONS:
        assert source in names, f"{source.value} is absent from the real feature vector"

    profile = engine.update(profile, vector)
    for dimension, _source in _SOURCE_DIMENSIONS:
        item = profile.statistics_for(dimension)
        assert item.observations == 1
        assert item.rejected == 0, "a real feature must not be counted as a defect"
        assert item.centre is not None
        assert item.spread is None, "one observation has no dispersion"


def test_an_absent_source_feature_is_absent_rather_than_rejected() -> None:
    """A vector missing the accuracy entry must not be counted as a data defect."""
    engine = BaselineEngine(FAST_LADDER, _prior())
    profile = engine.create_profile(LEARNER_ID, _START)
    vector = _vector_stream(LearnerArchetype.STABLE, 1)[0]

    trimmed = FeatureValue(
        feature_set_version=vector.feature_set_version,
        computed_at=vector.computed_at,
        values=tuple(
            entry
            for entry in vector.values
            if entry.name is not FeatureName.PERFORMANCE_RECENT_ACCURACY
        ),
        learner_id=vector.learner_id,
        session_id=vector.session_id,
        origins=vector.origins,
    )
    profile = engine.update(profile, trimmed)

    accuracy = profile.statistics_for(BaselineDimension.ACCURACY)
    assert accuracy.observations == 0
    assert accuracy.rejected == 0, "an absent feature is not a defect"
    assert profile.observation_count > 0, "the other dimensions were still folded in"


def test_cold_start_is_not_promoted_by_a_single_observation() -> None:
    """One real vector must not move a cold start off the population prior."""
    engine = BaselineEngine(FAST_LADDER, _prior())
    profile = engine.create_profile(LEARNER_ID, _START)

    cold = engine.reference(profile, BaselineDimension.ACCURACY)
    assert cold.maturity is BaselineMaturity.NEW
    assert cold.basis is InferenceBasis.POPULATION_PRIOR
    assert cold.centre == pytest.approx(0.7)

    warmed = engine.update(profile, _vector_stream(LearnerArchetype.STABLE, 1)[0])
    still_cold = engine.reference(warmed, BaselineDimension.ACCURACY)
    assert still_cold.maturity is BaselineMaturity.NEW
    assert still_cold.source is ReferenceSource.POPULATION_PRIOR
    assert still_cold.observations == 1
    assert still_cold.centre == pytest.approx(0.7), (
        "the prior centre must not be replaced by one observation's value"
    )
    assert warmed.statistics_for(BaselineDimension.ACCURACY).centre != pytest.approx(0.7), (
        "the personal centre is recorded even while it is not yet admissible for "
        "inference, so it is available without being used"
    )


def test_baseline_maturity_flows_back_into_the_context_and_features() -> None:
    """The Phase 5 placeholder becomes a measurement once the baseline exists.

    ``baseline_maturity_encoded`` was hard-coded to report insufficiency because no
    engine could produce a maturity. Now that one can, the value has to travel baseline to
    context to feature, and the context must stop declaring the blocking gap.

    The context is rebuilt through ``model_validate`` rather than
    ``model_copy(update=...)`` on purpose: ``missing`` is *derived* by an ``after``
    validator, and ``model_copy`` skips validators, which would leave a closed gap
    recorded as open. That is the same reason the baseline engine revalidates on every
    update instead of copying.
    """
    engine = BaselineEngine(FAST_LADDER, _prior())
    profile = engine.create_profile(LEARNER_ID, _START)
    for vector in _vector_stream(LearnerArchetype.STABLE, SESSIONS):
        profile = engine.update(profile, vector)

    reference = engine.reference(profile, BaselineDimension.ACCURACY)
    assert reference.centre is not None
    assert reference.maturity is not BaselineMaturity.NEW

    events = _session_events(LearnerArchetype.STABLE, 99)
    derived = _context_for(events)
    assert ContextKey.BASELINE_MATURITY in derived.missing, (
        "no maturity has been reported yet, so the blocking gap is correctly open"
    )

    enriched = ContextModel.model_validate(
        {**derived.model_dump(), "baseline_maturity": reference.maturity.value}
    )
    assert ContextKey.BASELINE_MATURITY not in enriched.missing, (
        "reporting a maturity must close the blocking gap it was blocking"
    )

    recomputed = compute_features(events, enriched)
    encoded = next(
        entry for entry in recomputed.values if entry.name is FeatureName.BASELINE_MATURITY_ENCODED
    )
    assert encoded.availability is FeatureAvailability.SYNTHETIC_ONLY, (
        "the vector's origins are purely synthetic, so no feature may claim to be observed"
    )
    assert encoded.availability is not FeatureAvailability.AVAILABLE
    assert encoded.value == reference.maturity.value


def test_replaying_the_whole_pipeline_reproduces_the_profile() -> None:
    """Determinism across layers, not just inside the baseline."""
    first = BaselineEngine(FAST_LADDER, _prior())
    profile = first.create_profile(LEARNER_ID, _START)
    for vector in _vector_stream(LearnerArchetype.NOISY, SESSIONS):
        profile = first.update(profile, vector)

    second = BaselineEngine(FAST_LADDER, _prior())
    replayed = second.create_profile(LEARNER_ID, _START)
    for vector in _vector_stream(LearnerArchetype.NOISY, SESSIONS):
        replayed = second.update(replayed, vector)

    assert profile.model_dump() == replayed.model_dump()
    assert first.settings_fingerprint() == second.settings_fingerprint()
