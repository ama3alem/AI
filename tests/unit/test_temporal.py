"""Tests for the temporal state engine (Phase 7).

The phase exists to separate a moment from a change, so the suite is organised around the
places where that separation can collapse:

* **Absence must stay absence.** A learner with nothing recorded, or with too little to
  interpret, reports ``INSUFFICIENT_DATA`` with a stated reason. A single measurement that
  happens to sit on the baseline is not evidence of stability, and a test asserts that the
  engine refuses to call it that.
* **One anomalous observation must never be a trajectory.** The single-observation case is
  driven explicitly at, below, and above the persistence boundary, and the model's own
  validator is exercised separately, so a trajectory cannot be fabricated by constructing
  one directly.
* **Unmeasurable observations must be retained, not dropped.** A learner whose baseline is
  still a population prior has real measurements that cannot be standardised. Discarding
  them would erase the difference between a learner nobody measured and one who was
  measured against the wrong reference.
* **Provenance and attribution must not be reconstructible by accident.** Time must move
  forwards, deviations must belong to the learner whose state they advance, and origin must
  be stated rather than defaulted.

Every test drives a :class:`~focus_engine.utils.clock.FixedClock`. Nothing here reads the
host clock, so the suite is deterministic and a failure is a real disagreement about
behaviour rather than a timing artefact.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import pstdev

import pytest
from pydantic import ValidationError

from focus_engine.baseline import (
    MAD_NORMAL_SCALE,
    BaselineDimension,
    BaselineMaturity,
    DeviationResult,
    ReferenceSource,
    StatisticsMethod,
    basis_for,
)
from focus_engine.configuration.thresholds import TemporalSettings
from focus_engine.schemas.primitives import DataOrigin, InferenceBasis
from focus_engine.temporal import (
    TEMPORAL_STATE_V1,
    TRACKED_DIMENSIONS,
    BehavioralEngagementState,
    ChangeKind,
    DeviationPoint,
    StateEvidence,
    TemporalEngine,
    TemporalObservation,
    TemporalState,
    TemporalStateError,
    TemporalTrack,
    TrajectoryChange,
    TrendCharacter,
    TrendDirection,
    change_per_observation,
    current_sign_run,
    deviation_dispersion,
    is_sustained,
    mean_standardised,
    signed_direction,
    weakest_maturity,
)
from focus_engine.utils.clock import FixedClock

pytestmark = pytest.mark.unit

BASE_TIME = datetime(2026, 9, 27, 9, 0, 0, tzinfo=UTC)
LEARNER_ID = "learner-0001"
OTHER_LEARNER_ID = "learner-0002"
ACCURACY = BaselineDimension.ACCURACY
RESPONSE = BaselineDimension.RESPONSE_SECONDS

STEP = timedelta(minutes=5)

PLAIN = TemporalSettings()
SYNTHETIC = frozenset({DataOrigin.SYNTHETIC})
ESTABLISHED = BaselineMaturity.ESTABLISHED


def _engine(settings: TemporalSettings = PLAIN) -> TemporalEngine:
    return TemporalEngine(settings=settings, clock=FixedClock(BASE_TIME))


def _at(offset: int) -> datetime:
    return BASE_TIME + STEP * offset


def _deviation(
    standardised: float | None,
    at: datetime,
    dimension: BaselineDimension = ACCURACY,
    maturity: BaselineMaturity | None = None,
) -> DeviationResult:
    """Build one deviation, standing in for what the baseline layer would produce.

    A ``standardised`` of ``None`` reproduces the baseline layer's genuine
    unmeasurable case: a dimension whose retained values are too uniform to estimate a
    dispersion from, so no comparison against the personal centre is possible.
    """
    measurable = standardised is not None
    resolved = (
        maturity
        if maturity is not None
        else (ESTABLISHED if measurable else BaselineMaturity.EARLY)
    )
    return DeviationResult(
        dimension=dimension,
        value=0.8,
        centre=0.75 if measurable else None,
        spread=0.15 if measurable else None,
        absolute=abs(standardised) if measurable else None,
        standardised=standardised,
        method=StatisticsMethod.ROBUST_WINSORISED_MAD if measurable else None,
        maturity=resolved,
        basis=basis_for(resolved),
        source=(ReferenceSource.PERSONAL_HISTORY if measurable else ReferenceSource.UNAVAILABLE),
        computed_at=at,
        reason=None if measurable else "the retained values are too uniform to estimate a spread",
    )


def _observation(
    standardised: float | None,
    offset: int,
    learner_id: str = LEARNER_ID,
    dimension: BaselineDimension = ACCURACY,
    maturity: BaselineMaturity | None = None,
    origins: frozenset[DataOrigin] = SYNTHETIC,
) -> TemporalObservation:
    return TemporalObservation(
        learner_id=learner_id,
        computed_at=_at(offset),
        deviations=(_deviation(standardised, _at(offset), dimension, maturity),),
        origins=origins,
    )


def _fold(
    engine: TemporalEngine,
    values: Sequence[float | None],
    learner_id: str = LEARNER_ID,
    maturity: BaselineMaturity | None = None,
) -> TemporalState:
    """Feed a series of standardised deviations through the engine."""
    state = engine.create_state(learner_id, BASE_TIME)
    for offset, value in enumerate(values, start=1):
        state = engine.ingest(state, _observation(value, offset, learner_id, maturity=maturity))
    return state


def _fold_both(
    engine: TemporalEngine,
    values: Sequence[float | None],
    learner_id: str = LEARNER_ID,
    maturity: BaselineMaturity | None = None,
) -> TemporalState:
    """Feed each value to every tracked dimension in one observation per instant.

    Maturity is taken as the weakest link across every contributing track, so a state that
    only ever sees one dimension stays as fresh as the others it never saw. This helper
    exists so a test can exercise a state whose maturity is genuinely established.
    """
    state = engine.create_state(learner_id, BASE_TIME)
    for offset, value in enumerate(values, start=1):
        state = engine.ingest(
            state,
            TemporalObservation(
                learner_id=learner_id,
                computed_at=_at(offset),
                deviations=tuple(
                    _deviation(value, _at(offset), dimension, maturity)
                    for dimension in TRACKED_DIMENSIONS
                ),
                origins=SYNTHETIC,
            ),
        )
    return state


def _trajectory(first: float, last: float, direction: TrendDirection) -> TrajectoryChange:
    return TrajectoryChange(
        dimension=ACCURACY,
        direction=direction,
        start=_at(1),
        end=_at(3),
        observations=3,
        first_standardised=first,
        last_standardised=last,
        mean_standardised=(first + last) / 2,
        peak_absolute=max(abs(first), abs(last)),
        rate_per_observation=(last - first) / 2,
        duration_seconds=600.0,
        maturity=ESTABLISHED,
        basis=basis_for(ESTABLISHED),
    )


def test_a_fresh_state_is_insufficient_data_and_says_why() -> None:
    state = _engine().create_state(LEARNER_ID, BASE_TIME)
    assert state.state is BehavioralEngagementState.INSUFFICIENT_DATA
    assert state.change is ChangeKind.NONE
    assert state.direction is TrendDirection.INSUFFICIENT_DATA
    assert state.evidence.reason
    assert state.observations == 0


def test_a_single_measurement_is_not_stability() -> None:
    state = _fold(_engine(), [0.0])
    assert state.state is BehavioralEngagementState.INSUFFICIENT_DATA
    assert state.evidence.consecutive == 0
    assert "1 measurable deviation" in (state.evidence.reason or "")


def test_the_state_minimum_is_configurable() -> None:
    permissive = _engine(TemporalSettings(state_min_observations=1))
    assert _fold(permissive, [0.0]).state is BehavioralEngagementState.STABLE
    assert _fold(_engine(), [0.0]).state is BehavioralEngagementState.INSUFFICIENT_DATA


def test_measurements_without_a_comparable_baseline_are_retained() -> None:
    state = _fold(_engine(), [None, None, None])
    assert state.observations == 3
    assert state.measurable_observations == 0
    assert state.state is BehavioralEngagementState.INSUFFICIENT_DATA
    assert state.evidence.observations == 0
    assert state.track_for(ACCURACY).points[0].reason


def test_a_learner_measurable_only_late_still_gets_a_state() -> None:
    state = _fold(_engine(), [None, None, 0.0, 0.0])
    assert state.observations == 4
    assert state.measurable_observations == 2
    assert state.state is BehavioralEngagementState.STABLE
    assert state.evidence.observations == 2


# A single anomalous observation is not a trajectory
# ---------------------------------------------------------------------------


def test_one_large_excursion_is_an_anomaly_and_not_a_trajectory() -> None:
    state = _fold(_engine(), [0.0, 0.0, 4.0])
    assert state.state is BehavioralEngagementState.SLIGHT_DEVIATION
    assert state.change is ChangeKind.ANOMALY
    assert state.character is TrendCharacter.ISOLATED
    assert state.trajectories == ()


def test_a_run_below_the_persistence_minimum_stays_an_anomaly() -> None:
    state = _fold(_engine(), [0.0, 1.5, 2.5])
    assert state.run == 2
    assert state.change is ChangeKind.ANOMALY
    assert state.trajectories == ()


# Retention
# ---------------------------------------------------------------------------


def _narrow(max_points: int) -> TemporalSettings:
    """Settings whose windows are sized to fit a deliberately small retention bound.

    Retention cannot be narrower than the windows that read it, so a test that wants a
    short track has to shorten the windows too, rather than configuring an impossible
    combination and working around the refusal.
    """
    return TemporalSettings(
        rate_of_change_window=1,
        stability_window=1,
        max_points_retained=max_points,
    )


def test_a_track_stops_growing_at_the_retention_bound() -> None:
    engine = _engine(_narrow(6))
    state = _fold(engine, [float(index) / 10.0 for index in range(20)])
    track = state.track_for(ACCURACY)
    assert len(track.points) == 6
    assert track.max_points_retained == 6
    assert track.observations == 20


def test_a_track_keeps_the_newest_points_and_not_the_oldest() -> None:
    engine = _engine(_narrow(4))
    state = _fold(engine, [float(index) for index in range(10)])
    track = state.track_for(ACCURACY)
    # Dropping the newest point would report a learner who stopped behaving, and every
    # decision drawn from the state would be about a moment the learner is no longer in.
    assert [point.standardised for point in track.points] == [6.0, 7.0, 8.0, 9.0]
    assert track.points[-1].computed_at == _at(10)
    assert track.latest_at == _at(10)


def test_a_track_records_everything_it_ever_saw_not_only_what_it_keeps() -> None:
    engine = _engine(_narrow(4))
    state = _fold(engine, [float(index) for index in range(10)])
    track = state.track_for(ACCURACY)
    assert track.observations == 10
    assert track.dropped == 6
    # The state is a window, not a tally. Reporting ten retained points when forty were
    # seen would let a long-lived learner look newly arrived on every fresh state.
    assert state.observations == 4
    assert state.measurable_observations == 4


def test_a_track_cannot_claim_fewer_observations_than_it_retains() -> None:
    point = DeviationPoint(
        dimension=ACCURACY,
        value=0.8,
        standardised=0.0,
        reason=None,
        maturity=ESTABLISHED,
        basis=basis_for(ESTABLISHED),
        computed_at=_at(1),
        observation_index=1,
    )
    with pytest.raises(ValidationError, match="retention cannot outrun"):
        TemporalTrack(
            dimension=ACCURACY,
            points=(point,),
            latest_at=_at(1),
            maturity=ESTABLISHED,
            basis=basis_for(ESTABLISHED),
            max_points_retained=4,
            recorded=0,
        )


def test_a_track_cannot_retain_more_points_than_its_bound_allows() -> None:
    points = tuple(
        DeviationPoint(
            dimension=ACCURACY,
            value=0.8,
            standardised=0.0,
            reason=None,
            maturity=ESTABLISHED,
            basis=basis_for(ESTABLISHED),
            computed_at=_at(index),
            observation_index=index,
        )
        for index in (1, 2, 3)
    )
    with pytest.raises(ValidationError, match="the bound is"):
        TemporalTrack(
            dimension=ACCURACY,
            points=points,
            latest_at=_at(3),
            maturity=ESTABLISHED,
            basis=basis_for(ESTABLISHED),
            max_points_retained=2,
            recorded=3,
        )


def test_a_track_must_keep_its_points_in_observation_order() -> None:
    points = tuple(
        DeviationPoint(
            dimension=ACCURACY,
            value=0.8,
            standardised=0.0,
            reason=None,
            maturity=ESTABLISHED,
            basis=basis_for(ESTABLISHED),
            computed_at=_at(index),
            observation_index=index,
        )
        for index in (2, 1)
    )
    with pytest.raises(ValidationError, match="strictly increasing observation order"):
        TemporalTrack(
            dimension=ACCURACY,
            points=points,
            latest_at=_at(2),
            maturity=ESTABLISHED,
            basis=basis_for(ESTABLISHED),
            max_points_retained=4,
            recorded=2,
        )


def test_the_retention_bound_is_configurable() -> None:
    state = _fold(_engine(_narrow(4)), [float(index) for index in range(10)])
    assert len(state.track_for(ACCURACY).points) == 4


def test_retention_cannot_be_narrower_than_the_windows_that_read_it() -> None:
    with pytest.raises(ValueError, match="max_points_retained"):
        TemporalSettings(rate_of_change_window=5, max_points_retained=4)
    with pytest.raises(ValueError, match="max_points_retained"):
        TemporalSettings(stability_window=5, max_points_retained=4)
    # Four times the widest window is enough for the shortest run the persistence rule
    # permits, so the configuration is accepted rather than refused on a technicality.
    TemporalSettings(rate_of_change_window=5, stability_window=5, max_points_retained=20)


def test_a_track_that_was_never_updated_is_not_current() -> None:
    state = _engine().create_state(LEARNER_ID, BASE_TIME)
    assert state.track_for(ACCURACY).observations == 0
    assert not state.track_for(ACCURACY).is_current
    updated = _fold(_engine(), [0.0])
    assert updated.track_for(ACCURACY).is_current
    assert not updated.track_for(RESPONSE).is_current


def test_a_sustained_run_produces_exactly_one_trajectory() -> None:
    state = _fold(_engine(), [0.0, 0.0, 1.0, 2.0, 3.0])
    assert len(state.trajectories) == 1
    trajectory = state.trajectories[0]
    assert trajectory.dimension is ACCURACY
    assert trajectory.observations == 3
    assert trajectory.first_standardised == 1.0
    assert trajectory.last_standardised == 3.0
    assert trajectory.rate_per_observation == 1.0
    assert trajectory.peak_absolute == 3.0
    assert trajectory.duration_seconds == 600.0
    assert trajectory.start == _at(3)
    assert trajectory.end == _at(5)


def test_a_continuing_run_does_not_accumulate_trajectories() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert state.run == 6
    assert len(state.trajectories) == 1


def test_a_second_separate_run_produces_a_second_trajectory() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, -1.0, -2.0, -3.0])
    assert len(state.trajectories) == 2
    assert state.trajectories[1].first_standardised == -1.0
    assert state.trajectories[1].last_standardised == -3.0


def test_a_dimension_absent_from_an_observation_does_not_re_emit_its_trajectory() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0, 1.0, 2.0, 3.0])
    other = engine.ingest(state, _observation(0.0, 6, dimension=RESPONSE))
    assert len(other.trajectories) == 1
    assert other.track_for(ACCURACY).points[-1].standardised == 3.0


def test_a_trajectory_cannot_describe_an_unmeasured_observation() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0, 1.0, 2.0, 3.0])
    fabricated = TrajectoryChange(
        dimension=ACCURACY,
        direction=TrendDirection.WORSENING,
        start=BASE_TIME + timedelta(seconds=1),
        end=_at(5),
        observations=3,
        first_standardised=0.5,
        last_standardised=3.0,
        mean_standardised=1.83,
        peak_absolute=3.0,
        rate_per_observation=1.25,
        duration_seconds=1199.0,
        maturity=ESTABLISHED,
        basis=basis_for(ESTABLISHED),
    )
    with pytest.raises(ValidationError, match="was never emitted for its track"):
        TemporalState(
            learner_id=LEARNER_ID,
            temporal_version=TEMPORAL_STATE_V1,
            reference_time=_at(5),
            created_at=BASE_TIME,
            state=state.state,
            evidence=state.evidence,
            direction=state.direction,
            character=state.character,
            change=state.change,
            run=state.run,
            run_started_at=state.run_started_at,
            origins=state.origins,
            settings_fingerprint=state.settings_fingerprint,
            tracks=state.tracks,
            trajectories=(fabricated,),
        )


def test_a_trajectory_cannot_have_no_direction() -> None:
    with pytest.raises(ValidationError, match="has a direction by construction"):
        TrajectoryChange(
            dimension=ACCURACY,
            direction=TrendDirection.INSUFFICIENT_DATA,
            start=_at(1),
            end=_at(3),
            observations=3,
            first_standardised=1.0,
            last_standardised=2.0,
            mean_standardised=1.5,
            peak_absolute=2.0,
            rate_per_observation=0.5,
            duration_seconds=600.0,
            maturity=ESTABLISHED,
            basis=basis_for(ESTABLISHED),
        )


def test_a_trajectory_cannot_change_sign_mid_run() -> None:
    with pytest.raises(ValidationError, match="cannot change sign"):
        TrajectoryChange(
            dimension=ACCURACY,
            direction=TrendDirection.IMPROVING,
            start=_at(1),
            end=_at(3),
            observations=3,
            first_standardised=-2.0,
            last_standardised=2.0,
            mean_standardised=0.5,
            peak_absolute=2.0,
            rate_per_observation=2.0,
            duration_seconds=600.0,
            maturity=ESTABLISHED,
            basis=basis_for(ESTABLISHED),
        )


def test_a_trajectory_cannot_overstate_its_own_peak() -> None:
    with pytest.raises(ValidationError, match="does not describe the run"):
        TrajectoryChange(
            dimension=ACCURACY,
            direction=TrendDirection.WORSENING,
            start=_at(1),
            end=_at(3),
            observations=3,
            first_standardised=1.0,
            last_standardised=3.0,
            mean_standardised=2.0,
            peak_absolute=1.5,
            rate_per_observation=1.0,
            duration_seconds=600.0,
            maturity=ESTABLISHED,
            basis=basis_for(ESTABLISHED),
        )


def test_a_trajectory_cannot_outrank_its_own_maturity() -> None:
    with pytest.raises(ValidationError, match="does not admit basis"):
        TrajectoryChange(
            dimension=ACCURACY,
            direction=TrendDirection.WORSENING,
            start=_at(1),
            end=_at(3),
            observations=3,
            first_standardised=1.0,
            last_standardised=3.0,
            mean_standardised=2.0,
            peak_absolute=3.0,
            rate_per_observation=1.0,
            duration_seconds=600.0,
            maturity=BaselineMaturity.NEW,
            basis=InferenceBasis.PERSONAL,
        )


def test_a_trajectory_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValidationError, match="before it starts"):
        TrajectoryChange(
            dimension=ACCURACY,
            direction=TrendDirection.WORSENING,
            start=_at(3),
            end=_at(1),
            observations=3,
            first_standardised=1.0,
            last_standardised=3.0,
            mean_standardised=2.0,
            peak_absolute=3.0,
            rate_per_observation=1.0,
            duration_seconds=600.0,
            maturity=ESTABLISHED,
            basis=basis_for(ESTABLISHED),
        )


def test_the_persistence_minimum_cannot_be_lowered_below_a_direction() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        TemporalSettings(sustained_change_min_observations=1)


def test_the_persistence_minimum_is_honoured_when_raised() -> None:
    engine = _engine(TemporalSettings(sustained_change_min_observations=4))
    state = _fold(engine, [0.0, 1.0, 2.0, 3.0])
    assert state.change is ChangeKind.ANOMALY
    assert state.trajectories == ()
    later = engine.ingest(state, _observation(4.0, 5))
    assert later.change is ChangeKind.TRAJECTORY
    assert later.trajectories[0].observations == 4


# Direction, rate, and recovery
# ---------------------------------------------------------------------------


def test_a_growing_departure_is_reported_as_worsening() -> None:
    state = _fold(_engine(), [0.0, 0.0, 0.5, 1.0, 1.5])
    assert state.direction is TrendDirection.WORSENING
    assert state.state is BehavioralEngagementState.DECLINING
    assert state.character is TrendCharacter.PERSISTENT


def test_a_large_sustained_departure_is_a_high_deviation() -> None:
    state = _fold(_engine(), [0.0, 0.0, 2.5, 3.0, 3.5])
    assert state.state is BehavioralEngagementState.HIGH_DEVIATION
    assert state.change is ChangeKind.TRAJECTORY


def test_the_high_deviation_threshold_is_configurable() -> None:
    strict = _engine(TemporalSettings(high_deviation_standardised=5.0))
    assert _fold(strict, [0.0, 0.0, 2.5, 3.0, 3.5]).state is (BehavioralEngagementState.DECLINING)
    assert _fold(_engine(), [0.0, 0.0, 2.5, 3.0, 3.5]).state is (
        BehavioralEngagementState.HIGH_DEVIATION
    )


def test_a_shrinking_departure_is_reported_as_recovery() -> None:
    state = _fold(_engine(), [0.0, 2.0, 3.0, 2.0, 1.0, 0.5])
    assert state.direction is TrendDirection.IMPROVING
    assert state.state is BehavioralEngagementState.RECOVERING
    assert state.change is ChangeKind.RECOVERY


def test_direction_is_read_on_distance_from_the_baseline_not_on_the_raw_sign() -> None:
    growing = _fold(_engine(), [0.0, -1.0, -2.0, -3.0])
    shrinking = _fold(_engine(), [0.0, -3.0, -2.0, -1.0])
    assert growing.direction is TrendDirection.WORSENING
    assert shrinking.direction is TrendDirection.IMPROVING


def test_a_sustained_flat_departure_is_a_trajectory_and_not_an_absence() -> None:
    state = _fold(_engine(), [0.0, 0.0, 1.0, 1.0, 1.0])
    assert state.run == 3
    assert state.direction is TrendDirection.STABLE
    assert state.change is ChangeKind.TRAJECTORY
    assert state.trajectories[0].direction is TrendDirection.STABLE
    assert state.trajectories[0].rate_per_observation == 0.0


def test_the_run_start_and_length_describe_the_same_dimension() -> None:
    state = _fold(_engine(), [0.0, 0.0, 1.0, 2.0, 3.0])
    assert state.run == 3
    assert state.run_started_at == _at(3)
    assert state.run_seconds == 600.0
    trajectory = state.trajectories[0]
    assert trajectory.start == state.run_started_at
    assert trajectory.observations == state.run


def test_a_long_run_is_not_capped_by_the_rate_window() -> None:
    state = _fold(_engine(), [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert state.run == 6
    assert state.run_started_at == _at(3)
    assert len(state.trajectories) == 1


def test_the_run_length_is_reported_by_the_evidence_the_state_is_read_from() -> None:
    state = _fold(_engine(), [0.0, 0.0, 1.0, 2.0, 3.0])
    assert state.evidence.consecutive == state.run
    assert state.transitions[-1].consecutive == state.run


def test_the_recovery_epsilon_is_validated() -> None:
    with pytest.raises(ValueError, match="recovery_epsilon"):
        TemporalSettings(recovery_epsilon=0.0)
    with pytest.raises(ValueError, match="recovery_epsilon"):
        TemporalSettings(recovery_epsilon=1.5)


def test_a_run_that_ends_is_not_reported_as_current() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 1.0, 2.0, 3.0])
    assert state.run == 3
    settled = engine.ingest(state, _observation(0.0, 5))
    assert settled.run == 0
    assert settled.run_started_at is None
    assert settled.state is BehavioralEngagementState.STABLE
    assert settled.run_seconds == 0.0


def test_a_sign_change_restarts_the_run_rather_than_continuing_it() -> None:
    state = _fold(_engine(), [0.0, 1.0, 2.0, 3.0, -1.0])
    assert state.run == 1
    assert state.evidence.consecutive == 1


# Aggregation across dimensions
# ---------------------------------------------------------------------------


def test_every_tracked_dimension_has_a_track_from_the_start() -> None:
    state = _engine().create_state(LEARNER_ID, BASE_TIME)
    assert tuple(track.dimension for track in state.tracks) == TRACKED_DIMENSIONS


def test_a_partial_observation_does_not_erase_the_other_dimensions() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0])
    partial = engine.ingest(state, _observation(1.5, 3, dimension=RESPONSE))
    assert partial.track_for(ACCURACY).points[-1].standardised == 0.0
    assert partial.track_for(RESPONSE).points[-1].standardised == 1.5
    assert partial.observations == 3


def test_the_state_is_read_from_one_dimension_so_its_figures_agree() -> None:
    engine = _engine()
    steps = (
        (0.0, ACCURACY),
        (0.0, ACCURACY),
        (0.0, RESPONSE),
        (1.0, ACCURACY),
        (0.0, RESPONSE),
        (2.0, ACCURACY),
        (0.0, RESPONSE),
    )
    state = engine.create_state(LEARNER_ID, BASE_TIME)
    for index, (value, dimension) in enumerate(steps, start=1):
        state = engine.ingest(state, _observation(value, index, dimension=dimension))
    # Response time has as many observations as accuracy but has never left the baseline,
    # and a learner with more unremarkable observations has not become more noteworthy.
    # The run length, the start time, the evidence, and the label must all describe the
    # accuracy departure, or the state describes a learner who does not exist.
    assert state.track_for(ACCURACY).observations == 4
    assert state.track_for(RESPONSE).observations == 3
    assert state.run == 2
    assert state.run_started_at == _at(4)
    assert state.evidence.consecutive == 2
    assert state.evidence.duration_seconds == 600.0
    # Two observations is a real run but not yet a sustained one, so the departure is
    # reported as an excursion. It is also not the high-deviation label, which requires
    # the same sustained run to be large as well as persistent.
    assert state.state is BehavioralEngagementState.SLIGHT_DEVIATION
    assert state.character is TrendCharacter.ISOLATED
    assert state.change is ChangeKind.ANOMALY
    assert not state.trajectories
    # Maturity is the one figure that is not read from the dominant dimension. A state
    # built partly from a freshly populated dimension is only as personal as that
    # dimension, however firmly the dominant one has departed.
    assert state.evidence.maturity is BaselineMaturity.NEW


def test_a_sustained_departure_outranks_a_longer_record_of_nothing() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0, 1.0])
    state = engine.ingest(state, _observation(0.0, 4, dimension=RESPONSE))
    state = engine.ingest(state, _observation(0.0, 5, dimension=RESPONSE))
    state = engine.ingest(state, _observation(0.0, 6, dimension=RESPONSE))
    assert state.track_for(RESPONSE).observations == 3
    assert state.track_for(ACCURACY).observations == 3
    assert state.run == 1
    assert state.evidence.consecutive == 1
    assert state.run_started_at == _at(3)


def test_a_repeated_dimension_in_a_track_tuple_is_refused() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0])
    track = state.track_for(ACCURACY)
    with pytest.raises(ValidationError, match="repeated dimensions"):
        TemporalState(
            learner_id=LEARNER_ID,
            temporal_version=TEMPORAL_STATE_V1,
            reference_time=_at(2),
            created_at=BASE_TIME,
            state=state.state,
            evidence=state.evidence,
            direction=state.direction,
            character=state.character,
            change=state.change,
            run=state.run,
            run_started_at=state.run_started_at,
            origins=state.origins,
            settings_fingerprint=state.settings_fingerprint,
            tracks=(*state.tracks, track),
        )


def test_a_state_cannot_assert_a_label_its_evidence_contradicts() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0])
    with pytest.raises(ValidationError, match="must agree"):
        TemporalState(
            learner_id=LEARNER_ID,
            temporal_version=TEMPORAL_STATE_V1,
            reference_time=_at(2),
            created_at=BASE_TIME,
            state=BehavioralEngagementState.DECLINING,
            evidence=state.evidence,
            direction=state.direction,
            character=state.character,
            change=state.change,
            run=state.run,
            run_started_at=state.run_started_at,
            origins=state.origins,
            settings_fingerprint=state.settings_fingerprint,
            tracks=state.tracks,
        )


def test_evidence_that_cannot_be_counted_from_the_tracks_is_refused() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0])
    inflated = StateEvidence(
        state=BehavioralEngagementState.STABLE,
        maturity=state.evidence.maturity,
        basis=state.evidence.basis,
        observations=99,
        consecutive=0,
        mean_standardised=0.0,
        dispersion=0.0,
        reason="claimed more observations than the tracks hold",
    )
    with pytest.raises(ValidationError, match="countable from the tracks"):
        TemporalState(
            learner_id=LEARNER_ID,
            temporal_version=TEMPORAL_STATE_V1,
            reference_time=_at(2),
            created_at=BASE_TIME,
            state=BehavioralEngagementState.STABLE,
            evidence=inflated,
            direction=TrendDirection.STABLE,
            character=TrendCharacter.FLAT,
            change=ChangeKind.STABLE,
            run=0,
            run_started_at=None,
            origins=state.origins,
            settings_fingerprint=state.settings_fingerprint,
            tracks=state.tracks,
        )


def test_a_state_cannot_assert_a_label_with_nothing_measurable_behind_it() -> None:
    engine = _engine()
    state = _fold(engine, [None, None])
    # Two points were recorded and retained, but neither could be compared against a
    # personal baseline. Asserting stability from them would report the absence of a
    # comparison as a measurement, so the evidence itself refuses to carry the claim and
    # the state falls back to declaring that it does not yet know.
    assert state.observations == 2
    assert state.measurable_observations == 0
    assert state.state is BehavioralEngagementState.INSUFFICIENT_DATA
    assert state.evidence.observations == 0
    assert state.evidence.consecutive == 0
    assert state.evidence.reason is not None
    with pytest.raises(ValidationError, match="requires at least one observation"):
        StateEvidence(
            state=BehavioralEngagementState.STABLE,
            maturity=BaselineMaturity.NEW,
            basis=InferenceBasis.POPULATION_PRIOR,
            observations=0,
            consecutive=0,
            mean_standardised=None,
            dispersion=None,
            reason="claimed stability from unmeasurable points",
        )


def test_insufficient_data_must_explain_itself() -> None:
    with pytest.raises(ValidationError, match="must state why"):
        StateEvidence(
            state=BehavioralEngagementState.INSUFFICIENT_DATA,
            maturity=BaselineMaturity.NEW,
            basis=InferenceBasis.POPULATION_PRIOR,
            observations=0,
        )


def test_insufficient_data_cannot_claim_a_run() -> None:
    with pytest.raises(ValidationError, match="cannot be paired with a run"):
        StateEvidence(
            state=BehavioralEngagementState.INSUFFICIENT_DATA,
            maturity=BaselineMaturity.NEW,
            basis=InferenceBasis.POPULATION_PRIOR,
            observations=1,
            consecutive=3,
            reason="claimed an absence while asserting a run",
        )


def test_the_state_carries_the_weakest_maturity_among_contributions() -> None:
    engine = _engine()
    state = engine.create_state(LEARNER_ID, BASE_TIME)
    state = engine.ingest(
        state, _observation(0.0, 1, dimension=ACCURACY, maturity=BaselineMaturity.NEW)
    )
    state = engine.ingest(state, _observation(0.0, 2, dimension=RESPONSE, maturity=ESTABLISHED))
    assert state.evidence.maturity is BaselineMaturity.NEW
    assert state.evidence.basis is basis_for(BaselineMaturity.NEW)


def test_weakest_maturity_is_the_weakest_link() -> None:
    assert (
        weakest_maturity([ESTABLISHED, BaselineMaturity.EARLY, BaselineMaturity.NEW])
        is BaselineMaturity.NEW
    )
    assert weakest_maturity([ESTABLISHED]) is ESTABLISHED


# Time, provenance, and replay
# ---------------------------------------------------------------------------


def test_an_observation_from_another_learner_cannot_advance_a_state() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0])
    intruder = _observation(1.0, 3).model_copy(update={"learner_id": "learner-0002"})
    with pytest.raises(TemporalStateError, match="another learner's behaviour"):
        engine.ingest(state, intruder)


def test_an_observation_from_the_past_cannot_advance_a_state() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0])
    with pytest.raises(TemporalStateError, match="function of observed time"):
        engine.ingest(state, _observation(1.0, 2))
    with pytest.raises(TemporalStateError, match="function of observed time"):
        engine.ingest(state, _observation(1.0, 1))


def test_replaying_the_same_observations_gives_the_same_state() -> None:
    observations = tuple(
        _observation(value, index)
        for index, value in enumerate((0.0, 0.0, 0.5, 1.0, 1.5, 2.0), start=1)
    )
    first = _engine().ingest_many(_engine().create_state(LEARNER_ID, BASE_TIME), observations)
    second = _engine().ingest_many(_engine().create_state(LEARNER_ID, BASE_TIME), observations)
    assert first == second
    assert first.settings_fingerprint == second.settings_fingerprint


def test_ingesting_one_at_a_time_matches_ingesting_them_together() -> None:
    values = (0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    observations = tuple(_observation(value, index) for index, value in enumerate(values, start=1))
    engine = _engine()
    stepwise = engine.create_state(LEARNER_ID, BASE_TIME)
    for observation in observations:
        stepwise = engine.ingest(stepwise, observation)
    assert engine.ingest_many(engine.create_state(LEARNER_ID, BASE_TIME), observations) == stepwise


def test_advancing_a_state_leaves_the_original_untouched() -> None:
    engine = _engine()
    before = _fold(engine, [0.0, 0.0])
    snapshot = before.model_dump()
    engine.ingest(before, _observation(5.0, 3))
    assert before.model_dump() == snapshot
    assert before.run == 0
    assert before.reference_time == _at(2)


def test_a_state_records_where_its_observations_came_from() -> None:
    engine = _engine()
    state = engine.create_state(LEARNER_ID, BASE_TIME)
    assert state.origins == frozenset()
    assert not state.is_synthetic_only
    state = engine.ingest(state, _observation(0.0, 1))
    assert state.origins == frozenset({DataOrigin.SYNTHETIC})
    assert state.is_synthetic_only


def test_synthetic_and_real_origins_can_be_told_apart() -> None:
    engine = _engine()
    state = engine.ingest(
        engine.create_state(LEARNER_ID, BASE_TIME),
        _observation(0.0, 1).model_copy(update={"origins": frozenset({DataOrigin.REAL})}),
    )
    assert state.origins == frozenset({DataOrigin.REAL})
    assert not state.is_synthetic_only


def test_a_state_that_seen_both_origins_is_not_labelled_synthetic_only() -> None:
    engine = _engine()
    state = engine.ingest(
        engine.create_state(LEARNER_ID, BASE_TIME),
        _observation(0.0, 1),
    )
    state = engine.ingest(
        state,
        _observation(0.0, 2).model_copy(update={"origins": frozenset({DataOrigin.REAL})}),
    )
    assert state.origins == frozenset({DataOrigin.SYNTHETIC, DataOrigin.REAL})
    assert not state.is_synthetic_only


def test_a_state_records_the_settings_that_produced_it() -> None:
    engine = _engine()
    state = _fold(engine, [0.0, 0.0])
    assert state.settings_fingerprint == engine.settings_fingerprint()
    assert (
        state.settings_fingerprint
        != _engine(TemporalSettings(sustained_change_min_observations=4)).settings_fingerprint()
    )


def test_a_state_carries_its_version() -> None:
    state = _fold(_engine(), [0.0, 0.0])
    assert state.temporal_version == TEMPORAL_STATE_V1
    assert state.to_summary_dict()["temporal_version"] == "TEMPORAL_STATE_V1"


def test_every_state_records_the_transitions_that_led_to_it() -> None:
    engine = _engine()
    state = engine.create_state(LEARNER_ID, BASE_TIME)
    assert state.transitions == ()
    state = _fold(engine, [0.0, 0.0, 1.0])
    assert len(state.transitions) == 3
    assert state.transitions[0].to_state is BehavioralEngagementState.INSUFFICIENT_DATA
    assert state.transitions[-1].to_state is state.state
    # A transition is recorded per advance, not per change. A system that only logged
    # changes would make a long stable stretch indistinguishable from a short one.
    assert all(item.consecutive == 0 for item in state.transitions[:2])


def test_a_recorded_state_can_be_handed_to_a_policy() -> None:
    state = _fold_both(_engine(), [0.0, 0.0, 1.0, 2.0, 3.0])
    summary = state.to_summary_dict()
    assert summary["learner_id"] == LEARNER_ID
    assert summary["state"] == "high_deviation"
    assert summary["direction"] == "worsening"
    assert summary["run"] == 3
    assert summary["change"] == "trajectory"
    assert summary["maturity"] == "established"
    assert summary["temporal_version"] == "TEMPORAL_STATE_V1"


def test_the_summary_counts_the_record_but_does_not_hand_over_its_contents() -> None:
    state = _fold(_engine(), [0.0, 0.0, 1.0, 2.0, 3.0])
    summary = state.to_summary_dict()
    # A consumer needs to know a trajectory was recorded, not be handed the deviations
    # behind it. The count says the claim is backed without exporting the evidence.
    assert summary["trajectories"] == 1
    assert isinstance(summary["trajectories"], int)
    assert "tracks" not in summary
    assert "points" not in summary
    assert not any(isinstance(value, (list, tuple, dict)) for value in summary.values())


# Statistics
# ---------------------------------------------------------------------------


def test_a_run_counts_only_the_consecutive_observations_on_one_side() -> None:
    assert current_sign_run((1.0, 2.0, 3.0), 0.5) == (3, 2.0)
    assert current_sign_run((0.0, 0.0), 0.5) == (0, 0.0)
    assert current_sign_run((1.0, 0.2, 3.0), 0.5) == (1, 3.0)
    assert current_sign_run((1.0, -2.0, -3.0), 0.5) == (2, -2.5)
    assert current_sign_run((), 0.5) == (0, 0.0)


def test_an_immaterial_observation_ends_the_run() -> None:
    # A deviation this close to zero is arithmetic rather than behaviour, and the run ends
    # there. Below the material threshold it is immaterial; above it, the same number is a
    # measurement. The threshold decides which, and it is configured rather than assumed.
    assert current_sign_run((1.0, 0.05, 2.0, 3.0), 0.5) == (2, 2.5)
    assert current_sign_run((1.0, 0.05, 2.0, 3.0), 0.01) == (4, 1.5125)


def test_the_material_threshold_is_inclusive_of_the_threshold_itself() -> None:
    # A deviation exactly at the threshold is material, so it does not break the run. The
    # test at or above it, and the test below it, are the two halves of that decision.
    assert current_sign_run((1.0, 0.5, 2.0), 0.5) == (3, pytest.approx(3.5 / 3.0))
    assert current_sign_run((1.0, 0.49, 2.0), 0.5) == (1, 2.0)


def test_direction_needs_two_points_to_have_one() -> None:
    assert signed_direction(()) is TrendDirection.INSUFFICIENT_DATA
    assert signed_direction((1.0,)) is TrendDirection.INSUFFICIENT_DATA
    assert signed_direction((1.0, 2.0)) is TrendDirection.WORSENING
    assert signed_direction((-1.0, -2.0)) is TrendDirection.WORSENING
    assert signed_direction((-3.0, -2.0, -1.0)) is TrendDirection.IMPROVING
    assert signed_direction((3.0, 2.0, 1.0)) is TrendDirection.IMPROVING
    assert signed_direction((1.0, 1.0, 1.0)) is TrendDirection.STABLE


def test_a_measurable_but_flat_pair_is_stable_rather_than_absent() -> None:
    # Two identical readings cannot show a slope, but they are not missing data either.
    # Reporting absence would describe a learner who gave no measurement.
    assert signed_direction((2.0, 2.0)) is TrendDirection.STABLE


def test_change_per_observation_needs_two_points() -> None:
    assert change_per_observation(()) is None
    assert change_per_observation((1.0,)) is None
    assert change_per_observation((1.0, 3.0)) == 2.0


def test_dispersion_needs_two_points() -> None:
    assert deviation_dispersion(()) is None
    assert deviation_dispersion((1.0,)) is None
    assert deviation_dispersion((1.0, 1.0)) == 0.0


def test_dispersion_is_scaled_to_be_comparable_with_a_standard_deviation() -> None:
    # A median absolute deviation is scaled so that it estimates the same quantity as a
    # standard deviation for normally distributed data. Without the scale, every
    # dispersion here would be 29% low and thresholds tuned against one would be tuned
    # against a different quantity than the thresholds describe.
    scaled = deviation_dispersion((1.0, 3.0))
    assert scaled is not None
    assert scaled == pytest.approx(1.4826, abs=0.0001)
    assert scaled == pytest.approx(MAD_NORMAL_SCALE * 1.0)


def test_an_excursion_cannot_inflate_the_scale_the_way_a_standard_deviation_would() -> None:
    ordinary = (0.0, 1.0, 0.5, -0.5, 0.2)
    with_excursion = (*ordinary, 400.0)
    # A single large excursion drags a standard deviation after it, which would then be
    # used to judge whether a genuine departure is large. The robust scale ignores it, so
    # the excursion stands out against the data rather than being absorbed into the ruler.
    assert deviation_dispersion(with_excursion) is not None
    assert deviation_dispersion(with_excursion) < 1.0
    assert pstdev(with_excursion) > 100.0
    assert pstdev(ordinary) < 1.0


def test_a_mean_of_nothing_is_not_zero() -> None:
    assert mean_standardised(()) is None
    assert mean_standardised((1.0, 3.0)) == 2.0


def test_sustainability_uses_the_configured_minimum() -> None:
    assert is_sustained(2, 2)
    assert is_sustained(3, 2)
    assert not is_sustained(1, 2)
    assert not is_sustained(0, 2)


# Dependency direction
# --------------------------------------------------------------------------------------


def test_the_temporal_layer_imports_no_later_layer() -> None:
    """A static check, because the boundary is easier to break than to notice."""
    allowed = {
        "focus_engine.baseline",
        "focus_engine.configuration",
        "focus_engine.context",
        "focus_engine.schemas",
        "focus_engine.temporal",
        "focus_engine.utils",
    }
    import focus_engine.temporal as package

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


# Time
# --------------------------------------------------------------------------------------


def test_time_enters_only_through_the_injected_clock() -> None:
    """Two engines whose clocks are years apart must produce the same temporal state.

    The engine is handed observations carrying their own instants, so the only place the
    host clock could leak in is the creation instant of a state the caller does not date. A
    state dated by wall clock would carry a different ``created_at`` on every run, and
    every duration derived from it would drift with the machine rather than with the
    learner. Feeding the same stream to two engines whose clocks differ by decades must
    therefore produce indistinguishable states.
    """
    far_future = TemporalEngine(settings=PLAIN, clock=FixedClock(_at(100_000)))
    near_past = TemporalEngine(settings=PLAIN, clock=FixedClock(_at(-100_000)))

    # An undated state takes its instant from the injected clock, and nothing else.
    assert far_future.create_state(LEARNER_ID).created_at == _at(100_000)
    assert near_past.create_state(LEARNER_ID).created_at == _at(-100_000)
    # A dated state ignores the clock entirely.
    assert far_future.create_state(LEARNER_ID, BASE_TIME).created_at == BASE_TIME

    values = [1.0, 1.5, 2.0, 2.5, 3.0]
    assert _fold(far_future, values) == _fold(near_past, values)
