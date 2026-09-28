"""End-to-end test: simulated events to a temporal state and back into the features.

The unit suite in ``tests/unit/test_temporal.py`` feeds the engine hand-built
:class:`~focus_engine.baseline.models.DeviationResult` objects. That proves the temporal
arithmetic, but not that it accepts what the baseline engine actually produces, and not
that its output reaches the layers that are supposed to consume it.

The seam that matters most is the one closing the loop. ``trajectory_encoded`` was a
declared absence in Phase 5 because no engine computed a trajectory. Phase 7 supplies one,
so the value has to travel simulator to context to features to baseline to deviation to
temporal state and back out to a feature vector. Nothing forces that path to exist: a
temporal engine that computed a correct state nobody reads would satisfy every unit test in
its own file and leave ``trajectory_encoded`` permanently absent. This test asserts the
whole traversal, because the traversal is the deliverable.

Three properties are checked that only appear once the layers are composed:

* the temporal engine accepts the deviations the baseline engine really produces, without
  a hand-built fixture papering over a type or range mismatch;
* a trajectory reported by the temporal engine closes the non-blocking ``trajectory`` gap
  and reaches ``trajectory_encoded`` as a real measurement;
* provenance survives every hop, so a state built from simulated events stays labelled
  synthetic and no feature built from it claims to be observed.

Everything here is synthetic. A temporal state computed from a simulator reflects that
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
    StatisticsMethod,
)
from focus_engine.configuration.thresholds import BaselineSettings, TemporalSettings
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
from focus_engine.temporal import (
    BehavioralEngagementState,
    ChangeKind,
    TemporalEngine,
    TemporalObservation,
    TemporalState,
)
from focus_engine.utils.clock import FixedClock

pytestmark = pytest.mark.integration

LEARNER_ID = "learner-integration-temporal-0001"
SESSIONS = 6
QUESTIONS_PER_SESSION = 12

_START = datetime(2026, 9, 27, 9, 0, 0, tzinfo=UTC)
_STRIDE = timedelta(hours=1)

#: The dimension carried through the whole traversal. One dimension is enough to prove the
#: path; four would only repeat it.
DIMENSION = BaselineDimension.ACCURACY
SOURCE = FeatureName.PERFORMANCE_RECENT_ACCURACY

#: A ladder reaching ``ESTABLISHED`` inside the session count above. These thresholds are
#: not the subject of the test — the wiring is. A real ladder would need twenty simulated
#: sessions before a personal value is admissible at all.
FAST_LADDER = BaselineSettings(
    min_samples_new=0,
    min_samples_early=2,
    min_samples_developing=3,
    min_samples_established=4,
)

#: Read from the shipped settings rather than written as a literal, so a future change to
#: the persistence minimum cannot leave this test asserting a threshold nothing uses.
SUSTAINED_MINIMUM = TemporalSettings().sustained_change_min_observations


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


def _source_value(vector: FeatureValue) -> float:
    """Read the one numeric feature the traversal follows.

    Args:
        vector: A real feature-engine output.

    Returns:
        The feature's value.

    Raises:
        AssertionError: If the feature is absent or not a number, which is a seam failure
            rather than a value to be interpreted.
    """
    entry = next(item for item in vector.values if item.name is SOURCE)
    assert isinstance(entry.value, (int, float)), (
        f"{SOURCE.value} must reach the baseline as a number; got {entry.value!r}"
    )
    return float(entry.value)


def _drive() -> tuple[TemporalState, FeatureValue, ContextModel, BaselineEngine]:
    """Run the full pipeline and return the state plus the seam it has to cross.

    One observation per session, each carrying the deviation of that session's own
    measurement against the baseline built from everything before it. The feature engine
    emits one vector per session, so this is one temporal observation per session rather
    than one per question.

    Returns:
        The final temporal state, the final feature vector, its derived context, and the
        baseline engine that produced the deviations.
    """
    baseline = BaselineEngine(FAST_LADDER, _prior())
    profile = baseline.create_profile(LEARNER_ID, _START)
    temporal = TemporalEngine(clock=FixedClock(_START))
    state = temporal.create_state(LEARNER_ID, _START)

    vector = None
    context = None
    for index in range(SESSIONS):
        events = _session_events(LearnerArchetype.STABLE, index)
        context = _context_for(events)
        vector = compute_features(events, context)

        profile = baseline.update(profile, vector)
        deviation = baseline.deviation(profile, DIMENSION, _source_value(vector))
        state = temporal.ingest(
            state,
            TemporalObservation(
                learner_id=LEARNER_ID,
                computed_at=vector.computed_at,
                deviations=(deviation,),
                origins=vector.origins,
            ),
        )

    assert vector is not None and context is not None
    return state, vector, context, baseline


def test_the_baseline_deviations_reach_a_real_temporal_state() -> None:
    """The seam the unit suite cannot check: real deviations, real standardised values.

    Hand-built fixtures prove the temporal arithmetic accepts a plausible
    ``DeviationResult``. Only a real one proves the baseline emits a shape the temporal
    engine will take, with a `computed_at` that advances and a `learner_id` that matches.
    """
    state, vector, _context, _baseline = _drive()

    assert state.learner_id == LEARNER_ID
    assert state.reference_time == vector.computed_at
    assert state.state is not BehavioralEngagementState.INSUFFICIENT_DATA, (
        f"the state never left cold start: {state.evidence.reason!r}. A trajectory this "
        "layer computes would be unreadable if real baseline output could not reach it."
    )

    track = state.track_for(DIMENSION)
    assert track.recorded == SESSIONS
    assert track.maturity is BaselineMaturity.ESTABLISHED
    assert track.basis is not InferenceBasis.POPULATION_PRIOR, (
        "by the last session the reference is personal, so a population answer here would "
        "mean the maturity ladder was not actually reached"
    )

    # The *state*'s maturity is the weakest link across every tracked dimension, and this
    # traversal only ever fed one of the four. The untouched dimensions are cold, so the
    # state is as cold as its least-observed dimension even though the one being read is
    # established. A state that reported ESTABLISHED here would be a personal inference
    # resting on three dimensions that have never been measured.
    assert state.evidence.maturity is BaselineMaturity.NEW
    assert state.evidence.basis is not InferenceBasis.PERSONAL, (
        "a state built from one mature dimension and three empty ones is not a personal "
        "inference, and the weakest-link rule is what stops it being presented as one"
    )
    assert state.evidence.observations >= 2, (
        "a direction needs at least two measurable points, and this is the floor for one"
    )


def test_an_anomaly_is_not_reported_as_a_trajectory() -> None:
    """The Phase 7 exit criterion, measured on a real stream rather than a fixture.

    A stable archetype's deviations scatter around the baseline. A single excursion must
    not accumulate into a sustained claim, and the current change kind must agree with the
    length of the *current* run.

    The comparison is against the run, not against the ledger. ``state.trajectories`` is
    every run that ever reached the persistence minimum, and ``state.change`` describes
    only the run in progress. A stream that recorded a trajectory and then crossed zero
    restarts the run, leaving a trajectory in the ledger beside a current change of
    ``ANOMALY``. Reading the ledger as the current claim would report that as sustained
    when the learner has already reversed.
    """
    state, _vector, _context, _baseline = _drive()

    assert state.change is not None, "an observable state must state its change kind"

    for trajectory in state.trajectories:
        assert trajectory.observations >= SUSTAINED_MINIMUM, (
            "a trajectory shorter than the persistence minimum would be an anomaly "
            "reported as a sustained change"
        )
        assert trajectory.maturity is BaselineMaturity.ESTABLISHED
        assert trajectory.basis is not InferenceBasis.POPULATION_PRIOR

    if state.change is ChangeKind.TRAJECTORY:
        assert state.evidence.consecutive >= SUSTAINED_MINIMUM, (
            "a sustained claim on a run shorter than the persistence minimum is exactly "
            "the defect this test exists to catch"
        )
    elif state.change is ChangeKind.ANOMALY:
        assert state.evidence.consecutive < SUSTAINED_MINIMUM, (
            f"an anomaly spanning {state.evidence.consecutive} consecutive observations is "
            "long enough to have been reported as a trajectory"
        )


def test_a_trajectory_reaches_the_context_and_becomes_a_measurement() -> None:
    """The loop closes: the value Phase 5 declared absent is now computed.

    ``trajectory_encoded`` reported insufficiency because no engine produced a
    trajectory. With the temporal engine in place the direction has to travel back into
    the context, close the non-blocking ``trajectory`` gap, and reach the feature set as a
    real encoding.

    The context is rebuilt through ``model_validate`` rather than
    ``model_copy(update=...)`` on purpose: ``missing`` is *derived* by an ``after``
    validator, and ``model_copy`` skips validators, which would leave a closed gap recorded
    as open. That is the same reason the baseline engine revalidates on every update.
    """
    state, _vector, context, _baseline = _drive()

    assert state.direction is not None
    assert ContextKey.TRAJECTORY in context.missing, (
        "a context built from events alone has no trajectory, so the gap is correctly open"
    )

    enriched = ContextModel.model_validate(
        {**context.model_dump(), "trajectory": state.direction.value}
    )
    assert ContextKey.TRAJECTORY not in enriched.missing, (
        "reporting a trajectory must close the non-blocking gap it was blocking"
    )

    events = _session_events(LearnerArchetype.STABLE, 200)
    recomputed = compute_features(events, enriched)
    encoded = next(
        entry for entry in recomputed.values if entry.name is FeatureName.TRAJECTORY_ENCODED
    )

    assert encoded.availability is FeatureAvailability.SYNTHETIC_ONLY, (
        "the vector's origins are purely synthetic, so no feature may claim to be observed"
    )
    assert encoded.availability is not FeatureAvailability.INSUFFICIENT_DATA, (
        "the Phase 5 absence must have closed now that a temporal engine exists"
    )
    assert encoded.value in ("improving", "stable", "declining", "other")
    assert encoded.value == _expected_encoding(state.direction.value)


def test_a_context_without_a_trajectory_still_reports_absence() -> None:
    """The other half: the seam must not fabricate a value when nothing reported one.

    A closed loop is only trustworthy if it is also openable. With no trajectory reported
    the feature must stay absent, because an encoding invented from a missing input is
    indistinguishable downstream from one derived from an actual temporal state.
    """
    events = _session_events(LearnerArchetype.STABLE, 201)
    context = _context_for(events)
    vector = compute_features(events, context)

    encoded = next(entry for entry in vector.values if entry.name is FeatureName.TRAJECTORY_ENCODED)
    assert encoded.availability is FeatureAvailability.INSUFFICIENT_DATA
    assert "trajectory" in (encoded.reason or "")


def test_synthetic_provenance_survives_every_hop() -> None:
    """Nothing in the traversal may launder a simulated stream into an observed one."""
    state, vector, _context, _baseline = _drive()

    assert state.origins == frozenset({DataOrigin.SYNTHETIC})
    assert state.is_synthetic_only, (
        "a state built only from simulated events must stay labelled synthetic"
    )
    assert vector.origins == frozenset({DataOrigin.SYNTHETIC})
    assert state.settings_fingerprint, "the settings that shaped the state are recorded"


def test_replaying_the_whole_pipeline_reproduces_the_state() -> None:
    """Determinism across every layer, not just inside the temporal engine."""
    first_state, first_vector, _context, _baseline = _drive()
    second_state, second_vector, _context, _baseline = _drive()

    assert first_state.model_dump() == second_state.model_dump()
    assert first_vector.model_dump() == second_vector.model_dump()


def _expected_encoding(direction: str) -> str:
    """Map a temporal direction onto the feature set's closed encoding vocabulary.

    Args:
        direction: The direction the temporal engine reported.

    Returns:
        The encoding a consumer of the feature set will see.
    """
    normalized = direction.lower().strip()
    if normalized in ("improving", "recovering"):
        return "improving"
    if normalized in ("stable", "flat"):
        return "stable"
    if normalized in ("declining", "deteriorating"):
        return "declining"
    return "other"
