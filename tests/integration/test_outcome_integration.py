"""End-to-end: simulated sessions, a real policy history, a real temporal state, one record.

The unit suite in ``tests/unit/outcomes`` hands the engine synthetic event slices built for
one measure at a time. That proves each measure's arithmetic, and it leaves three things
unchecked that only appear once the layers are composed:

* the outcome engine accepts the ``TemporalState`` the temporal engine *really* produces
  from a real baseline deviation, rather than a hand-built fixture that agrees with it by
  construction;
* the policy layer's history, derived from a real event stream, supplies a delivery the
  outcome layer can locate and measure - so ``SUBSEQUENT_TRAJECTORY`` is a reading rather
  than the declared absence it was when no engine computed a trajectory;
* provenance survives the whole traversal, so a record built from simulator output is
  labelled synthetic at every level and cannot be read as a measurement of a person.

The delivery is placed two minutes into the last simulated session rather than at a session
boundary, because that is where a recap actually lands: the before window reaches back
through the learner's own earlier work, the immediate window catches the first two minutes of
the response, and the session ends inside the after window. That last detail is deliberate.
It is the composition case for the caveat the layer exists to insist on - the session-end
reading is measured, and the record still has to say the events do not show the prompt is
why - and it only arises when a real session's end event lands inside a real window.

Everything here is synthetic. A record computed from a simulator reflects that simulator's
assumptions and establishes nothing about a real learner.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from focus_engine.baseline import (
    BaselineDimension,
    BaselineEngine,
    PopulationPrior,
    PopulationPriorSet,
    StatisticsMethod,
)
from focus_engine.configuration.thresholds import BaselineSettings
from focus_engine.context import ContextEngine, ContextModel
from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    InterventionCompletedPayload,
    InterventionStartedPayload,
)
from focus_engine.features import FeatureName, FeatureValue, compute_features
from focus_engine.outcomes import (
    OUTCOME_MEASURES,
    MeasurementStatus,
    OutcomeEngine,
    OutcomeError,
    OutcomeMeasure,
    OutcomeRecord,
)
from focus_engine.policy.history import InterventionHistory, history_from_events
from focus_engine.schemas.primitives import (
    DataOrigin,
    Provenance,
    SyntheticDataStamp,
)
from focus_engine.simulator import (
    SYNTHETIC_WARNING,
    LearnerArchetype,
    SimulationConfig,
    generate_session,
)
from focus_engine.temporal import TemporalEngine, TemporalObservation, TemporalState
from focus_engine.utils.clock import FixedClock

pytestmark = pytest.mark.integration

LEARNER_ID = "learner-integration-outcome-0001"
OTHER_LEARNER_ID = "learner-integration-outcome-0002"
SESSIONS = 6
QUESTIONS_PER_SESSION = 12

_START = datetime(2026, 9, 27, 9, 0, 0, tzinfo=UTC)

#: Twenty minutes between session starts. Longer than the fifteen-minute after window, so
#: the after window of a delivery inside one session cannot reach into the next one and
#: report another session's activity as this delivery's outcome.
_STRIDE = timedelta(minutes=20)

#: Where the recap lands: two minutes into the final session, which is after the learner has
#: been working for a while and before the session ends.
_DELIVERY_OFFSET = timedelta(minutes=2)

DIMENSION = BaselineDimension.ACCURACY
SOURCE = FeatureName.PERFORMANCE_RECENT_ACCURACY

INTERVENTION_ID = "intervention-integration-outcome-0001"
INTERVENTION_TYPE = "recap"

#: A second delivery, in the session before the first, for the many-deliveries case.
SECOND_ID = "intervention-integration-outcome-0002"

#: The delivery-side label, which the policy layer classifies. ``accepted`` is the
#: delivery component's own vocabulary; the outcome layer never interprets it directly.
ACCEPTED = "accepted"

#: A ladder reaching ``ESTABLISHED`` inside the session count above. These thresholds are
#: not the subject of the test - the wiring is. A real ladder would need twenty simulated
#: sessions before a personal value is admissible at all.
FAST_LADDER = BaselineSettings(
    min_samples_new=0,
    min_samples_early=2,
    min_samples_developing=3,
    min_samples_established=4,
)


def _session_events(index: int) -> tuple[EventEnvelope, ...]:
    """Generate one synthetic session.

    Args:
        index: Zero-based session index, used to separate seeds and clock positions.

    Returns:
        The session's events in temporal order.
    """
    result = generate_session(
        SimulationConfig(
            archetype=LearnerArchetype.STABLE,
            learner_id=LEARNER_ID,
            session_id=f"session-outcome-{index:04d}",
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
        A prior set covering the accuracy dimension.
    """
    return PopulationPriorSet(
        source="simulator-cohort-v1",
        version="COHORT_V1",
        n_reference=50,
        data_origin=DataOrigin.SYNTHETIC,
        priors=(
            PopulationPrior(
                dimension=DIMENSION,
                centre=0.7,
                spread=0.15,
                method=StatisticsMethod.ROBUST_WINSORISED_MAD,
            ),
        ),
    )


def _source_value(vector: FeatureValue) -> float:
    """Read the one numeric feature the temporal traversal follows.

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


def _state_after_every_session() -> tuple[TemporalState, tuple[EventEnvelope, ...]]:
    """Build a real temporal state by driving the whole pipeline, one session at a time.

    One observation per session, each carrying that session's own deviation against the
    baseline built from everything before it. The feature engine emits one vector per
    session, so this is one temporal observation per session rather than one per question.

    Returns:
        The state as of the end of the final session, and every session's events flattened
        into one stream.
    """
    baseline = BaselineEngine(FAST_LADDER, _prior())
    profile = baseline.create_profile(LEARNER_ID, _START)
    temporal = TemporalEngine(clock=FixedClock(_START))
    state = temporal.create_state(LEARNER_ID, _START)

    everything: list[EventEnvelope] = []
    for index in range(SESSIONS):
        events = _session_events(index)
        everything.extend(events)
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
    return state, tuple(everything)


def _stamp() -> SyntheticDataStamp:
    """A synthetic stamp for an injected event.

    Returns:
        A stamp carrying the generator and seed the envelope schema requires of any
        synthetic event.
    """
    return SyntheticDataStamp(generator="focus_engine.outcomes.integration-test", seed=4242)


def _injected(
    event_type: EventType,
    timestamp: datetime,
    session_id: str,
    payload: object,
) -> EventEnvelope:
    """Build one synthetic envelope for an event the simulator does not emit.

    The delivery component's own events, and the content events a real LMS would send, are
    absent from the simulator's output by design - it models the learner, not the platform.
    They are built here against the same payload models and the same validation the engine
    will read, so nothing about them is a test-only shortcut.

    Args:
        event_type: Which event to build.
        timestamp: When it happened.
        session_id: The session it belongs to.
        payload: The payload model.

    Returns:
        The envelope.
    """
    return EventEnvelope(
        event_id=f"event-injected-{event_type.value}-{timestamp.timestamp():.0f}",
        learner_id=LEARNER_ID,
        session_id=session_id,
        timestamp=timestamp,
        event_type=event_type,
        payload=payload,
        origin=DataOrigin.SYNTHETIC,
        provenance=Provenance.SYNTHETIC_LABEL,
        synthetic_stamp=_stamp(),
    )


def _delivery_events(delivered_at: datetime, session_id: str) -> tuple[EventEnvelope, ...]:
    """Build the delivery and its completion at one instant.

    Args:
        delivered_at: The delivery instant.
        session_id: The session the delivery belongs to.

    Returns:
        The delivery and completion envelopes.
    """
    return (
        _injected(
            EventType.INTERVENTION_STARTED,
            delivered_at,
            session_id,
            InterventionStartedPayload(
                intervention_id=INTERVENTION_ID,
                intervention_type=INTERVENTION_TYPE,
                trigger_probability=0.82,
                trigger_state="declining_engagement",
            ),
        ),
        _injected(
            EventType.INTERVENTION_COMPLETED,
            delivered_at + timedelta(seconds=25),
            session_id,
            InterventionCompletedPayload(
                intervention_id=INTERVENTION_ID,
                outcome=ACCEPTED,
                response_seconds=9.0,
            ),
        ),
    )


def _history_and_events() -> tuple[
    InterventionHistory, tuple[EventEnvelope, ...], TemporalState, datetime
]:
    """Compose the whole traversal and return what the engine will be handed.

    Returns:
        The history derived from the real event stream, the stream itself, the temporal
        state, and the delivery instant.
    """
    state, events = _state_after_every_session()
    last_session = _session_events(SESSIONS - 1)
    delivered_at = last_session[0].timestamp + _DELIVERY_OFFSET

    stream = sorted(
        (*events, *_delivery_events(delivered_at, f"session-outcome-{SESSIONS - 1:04d}")),
        key=lambda item: item.timestamp,
    )
    return history_from_events(LEARNER_ID, stream), stream, state, delivered_at


def _record() -> OutcomeRecord:
    """Produce the record under test from a real history, stream, and temporal state.

    Returns:
        One measured record.
    """
    history, stream, state, _delivered_at = _history_and_events()
    engine = OutcomeEngine(clock=FixedClock(_START + SESSIONS * _STRIDE))
    return engine.measure(history, INTERVENTION_ID, stream, state=state)


class TestTheTraversalCloses:
    """The seams that only exist once the layers are composed."""

    def test_a_simulated_session_produces_a_record_over_every_measure(self) -> None:
        """The deliverable: one delivery, one record, all seven measures present.

        A record with a hole in it is indistinguishable from one where the measure was
        considered and found to be noise, so the coverage is required rather than expected.
        """
        record = _record()

        assert tuple(item.measure for item in record.measurements) == OUTCOME_MEASURES
        assert record.response_class.value == "response"
        assert record.intervention_type == INTERVENTION_TYPE

    def test_the_rate_and_accuracy_measures_are_read_from_real_events(self) -> None:
        """The learner's own simulated behaviour produces measured readings.

        Each of these is measured on real simulator output rather than a hand-built slice, so
        the sample floor, the window geometry, and the tolerance are all exercised against
        event timestamps nobody arranged for the test.
        """
        record = _record()

        for measure in (
            OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE,
            OutcomeMeasure.RESPONSE_LATENCY,
            OutcomeMeasure.ACCURACY,
            OutcomeMeasure.CONTINUED_ACTIVITY,
        ):
            reading = record.measurement(measure)
            assert reading.status is MeasurementStatus.MEASURED, (
                f"{measure.value} came back {reading.status.value} on a simulated session: "
                f"{reading.reason}"
            )
            assert reading.after_value is not None

    def test_the_trajectory_reading_carries_the_temporal_layers_own_verdict(self) -> None:
        """The seam the unit suite cannot check: a real state, not a plausible one.

        ``tests/unit/outcomes`` builds ``TemporalState`` objects by hand, so it proves the
        measure maps a verdict and not that the temporal engine's real output survives the
        trip. The observation count is compared directly, because a trajectory that had been
        rebuilt from raw deviations on this side would carry a count of its own inventing.
        """
        _history, _stream, state, _delivered_at = _history_and_events()
        record = _record()

        reading = record.measurement(OutcomeMeasure.SUBSEQUENT_TRAJECTORY)

        assert reading.status is MeasurementStatus.MEASURED, reading.reason
        assert reading.after_value is not None
        assert reading.after_samples == state.evidence.observations
        assert reading.unit == "direction_code"

    def test_a_session_ending_inside_the_window_is_measured_with_its_caveat(self) -> None:
        """The composed caveat: measured direction, and no causal claim attached.

        The simulated session ends a few minutes after the prompt, inside the after window.
        The record therefore measures that the session did not continue - and the reading
        carries the reason saying the events do not show the prompt is why. A record that
        stated the direction alone would be handing the causal reading to whoever reads it,
        which is the one conclusion this layer exists not to supply.
        """
        record = _record()
        reading = record.measurement(OutcomeMeasure.SESSION_CONTINUATION)

        assert reading.status is MeasurementStatus.MEASURED, reading.reason
        assert reading.direction.value == "deteriorated"
        assert reading.reason is not None
        assert "is not something the events can answer" in reading.reason

    def test_a_measure_with_no_evidence_is_reported_as_such(self) -> None:
        """The simulator emits no content events, so task persistence is not applicable.

        Nothing about the platform's own events is invented to fill the gap: the measure says
        there was no open task and explains itself, rather than reporting a learner who
        abandoned something the record never saw them open.
        """
        record = _record()
        reading = record.measurement(OutcomeMeasure.TASK_PERSISTENCE)

        assert reading.status is MeasurementStatus.NOT_APPLICABLE
        assert reading.direction.value == "not_assessed"
        assert reading.reason is not None
        assert record.is_complete is False


class TestProvenanceSurvives:
    """A simulated record cannot be read as a measurement of a person."""

    def test_every_reading_is_labelled_synthetic(self) -> None:
        """The label is derived from the events and travels to every level of the record."""
        record = _record()

        assert record.data_origin is DataOrigin.SYNTHETIC
        assert record.is_synthetic_only
        assert all(item.data_origin is DataOrigin.SYNTHETIC for item in record.measurements)

    def test_the_record_is_an_observation_and_says_so(self) -> None:
        """Provenance is constrained to observed at the type, whatever the caller wanted."""
        record = _record()

        assert record.provenance is Provenance.OBSERVED

    def test_the_summary_counts_the_measures_that_were_never_read(self) -> None:
        """A summary that omitted the unmeasured count would present absences as stability."""
        summary = _record().to_summary_dict()

        assert summary["measures_total"] == len(OUTCOME_MEASURES)
        assert summary["measures_measured"] < summary["measures_total"]
        assert summary["data_origin"] == "synthetic"
        assert summary["response_class"] == "response"


class TestTheRecordIsReproducible:
    """Determinism, with the one non-deterministic input pinned."""

    def test_the_same_events_produce_the_same_record(self) -> None:
        """Two measurements of one delivery agree exactly, timestamp included.

        The record is compared in full rather than field by field, so a change that made any
        part of it irreproducible - a window width, an origin, the clock - would show up.
        """
        first = _record()
        second = _record()

        assert first.model_dump_json() == second.model_dump_json()

    def test_a_different_window_configuration_changes_the_fingerprint(self) -> None:
        """A record cannot later be read as though it had been produced under other widths.

        The fingerprint is what makes that detectable, and it is derived from the settings
        rather than accepted from the caller.
        """
        from focus_engine.configuration.thresholds import OutcomeSettings

        history, stream, state, _delivered_at = _history_and_events()
        narrow = OutcomeEngine(
            settings=OutcomeSettings(before_window_minutes=20.0),
            clock=FixedClock(_START + SESSIONS * _STRIDE),
        )

        record = narrow.measure(history, INTERVENTION_ID, stream, state=state)
        default = _record()

        assert record.settings_fingerprint != default.settings_fingerprint


class TestTheRefusalsSurviveComposition:
    """The engine's guards, exercised with real envelopes rather than unit fixtures."""

    def test_events_from_another_learner_are_refused(self) -> None:
        """A routing bug reaches the engine as a refusal, not as a thin measurement.

        Silently dropping another learner's events would produce a plausible record with too
        little evidence in it, which is the harder failure to notice.
        """
        history, stream, state, _delivered_at = _history_and_events()
        foreign = stream[0].model_copy(update={"learner_id": OTHER_LEARNER_ID})

        with pytest.raises(OutcomeError, match="belongs to learner"):
            OutcomeEngine().measure(history, INTERVENTION_ID, (*stream, foreign), state=state)

    def test_a_temporal_state_for_another_learner_is_refused(self) -> None:
        """A trajectory about someone else is not a gap in the data."""
        history, stream, state, _delivered_at = _history_and_events()

        with pytest.raises(OutcomeError, match="is a record of the wrong person"):
            OutcomeEngine().measure(
                history,
                INTERVENTION_ID,
                stream,
                state=state.model_copy(update={"learner_id": OTHER_LEARNER_ID}),
            )

    def test_a_delivery_that_was_never_made_cannot_be_measured(self) -> None:
        """An identifier with no delivery raises rather than yielding a record.

        A measurement attached to an intervention that was never delivered is exactly the
        fabricated finding this layer exists to prevent.
        """
        history, stream, state, _delivered_at = _history_and_events()

        with pytest.raises(OutcomeError, match="no delivery"):
            OutcomeEngine().measure(
                history, "intervention-integration-outcome-9999", stream, state=state
            )

    def test_measuring_many_covers_every_delivery_in_the_history(self) -> None:
        """A two-delivery history produces two records, in the history's own order.

        The earlier delivery is in the second-to-last session and the later one in the last,
        so the pair also checks that the history's chronological order is the order the
        records come back in rather than an order re-derived here.
        """
        _history, stream, _state, _delivered_at = _history_and_events()
        earlier = _session_events(SESSIONS - 2)
        second_session = f"session-outcome-{SESSIONS - 2:04d}"
        second = tuple(
            event.model_copy(
                update={
                    "event_id": f"event-injected-second-{index:02d}",
                    "payload": (
                        event.payload.model_copy(update={"intervention_id": SECOND_ID})
                        if event.event_type
                        in (EventType.INTERVENTION_STARTED, EventType.INTERVENTION_COMPLETED)
                        else event.payload
                    ),
                }
            )
            for index, event in enumerate(
                _delivery_events(earlier[0].timestamp + _DELIVERY_OFFSET, second_session)
            )
        )
        both = sorted((*stream, *second), key=lambda item: item.timestamp)
        history = history_from_events(LEARNER_ID, both)

        records = OutcomeEngine().measure_many(history, both)

        assert [item.intervention_id for item in records] == [SECOND_ID, INTERVENTION_ID]
        assert [item.delivered_at for item in records] == sorted(
            item.delivered_at for item in records
        )
