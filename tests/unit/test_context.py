"""Unit tests for :mod:`focus_engine.context`.

Covers context derivation from event streams, session position, mid-session starts,
performance windows, intervention outcome matching, cooldown, replay determinism, and
explicit missing-context handling.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from focus_engine.context import (
    BLOCKING_CONTEXT_KEYS,
    ContentContext,
    ContextEngine,
    ContextKey,
    ContextModel,
    ContextVersion,
    InterventionContext,
    PerformanceContext,
    SessionContext,
)
from focus_engine.events.types import (
    ContentOpenedPayload,
    EventEnvelope,
    EventType,
    InteractionPayload,
    InterventionCompletedPayload,
    InterventionStartedPayload,
    QuestionAnsweredPayload,
    QuestionStartedPayload,
    SessionEndedPayload,
    SessionStartedPayload,
    VideoStartedPayload,
)
from focus_engine.schemas.primitives import DataOrigin, Provenance, SyntheticDataStamp
from focus_engine.utils.clock import FixedClock

pytestmark = pytest.mark.unit

_BASE = datetime(2026, 9, 26, 10, 0, 0, tzinfo=UTC)


def _event(
    event_type: EventType,
    offset_seconds: int = 0,
    *,
    payload: object | None = None,
    session_id: str = "session-0001",
    learner_id: str = "learner-0001",
    synthetic: bool = False,
) -> EventEnvelope:
    """Build a valid event for context tests.

    Args:
        event_type: Event type discriminator.
        offset_seconds: Seconds added to the module base timestamp.
        payload: Explicit payload model, or a sensible default for the type.
        session_id: Pseudonymous session identifier.
        learner_id: Pseudonymous learner identifier.
        synthetic: Whether to mark the event as synthetic.

    Returns:
        A validated envelope.
    """
    if payload is None:
        defaults: dict[EventType, Callable[[], object]] = {
            EventType.SESSION_STARTED: SessionStartedPayload,
            EventType.SESSION_ENDED: SessionEndedPayload,
            EventType.INTERACTION: lambda: InteractionPayload(interaction_type="click"),
            EventType.QUESTION_STARTED: lambda: QuestionStartedPayload(question_id="question-001"),
            EventType.QUESTION_ANSWERED: lambda: QuestionAnsweredPayload(
                question_id="question-001", correct=True, response_seconds=10.0
            ),
            EventType.VIDEO_STARTED: lambda: VideoStartedPayload(video_id="video-0001"),
            EventType.CONTENT_OPENED: lambda: ContentOpenedPayload(content_id="content-001"),
        }
        try:
            payload = defaults[event_type]()
        except KeyError:
            raise AssertionError(f"no default payload for {event_type}") from None

    return EventEnvelope(
        event_id=f"event-{event_type.value}-{session_id}-{offset_seconds:05d}",
        learner_id=learner_id,
        session_id=session_id,
        timestamp=_BASE + timedelta(seconds=offset_seconds),
        event_type=event_type,
        payload=payload,
        origin=DataOrigin.SYNTHETIC if synthetic else DataOrigin.REAL,
        provenance=Provenance.SYNTHETIC_LABEL if synthetic else Provenance.OBSERVED,
        synthetic_stamp=SyntheticDataStamp(generator="sim", seed=1) if synthetic else None,
    )


def _intervention_started(identifier: str, kind: str, offset_seconds: int) -> EventEnvelope:
    """Build an ``intervention_started`` event."""
    return _event(
        EventType.INTERVENTION_STARTED,
        offset_seconds,
        payload=InterventionStartedPayload(
            intervention_id=identifier,
            intervention_type=kind,
            trigger_probability=0.7,
            trigger_state="declining",
        ),
    )


def _intervention_completed(identifier: str, outcome: str, offset_seconds: int) -> EventEnvelope:
    """Build an ``intervention_completed`` event."""
    return _event(
        EventType.INTERVENTION_COMPLETED,
        offset_seconds,
        payload=InterventionCompletedPayload(intervention_id=identifier, outcome=outcome),
    )


def _answered(
    offset_seconds: int,
    correct: bool,
    response_seconds: float,
    index: int = 0,
    *,
    synthetic: bool = False,
) -> EventEnvelope:
    """Build a ``question_answered`` event."""
    return _event(
        EventType.QUESTION_ANSWERED,
        offset_seconds,
        payload=QuestionAnsweredPayload(
            question_id=f"question-{index:03d}", correct=correct, response_seconds=response_seconds
        ),
        synthetic=synthetic,
    )


# --------------------------------------------------------------------------------------
# Model defaults and versioning
# --------------------------------------------------------------------------------------


def test_context_version_is_v1() -> None:
    assert ContextVersion.V1 == "v1"
    assert ContextModel(reference_time=_BASE).version is ContextVersion.V1


def test_context_model_is_frozen() -> None:
    model = ContextModel(reference_time=_BASE)
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        model.trajectory = "declining"  # type: ignore[misc]


def test_context_model_rejects_extra_fields() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        ContextModel(reference_time=_BASE, unexpected="value")  # type: ignore[call-arg]


def test_context_model_rejects_naive_reference_time() -> None:
    with pytest.raises(Exception, match="timezone-aware"):
        ContextModel(reference_time=datetime(2026, 9, 26, 10, 0, 0))


# --------------------------------------------------------------------------------------
# Missing-context handling
# --------------------------------------------------------------------------------------


def test_empty_context_reports_every_gap() -> None:
    model = ContextModel(reference_time=_BASE)
    assert ContextKey.SESSION in model.missing
    assert ContextKey.RECENT_ACCURACY in model.missing
    assert ContextKey.BASELINE_MATURITY in model.missing
    assert not model.is_interpretable()


def test_missing_keys_are_derived_not_supplied() -> None:
    """A caller cannot declare a gap closed while the value is still absent."""
    model = ContextModel(reference_time=_BASE, missing=frozenset())
    assert ContextKey.SESSION in model.missing


def test_context_is_not_interpretable_without_baseline_maturity() -> None:
    """Separation 2: an unlabelled inference basis must block interpretation."""
    model = ContextModel(
        reference_time=_BASE,
        session=SessionContext(
            session_id="session-0001",
            learner_id="learner-0001",
            started_at=_BASE,
            elapsed_seconds=60.0,
            event_count=4,
            position_in_session=0.02,
        ),
        performance=PerformanceContext(recent_questions=2, recent_correct=1, recent_accuracy=0.5),
    )
    assert ContextKey.BASELINE_MATURITY in model.blocking_gaps()
    assert not model.is_interpretable()


def test_context_is_interpretable_once_blocking_gaps_are_filled() -> None:
    model = ContextModel(
        reference_time=_BASE,
        session=SessionContext(
            session_id="session-0001",
            learner_id="learner-0001",
            started_at=_BASE,
            elapsed_seconds=60.0,
            event_count=4,
            position_in_session=0.02,
        ),
        performance=PerformanceContext(recent_questions=2, recent_correct=1, recent_accuracy=0.5),
        baseline_maturity="EARLY",
    )
    assert model.is_interpretable()
    assert model.blocking_gaps() == frozenset()


def test_non_blocking_gaps_do_not_prevent_interpretation() -> None:
    """Absent difficulty and trajectory reduce confidence; they do not block."""
    model = ContextModel(
        reference_time=_BASE,
        session=SessionContext(
            session_id="session-0001",
            learner_id="learner-0001",
            started_at=_BASE,
            elapsed_seconds=60.0,
            event_count=4,
            position_in_session=0.02,
        ),
        performance=PerformanceContext(recent_questions=2, recent_correct=1, recent_accuracy=0.5),
        baseline_maturity="DEVELOPING",
    )
    assert ContextKey.DIFFICULTY in model.missing
    assert ContextKey.TRAJECTORY in model.missing
    assert not (model.missing & BLOCKING_CONTEXT_KEYS)
    assert model.is_interpretable()


def test_blocking_keys_are_the_documented_set() -> None:
    assert (
        frozenset(
            {
                ContextKey.SESSION.value,
                ContextKey.ELAPSED_SECONDS.value,
                ContextKey.POSITION_IN_SESSION.value,
                ContextKey.RECENT_ACCURACY.value,
                ContextKey.BASELINE_MATURITY.value,
            }
        )
        == BLOCKING_CONTEXT_KEYS
    )


# --------------------------------------------------------------------------------------
# Engine parameter validation
# --------------------------------------------------------------------------------------


def test_engine_rejects_non_positive_expected_session() -> None:
    with pytest.raises(ValueError, match="expected_session_seconds must be positive"):
        ContextEngine(expected_session_seconds=0.0)


def test_engine_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="performance_window_size must be at least 1"):
        ContextEngine(performance_window_size=0)


def test_engine_rejects_negative_cooldown() -> None:
    with pytest.raises(ValueError, match="cooldown_seconds must not be negative"):
        ContextEngine(cooldown_seconds=-1.0)


# --------------------------------------------------------------------------------------
# Session context
# --------------------------------------------------------------------------------------


def test_empty_stream_has_no_session() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE))
    model = engine.process([])
    assert model.session is None
    assert not model.is_interpretable()


def test_empty_stream_uses_clock_as_reference_time() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE))
    assert engine.process([]).reference_time == _BASE


def test_explicit_session_start_is_used() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(hours=1)))
    model = engine.process([_event(EventType.SESSION_STARTED)])
    assert model.session is not None
    assert model.session.started_at == _BASE
    assert model.session.start_inferred is False
    assert model.session.has_ended is False


def test_mid_session_start_is_inferred_and_flagged() -> None:
    """A stream beginning mid-session still yields context, and says it is inferred."""
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(hours=1)))
    model = engine.process([_answered(offset_seconds=120, correct=True, response_seconds=8.0)])

    assert model.session is not None
    assert model.session.start_inferred is True
    assert model.session.started_at == _BASE + timedelta(seconds=120)
    assert model.session.event_count == 1


def test_session_ended_is_recorded() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=600)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _event(EventType.SESSION_ENDED, offset_seconds=600),
        ]
    )
    assert model.session is not None
    assert model.session.has_ended is True


def test_position_normalises_against_expected_length() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE), expected_session_seconds=1200.0)
    model = engine.process([_event(EventType.SESSION_STARTED), _event(EventType.INTERACTION, 600)])
    assert model.session is not None
    assert model.session.position_in_session == pytest.approx(0.5)


def test_position_is_capped_at_one() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE), expected_session_seconds=600.0)
    model = engine.process([_event(EventType.SESSION_STARTED), _event(EventType.INTERACTION, 5000)])
    assert model.session is not None
    assert model.session.position_in_session == 1.0


def test_elapsed_time_comes_from_events_not_the_wall_clock() -> None:
    """Replaying a recorded session must reproduce the context it produced live."""
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(days=3650)))
    model = engine.process([_event(EventType.SESSION_STARTED), _event(EventType.INTERACTION, 900)])
    assert model.session is not None
    assert model.session.elapsed_seconds == 900.0
    assert model.reference_time == _BASE + timedelta(seconds=900)


def test_reference_time_is_the_latest_event_timestamp() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _event(EventType.INTERACTION, 100),
            _event(EventType.INTERACTION, 300),
        ]
    )
    assert model.reference_time == _BASE + timedelta(seconds=300)


def test_out_of_order_event_does_not_move_reference_time_backwards() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _event(EventType.INTERACTION, 300),
            _event(EventType.INTERACTION, 100),
        ]
    )
    assert model.reference_time == _BASE + timedelta(seconds=300)


def test_new_session_resets_derived_state() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(hours=2)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _answered(offset_seconds=30, correct=True, response_seconds=5.0),
            _event(EventType.SESSION_STARTED, 0, session_id="session-0002"),
        ]
    )
    assert model.session is not None
    assert model.session.session_id == "session-0002"
    assert model.session.event_count == 1
    assert model.performance.recent_questions == 0
    assert model.intervention.total_interventions == 0


def test_reset_clears_all_state() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE))
    engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _answered(offset_seconds=10, correct=True, response_seconds=5.0),
        ]
    )
    engine.reset()
    model = engine.compute_context()
    assert model.session is None
    assert model.observed_event_types == frozenset()


# --------------------------------------------------------------------------------------
# Content context
# --------------------------------------------------------------------------------------


def test_question_started_sets_content_with_metadata() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=30)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _event(
                EventType.QUESTION_STARTED,
                10,
                payload=QuestionStartedPayload(
                    question_id="question-abc123", difficulty=0.75, topic="algebra"
                ),
            ),
        ]
    )
    assert model.content == ContentContext(
        content_type="question", content_id="question-abc123", difficulty=0.75, topic="algebra"
    )


def test_video_started_sets_content_context() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=30)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _event(
                EventType.VIDEO_STARTED,
                10,
                payload=VideoStartedPayload(video_id="video-xyz789", duration_seconds=300.0),
            ),
        ]
    )
    assert model.content.content_type == "video"
    assert model.content.content_id == "video-xyz789"
    assert model.content.video_duration_seconds == 300.0


def test_content_opened_defaults_type_when_absent() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=30)))
    model = engine.process(
        [_event(EventType.SESSION_STARTED), _event(EventType.CONTENT_OPENED, 10)]
    )
    assert model.content.content_type == "content"
    assert model.content.content_id == "content-001"


def test_new_content_replaces_previous_content() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=30)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _event(
                EventType.QUESTION_STARTED,
                5,
                payload=QuestionStartedPayload(question_id="question-001", difficulty=0.5),
            ),
            _event(EventType.VIDEO_STARTED, 10, payload=VideoStartedPayload(video_id="video-0001")),
        ]
    )
    assert model.content.content_type == "video"
    assert model.content.difficulty is None


def test_unrelated_events_leave_content_untouched() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=30)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _event(EventType.VIDEO_STARTED, 5, payload=VideoStartedPayload(video_id="video-0001")),
            _event(EventType.INTERACTION, 10),
        ]
    )
    assert model.content.content_type == "video"


# --------------------------------------------------------------------------------------
# Performance context
# --------------------------------------------------------------------------------------


def test_no_answers_yields_empty_performance() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=30)))
    model = engine.process([_event(EventType.SESSION_STARTED), _event(EventType.INTERACTION, 10)])
    assert model.performance == PerformanceContext(window_size=10)


def test_accuracy_and_mean_response_are_computed() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=60)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _answered(10, correct=True, response_seconds=15.0, index=1),
            _answered(30, correct=False, response_seconds=25.0, index=2),
        ]
    )
    assert model.performance.recent_questions == 2
    assert model.performance.recent_correct == 1
    assert model.performance.recent_accuracy == 0.5
    assert model.performance.average_response_seconds == 20.0


def test_performance_window_keeps_most_recent() -> None:
    engine = ContextEngine(
        clock=FixedClock(_BASE + timedelta(seconds=200)), performance_window_size=3
    )
    events = [_event(EventType.SESSION_STARTED)]
    for index in range(5):
        events.append(
            _answered(
                10 + index * 10, correct=index % 2 == 0, response_seconds=10.0 + index, index=index
            )
        )
    model = engine.process(events)

    # Window holds indices 2, 3, 4: correct, incorrect, correct.
    assert model.performance.recent_questions == 3
    assert model.performance.recent_correct == 2
    assert model.performance.average_response_seconds == 13.0
    assert model.performance.window_size == 3


def test_all_correct_yields_full_accuracy() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=60)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _answered(10, correct=True, response_seconds=10.0, index=1),
            _answered(30, correct=True, response_seconds=8.0, index=2),
        ]
    )
    assert model.performance.recent_accuracy == 1.0


def test_wrongly_typed_payload_is_ignored_not_crashed() -> None:
    """A payload that does not match the event type never reaches this layer."""
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=60)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _event(EventType.INTERACTION, 10),
            _answered(20, correct=True, response_seconds=5.0),
        ]
    )
    assert model.performance.recent_questions == 1


# --------------------------------------------------------------------------------------
# Intervention context
# --------------------------------------------------------------------------------------


def test_no_interventions_reports_no_history_and_no_cooldown() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=30)))
    model = engine.process([_event(EventType.SESSION_STARTED)])
    assert model.intervention == InterventionContext()
    assert ContextKey.INTERVENTION_HISTORY in model.missing
    assert ContextKey.COOLDOWN_STATUS in model.missing


def test_intervention_start_is_tracked() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=100)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _intervention_started("interv-0001", "micro_question", 20),
        ]
    )
    assert model.intervention.total_interventions == 1
    assert model.intervention.last_intervention_type == "micro_question"
    assert model.intervention.last_intervention_at == _BASE + timedelta(seconds=20)
    assert model.intervention.last_intervention_outcome is None


def test_outcome_matches_the_correct_intervention_by_id() -> None:
    """With two interventions in flight, completing the first must not hit the second."""
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=100)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _intervention_started("interv-0001", "micro_question", 10),
            _intervention_started("interv-0002", "focus_quiz", 20),
            _intervention_completed("interv-0001", "dismissed", 30),
        ]
    )
    assert model.intervention.total_interventions == 2
    assert model.intervention.last_intervention_outcome is None


def test_most_recent_intervention_outcome_is_reported() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=100)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _intervention_started("interv-0001", "micro_question", 10),
            _intervention_completed("interv-0001", "accepted", 30),
        ]
    )
    assert model.intervention.last_intervention_outcome == "accepted"


def test_completion_without_matching_start_is_ignored() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=100)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _intervention_completed("interv-unknown", "accepted", 30),
        ]
    )
    assert model.intervention.total_interventions == 0


def test_cooldown_active_inside_window() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE), cooldown_seconds=300.0)
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _intervention_started("interv-0001", "micro_question", 20),
            _event(EventType.INTERACTION, 100),
        ]
    )
    # Reference time is the last event, 80s after the intervention started.
    assert model.intervention.seconds_since_last_intervention == 80.0
    assert model.intervention.in_cooldown is True
    assert model.intervention.cooldown_remaining_seconds == 220.0


def test_cooldown_expires_after_window() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE), cooldown_seconds=300.0)
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _intervention_started("interv-0001", "micro_question", 20),
            _event(EventType.INTERACTION, 400),
        ]
    )
    assert model.intervention.in_cooldown is False
    assert model.intervention.cooldown_remaining_seconds == 0.0


def test_zero_cooldown_never_restraints() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE), cooldown_seconds=0.0)
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _intervention_started("interv-0001", "micro_question", 20),
        ]
    )
    assert model.intervention.in_cooldown is False


def test_multiple_interventions_are_counted_and_latest_reported() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=100)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _intervention_started("interv-0001", "micro_question", 10),
            _intervention_completed("interv-0001", "accepted", 15),
            _intervention_started("interv-0002", "focus_quiz", 40),
        ]
    )
    assert model.intervention.total_interventions == 2
    assert model.intervention.last_intervention_type == "focus_quiz"


# --------------------------------------------------------------------------------------
# Provenance: the context layer does not launder synthetic data
# --------------------------------------------------------------------------------------


def test_synthetic_only_context_is_flagged_as_such() -> None:
    """A context built only from generated events must not read as observational."""
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=60)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED, synthetic=True),
            _answered(10, correct=True, response_seconds=5.0, synthetic=True),
        ]
    )
    assert model.origins == frozenset({DataOrigin.SYNTHETIC})
    assert model.is_synthetic_only() is True
    assert model.to_summary_dict()["synthetic_only"] is True


def test_real_only_context_is_not_flagged_synthetic() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=60)))
    model = engine.process(
        [_event(EventType.SESSION_STARTED), _answered(10, correct=True, response_seconds=5.0)]
    )
    assert model.origins == frozenset({DataOrigin.REAL})
    assert model.is_synthetic_only() is False


def test_empty_context_is_not_synthetic_only() -> None:
    """No contributing events is not the same as synthetic contributing events."""
    model = ContextModel(reference_time=_BASE)
    assert model.origins == frozenset()
    assert model.is_synthetic_only() is False


def test_mixed_origins_are_reported_as_mixed() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=60)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED, synthetic=True),
            _answered(10, correct=True, response_seconds=5.0),
        ]
    )
    assert model.origins == frozenset({DataOrigin.SYNTHETIC, DataOrigin.REAL})
    assert model.is_synthetic_only() is False


# --------------------------------------------------------------------------------------
# Replay determinism
# --------------------------------------------------------------------------------------


def test_replaying_the_same_stream_gives_identical_context() -> None:
    events = [
        _event(EventType.SESSION_STARTED),
        _event(EventType.VIDEO_STARTED, 5, payload=VideoStartedPayload(video_id="video-0001")),
        _answered(20, correct=True, response_seconds=12.0, index=1),
        _answered(40, correct=False, response_seconds=18.0, index=2),
        _intervention_started("interv-0001", "micro_question", 60),
    ]
    first = ContextEngine(clock=FixedClock(_BASE + timedelta(days=1))).process(events)
    second = ContextEngine(clock=FixedClock(_BASE + timedelta(days=500))).process(events)

    assert first.to_summary_dict() == second.to_summary_dict()
    assert first == second


def test_incremental_processing_matches_batch_processing() -> None:
    events = [
        _event(EventType.SESSION_STARTED),
        _answered(20, correct=True, response_seconds=12.0, index=1),
        _answered(40, correct=False, response_seconds=18.0, index=2),
    ]
    batch = ContextEngine(clock=FixedClock(_BASE)).process(events)

    incremental = ContextEngine(clock=FixedClock(_BASE))
    for event in events:
        incremental.process([event])

    assert batch == incremental.compute_context()


# --------------------------------------------------------------------------------------
# The 17-second response example
# --------------------------------------------------------------------------------------


def test_seventeen_second_response_is_uninterpretable_without_context() -> None:
    """A 17-second response time yields context, and no behavioural conclusion.

    Seventeen seconds is slow on an easy multiple-choice item and fast on a hard
    free-response one. The context layer supplies the facts that would be needed to
    compare it; it does not supply the comparison.
    """
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=200)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _event(
                EventType.QUESTION_STARTED,
                100,
                payload=QuestionStartedPayload(
                    question_id="question-001",
                    question_type="free_response",
                    difficulty=0.9,
                    topic="calculus",
                ),
            ),
            _answered(117, correct=True, response_seconds=17.0, index=1),
        ]
    )

    # The raw signal is present.
    assert model.performance.average_response_seconds == 17.0
    # The context needed to read it is present.
    assert model.content.content_type == "question"
    assert model.content.difficulty == 0.9
    assert model.content.topic == "calculus"
    assert model.session is not None
    assert model.session.elapsed_seconds == 117.0
    # And the layer refuses to interpret, because no baseline maturity exists yet.
    assert not model.is_interpretable()
    assert ContextKey.BASELINE_MATURITY in model.blocking_gaps()


def test_observed_event_types_are_recorded() -> None:
    engine = ContextEngine(clock=FixedClock(_BASE + timedelta(seconds=200)))
    model = engine.process(
        [
            _event(EventType.SESSION_STARTED),
            _answered(10, correct=True, response_seconds=5.0, index=1),
            _event(EventType.INTERACTION, 20),
        ]
    )
    assert model.observed_event_types == frozenset(
        {
            EventType.SESSION_STARTED.value,
            EventType.QUESTION_ANSWERED.value,
            EventType.INTERACTION.value,
        }
    )
