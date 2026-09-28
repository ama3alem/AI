"""Unit tests for :mod:`focus_engine.events.types`.

Covers the event type taxonomy, per-type payload schemas, the envelope validator, and
the invariants that prevent a synthetic event from being represented as an observed one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from focus_engine.events.types import (
    ALL_EVENT_TYPES,
    ContentClosedPayload,
    ContentOpenedPayload,
    EventEnvelope,
    EventType,
    InactivityEndedPayload,
    InactivityStartedPayload,
    InteractionPayload,
    InterventionCompletedPayload,
    InterventionStartedPayload,
    PageChangedPayload,
    QuestionAnsweredPayload,
    QuestionReviewedPayload,
    QuestionSkippedPayload,
    QuestionStartedPayload,
    SessionEndedPayload,
    SessionStartedPayload,
    VideoCompletedPayload,
    VideoPausedPayload,
    VideoResumedPayload,
    VideoStartedPayload,
)
from focus_engine.schemas.primitives import DataOrigin, Provenance, SyntheticDataStamp, utc_now

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------
# Event type taxonomy
# --------------------------------------------------------------------------------------


def test_all_event_types_matches_enum() -> None:
    assert set(ALL_EVENT_TYPES) == {et.value for et in EventType}


def test_no_event_type_implies_mental_state() -> None:
    """Event type names must describe observable behaviour, not cognition."""
    forbidden = {"focus", "attention", "distracted", "mind", "aware", "engaged_mind"}
    for et in EventType:
        for term in forbidden:
            assert term not in et.value


# --------------------------------------------------------------------------------------
# Session events
# --------------------------------------------------------------------------------------


def test_session_started_payload_accepts_optional_fields() -> None:
    payload = SessionStartedPayload(platform="web", entry_point="course/lesson/123")
    assert payload.platform == "web"
    assert payload.entry_point == "course/lesson/123"


def test_session_started_payload_accepts_minimal() -> None:
    payload = SessionStartedPayload()
    assert payload.platform is None
    assert payload.entry_point is None


def test_session_ended_payload_requires_non_negative_duration() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SessionEndedPayload(duration_seconds=-1.0)


def test_session_ended_payload_accepts_unknown_reason() -> None:
    payload = SessionEndedPayload(duration_seconds=120.0, reason="timeout")
    assert payload.reason == "timeout"


# --------------------------------------------------------------------------------------
# Question events
# --------------------------------------------------------------------------------------


def test_question_started_payload_requires_question_id() -> None:
    with pytest.raises(ValidationError, match="question_id"):
        QuestionStartedPayload()


def test_question_started_payload_accepts_optional_metadata() -> None:
    payload = QuestionStartedPayload(
        question_id="question-001",
        question_type="multiple_choice",
        difficulty=0.7,
        topic="algebra",
    )
    assert payload.question_id == "question-001"
    assert payload.difficulty == 0.7


def test_question_started_payload_rejects_out_of_range_difficulty() -> None:
    with pytest.raises(ValidationError):
        QuestionStartedPayload(question_id="q1", difficulty=1.5)


def test_question_answered_payload_requires_correct_and_response_time() -> None:
    payload = QuestionAnsweredPayload(
        question_id="question-001", correct=True, response_seconds=15.3
    )
    assert payload.correct is True
    assert payload.response_seconds == 15.3
    assert payload.attempt_number == 1


def test_question_answered_payload_rejects_non_positive_attempt() -> None:
    with pytest.raises(ValidationError):
        QuestionAnsweredPayload(
            question_id="question-001", correct=False, response_seconds=5.0, attempt_number=0
        )


def test_question_answered_payload_rejects_short_content_id() -> None:
    """Opaque content identifiers are length-bounded to avoid ad-hoc short keys."""
    with pytest.raises(ValidationError):
        QuestionAnsweredPayload(question_id="q1", correct=False, response_seconds=5.0)


def test_question_skipped_payload_requires_time_on_question() -> None:
    payload = QuestionSkippedPayload(question_id="question-001", time_on_question_seconds=30.0)
    assert payload.time_on_question_seconds == 30.0


def test_question_reviewed_payload_accepts_optional_previous_correct() -> None:
    payload = QuestionReviewedPayload(question_id="question-001", previous_answer_correct=False)
    assert payload.previous_answer_correct is False


# --------------------------------------------------------------------------------------
# Video events
# --------------------------------------------------------------------------------------


def test_video_started_payload_requires_video_id() -> None:
    payload = VideoStartedPayload(video_id="video-001", duration_seconds=300.0)
    assert payload.video_id == "video-001"
    assert payload.duration_seconds == 300.0
    assert payload.playback_rate == 1.0


def test_video_paused_payload_requires_position_and_watch_time() -> None:
    payload = VideoPausedPayload(
        video_id="video-001",
        position_seconds=45.0,
        watch_time_seconds=50.0,
        pause_count=2,
    )
    assert payload.position_seconds == 45.0
    assert payload.pause_count == 2


def test_video_resumed_payload_requires_pause_duration() -> None:
    payload = VideoResumedPayload(
        video_id="video-001", position_seconds=45.0, pause_duration_seconds=10.0
    )
    assert payload.pause_duration_seconds == 10.0


def test_video_completed_payload_requires_watch_time() -> None:
    payload = VideoCompletedPayload(
        video_id="video-001", total_watch_time_seconds=280.0, completion_ratio=0.93
    )
    assert payload.completion_ratio == 0.93


# --------------------------------------------------------------------------------------
# Content and navigation events
# --------------------------------------------------------------------------------------


def test_content_opened_payload_requires_content_id() -> None:
    with pytest.raises(ValidationError, match="content_id"):
        ContentOpenedPayload()
    with pytest.raises(ValidationError, match="content_id"):
        ContentOpenedPayload(content_id="short")


def test_content_closed_payload_requires_time_open() -> None:
    with pytest.raises(ValidationError, match="time_open_seconds"):
        ContentClosedPayload(content_id="content-001")
    assert (
        ContentClosedPayload(content_id="content-001", time_open_seconds=90.0).time_open_seconds
        == 90.0
    )


def test_page_changed_payload_requires_to_page() -> None:
    payload = PageChangedPayload(to_page="/lesson/2", from_page="/lesson/1")
    assert payload.to_page == "/lesson/2"


def test_interaction_payload_requires_type() -> None:
    payload = InteractionPayload(interaction_type="click", element_id="submit-btn")
    assert payload.interaction_type == "click"


def test_interaction_payload_accepts_coordinates() -> None:
    payload = InteractionPayload(interaction_type="click", coordinates=(120.0, 340.0))
    assert payload.coordinates == (120.0, 340.0)


# --------------------------------------------------------------------------------------
# Inactivity events
# --------------------------------------------------------------------------------------


def test_inactivity_started_payload_accepts_optional_last_interaction() -> None:
    ts = datetime(2026, 9, 26, 10, 5, 0, tzinfo=UTC)
    payload = InactivityStartedPayload(last_interaction_type="click", last_interaction_timestamp=ts)
    assert payload.last_interaction_timestamp == ts


def test_inactivity_ended_payload_requires_duration() -> None:
    payload = InactivityEndedPayload(
        inactivity_duration_seconds=120.0, resume_interaction_type="scroll"
    )
    assert payload.inactivity_duration_seconds == 120.0


# --------------------------------------------------------------------------------------
# Intervention events
# --------------------------------------------------------------------------------------


def test_intervention_started_payload_requires_all_fields() -> None:
    payload = InterventionStartedPayload(
        intervention_id="interv-001",
        intervention_type="micro_question",
        trigger_probability=0.72,
        trigger_state="declining",
    )
    assert payload.intervention_type == "micro_question"
    assert payload.trigger_probability == 0.72


def test_intervention_completed_payload_requires_outcome() -> None:
    payload = InterventionCompletedPayload(
        intervention_id="interv-001", outcome="accepted", response_seconds=8.5
    )
    assert payload.outcome == "accepted"


# --------------------------------------------------------------------------------------
# Event envelope
# --------------------------------------------------------------------------------------


def test_event_envelope_requires_matching_payload_type() -> None:
    with pytest.raises(ValidationError, match="does not match event_type"):
        EventEnvelope(
            event_id="event-mismatch-001",
            learner_id="learner-0001",
            session_id="session-0001",
            event_type=EventType.QUESTION_ANSWERED,
            payload=VideoStartedPayload(video_id="video-0001"),  # wrong type
        )


def test_event_envelope_accepts_valid_event() -> None:
    now = utc_now()
    event = EventEnvelope(
        event_id="event-001",
        learner_id="learner-001",
        session_id="session-001",
        timestamp=now,
        event_type=EventType.QUESTION_ANSWERED,
        payload=QuestionAnsweredPayload(
            question_id="question-001", correct=True, response_seconds=12.0
        ),
    )
    assert event.event_id == "event-001"
    assert event.timestamp == now
    assert isinstance(event.payload, QuestionAnsweredPayload)


def test_event_envelope_defaults_to_real_observed() -> None:
    event = EventEnvelope(
        event_id="event-envelope-001",
        learner_id="learner-0001",
        session_id="session-0001",
        event_type=EventType.SESSION_STARTED,
        payload=SessionStartedPayload(),
    )
    assert event.origin is DataOrigin.REAL
    assert event.provenance is Provenance.OBSERVED


def test_event_envelope_normalises_aware_timestamp_to_utc() -> None:
    tz = timezone(timedelta(hours=5, minutes=30))
    event = EventEnvelope(
        event_id="event-envelope-001",
        learner_id="learner-0001",
        session_id="session-0001",
        timestamp=datetime(2026, 9, 26, 15, 30, 0, tzinfo=tz),
        event_type=EventType.SESSION_STARTED,
        payload=SessionStartedPayload(),
    )
    assert event.timestamp.tzinfo is UTC


def test_event_envelope_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        EventEnvelope(
            event_id="event-synthetic-001",
            learner_id="learner-0001",
            session_id="session-0001",
            timestamp=datetime(2026, 9, 26, 10, 0, 0),
            event_type=EventType.SESSION_STARTED,
            payload=SessionStartedPayload(),
        )


# --------------------------------------------------------------------------------------
# Synthetic-origin invariants
# --------------------------------------------------------------------------------------


def test_synthetic_event_requires_stamp() -> None:
    with pytest.raises(ValidationError, match="synthetic_stamp is required"):
        EventEnvelope(
            event_id="event-synthetic-001",
            learner_id="learner-0001",
            session_id="session-0001",
            event_type=EventType.SESSION_STARTED,
            payload=SessionStartedPayload(),
            origin=DataOrigin.SYNTHETIC,
        )


def test_synthetic_event_cannot_claim_observed_provenance() -> None:
    with pytest.raises(ValidationError, match="provenance must not be 'observed'"):
        EventEnvelope(
            event_id="event-synthetic-001",
            learner_id="learner-0001",
            session_id="session-0001",
            event_type=EventType.SESSION_STARTED,
            payload=SessionStartedPayload(),
            origin=DataOrigin.SYNTHETIC,
            provenance=Provenance.OBSERVED,
            synthetic_stamp=SyntheticDataStamp(generator="sim", seed=1),
        )


def test_synthetic_event_with_correct_stamp_is_accepted() -> None:
    event = EventEnvelope(
        event_id="event-envelope-001",
        learner_id="learner-0001",
        session_id="session-0001",
        event_type=EventType.SESSION_STARTED,
        payload=SessionStartedPayload(),
        origin=DataOrigin.SYNTHETIC,
        provenance=Provenance.SYNTHETIC_LABEL,
        synthetic_stamp=SyntheticDataStamp(generator="simulator_v1", seed=42),
    )
    assert event.origin is DataOrigin.SYNTHETIC
    assert event.synthetic_stamp is not None
    assert event.synthetic_stamp.seed == 42


def test_real_event_does_not_require_stamp() -> None:
    event = EventEnvelope(
        event_id="event-envelope-001",
        learner_id="learner-0001",
        session_id="session-0001",
        event_type=EventType.SESSION_STARTED,
        payload=SessionStartedPayload(),
        origin=DataOrigin.REAL,
        provenance=Provenance.OBSERVED,
    )
    assert event.synthetic_stamp is None
