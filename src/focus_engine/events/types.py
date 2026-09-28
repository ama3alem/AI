"""Event type taxonomy and per-type payload definitions.

Every behavioural event that enters the engine is validated against this taxonomy. An
unrecognised `event_type` is rejected at the boundary, not logged as a warning and
passed through. The reason is auditability: if a downstream layer sees an event, it is
guaranteed to be a known type with a known payload schema.

Design rules enforced here:

* **Pseudonymous identifiers only.** Every event carries `learner_id`, `session_id`, and
  `event_id`. No name, email, device fingerprint, or other directly identifying value is
  representable.
* **Timezone-aware timestamps only.** Naive datetimes are rejected; the engine never has
  to guess an offset.
* **Typed payloads.** Each `event_type` has a dedicated payload model. A `question_answered`
  event must carry `question_id` and `correct`; a `video_paused` event must carry
  `video_id` and `position_seconds`. The union discriminator ensures a mismatched payload
  is a validation error, not a silent field omission.
* **Content identifiers are opaque tokens.** `question_id`, `video_id`, and `content_id`
  carry no semantic content. They are not foreign keys into a content database the
  engine does not own, and they are not human-readable titles.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.schemas.primitives import (
    DataOrigin,
    EventId,
    LearnerId,
    Provenance,
    SessionId,
    SyntheticDataStamp,
    Timestamp,
    utc_now,
)

__all__ = [
    "ALL_EVENT_TYPES",
    "ContentClosedPayload",
    "ContentOpenedPayload",
    "EventEnvelope",
    "EventType",
    "InactivityEndedPayload",
    "InactivityStartedPayload",
    "InteractionPayload",
    "InterventionCompletedPayload",
    "InterventionStartedPayload",
    "PageChangedPayload",
    "QuestionAnsweredPayload",
    "QuestionReviewedPayload",
    "QuestionSkippedPayload",
    "QuestionStartedPayload",
    "SessionEndedPayload",
    "SessionStartedPayload",
    "VideoCompletedPayload",
    "VideoPausedPayload",
    "VideoResumedPayload",
    "VideoStartedPayload",
]


class EventType(StrEnum):
    """Recognised behavioural event types.

    The closed vocabulary is deliberate. An open-ended string would permit a downstream
    layer to silently ignore a typo, or to handle two differently-spelt synonyms
    inconsistently. Here, a typo is a validation error at ingest, and the set of
    handled types is exhaustively enumerable.
    """

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    QUESTION_STARTED = "question_started"
    QUESTION_ANSWERED = "question_answered"
    QUESTION_SKIPPED = "question_skipped"
    QUESTION_REVIEWED = "question_reviewed"
    VIDEO_STARTED = "video_started"
    VIDEO_PAUSED = "video_paused"
    VIDEO_RESUMED = "video_resumed"
    VIDEO_COMPLETED = "video_completed"
    CONTENT_OPENED = "content_opened"
    CONTENT_CLOSED = "content_closed"
    PAGE_CHANGED = "page_changed"
    INTERACTION = "interaction"
    INACTIVITY_STARTED = "inactivity_started"
    INACTIVITY_ENDED = "inactivity_ended"
    INTERVENTION_STARTED = "intervention_started"
    INTERVENTION_COMPLETED = "intervention_completed"


#: The complete, closed set of event types. Nothing outside this tuple may appear in an
#: event envelope, which keeps downstream switches exhaustive.
ALL_EVENT_TYPES: Final[tuple[str, ...]] = tuple(et.value for et in EventType)


# --------------------------------------------------------------------------------------
# Per-type payload definitions
# --------------------------------------------------------------------------------------

_ContentId = Annotated[str, Field(min_length=8, max_length=128)]


class SessionStartedPayload(BaseModel):
    """Payload for `session_started` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    platform: str | None = Field(default=None, max_length=64)
    """Optional platform or client identifier, e.g. ``web``, ``mobile``, ``desktop``."""

    entry_point: str | None = Field(default=None, max_length=128)
    """Optional entry point, e.g. ``course/lesson/123``."""


class SessionEndedPayload(BaseModel):
    """Payload for `session_ended` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    duration_seconds: float | None = Field(default=None, ge=0.0)
    """Session duration if known. May be ``None`` if the session was terminated
    abnormally."""

    reason: str | None = Field(default=None, max_length=64)
    """Optional termination reason, e.g. ``logout``, ``timeout``, ``error``."""


class QuestionStartedPayload(BaseModel):
    """Payload for `question_started` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: _ContentId
    """Opaque identifier for the question."""

    question_type: str | None = Field(default=None, max_length=64)
    """Optional question type, e.g. ``multiple_choice``, ``short_answer``."""

    difficulty: float | None = Field(default=None, ge=0.0, le=1.0)
    """Optional normalised difficulty in ``[0, 1]``."""

    topic: str | None = Field(default=None, max_length=128)
    """Optional topic or skill identifier."""


class QuestionAnsweredPayload(BaseModel):
    """Payload for `question_answered` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: _ContentId
    """Opaque identifier for the question."""

    correct: bool
    """Whether the answer was correct."""

    response_seconds: float = Field(ge=0.0)
    """Time taken to answer, in seconds."""

    attempt_number: int = Field(default=1, ge=1)
    """Attempt number for this question within the session."""

    answer_value: str | None = Field(default=None, max_length=1024)
    """Optional answer representation. Not evaluated by the engine."""


class QuestionSkippedPayload(BaseModel):
    """Payload for `question_skipped` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: _ContentId
    """Opaque identifier for the skipped question."""

    time_on_question_seconds: float = Field(ge=0.0)
    """Time spent on the question before skipping."""


class QuestionReviewedPayload(BaseModel):
    """Payload for `question_reviewed` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: _ContentId
    """Opaque identifier for the reviewed question."""

    previous_answer_correct: bool | None = None
    """Whether the previous answer was correct, if applicable."""


class VideoStartedPayload(BaseModel):
    """Payload for `video_started` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: _ContentId
    """Opaque identifier for the video."""

    duration_seconds: float | None = Field(default=None, ge=0.0)
    """Total video duration if known."""

    playback_rate: float = Field(default=1.0, gt=0.0)
    """Initial playback rate."""


class VideoPausedPayload(BaseModel):
    """Payload for `video_paused` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: _ContentId
    """Opaque identifier for the video."""

    position_seconds: float = Field(ge=0.0)
    """Pause position within the video."""

    watch_time_seconds: float = Field(ge=0.0)
    """Cumulative watch time for this video in the session."""

    pause_count: int = Field(default=1, ge=1)
    """Number of pauses so far."""


class VideoResumedPayload(BaseModel):
    """Payload for `video_resumed` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: _ContentId
    """Opaque identifier for the video."""

    position_seconds: float = Field(ge=0.0)
    """Resume position within the video."""

    pause_duration_seconds: float = Field(ge=0.0)
    """Duration of the pause."""


class VideoCompletedPayload(BaseModel):
    """Payload for `video_completed` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: _ContentId
    """Opaque identifier for the video."""

    total_watch_time_seconds: float = Field(ge=0.0)
    """Total time spent watching this video."""

    completion_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    """Fraction of video watched."""


class ContentOpenedPayload(BaseModel):
    """Payload for `content_opened` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content_id: _ContentId
    """Opaque identifier for the content."""

    content_type: str | None = Field(default=None, max_length=64)
    """Optional content type, e.g. ``reading``, ``exercise``, ``external_link``."""


class ContentClosedPayload(BaseModel):
    """Payload for `content_closed` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content_id: _ContentId
    """Opaque identifier for the content."""

    time_open_seconds: float = Field(ge=0.0)
    """Time the content was open."""


class PageChangedPayload(BaseModel):
    """Payload for `page_changed` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_page: str | None = Field(default=None, max_length=256)
    """Page or route being left."""

    to_page: str = Field(max_length=256)
    """Page or route being entered."""

    navigation_type: str | None = Field(default=None, max_length=64)
    """Optional navigation type, e.g. ``click``, ``browser_back``, ``auto_redirect``."""


class InteractionPayload(BaseModel):
    """Payload for generic `interaction` events.

    Used for clicks, scrolls, keystrokes, or other interactions that do not warrant a
    dedicated event type. This is a catch-all; the feature engine may or may not use it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    interaction_type: str = Field(max_length=64)
    """Type of interaction, e.g. ``click``, ``scroll``, ``keypress``."""

    element_id: str | None = Field(default=None, max_length=128)
    """Optional element identifier."""

    coordinates: tuple[float, float] | None = None
    """Optional (x, y) coordinates for pointer events."""

    value: str | None = Field(default=None, max_length=256)
    """Optional value, e.g. keystroke character (non-sensitive)."""


class InactivityStartedPayload(BaseModel):
    """Payload for `inactivity_started` events.

    Inactivity is a session-level property: the learner has stopped interacting with the
    platform. The threshold that triggers this event is a configuration concern, not a
    schema concern.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    last_interaction_type: str | None = Field(default=None, max_length=64)
    """Type of the last interaction before inactivity."""

    last_interaction_timestamp: Timestamp | None = None
    """Timestamp of the last interaction."""


class InactivityEndedPayload(BaseModel):
    """Payload for `inactivity_ended` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inactivity_duration_seconds: float = Field(ge=0.0)
    """Duration of the inactivity period."""

    resume_interaction_type: str | None = Field(default=None, max_length=64)
    """Type of the interaction that ended the inactivity."""


class InterventionStartedPayload(BaseModel):
    """Payload for `intervention_started` events.

    An intervention is a deliberate action taken by the system in response to a predicted
    behavioural state. The `intervention_type` is a controlled vocabulary that the
    intervention engine owns.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intervention_id: str = Field(min_length=8, max_length=128)
    """Unique identifier for this intervention instance."""

    intervention_type: str = Field(max_length=64)
    """Type of intervention, e.g. ``micro_question``, ``focus_quiz``, ``recap``."""

    trigger_probability: float = Field(ge=0.0, le=1.0)
    """The predicted decline probability that triggered the intervention."""

    trigger_state: str = Field(max_length=64)
    """The behavioural state that triggered the intervention."""


class InterventionCompletedPayload(BaseModel):
    """Payload for `intervention_completed` events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intervention_id: str = Field(min_length=8, max_length=128)
    """Identifier of the intervention instance."""

    outcome: str = Field(max_length=64)
    """Outcome, e.g. ``accepted``, ``dismissed``, ``ignored``, ``timeout``."""

    response_seconds: float | None = Field(default=None, ge=0.0)
    """Time taken to respond, if applicable."""


# --------------------------------------------------------------------------------------
# Event envelope
# --------------------------------------------------------------------------------------

_EventPayload = (
    SessionStartedPayload
    | SessionEndedPayload
    | QuestionStartedPayload
    | QuestionAnsweredPayload
    | QuestionSkippedPayload
    | QuestionReviewedPayload
    | VideoStartedPayload
    | VideoPausedPayload
    | VideoResumedPayload
    | VideoCompletedPayload
    | ContentOpenedPayload
    | ContentClosedPayload
    | PageChangedPayload
    | InteractionPayload
    | InactivityStartedPayload
    | InactivityEndedPayload
    | InterventionStartedPayload
    | InterventionCompletedPayload
)


class EventEnvelope(BaseModel):
    """The top-level event schema.

    Every behavioural event enters the engine in this envelope. Validation ensures that
    the `event_type` and `payload` match, and that all identifiers and timestamps are
    well-formed.

    Attributes:
        event_id: Unique identifier for this event.
        learner_id: Pseudonymous learner identifier.
        session_id: Pseudonymous session identifier.
        timestamp: Timezone-aware event timestamp.
        event_type: Discriminator determining the payload type.
        payload: Type-specific event data.
        origin: Whether this is real or synthetic data.
        provenance: How this event came to exist.
        synthetic_stamp: Present and required when ``origin`` is ``SYNTHETIC``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: EventId
    learner_id: LearnerId
    session_id: SessionId
    timestamp: Timestamp = Field(default_factory=utc_now)
    event_type: EventType
    payload: _EventPayload
    origin: DataOrigin = DataOrigin.REAL
    provenance: Provenance = Provenance.OBSERVED
    synthetic_stamp: SyntheticDataStamp | None = None

    @model_validator(mode="after")
    def _validate_payload_type_match(self) -> EventEnvelope:
        """Ensure the payload type matches the event_type discriminator.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the payload type does not correspond to the event_type.
        """
        expected: dict[EventType, type[BaseModel]] = {
            EventType.SESSION_STARTED: SessionStartedPayload,
            EventType.SESSION_ENDED: SessionEndedPayload,
            EventType.QUESTION_STARTED: QuestionStartedPayload,
            EventType.QUESTION_ANSWERED: QuestionAnsweredPayload,
            EventType.QUESTION_SKIPPED: QuestionSkippedPayload,
            EventType.QUESTION_REVIEWED: QuestionReviewedPayload,
            EventType.VIDEO_STARTED: VideoStartedPayload,
            EventType.VIDEO_PAUSED: VideoPausedPayload,
            EventType.VIDEO_RESUMED: VideoResumedPayload,
            EventType.VIDEO_COMPLETED: VideoCompletedPayload,
            EventType.CONTENT_OPENED: ContentOpenedPayload,
            EventType.CONTENT_CLOSED: ContentClosedPayload,
            EventType.PAGE_CHANGED: PageChangedPayload,
            EventType.INTERACTION: InteractionPayload,
            EventType.INACTIVITY_STARTED: InactivityStartedPayload,
            EventType.INACTIVITY_ENDED: InactivityEndedPayload,
            EventType.INTERVENTION_STARTED: InterventionStartedPayload,
            EventType.INTERVENTION_COMPLETED: InterventionCompletedPayload,
        }
        if not isinstance(self.payload, expected[self.event_type]):
            raise ValueError(
                f"payload type {type(self.payload).__name__} does not match "
                f"event_type {self.event_type.value}; expected "
                f"{expected[self.event_type].__name__}"
            )
        return self

    @model_validator(mode="after")
    def _require_synthetic_stamp_when_origin_is_synthetic(self) -> EventEnvelope:
        """Require ``synthetic_stamp`` when ``origin`` is ``SYNTHETIC``.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If ``origin`` is ``SYNTHETIC`` and ``synthetic_stamp`` is missing.
        """
        if self.origin is DataOrigin.SYNTHETIC and self.synthetic_stamp is None:
            raise ValueError(
                "synthetic_stamp is required when origin is SYNTHETIC. "
                "Every synthetic event must carry the in-band warning stamp."
            )
        return self

    @model_validator(mode="after")
    def _reject_synthetic_origin_with_observed_provenance(self) -> EventEnvelope:
        """Reject ``origin=SYNTHETIC`` combined with ``provenance=OBSERVED``.

        A synthetic event is never an observation; it is generated by a simulator.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the combination is inconsistent.
        """
        if self.origin is DataOrigin.SYNTHETIC and self.provenance is Provenance.OBSERVED:
            raise ValueError(
                "provenance must not be 'observed' when origin is SYNTHETIC; "
                "use 'synthetic_label' or another appropriate provenance."
            )
        return self
