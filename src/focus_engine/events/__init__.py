"""Event layer: typed behavioural event vocabulary, validation, and ingestion.

Every event that enters the engine passes through this layer. Validation enforces the
type–payload correspondence, pseudonymous identifiers, timezone-aware timestamps, and
the synthetic-origin invariants.
"""

from __future__ import annotations

from focus_engine.events.store import (
    EventQuery,
    EventStore,
    EventStoreError,
    InMemoryEventStore,
    JsonlEventStore,
    SessionBounds,
)
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
from focus_engine.events.validation import (
    EventIngestor,
    EventIngestorConfig,
    IngestError,
    IngestResult,
    validate_event,
)

__all__ = [
    "ALL_EVENT_TYPES",
    "ContentClosedPayload",
    "ContentOpenedPayload",
    "EventEnvelope",
    "EventIngestor",
    "EventIngestorConfig",
    "EventQuery",
    "EventStore",
    "EventStoreError",
    "EventType",
    "InactivityEndedPayload",
    "InactivityStartedPayload",
    "InMemoryEventStore",
    "IngestError",
    "IngestResult",
    "InteractionPayload",
    "InterventionCompletedPayload",
    "InterventionStartedPayload",
    "JsonlEventStore",
    "PageChangedPayload",
    "QuestionAnsweredPayload",
    "QuestionReviewedPayload",
    "QuestionSkippedPayload",
    "QuestionStartedPayload",
    "SessionBounds",
    "SessionEndedPayload",
    "SessionStartedPayload",
    "VideoCompletedPayload",
    "VideoPausedPayload",
    "VideoResumedPayload",
    "VideoStartedPayload",
    "validate_event",
]
