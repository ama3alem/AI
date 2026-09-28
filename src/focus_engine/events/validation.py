"""Event validation and ingestion.

The event layer is the boundary between untrusted input and the rest of the engine.
Every event passes through validation before any other layer sees it, so a downstream
component can assume a well-formed, type-consistent event rather than guarding against
missing fields and malformed values.

Two entry points exist:

* :func:`validate_event` — parse a dictionary into an :class:`EventEnvelope` or raise
  with a structured error.
* :class:`EventIngestor` — batch validation with duplicate detection and error
  aggregation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from focus_engine.events.types import EventEnvelope
from focus_engine.schemas.primitives import EventId

__all__ = [
    "EventIngestor",
    "EventIngestorConfig",
    "IngestError",
    "IngestResult",
    "validate_event",
]


class IngestError(Exception):
    """Structured error from event ingestion.

    Carries enough context to be useful in logs and in API responses: the field that
    failed, the value that was rejected, and the reason.

    Attributes:
        message: Human-readable error description.
        field: Dot-separated path to the failing field, e.g. ``payload.question_id``.
        value: The rejected value, serialised safely.
        error_type: Category of error, e.g. ``validation``, ``duplicate``.
    """

    __slots__ = ("message", "field", "value", "error_type")

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: Any = None,
        error_type: str = "validation",
    ) -> None:
        """Create an ingest error.

        Args:
            message: Human-readable description.
            field: Path to the failing field.
            value: The rejected value.
            error_type: Error category.
        """
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value
        self.error_type = error_type

    def to_dict(self) -> dict[str, Any]:
        """Render the error as a JSON-serialisable dictionary.

        Returns:
            A dictionary with ``message``, ``field``, ``value``, and ``error_type`` keys.
        """
        return {
            "message": self.message,
            "field": self.field,
            "value": self.value,
            "error_type": self.error_type,
        }


def validate_event(data: dict[str, Any]) -> EventEnvelope:
    """Validate a raw dictionary as an event envelope.

    This is the single entry point for turning untrusted input into a validated event.
    All other layers receive :class:`EventEnvelope` objects, not dictionaries.

    Args:
        data: A dictionary containing event fields.

    Returns:
        A validated :class:`EventEnvelope`.

    Raises:
        IngestError: If validation fails. The error carries the field path and rejected
            value for debugging.
    """
    try:
        return EventEnvelope.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        field_path = ".".join(str(loc) for loc in first.get("loc", ()))
        raise IngestError(
            message=first.get("msg", "validation error"),
            field=field_path or None,
            value=first.get("input"),
            error_type=first.get("type", "validation"),
        ) from exc


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Result of ingesting a batch of events.

    Attributes:
        accepted: Events that passed validation and duplicate checks.
        rejected: Errors for events that failed validation.
        duplicate_count: Number of events rejected as duplicates.
    """

    accepted: tuple[EventEnvelope, ...]
    rejected: tuple[IngestError, ...]
    duplicate_count: int = 0

    @property
    def total_processed(self) -> int:
        """Total events processed, both accepted and rejected.

        Returns:
            The sum of accepted, rejected, and duplicate counts.
        """
        return len(self.accepted) + len(self.rejected) + self.duplicate_count

    @property
    def acceptance_rate(self) -> float:
        """Fraction of processed events that were accepted.

        Returns:
            Acceptance rate in ``[0, 1]``, or ``0.0`` if no events were processed.
        """
        total = self.total_processed
        return len(self.accepted) / total if total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class EventIngestorConfig:
    """Configuration for the event ingestor.

    Attributes:
        reject_duplicates: Whether to reject events with an ``event_id`` that has already
            been seen in this batch or in a previous batch for this ingestor instance.
        max_batch_size: Maximum number of events accepted in a single batch. Events
            beyond this count are rejected with ``batch_size_exceeded``.
        strict_mode: When ``True``, the first validation error raises an exception.
            When ``False``, errors are collected and returned in ``rejected``.
    """

    reject_duplicates: bool = True
    max_batch_size: int = 1000
    strict_mode: bool = False


@dataclass
class EventIngestor:
    """Batch event validation with duplicate detection.

    The ingestor maintains a set of seen ``event_id`` values for the lifetime of the
    instance. This is suitable for a single ingestion session, not for persistent
    deduplication across restarts.

    Attributes:
        config: Ingestor configuration.
        seen_event_ids: Set of event IDs seen so far.
    """

    config: EventIngestorConfig = field(default_factory=EventIngestorConfig)
    seen_event_ids: set[EventId] = field(default_factory=set)

    def ingest(self, events: Iterable[dict[str, Any]]) -> IngestResult:
        """Validate and ingest a batch of events.

        Args:
            events: Iterable of raw event dictionaries.

        Returns:
            An :class:`IngestResult` with accepted events, rejected errors, and the
            duplicate count.

        Raises:
            IngestError: In strict mode, the first validation error raises rather than
                being collected.
        """
        accepted: list[EventEnvelope] = []
        rejected: list[IngestError] = []
        duplicates = 0

        for idx, raw in enumerate(events):
            if idx >= self.config.max_batch_size:
                rejected.append(
                    IngestError(
                        message=f"batch size exceeds configured maximum of {self.config.max_batch_size}",
                        field="batch",
                        value=idx,
                        error_type="batch_size_exceeded",
                    )
                )
                continue

            try:
                event = validate_event(raw)
            except IngestError as exc:
                if self.config.strict_mode:
                    raise
                rejected.append(exc)
                continue

            if self.config.reject_duplicates and event.event_id in self.seen_event_ids:
                duplicates += 1
                rejected.append(
                    IngestError(
                        message="duplicate event_id",
                        field="event_id",
                        value=event.event_id,
                        error_type="duplicate",
                    )
                )
                continue

            if self.config.reject_duplicates:
                self.seen_event_ids.add(event.event_id)

            accepted.append(event)

        return IngestResult(
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            duplicate_count=duplicates,
        )

    def reset_seen_ids(self) -> None:
        """Clear the set of seen event IDs.

        This is useful when the ingestor is reused across independent sessions.
        """
        self.seen_event_ids.clear()
