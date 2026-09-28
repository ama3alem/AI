"""Append-only event store with session-scoped queries.

The store is the persistence boundary for events. It is deliberately boring: an
append-only log, no updates, no deletes. Events are immutable facts about behaviour; if
one is wrong, the fix is a correction event, not an edit. Overwriting history would make
the record unauditable, and an engine that reasons about sequences cannot reason about
a log it can silently rewrite.

Two implementations satisfy the same protocol:

* :class:`InMemoryEventStore` — default, used in tests and in-process pipelines.
* :class:`JsonlEventStore` — newline-delimited JSON on disk, one event per line.

Neither exposes an update or delete method. That omission is the design.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from focus_engine.events.types import EventEnvelope, EventType
from focus_engine.schemas.primitives import DataOrigin, LearnerId, SessionId

__all__ = [
    "EventQuery",
    "EventStore",
    "EventStoreError",
    "InMemoryEventStore",
    "JsonlEventStore",
    "SessionBounds",
]


class EventStoreError(Exception):
    """Raised when an event store operation cannot be completed."""


@dataclass(frozen=True, slots=True)
class SessionBounds:
    """First and last event time for a session.

    Attributes:
        session_id: The session these bounds describe.
        first_event_at: Timestamp of the earliest event.
        last_event_at: Timestamp of the latest event.
        event_count: Number of events in the session.
    """

    session_id: SessionId
    first_event_at: datetime
    last_event_at: datetime
    event_count: int


@dataclass(frozen=True, slots=True)
class EventQuery:
    """Filter for reading events back from a store.

    All fields are optional and combine with AND. An unset filter matches everything.

    Attributes:
        learner_id: Restrict to one learner.
        session_id: Restrict to one session.
        event_types: Restrict to these event types.
        start_time: Inclusive lower bound on event time.
        end_time: Exclusive upper bound on event time.
        origin: Restrict to real or synthetic events.
        limit: Maximum events to return. ``None`` means unlimited.
    """

    learner_id: LearnerId | None = None
    session_id: SessionId | None = None
    event_types: frozenset[EventType] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    origin: DataOrigin | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        """Validate the query itself.

        Raises:
            EventStoreError: If the time range is inverted or the limit is not positive.
        """
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time > self.end_time
        ):
            raise EventStoreError("start_time must not be after end_time")
        if self.limit is not None and self.limit < 1:
            raise EventStoreError("limit must be at least 1")

    def matches(self, event: EventEnvelope) -> bool:
        """Check whether an event satisfies this query.

        Args:
            event: The candidate event.

        Returns:
            ``True`` if the event matches every populated filter.
        """
        if self.learner_id is not None and event.learner_id != self.learner_id:
            return False
        if self.session_id is not None and event.session_id != self.session_id:
            return False
        if self.event_types is not None and event.event_type not in self.event_types:
            return False
        if self.start_time is not None and event.timestamp < self.start_time:
            return False
        if self.end_time is not None and event.timestamp >= self.end_time:
            return False
        return self.origin is None or event.origin is self.origin


@runtime_checkable
class EventStore(Protocol):
    """Read/write protocol for append-only event storage.

    Implementations must preserve insertion order and must not permit mutation of stored
    events.
    """

    def append(self, event: EventEnvelope) -> None:
        """Append a single event.

        Args:
            event: The validated event to store.
        """
        ...

    def extend(self, events: Sequence[EventEnvelope]) -> None:
        """Append a batch of events, preserving order.

        Args:
            events: Events to append in order.
        """
        ...

    def query(self, query: EventQuery | None = None) -> tuple[EventEnvelope, ...]:
        """Return events matching a query, in insertion order.

        Args:
            query: Filter to apply. ``None`` returns all events.

        Returns:
            The matching events.
        """
        ...

    def __len__(self) -> int:
        """Number of stored events.

        Returns:
            The event count.
        """
        ...


@dataclass
class InMemoryEventStore:
    """In-memory append-only event store.

    Intended for tests, notebooks, and single-process pipelines where the log does not
    need to outlive the process.

    Attributes:
        _events: The append-only event log.
    """

    _events: list[EventEnvelope] = field(default_factory=list, repr=False)

    def append(self, event: EventEnvelope) -> None:
        """Append a single event.

        Args:
            event: The validated event to store.
        """
        self._events.append(event)

    def extend(self, events: Sequence[EventEnvelope]) -> None:
        """Append a batch of events, preserving order.

        Args:
            events: Events to append in order.
        """
        self._events.extend(events)

    def query(self, query: EventQuery | None = None) -> tuple[EventEnvelope, ...]:
        """Return events matching a query, in insertion order.

        Args:
            query: Filter to apply. ``None`` returns all events.

        Returns:
            The matching events.
        """
        if query is None:
            return tuple(self._events)
        results: list[EventEnvelope] = []
        for event in self._events:
            if query.matches(event):
                results.append(event)
                if query.limit is not None and len(results) >= query.limit:
                    break
        return tuple(results)

    def sessions(self) -> tuple[SessionBounds, ...]:
        """Summarise the time span of every session in the store.

        Returns:
            One :class:`SessionBounds` per session, ordered by first event time.
        """
        grouped: dict[SessionId, list[EventEnvelope]] = {}
        for event in self._events:
            grouped.setdefault(event.session_id, []).append(event)
        bounds = [
            SessionBounds(
                session_id=session_id,
                first_event_at=min(e.timestamp for e in events),
                last_event_at=max(e.timestamp for e in events),
                event_count=len(events),
            )
            for session_id, events in grouped.items()
        ]
        bounds.sort(key=lambda b: b.first_event_at)
        return tuple(bounds)

    def __len__(self) -> int:
        """Number of stored events.

        Returns:
            The event count.
        """
        return len(self._events)

    def __iter__(self) -> Iterator[EventEnvelope]:
        """Iterate events in insertion order.

        Returns:
            An iterator over the event log.
        """
        return iter(self._events)


@dataclass
class JsonlEventStore:
    """Append-only newline-delimited JSON event store.

    One event per line, UTF-8 encoded. JSONL is used rather than a database because the
    write pattern is strictly append-only: it is crash-tolerant, human-inspectable, and
    needs no server.

    The file is opened and closed per write, so the store holds no long-lived handle.
    This costs some throughput and buys the guarantee that data is flushed before
    :meth:`append` returns.

    Attributes:
        path: The log file path.
        skipped_lines: Count of malformed lines encountered during reads.
    """

    path: Path
    skipped_lines: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Create the parent directory if it does not exist.

        Raises:
            EventStoreError: If the path cannot be prepared.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EventStoreError(f"cannot create event log directory: {exc}") from exc

    def append(self, event: EventEnvelope) -> None:
        """Append a single event as one JSON line.

        Args:
            event: The validated event to store.

        Raises:
            EventStoreError: If the event cannot be serialised or written.
        """
        self.extend((event,))

    def extend(self, events: Sequence[EventEnvelope]) -> None:
        """Append a batch of events as JSON lines, preserving order.

        The batch is serialised before the file is opened, so a serialisation failure
        never leaves a partially written line.

        Args:
            events: Events to append in order.

        Raises:
            EventStoreError: If the events cannot be serialised or written.
        """
        if not events:
            return
        lines: list[str] = []
        for event in events:
            try:
                record = event.model_dump(mode="json")
            except Exception as exc:  # pragma: no cover - pydantic models are serialisable
                raise EventStoreError(f"event {event.event_id} is not serialisable: {exc}") from exc
            lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))

        payload = "\n".join(lines) + "\n"
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
        except OSError as exc:
            raise EventStoreError(f"cannot write event log: {exc}") from exc

    def query(self, query: EventQuery | None = None) -> tuple[EventEnvelope, ...]:
        """Return events matching a query, in file order.

        Malformed lines are skipped and counted in :attr:`skipped_lines` rather than
        raising. One corrupt line should not make an entire session unreadable, and the
        count is surfaced so the corruption is still visible.

        Args:
            query: Filter to apply. ``None`` returns all readable events.

        Returns:
            The matching events.
        """
        results: list[EventEnvelope] = []
        if not self.path.exists():
            return ()
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = EventEnvelope.model_validate_json(stripped)
                except Exception:
                    self.skipped_lines += 1
                    continue
                if query is None or query.matches(event):
                    results.append(event)
                    if (
                        query is not None
                        and query.limit is not None
                        and len(results) >= query.limit
                    ):
                        break
        return tuple(results)

    def __len__(self) -> int:
        """Number of readable events in the log.

        Returns:
            The event count.
        """
        return len(self.query())
