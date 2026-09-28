"""Unit tests for :mod:`focus_engine.events.store`.

Covers both store implementations, query filtering, append-only semantics, and JSONL
persistence including corrupt-line handling.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from focus_engine.events.store import (
    EventQuery,
    EventStore,
    EventStoreError,
    InMemoryEventStore,
    JsonlEventStore,
)
from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    QuestionAnsweredPayload,
    QuestionStartedPayload,
    SessionEndedPayload,
    SessionStartedPayload,
    VideoPausedPayload,
    VideoStartedPayload,
)
from focus_engine.schemas.primitives import DataOrigin, Provenance, SyntheticDataStamp

pytestmark = pytest.mark.unit

_BASE = datetime(2026, 9, 26, 9, 0, 0, tzinfo=UTC)


def _event(
    event_id: str,
    event_type: EventType = EventType.SESSION_STARTED,
    *,
    payload: object | None = None,
    learner_id: str = "learner-0001",
    session_id: str = "session-0001",
    offset_seconds: int = 0,
    origin: DataOrigin = DataOrigin.REAL,
    provenance: Provenance = Provenance.OBSERVED,
    synthetic_stamp: SyntheticDataStamp | None = None,
) -> EventEnvelope:
    """Build a valid envelope for store tests.

    Args:
        event_id: Unique event identifier.
        event_type: Event type discriminator.
        payload: Payload model instance. Defaults per event type.
        learner_id: Pseudonymous learner identifier.
        session_id: Pseudonymous session identifier.
        offset_seconds: Seconds added to the base timestamp.
        origin: Real or synthetic.
        provenance: How the event came to exist.
        synthetic_stamp: Required when origin is synthetic.

    Returns:
        A validated event envelope.
    """
    if payload is None:
        payload = {
            EventType.SESSION_STARTED: SessionStartedPayload,
            EventType.SESSION_ENDED: SessionEndedPayload,
            EventType.QUESTION_STARTED: lambda: QuestionStartedPayload(question_id="question-001"),
            EventType.QUESTION_ANSWERED: lambda: QuestionAnsweredPayload(
                question_id="question-001", correct=True, response_seconds=10.0
            ),
            EventType.VIDEO_STARTED: lambda: VideoStartedPayload(video_id="video-0001"),
            EventType.VIDEO_PAUSED: lambda: VideoPausedPayload(
                video_id="video-0001", position_seconds=10.0, watch_time_seconds=10.0
            ),
        }[event_type]()
    return EventEnvelope(
        event_id=event_id,
        learner_id=learner_id,
        session_id=session_id,
        timestamp=_BASE + timedelta(seconds=offset_seconds),
        event_type=event_type,
        payload=payload,
        origin=origin,
        provenance=provenance,
        synthetic_stamp=synthetic_stamp,
    )


# --------------------------------------------------------------------------------------
# EventQuery validation
# --------------------------------------------------------------------------------------


def test_query_rejects_inverted_time_range() -> None:
    with pytest.raises(EventStoreError, match="must not be after"):
        EventQuery(start_time=_BASE, end_time=_BASE - timedelta(seconds=1))


def test_query_rejects_non_positive_limit() -> None:
    with pytest.raises(EventStoreError, match="at least 1"):
        EventQuery(limit=0)


def test_query_allows_equal_start_and_end() -> None:
    EventQuery(start_time=_BASE, end_time=_BASE)


# --------------------------------------------------------------------------------------
# InMemoryEventStore
# --------------------------------------------------------------------------------------


def test_empty_store_has_zero_length() -> None:
    store = InMemoryEventStore()
    assert len(store) == 0
    assert store.query() == ()


def test_append_preserves_insertion_order() -> None:
    store = InMemoryEventStore()
    events = [_event(f"event-{i:04d}", offset_seconds=i) for i in range(5)]
    store.extend(events)
    assert [e.event_id for e in store.query()] == [e.event_id for e in events]


def test_query_filters_by_learner() -> None:
    store = InMemoryEventStore()
    store.extend(
        [
            _event("event-0001", learner_id="learner-0001"),
            _event("event-0002", learner_id="learner-0002"),
        ]
    )
    result = store.query(EventQuery(learner_id="learner-0002"))
    assert [e.event_id for e in result] == ["event-0002"]


def test_query_filters_by_session() -> None:
    store = InMemoryEventStore()
    store.extend(
        [
            _event("event-0001", session_id="session-0001"),
            _event("event-0002", session_id="session-0002"),
        ]
    )
    result = store.query(EventQuery(session_id="session-0002"))
    assert [e.event_id for e in result] == ["event-0002"]


def test_query_filters_by_event_type() -> None:
    store = InMemoryEventStore()
    store.extend(
        [
            _event("event-0001", EventType.SESSION_STARTED),
            _event("event-0002", EventType.QUESTION_STARTED),
            _event("event-0003", EventType.SESSION_ENDED),
        ]
    )
    result = store.query(
        EventQuery(event_types=frozenset({EventType.SESSION_STARTED, EventType.SESSION_ENDED}))
    )
    assert [e.event_id for e in result] == ["event-0001", "event-0003"]


def test_query_time_range_is_start_inclusive_end_exclusive() -> None:
    store = InMemoryEventStore()
    store.extend([_event(f"event-{i:04d}", offset_seconds=i * 10) for i in range(5)])
    result = store.query(
        EventQuery(start_time=_BASE + timedelta(seconds=10), end_time=_BASE + timedelta(seconds=30))
    )
    assert [e.event_id for e in result] == ["event-0001", "event-0002"]


def test_query_filters_by_origin() -> None:
    stamp = SyntheticDataStamp(generator="sim", seed=7)
    store = InMemoryEventStore()
    store.extend(
        [
            _event("event-0001"),
            _event(
                "event-0002",
                origin=DataOrigin.SYNTHETIC,
                provenance=Provenance.SYNTHETIC_LABEL,
                synthetic_stamp=stamp,
            ),
        ]
    )
    synthetic = store.query(EventQuery(origin=DataOrigin.SYNTHETIC))
    assert [e.event_id for e in synthetic] == ["event-0002"]
    real = store.query(EventQuery(origin=DataOrigin.REAL))
    assert [e.event_id for e in real] == ["event-0001"]


def test_query_limit_stops_early() -> None:
    store = InMemoryEventStore()
    store.extend([_event(f"event-{i:04d}", offset_seconds=i) for i in range(10)])
    assert len(store.query(EventQuery(limit=3))) == 3


def test_query_combines_filters_with_and() -> None:
    store = InMemoryEventStore()
    store.extend(
        [
            _event("event-0001", EventType.QUESTION_ANSWERED, learner_id="learner-0001"),
            _event("event-0002", EventType.QUESTION_ANSWERED, learner_id="learner-0002"),
            _event("event-0003", EventType.VIDEO_STARTED, learner_id="learner-0001"),
        ]
    )
    result = store.query(
        EventQuery(learner_id="learner-0001", event_types=frozenset({EventType.QUESTION_ANSWERED}))
    )
    assert [e.event_id for e in result] == ["event-0001"]


def test_sessions_summarises_time_span() -> None:
    store = InMemoryEventStore()
    store.extend(
        [
            _event("event-0001", session_id="session-0001", offset_seconds=0),
            _event("event-0002", session_id="session-0001", offset_seconds=120),
            _event("event-0003", session_id="session-0002", offset_seconds=60),
        ]
    )
    bounds = store.sessions()
    assert len(bounds) == 2
    assert bounds[0].session_id == "session-0001"
    assert bounds[0].event_count == 2
    assert bounds[0].last_event_at - bounds[0].first_event_at == timedelta(seconds=120)
    assert bounds[1].session_id == "session-0002"


def test_store_exposes_no_mutation_api() -> None:
    """Append-only is enforced by omission, not by raising.

    Args:
        None: Not applicable.
    """
    store = InMemoryEventStore()
    assert not hasattr(store, "update")
    assert not hasattr(store, "delete")
    assert not hasattr(store, "remove")


def test_stored_event_is_immutable() -> None:
    store = InMemoryEventStore()
    event = _event("event-0001")
    store.append(event)
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        event.event_id = "event-9999"  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# JsonlEventStore
# --------------------------------------------------------------------------------------


def test_jsonl_store_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "events.jsonl"
    store = JsonlEventStore(path=target)
    store.append(_event("event-0001"))
    assert target.exists()


def test_jsonl_store_round_trips_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    events = [
        _event("event-0001", EventType.QUESTION_ANSWERED, offset_seconds=0),
        _event("event-0002", EventType.VIDEO_PAUSED, offset_seconds=30),
    ]
    store = JsonlEventStore(path=path)
    store.extend(events)

    reloaded = JsonlEventStore(path=path)
    result = reloaded.query()
    assert len(result) == 2
    assert [e.event_id for e in result] == ["event-0001", "event-0002"]
    assert result[0] == events[0]
    assert result[0].payload == events[0].payload


def test_jsonl_store_preserves_synthetic_stamp(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    stamp = SyntheticDataStamp(generator="simulator_v1", seed=99)
    store = JsonlEventStore(path=path)
    store.append(
        _event(
            "event-0001",
            origin=DataOrigin.SYNTHETIC,
            provenance=Provenance.SYNTHETIC_LABEL,
            synthetic_stamp=stamp,
        )
    )
    reloaded = JsonlEventStore(path=path).query()[0]
    assert reloaded.origin is DataOrigin.SYNTHETIC
    assert reloaded.synthetic_stamp is not None
    assert reloaded.synthetic_stamp.seed == 99


def test_jsonl_store_appends_rather_than_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path=path)
    store.append(_event("event-0001"))
    store.append(_event("event-0002"))
    assert len(JsonlEventStore(path=path).query()) == 2
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_jsonl_store_writes_one_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path=path)
    store.extend([_event(f"event-{i:04d}", offset_seconds=i) for i in range(4)])
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    assert all(line.startswith("{") for line in lines)


def test_jsonl_store_skips_corrupt_lines_and_counts_them(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path=path)
    store.append(_event("event-0001"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
        handle.write("\n")
    store.append(_event("event-0002"))

    reloaded = JsonlEventStore(path=path)
    result = reloaded.query()
    assert [e.event_id for e in result] == ["event-0001", "event-0002"]
    assert reloaded.skipped_lines == 1


def test_jsonl_store_skips_schema_invalid_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path=path)
    store.append(_event("event-0001"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_id": "short", "learner_id": "x", "session_id": "y"}\n')

    reloaded = JsonlEventStore(path=path)
    assert len(reloaded.query()) == 1
    assert reloaded.skipped_lines == 1


def test_jsonl_store_empty_batch_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path=path)
    store.extend([])
    assert not path.exists() or path.read_text(encoding="utf-8") == ""


def test_jsonl_store_query_applies_filters(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path=path)
    store.extend(
        [
            _event("event-0001", EventType.SESSION_STARTED, learner_id="learner-0001"),
            _event("event-0002", EventType.QUESTION_ANSWERED, learner_id="learner-0002"),
        ]
    )
    reloaded = JsonlEventStore(path=path)
    result = reloaded.query(EventQuery(event_types=frozenset({EventType.QUESTION_ANSWERED})))
    assert [e.event_id for e in result] == ["event-0002"]


def test_jsonl_store_missing_file_returns_empty(tmp_path: Path) -> None:
    store = JsonlEventStore(path=tmp_path / "absent" / "events.jsonl")
    assert store.query() == ()


def test_jsonl_store_len_counts_readable_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path=path)
    store.extend([_event("event-0001"), _event("event-0002"), _event("event-0003")])
    assert len(JsonlEventStore(path=path)) == 3


# --------------------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------------------


def test_both_stores_satisfy_protocol(tmp_path: Path) -> None:
    assert isinstance(InMemoryEventStore(), EventStore)
    assert isinstance(JsonlEventStore(path=tmp_path / "events.jsonl"), EventStore)


def test_both_stores_behave_identically_for_appends(tmp_path: Path) -> None:
    events = [_event(f"event-{i:04d}", offset_seconds=i) for i in range(3)]
    memory = InMemoryEventStore()
    memory.extend(events)
    disk = JsonlEventStore(path=tmp_path / "events.jsonl")
    disk.extend(events)
    assert memory.query() == disk.query()


def test_in_memory_store_exposes_iterator() -> None:
    store = InMemoryEventStore()
    store.extend([_event("event-0001"), _event("event-0002")])
    assert [e.event_id for e in store] == ["event-0001", "event-0002"]
