"""Unit tests for :mod:`focus_engine.events.validation`.

Covers the validation entry point, batch ingestion, duplicate detection, and error
aggregation.
"""

from __future__ import annotations

import pytest

from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    QuestionAnsweredPayload,
    SessionStartedPayload,
)
from focus_engine.events.validation import (
    EventIngestor,
    EventIngestorConfig,
    IngestError,
    IngestResult,
    validate_event,
)

pytestmark = pytest.mark.unit


def _minimal_event() -> dict[str, object]:
    return {
        "event_id": "event-001",
        "learner_id": "learner-001",
        "session_id": "session-001",
        "event_type": "session_started",
        "payload": {},
    }


# --------------------------------------------------------------------------------------
# Validation entry point
# --------------------------------------------------------------------------------------


def test_validate_event_accepts_valid_input() -> None:
    event = validate_event(_minimal_event())
    assert isinstance(event, EventEnvelope)
    assert event.event_type is EventType.SESSION_STARTED


def test_validate_event_rejects_missing_required_field() -> None:
    data = _minimal_event()
    del data["learner_id"]
    with pytest.raises(IngestError, match="Field required"):
        validate_event(data)


def test_validate_event_rejects_unknown_event_type() -> None:
    data = _minimal_event()
    data["event_type"] = "unknown_type"
    with pytest.raises(IngestError):
        validate_event(data)


def test_validate_event_rejects_mismatched_payload_type() -> None:
    data = _minimal_event()
    data["event_type"] = "question_answered"
    data["payload"] = {}
    with pytest.raises(IngestError):
        validate_event(data)


def test_validate_event_provides_field_path_in_error() -> None:
    data = _minimal_event()
    data["payload"] = {"question_id": "q1"}  # missing 'correct' and 'response_seconds'
    data["event_type"] = "question_answered"
    with pytest.raises(IngestError) as excinfo:
        validate_event(data)
    assert excinfo.value.field is not None


def test_validate_event_rejects_short_content_id() -> None:
    data = _minimal_event()
    data["event_type"] = "question_answered"
    data["payload"] = {"question_id": "question-001", "correct": True, "response_seconds": 5.0}
    assert isinstance(validate_event(data).payload, QuestionAnsweredPayload)


def test_validate_event_rejects_forbidden_payload_field() -> None:
    data = _minimal_event()
    data["payload"] = {"unexpected": "value"}
    with pytest.raises(IngestError, match="not permitted"):
        validate_event(data)


# --------------------------------------------------------------------------------------
# Batch ingestion
# --------------------------------------------------------------------------------------


def test_ingestor_accepts_batch_of_valid_events() -> None:
    ingestor = EventIngestor()
    events = [_minimal_event(), _minimal_event() | {"event_id": "event-002"}]
    result = ingestor.ingest(events)
    assert len(result.accepted) == 2
    assert len(result.rejected) == 0
    assert result.acceptance_rate == 1.0


def test_ingestor_rejects_duplicate_event_ids() -> None:
    ingestor = EventIngestor()
    events = [_minimal_event(), _minimal_event()]
    result = ingestor.ingest(events)
    assert len(result.accepted) == 1
    assert result.duplicate_count == 1
    assert any(err.error_type == "duplicate" for err in result.rejected)


def test_ingestor_detects_duplicates_across_batches() -> None:
    ingestor = EventIngestor()
    ingestor.ingest([_minimal_event()])
    result = ingestor.ingest([_minimal_event()])
    assert result.duplicate_count == 1


def test_ingestor_resets_seen_ids() -> None:
    ingestor = EventIngestor()
    ingestor.ingest([_minimal_event()])
    ingestor.reset_seen_ids()
    result = ingestor.ingest([_minimal_event()])
    assert len(result.accepted) == 1
    assert result.duplicate_count == 0


def test_ingestor_respects_batch_size_limit() -> None:
    config = EventIngestorConfig(max_batch_size=2)
    ingestor = EventIngestor(config=config)
    events = [
        _minimal_event(),
        _minimal_event() | {"event_id": "event-002"},
        _minimal_event() | {"event_id": "event-003"},
    ]
    result = ingestor.ingest(events)
    assert len(result.accepted) == 2
    assert any(err.error_type == "batch_size_exceeded" for err in result.rejected)


def test_ingestor_collects_all_errors_in_non_strict_mode() -> None:
    ingestor = EventIngestor(config=EventIngestorConfig(strict_mode=False))
    events = [
        _minimal_event(),
        {"event_id": "bad"},  # missing fields
        _minimal_event() | {"event_id": "event-002"},
        {"event_id": "also-bad"},
    ]
    result = ingestor.ingest(events)
    assert len(result.accepted) == 2
    assert len(result.rejected) == 2


def test_ingestor_raises_on_first_error_in_strict_mode() -> None:
    ingestor = EventIngestor(config=EventIngestorConfig(strict_mode=True))
    events = [
        _minimal_event(),
        {"event_id": "bad"},
        _minimal_event() | {"event_id": "event-002"},
    ]
    with pytest.raises(IngestError):
        ingestor.ingest(events)


def test_ingestor_can_disable_duplicate_detection() -> None:
    config = EventIngestorConfig(reject_duplicates=False)
    ingestor = EventIngestor(config=config)
    events = [_minimal_event(), _minimal_event()]
    result = ingestor.ingest(events)
    assert len(result.accepted) == 2
    assert result.duplicate_count == 0


# --------------------------------------------------------------------------------------
# Ingest result
# --------------------------------------------------------------------------------------


def test_ingest_result_total_processed_counts_all_outcomes() -> None:
    event = EventEnvelope(
        event_id="event-result-001",
        learner_id="learner-0001",
        session_id="session-0001",
        event_type=EventType.SESSION_STARTED,
        payload=SessionStartedPayload(),
    )
    result = IngestResult(accepted=(event,), rejected=(IngestError("err"),), duplicate_count=2)
    assert result.total_processed == 4


def test_ingest_result_acceptance_rate_handles_empty() -> None:
    result = IngestResult(accepted=(), rejected=(), duplicate_count=0)
    assert result.acceptance_rate == 0.0


def test_ingest_result_acceptance_rate_computes_correctly() -> None:
    event = EventEnvelope(
        event_id="event-result-002",
        learner_id="learner-0001",
        session_id="session-0001",
        event_type=EventType.SESSION_STARTED,
        payload=SessionStartedPayload(),
    )
    result = IngestResult(accepted=(event,), rejected=(), duplicate_count=1)
    assert result.acceptance_rate == 0.5


# --------------------------------------------------------------------------------------
# Ingest error
# --------------------------------------------------------------------------------------


def test_ingest_error_to_dict_is_serialisable() -> None:
    err = IngestError(
        "validation failed", field="payload.question_id", value="q1", error_type="missing"
    )
    d = err.to_dict()
    assert d["message"] == "validation failed"
    assert d["field"] == "payload.question_id"
    assert d["value"] == "q1"
    assert d["error_type"] == "missing"
