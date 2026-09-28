"""Lab event-contract tests: the Lab must speak the engine's native event language.

The Lab had a defect that only a test of this shape would have caught: it accepted
flattened pseudo-events shaped like ``{"event_type": ..., "subject": ...}`` and surfaced
pydantic's raw union errors, so a caller who pasted a *correct* event was told
``payload: Field required`` and ``subject: Extra inputs are not permitted``.

The governing rule, asserted repeatedly below: **the engine decides validity.** The Lab
decides what to *say* about a refusal. Nothing in this package re-declares a schema,
relaxes a constraint, or accepts an event the engine would reject.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from api.lab import validation as lab_validation
from api.lab.scenarios import (
    DEFAULT_SAMPLE_SEED,
    canonical_example,
    generate_sample_session,
)
from api.lab.validation import (
    coerce_events,
    expected_schema_for,
    validate_dataset,
)
from pydantic import BaseModel, ValidationError

from focus_engine.events.types import ALL_EVENT_TYPES, EventEnvelope, EventType
from focus_engine.events.validation import EventIngestor, validate_event

LEARNER = "lab-sample-0001"
SESSION = "lab-sample-0001-s01"


#: Finding codes that would indicate a schema-level complaint. A valid session must produce
#: none of them, so this set is asserted empty rather than enumerated case by case.
SCHEMA_COMPLAINT_CODES = frozenset(
    {
        "schema_invalid",
        "flattened_event",
        "missing_field",
        "unknown_event_type",
        "invalid_timestamp",
        "naive_timestamp",
    }
)


def _validate(events: list[dict[str, Any]]):
    """Run the Lab's paste path over a list of raw event dicts."""
    parsed = coerce_events(json.dumps({"events": events}))
    return validate_dataset(parsed.rows, submitted=parsed.presented)


def _valid_session() -> list[dict[str, Any]]:
    """A minimal, engine-valid four-event session, built from the Lab's own example."""
    return canonical_example()["events"]


# ----------------------------------------------------------------------------------
# 1. A valid session is accepted.
# ----------------------------------------------------------------------------------


def test_valid_native_session_is_accepted() -> None:
    report = _validate(_valid_session())
    assert report.can_proceed
    assert report.accepted == 4
    assert report.rejected == 0
    assert report.errors == ()
    for event in report.events:
        assert isinstance(event, EventEnvelope)


def test_valid_session_reports_no_schema_findings() -> None:
    report = _validate(_valid_session())
    codes = {finding.code.value for finding in report.findings}
    assert codes & SCHEMA_COMPLAINT_CODES == set(), (
        f"a valid session produced schema complaints: {codes & SCHEMA_COMPLAINT_CODES}"
    )


# ----------------------------------------------------------------------------------
# 2. At least three valid events reach the engine.
# ----------------------------------------------------------------------------------


def test_three_or_more_valid_events_reach_the_engine() -> None:
    report = _validate(_valid_session())
    assert len(report.events) >= 3, "the Lab must let a minimum viable session through"
    # And they are the engine's own objects, not dictionaries or lab-local stand-ins.
    assert all(type(event) is EventEnvelope for event in report.events)


def test_minimum_dataset_threshold_is_three() -> None:
    two = _valid_session()[:2]
    report = _validate(two)
    assert report.accepted == 2
    assert not report.can_proceed
    assert "too_few_events" in {f.code.value for f in report.findings}

    three = _valid_session()[:3]
    report_three = _validate(three)
    assert report_three.accepted == 3
    assert report_three.can_proceed


# ----------------------------------------------------------------------------------
# 3. An invalid flattened event is rejected.
# ----------------------------------------------------------------------------------


def test_flattened_pseudo_event_is_rejected() -> None:
    """The original bug: a single flat object with payload fields at the top level."""
    flattened = {
        "event_id": "evt-00000001",
        "learner_id": LEARNER,
        "session_id": SESSION,
        "timestamp": "2026-06-01T09:00:00Z",
        "event_type": "question_answered",
        "subject": "algebra",
        "activity_id": "act-0077",
        "correct": True,
        "response_time_seconds": 12.5,
    }
    report = _validate([flattened])
    assert not report.can_proceed
    assert report.accepted == 0
    codes = {f.code.value for f in report.findings}
    assert "flattened_event" in codes
    assert "schema_invalid" in codes


def test_flattened_event_names_the_misplaced_and_foreign_fields() -> None:
    flattened = {
        "event_id": "evt-00000001",
        "learner_id": LEARNER,
        "session_id": SESSION,
        "timestamp": "2026-06-01T09:00:00Z",
        "event_type": "question_answered",
        "subject": "algebra",
        "correct": True,
    }
    report = _validate([flattened])
    hint = next(f for f in report.findings if f.code.value == "flattened_event")
    assert "correct" in hint.message, "misplaced payload field must be named"
    assert "subject" in hint.message, "unrecognised field must be named"
    assert "payload" in hint.message


def test_valid_event_with_extra_field_is_rejected() -> None:
    """``extra='forbid'`` must survive the trip through the Lab."""
    event = dict(_valid_session()[2], unexpected_field="surprise")
    report = _validate([event])
    assert report.accepted == 0
    assert "schema_invalid" in {f.code.value for f in report.findings}


# ----------------------------------------------------------------------------------
# 4. ``response_seconds`` is the required name, and the report says so.
# ----------------------------------------------------------------------------------


def test_response_seconds_is_required_and_response_time_seconds_is_not() -> None:
    """``response_time_seconds`` is the near-miss that produced the original confusion."""
    good = _valid_session()[2]
    assert "response_seconds" in good["payload"]

    # Swap in the wrong name; the event must be refused.
    bad = dict(
        good,
        payload={
            "question_id": good["payload"]["question_id"],
            "correct": True,
            "response_time_seconds": 12.5,
        },
    )
    report = _validate([bad])
    assert report.accepted == 0

    fields = {f.detail.get("field") for f in report.findings}
    assert "payload.response_seconds" in fields, "the missing required field must be named"
    assert "payload.response_time_seconds" in fields, "the extra field must be named"


def test_schema_report_names_the_expected_payload_schema() -> None:
    good = _valid_session()[2]
    bad = dict(good, payload={"question_id": "qst-000001", "correct": True})
    report = _validate([bad])
    findings = [f for f in report.findings if f.detail.get("expected_schema")]
    assert findings, "a payload failure must carry the schema the engine expects"
    schema = findings[0].detail["expected_schema"]
    assert "QuestionAnsweredPayload" in schema
    assert "response_seconds" in schema


def test_union_errors_are_collapsed_not_dumped() -> None:
    """A wrong field name yields 72 raw pydantic errors. The Lab must not pass them on."""
    good = _valid_session()[2]
    bad = dict(
        good, payload={"question_id": "qst-000001", "correct": True, "response_time_seconds": 12.5}
    )
    with pytest.raises(ValidationError) as raw:
        EventEnvelope.model_validate(bad)
    assert len(raw.value.errors()) > 20, "precondition: the engine really does emit a flood"

    report = _validate([bad])
    schema_findings = [f for f in report.findings if f.code.value == "schema_invalid"]
    assert len(schema_findings) <= 6, "the Lab must bound the flood, not relay it"


# ----------------------------------------------------------------------------------
# 5. Identifiers shorter than 8 characters are rejected.
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["event_id", "learner_id", "session_id"])
@pytest.mark.parametrize("short", ["e1", "abc", "1234567"])
def test_short_identifiers_are_rejected(field: str, short: str) -> None:
    event = dict(_valid_session()[0])
    event[field] = short
    report = _validate([event])
    assert report.accepted == 0, f"{field}={short!r} must be refused"
    messages = " ".join(f.message for f in report.findings)
    assert "at least 8 characters" in messages


def test_eight_character_identifier_is_accepted() -> None:
    event = dict(_valid_session()[0], event_id="12345678")
    report = _validate([event])
    assert report.accepted == 1, "the boundary is inclusive: 8 is allowed, 7 is not"


def test_short_question_id_in_payload_is_rejected() -> None:
    event = dict(_valid_session()[1], payload={"question_id": "q1", "difficulty": 0.5})
    report = _validate([event])
    assert report.accepted == 0
    messages = " ".join(f.message for f in report.findings)
    assert "at least 8 characters" in messages


# ----------------------------------------------------------------------------------
# 6. The real engine EventIngestor is the authority.
# ----------------------------------------------------------------------------------


def test_validate_dataset_uses_the_engine_ingestor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Lab must not validate on its own; it must go through ``EventIngestor``."""
    calls: list[list[dict[str, Any]]] = []
    real = lab_validation.EventIngestor

    class SpyIngestor(real):  # type: ignore[misc, valid-type]
        def ingest(self, events: Any) -> Any:
            calls.append(list(events))
            return real.ingest(self, events)

    monkeypatch.setattr(lab_validation, "EventIngestor", SpyIngestor)
    report = _validate(_valid_session())
    assert report.can_proceed
    assert len(calls) == 1, "the ingestor must be consulted exactly once per dataset"
    assert len(calls[0]) == 4, "every submitted row must reach the ingestor"


def test_ingestor_rejection_is_not_overridden_by_the_lab() -> None:
    """Whatever the ingestor refuses, the Lab must refuse too."""
    bad = {
        "event_id": "evt-00000001",
        "learner_id": LEARNER,
        "session_id": SESSION,
        "timestamp": "2026-06-01T09:00:00Z",
        "event_type": "question_answered",
        "payload": {"question_id": "qst-000001", "correct": True},
    }
    ingestor_result = EventIngestor().ingest([bad])
    assert len(ingestor_result.accepted) == 0

    report = _validate([bad])
    assert report.accepted == 0


def test_lab_acceptance_matches_engine_acceptance_case_by_case() -> None:
    """The strongest form of "no rules were weakened": identical decisions, event by event.

    Every event below is fed to the engine's own ``EventIngestor`` and to the Lab. The two
    must agree on validity. If the Lab ever accepts something the engine rejects, or
    rejects something the engine accepts, this fails.
    """
    base = _valid_session()
    cases: list[dict[str, Any]] = [
        base[0],  # valid
        base[1],  # valid
        base[2],  # valid
        dict(base[0], event_id="short"),  # short id
        dict(base[0], event_type="not_a_real_type"),  # unknown type
        dict(base[0], timestamp="2026-06-01T09:00:00"),  # naive timestamp
        dict(base[0], payload={}),  # missing required
        dict(base[0], payload=None),  # null payload
        dict(base[0], payload="nonsense"),  # wrong payload type
        dict(base[0], payload={"platform": "web", "surprise": 1}),  # extra in payload
        dict(base[0], payload={"platform": "x" * 65}),  # max_length violated
        dict(base[1], payload={"question_id": "qst-000001", "difficulty": 1.5}),  # le=1.0
        dict(
            base[2],
            payload={"question_id": "qst-000001", "correct": True, "response_seconds": -1.0},
        ),  # ge=0.0
    ]

    for index, event in enumerate(cases):
        engine_ok = bool(EventIngestor().ingest([event]).accepted)
        lab_ok = _validate([event]).accepted == 1
        assert lab_ok == engine_ok, (
            f"case {index} disagrees: engine accepted={engine_ok}, lab accepted={lab_ok}"
        )


# ----------------------------------------------------------------------------------
# 7. No engine validation rule was weakened.
# ----------------------------------------------------------------------------------


def test_event_envelope_still_forbids_extra_fields() -> None:
    assert EventEnvelope.model_config.get("extra") == "forbid"


def test_every_payload_model_still_forbids_extra_fields() -> None:
    from api.lab.validation import _PAYLOAD_MODELS

    assert len(_PAYLOAD_MODELS) == len(ALL_EVENT_TYPES)
    for event_type, model in _PAYLOAD_MODELS.items():
        assert issubclass(model, BaseModel), event_type
        assert model.model_config.get("extra") == "forbid", (
            f"{model.__name__} no longer forbids extra fields"
        )


def test_payload_map_covers_every_event_type_exactly_once() -> None:
    from api.lab.validation import _PAYLOAD_MODELS

    assert set(_PAYLOAD_MODELS) == set(EventType)
    assert len(set(_PAYLOAD_MODELS.values())) == len(EventType), "must be a bijection"


def test_expected_schema_matches_the_engine_for_every_event_type() -> None:
    """The Lab's own description of a schema must agree with the engine's model.

    Checked field by field for all eighteen types, so the description the UI shows a
    rejected event cannot name a required field the engine does not require, or hide one
    it does.
    """
    from api.lab.validation import _PAYLOAD_MODELS

    for event_type, model in _PAYLOAD_MODELS.items():
        schema = expected_schema_for(event_type)
        assert schema.startswith(f"{model.__name__}("), event_type

        required_clause, _, optional_clause = schema.partition("; optional:")
        required_clause = required_clause.split("required:", 1)[1]
        for name, info in model.model_fields.items():
            if info.is_required():
                assert name in required_clause, f"{model.__name__}: {name} must be required"
            else:
                assert name in optional_clause, f"{model.__name__}: {name} must be optional"


def test_minimum_id_length_is_the_engines_eight() -> None:
    from focus_engine.schemas import primitives

    assert primitives._MIN_ID_LENGTH == 8
    assert primitives._ID_PATTERN.match("abc-123_XYZ")
    assert not primitives._ID_PATTERN.match("has space")
    assert not primitives._ID_PATTERN.match("a@b.com")


# ----------------------------------------------------------------------------------
# 8. The generated sample session is valid and deterministic.
# ----------------------------------------------------------------------------------


def test_canonical_example_is_valid_by_construction() -> None:
    document = canonical_example()
    assert len(document["events"]) >= 3
    for raw in document["events"]:
        event = validate_event(raw)  # the engine's own single-event entry point
        assert event.event_id == raw["event_id"]


def test_generated_sample_session_is_deterministic_for_a_seed() -> None:
    first = generate_sample_session(seed=DEFAULT_SAMPLE_SEED)
    second = generate_sample_session(seed=DEFAULT_SAMPLE_SEED)
    assert first == second, "the same seed must produce byte-identical output"
    assert first["event_count"] >= 3


def test_generated_sample_session_changes_with_the_seed() -> None:
    first = generate_sample_session(seed=1)
    second = generate_sample_session(seed=2)
    assert first != second, "a different seed must produce different data"


def test_generated_sample_session_passes_the_labs_own_validation() -> None:
    generated = generate_sample_session(seed=DEFAULT_SAMPLE_SEED)
    report = _validate(generated["events"])
    assert report.can_proceed
    assert report.accepted == generated["event_count"]
    assert report.errors == ()


def test_generated_sample_session_uses_native_payloads() -> None:
    generated = generate_sample_session(seed=DEFAULT_SAMPLE_SEED)
    for raw in generated["events"]:
        assert "payload" in raw
        assert raw["event_type"] in ALL_EVENT_TYPES
        event = validate_event(raw)
        assert type(event.payload).__name__ in expected_schema_for(event.event_type)


def test_generator_refuses_a_seed_below_the_engines_floor() -> None:
    with pytest.raises(ValueError):
        generate_sample_session(seed=-1)


def test_generator_refuses_fewer_questions_than_the_simulator_allows() -> None:
    with pytest.raises(ValueError):
        generate_sample_session(seed=DEFAULT_SAMPLE_SEED, questions=2)
