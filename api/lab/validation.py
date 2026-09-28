"""Phase K: data-quality checks that report rather than repair.

The engine's own validation (:func:`focus_engine.events.validate_event`) already refuses a
malformed event at the boundary. This module covers the checks that sit *above* a single
event — the ones about a whole dataset that a per-event validator cannot see, because each
event is individually well-formed and the set is not:

* an event outside its own session's bounds;
* the same ``event_id`` submitted twice;
* timestamps that are not timezone-aware, which the engine refuses but which are worth
  naming in a dataset-level report so the user learns *why*;
* response times that are individually valid (``ge=0``) and collectively impossible;
* a dataset too small to support any determination.

**Nothing here repairs anything.** A duplicate is reported, not deduplicated; an
out-of-bounds event is reported, not dropped; an impossible response time is reported, not
clamped. This is the same discipline the baseline layer applies to an out-of-range
measurement (Phase 6: *"clamping an impossible measurement to the boundary replaces a
visible defect with an invisible plausible number"*), and the same one the feature engine
applies to a missing input (Phase 5: absent yields ``INSUFFICIENT_DATA`` with a reason,
never a substituted constant).

A lab that silently cleaned its input would be the most dangerous thing in this repository:
it would display a clean pipeline run over data the user never supplied, and attribute the
result to their data.

Severity is reported per finding. ``ERROR`` means the engine cannot proceed with that item.
``WARNING`` means the engine can proceed and the user should know what they are looking at.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Final, get_args

from pydantic import BaseModel, ValidationError

from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    QuestionAnsweredPayload,
)
from focus_engine.events.validation import EventIngestor, EventIngestorConfig
from focus_engine.schemas.primitives import DataOrigin

__all__ = [
    "FINDING_CODES",
    "Finding",
    "FindingCode",
    "ParsedDocument",
    "RootShape",
    "Severity",
    "ValidationReport",
    "MAX_PLAUSIBLE_RESPONSE_SECONDS",
    "coerce_events",
    "dataset_summary",
    "expected_schema_for",
    "validate_dataset",
]

#: A response longer than this is almost always a client clock bug, an abandoned tab, or a
#: learner who left. It is *individually* schema-valid, so no per-event validator will
#: refuse it, and it is exactly the kind of value that silently widens a baseline's spread
#: and hides a real change. Reported, never clamped.
MAX_PLAUSIBLE_RESPONSE_SECONDS: Final[float] = 3600.0

#: Below this, no layer of the engine can support a determination, and the UI must say so
#: rather than showing a verdict derived from almost nothing.
MIN_DATASET_EVENTS: Final[int] = 3

#: Cap on schema findings reported for one event. A single mis-shaped payload can make
#: pydantic emit one error per union member per field; the report exists to point at the
#: mistake, not to enumerate the engine's failure modes.
_MAX_SCHEMA_FINDINGS_PER_EVENT: Final[int] = 6


def _derive_payload_models() -> dict[EventType, type[BaseModel]]:
    """Build the event_type -> payload model map *from the engine's own types*.

    ``EventEnvelope.payload`` is an 18-member untagged union. Pydantic validates a payload
    against every member before reporting, so a wrong field name produces one error per
    member: 72 errors to say "you wrote ``response_time_seconds``, the field is
    ``response_seconds``". The engine's own check
    (:meth:`EventEnvelope._validate_payload_type_match`) runs only *after* a payload
    validates, so it cannot disambiguate the union errors for us.

    To report a useful message we need to know which member the declared ``event_type``
    claims. That map is derived here by name correspondence and then checked for being a
    bijection onto the real union members, so if the engine ever adds a type, renames a
    payload, or reorders the union, this module fails loudly at import instead of quietly
    reporting the wrong expected schema. Nothing here re-declares a schema: the assertion
    guarantees the map cannot drift from the engine.
    """
    members = tuple(
        arg
        for arg in get_args(EventEnvelope.model_fields["payload"].annotation)
        if isinstance(arg, type) and issubclass(arg, BaseModel)
    )
    by_name = {member.__name__: member for member in members}
    derived: dict[EventType, type[BaseModel]] = {}
    for event_type in EventType:
        expected_name = event_type.name.title().replace("_", "") + "Payload"
        model = by_name.get(expected_name)
        if model is None:
            raise RuntimeError(
                f"EventType.{event_type.name} has no matching payload model "
                f"{expected_name!r} in EventEnvelope.payload. The engine's event taxonomy "
                "and this module's error reporting have diverged; fix the lab rather than "
                "the engine."
            )
        derived[event_type] = model
    if len(derived) != len(members):
        raise RuntimeError(
            f"Derived {len(derived)} payload models for {len(EventType)} event types, but "
            f"EventEnvelope.payload is a union of {len(members)}. The engine's event "
            "taxonomy and this module's error reporting have diverged."
        )
    return derived


_PAYLOAD_MODELS: Final[dict[EventType, type[BaseModel]]] = _derive_payload_models()

#: Every field name that belongs to some payload model, collected from the engine's models
#: rather than hand-listed. Used to recognise a flattened event in order to explain it.
_PAYLOAD_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    name for model in _PAYLOAD_MODELS.values() for name in model.model_fields
)

#: The envelope's own field names, read from the engine. Anything at the top level that is
#: in neither set belongs to no part of the event contract.
_ENVELOPE_FIELD_NAMES: Final[frozenset[str]] = frozenset(EventEnvelope.model_fields)


def expected_schema_for(event_type: EventType) -> str:
    """Describe the payload schema an event of this type must satisfy.

    Args:
        event_type: The declared discriminator.

    Returns:
        A compact, human-readable schema line naming the required fields, the optional
        fields, and any constraint the type's fields carry. Derived from the engine's model
        metadata, so it cannot describe a schema the engine does not enforce.
    """
    model = _PAYLOAD_MODELS[event_type]
    required: list[str] = []
    optional: list[str] = []
    for name, info in model.model_fields.items():
        annotation = getattr(info.annotation, "__name__", None) or str(info.annotation)
        entry = f"{name}: {annotation}"
        if info.is_required():
            required.append(entry)
        else:
            optional.append(entry)
    schema = f"{model.__name__}(required: {', '.join(required) or 'none'}"
    if optional:
        schema += f"; optional: {', '.join(optional)}"
    return schema + ")"


def _clean_message(message: str) -> str:
    """Strip pydantic's ``Value error, `` prefix from a custom validator message."""
    prefix = "Value error, "
    return message[len(prefix) :] if message.startswith(prefix) else message


class Severity(StrEnum):
    """How much a finding constrains what the lab can honestly show."""

    ERROR = "error"
    WARNING = "warning"


class FindingCode(StrEnum):
    """Stable identifiers, so the UI and the tests can match on the code not the prose."""

    NOT_JSON = "not_json"
    NOT_AN_OBJECT = "not_an_object"
    EMPTY_DATASET = "empty_dataset"
    EMPTY_COLLECTION = "empty_collection"
    TOO_FEW_EVENTS = "too_few_events"
    MISSING_FIELD = "missing_field"
    INVALID_TIMESTAMP = "invalid_timestamp"
    NAIVE_TIMESTAMP = "naive_timestamp"
    DUPLICATE_EVENT_ID = "duplicate_event_id"
    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    SCHEMA_INVALID = "schema_invalid"
    OUTSIDE_SESSION_BOUNDS = "outside_session_bounds"
    IMPLAUSIBLE_RESPONSE_TIME = "implausible_response_time"
    MULTIPLE_LEARNERS = "multiple_learners"
    MULTIPLE_SESSIONS = "multiple_sessions"
    OUT_OF_ORDER = "out_of_order"
    MISSING_SESSION_START = "missing_session_start"
    MISSING_SESSION_END = "missing_session_end"
    SINGLE_EVENT_OBJECT = "single_event_object"
    COLLECTION_NOT_A_LIST = "collection_not_a_list"
    SCALAR_ROOT = "scalar_root"
    FLATTENED_EVENT = "flattened_event"
    BATCH_SIZE_EXCEEDED = "batch_size_exceeded"


FINDING_CODES: Final[tuple[str, ...]] = tuple(code.value for code in FindingCode)


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong with the submitted data, stated and not fixed."""

    code: FindingCode
    severity: Severity
    message: str
    index: int | None = None
    event_id: str | None = None
    timestamp: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the UI. ``detail`` is included so a user can locate the row."""
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "index": self.index,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The outcome of checking a dataset. Never a cleaned dataset.

    ``events`` holds only what the engine accepted. ``findings`` explains everything that
    was not accepted and everything suspicious that was. A caller that wants to know
    whether the engine may proceed checks ``can_proceed``; a caller that wants to show the
    user what is wrong with their paste reads ``findings``.

    The three counts are deliberately separate. ``submitted`` is what the caller handed
    over, ``accepted`` is what survived the engine's own schema, and ``rejected`` is the
    difference. Collapsing them into a single "event count" is what produced the
    misleading "No events supplied" reply to a caller who had pasted one perfectly
    well-formed event object: the collection was rejected at the root, so zero rows
    reached per-event validation, and the only surviving finding was the downstream
    consequence rather than the cause.
    """

    events: tuple[EventEnvelope, ...]
    findings: tuple[Finding, ...]
    submitted: int = 0

    @property
    def can_proceed(self) -> bool:
        """True when at least one event survived and nothing fatal was found.

        ``WARNING`` findings do not block. A duplicate id or an implausible response time
        is something the user should see, not a reason to refuse to show them a pipeline.
        """
        return bool(self.events) and not any(
            finding.severity is Severity.ERROR for finding in self.findings
        )

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def accepted(self) -> int:
        return len(self.events)

    @property
    def rejected(self) -> int:
        return max(self.submitted - len(self.events), 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_proceed": self.can_proceed,
            "events_submitted": self.submitted,
            "events_accepted": self.accepted,
            "events_rejected": self.rejected,
            "event_count": len(self.events),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _finding(
    code: FindingCode,
    severity: Severity,
    message: str,
    *,
    index: int | None = None,
    event_id: str | None = None,
    timestamp: str | None = None,
    **detail: Any,
) -> Finding:
    return Finding(
        code=code,
        severity=severity,
        message=message,
        index=index,
        event_id=event_id,
        timestamp=timestamp,
        detail=detail,
    )


class RootShape(StrEnum):
    """The shape of the submitted document's root, before any event was validated.

    Recorded so the UI can say *why* zero events arrived, instead of reporting the
    downstream consequence of a decision made at the root.
    """

    INVALID_JSON = "invalid_json"
    BARE_ARRAY = "bare_array"
    ENVELOPE = "envelope"
    SINGLE_EVENT = "single_event"
    NOT_A_COLLECTION = "not_a_collection"
    UNRECOGNISED = "unrecognised"


#: Keys that only an event envelope carries. Used to tell "the caller pasted one event"
#: apart from "the caller pasted some other object", so the message can be specific.
_EVENT_MARKERS: Final[frozenset[str]] = frozenset(
    {"event_type", "event_id", "session_id", "learner_id", "timestamp", "payload"}
)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """A submitted document after its root shape was understood, not repaired.

    ``presented`` is how many event positions the root claimed to carry. It is 1 for a
    lone event object and 0 for a root that is not a collection at all, which is what lets
    the report say "1 submitted, 0 accepted, 1 rejected" instead of the bare "0 events".
    """

    root_shape: RootShape
    rows: tuple[dict[str, Any], ...]
    findings: tuple[Finding, ...]
    presented: int

    @property
    def can_proceed(self) -> bool:
        return bool(self.rows) and not any(
            finding.severity is Severity.ERROR for finding in self.findings
        )


def _looks_like_event(document: dict[str, Any]) -> bool:
    """Whether an object is plausibly one event envelope rather than some other payload.

    A structural guess, used only to choose *which question to ask*. A single unambiguous
    event marker is enough: ``event_type`` is the discriminator, because the engine's
    vocabulary is closed and no other object in this contract carries that key. The guess
    never decides whether the object is a valid event - that is the engine's schema, and it
    runs unchanged afterwards.
    """
    return "event_type" in document


def coerce_events(raw: str) -> ParsedDocument:
    """Parse a pasted JSON document into event-shaped dicts, reporting what is wrong.

    Accepts a bare array of events or ``{"events": [...]}``, because both are things a
    person will reasonably paste. A *single* event object is recognised and reported as
    such, but is deliberately not silently promoted into a one-element collection: the
    contract asks for a collection, and quietly reinterpreting the root would mean the
    Lab accepted a shape its own documentation says it does not accept. The finding names
    the wrapper to use.

    Returns findings rather than raising, so the UI can show the problem next to the
    textarea and distinguish "you pasted the wrong shape" from "your events are invalid".
    """
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        return ParsedDocument(
            root_shape=RootShape.INVALID_JSON,
            rows=(),
            findings=(
                _finding(
                    FindingCode.NOT_JSON,
                    Severity.ERROR,
                    f"Not valid JSON: {error.msg} (line {error.lineno}, column "
                    f"{error.colno}). Nothing was submitted to the engine.",
                ),
            ),
            presented=0,
        )

    if isinstance(document, dict):
        if "events" not in document:
            if _looks_like_event(document):
                return ParsedDocument(
                    root_shape=RootShape.SINGLE_EVENT,
                    rows=(),
                    findings=(
                        _finding(
                            FindingCode.SINGLE_EVENT_OBJECT,
                            Severity.ERROR,
                            "The root object is a single event, but the contract expects a "
                            'collection. Wrap it: {"events": [<this object>]}. The object '
                            "was not unwrapped automatically because silently reinterpreting "
                            "the root would accept a shape the contract does not define. "
                            "Note this is a root-shape error, not an event error: the event "
                            "itself has not been validated yet.",
                            event_keys=sorted(k for k in document if k in _EVENT_MARKERS),
                        ),
                    ),
                    presented=1,
                )
            return ParsedDocument(
                root_shape=RootShape.UNRECOGNISED,
                rows=(),
                findings=(
                    _finding(
                        FindingCode.NOT_AN_OBJECT,
                        Severity.ERROR,
                        'Object supplied but it has no "events" key and does not look like '
                        "an event. Send a JSON array of events, or an object with an "
                        '"events" array.',
                    ),
                ),
                presented=0,
            )
        collection = document["events"]
        if isinstance(collection, dict):
            return ParsedDocument(
                root_shape=RootShape.SINGLE_EVENT,
                rows=(),
                findings=(
                    _finding(
                        FindingCode.SINGLE_EVENT_OBJECT,
                        Severity.ERROR,
                        'The "events" key holds a single object rather than an array. Wrap '
                        'it in an array: {"events": [<this object>]}.',
                    ),
                ),
                presented=1,
            )
        if not isinstance(collection, list):
            return ParsedDocument(
                root_shape=RootShape.NOT_A_COLLECTION,
                rows=(),
                findings=(
                    _finding(
                        FindingCode.COLLECTION_NOT_A_LIST,
                        Severity.ERROR,
                        f'"events" must be a JSON array, got {type(collection).__name__}.',
                        expected="array",
                        observed=type(collection).__name__,
                    ),
                ),
                presented=0,
            )
        return _collect_rows(collection, RootShape.ENVELOPE)

    if not isinstance(document, list):
        # A scalar root and a container root are different mistakes. ``42`` is a perfectly
        # good JSON value that cannot hold events; an object at least claims to be a
        # payload. Saying "not an object" about a number describes the wrong thing.
        scalar = not isinstance(document, dict)
        return ParsedDocument(
            root_shape=RootShape.NOT_A_COLLECTION,
            rows=(),
            findings=(
                _finding(
                    FindingCode.SCALAR_ROOT if scalar else FindingCode.NOT_AN_OBJECT,
                    Severity.ERROR,
                    f"Expected a JSON array of events, got {type(document).__name__}.",
                    expected="array",
                    received=type(document).__name__,
                ),
            ),
            presented=0,
        )

    return _collect_rows(document, RootShape.BARE_ARRAY)


def _collect_rows(document: list[Any], shape: RootShape) -> ParsedDocument:
    """Split a recognised collection into event-shaped dicts plus per-item findings."""
    rows: list[dict[str, Any]] = []
    findings: list[Finding] = []
    for index, item in enumerate(document):
        if not isinstance(item, dict):
            findings.append(
                _finding(
                    FindingCode.NOT_AN_OBJECT,
                    Severity.ERROR,
                    f"Event {index} is {type(item).__name__}, not an object.",
                    index=index,
                )
            )
            continue
        rows.append(item)
    if not document:
        findings.append(
            _finding(
                FindingCode.EMPTY_COLLECTION,
                Severity.ERROR,
                "The event collection is empty. An array with no events cannot be analysed; "
                "this is a root-shape outcome, not a per-event one.",
            )
        )
    return ParsedDocument(
        root_shape=shape,
        rows=tuple(rows),
        findings=tuple(findings),
        presented=len(document),
    )


def _check_timestamps(
    row: dict[str, Any], index: int
) -> tuple[datetime | None, tuple[Finding, ...]]:
    """Read a timestamp, reporting absent / unparseable / naive separately.

    Three distinct problems that all present as "the timestamp is wrong", so they are
    reported separately: a caller who fixes only one of them should be able to see that the
    other two remain.
    """
    raw = row.get("timestamp")
    if raw is None:
        return None, (
            _finding(
                FindingCode.MISSING_FIELD,
                Severity.ERROR,
                "Event has no 'timestamp'.",
                index=index,
                field="timestamp",
            ),
        )
    if not isinstance(raw, str):
        return None, (
            _finding(
                FindingCode.INVALID_TIMESTAMP,
                Severity.ERROR,
                f"'timestamp' must be an ISO-8601 string, got {type(raw).__name__}.",
                index=index,
                field="timestamp",
            ),
        )
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as error:
        return None, (
            _finding(
                FindingCode.INVALID_TIMESTAMP,
                Severity.ERROR,
                f"'timestamp' is not ISO-8601: {error}. Naive local times are not accepted; "
                "include an offset such as 'Z' or '+00:00'.",
                index=index,
                timestamp=raw,
                field="timestamp",
            ),
        )
    if parsed.tzinfo is None:
        return None, (
            _finding(
                FindingCode.NAIVE_TIMESTAMP,
                Severity.ERROR,
                f"'timestamp' {raw!r} has no timezone. The engine refuses naive datetimes "
                "because a naive time is ambiguous by exactly the amount that matters here.",
                index=index,
                timestamp=raw,
                field="timestamp",
            ),
        )
    return parsed, ()


def _session_bounds(
    events: Sequence[EventEnvelope],
) -> tuple[datetime | None, datetime | None]:
    """The declared session start/end, if the dataset carries them.

    Bounds come from the events themselves rather than from ``min``/``max`` of all
    timestamps, because "the first thing that happened" and "when the session was declared
    to start" are different claims and only the second one can be violated.
    """
    start: datetime | None = None
    end: datetime | None = None
    for event in events:
        if event.event_type is EventType.SESSION_STARTED:
            if start is None or event.timestamp < start:
                start = event.timestamp
        elif event.event_type is EventType.SESSION_ENDED and (end is None or event.timestamp > end):
            end = event.timestamp
    return start, end


def _declared_event_type(value: str) -> EventType | None:
    """Return the :class:`EventType` for a declared type string, or ``None`` if unknown."""
    try:
        return EventType(value)
    except ValueError:
        return None


def _flattened_hint(row: dict[str, Any]) -> str | None:
    """Recognise the most common Lab mis-shape: a native event flattened into one object.

    The engine's envelope is two levels deep — envelope fields plus a typed ``payload`` — so
    a caller who hand-writes a single flat object gets the same error for every payload
    field at once. Naming the cause is worth a dedicated message, because the fix is
    structural (wrap the type-specific fields in ``payload``) rather than a typo correction.

    Two kinds of stray key are named. Keys that belong to *some* payload model (``correct``)
    were written one level too high. Keys that belong to nothing in the engine at all
    (``subject``, ``activity_id``) come from a foreign format and would be rejected even
    after the payload is wrapped. Both sets are read from the engine's own model metadata,
    so this can never suggest a field the engine does not define.
    """
    if "payload" in row:
        return None
    misplaced = sorted(_PAYLOAD_FIELD_NAMES.intersection(row))
    foreign = sorted(
        key for key in row if key not in _ENVELOPE_FIELD_NAMES and key not in _PAYLOAD_FIELD_NAMES
    )
    if not misplaced and not foreign:
        return None
    parts = []
    if misplaced:
        parts.append(
            f"{', '.join(misplaced)} "
            f"{'belongs' if len(misplaced) == 1 else 'belong'} inside 'payload'"
        )
    if foreign:
        parts.append(
            f"{', '.join(foreign)} "
            f"{'is' if len(foreign) == 1 else 'are'} not a field of any engine event or payload"
        )
    return (
        "; ".join(parts)
        + ". A native event is an envelope (event_id, learner_id, session_id, timestamp, "
        "event_type, payload) where payload holds the type-specific fields."
    )


def _schema_findings(
    row: dict[str, Any],
    event_type: str,
    error: ValidationError,
    index: int,
) -> list[Finding]:
    """Turn a pydantic failure into a few findings a reader can act on.

    The engine remains the sole authority on validity; this function only decides what to
    *say* about its refusal. The filtering matters: ``payload`` is an 18-member untagged
    union, so pydantic reports one error per member and a single wrong field name yields 72
    of them. Only the member matching the declared ``event_type`` is relevant, and saying
    so turns an unreadable wall of text into "this field, this name, this schema".

    Args:
        row: The submitted event dictionary.
        event_type: The declared ``event_type`` string.
        error: The engine's own ``ValidationError``.
        index: Position of the event in the submitted array.

    Returns:
        Findings with precise field paths, capped at
        ``_MAX_SCHEMA_FINDINGS_PER_EVENT``.
    """
    declared = _declared_event_type(event_type)
    member = _PAYLOAD_MODELS.get(declared) if declared is not None else None
    member_name = member.__name__ if member is not None else None

    envelope: list[dict[str, Any]] = []
    payload: list[dict[str, Any]] = []
    for item in error.errors():
        loc = tuple(str(part) for part in item.get("loc", ()))
        if not loc or loc[0] != "payload":
            envelope.append(item)
        elif member_name is not None and len(loc) >= 2 and loc[1] == member_name:
            payload.append(item)

    def _path(item: dict[str, Any]) -> str:
        loc = tuple(str(part) for part in item.get("loc", ()))
        # Strip the union member name so the path is the user's, not pydantic's.
        if len(loc) >= 3 and loc[0] == "payload" and loc[1] == member_name:
            return "payload." + ".".join(loc[2:])
        return ".".join(loc) or "(root)"

    # Deduplicate: a single mistake can surface the same path twice, and repeating it
    # hides the other mistakes in the same event.
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in envelope + payload:
        key = (_path(item), _clean_message(str(item.get("msg", ""))))
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)

    # Payload errors whose loc never named the expected member mean the payload was not an
    # object at all (or failed every member identically). Report that shape problem once
    # rather than dropping it silently.
    if not payload and not envelope and declared is not None:
        selected.append(
            {
                "loc": ("payload",),
                "msg": f"payload does not satisfy {member_name}",
                "type": "payload_shape",
            }
        )

    truncated = len(selected) > _MAX_SCHEMA_FINDINGS_PER_EVENT
    findings: list[Finding] = []
    for item in selected[:_MAX_SCHEMA_FINDINGS_PER_EVENT]:
        path = _path(item)
        metadata: dict[str, Any] = {
            "index": index,
            "field": path,
            "event_type": event_type,
            "error_type": str(item.get("type", "validation")),
        }
        if row.get("event_id"):
            metadata["event_id"] = str(row["event_id"])
        if row.get("timestamp"):
            metadata["timestamp"] = str(row["timestamp"])
        if declared is not None and path.startswith("payload"):
            # Name the schema the engine actually enforces for this discriminator.
            metadata["expected_schema"] = expected_schema_for(declared)
        if item.get("input") is not None and not isinstance(item.get("input"), (dict, list)):
            metadata["received"] = str(item.get("input"))[:80]
        findings.append(
            _finding(
                FindingCode.SCHEMA_INVALID,
                Severity.ERROR,
                f"{path}: {_clean_message(str(item.get('msg', 'validation error')))}",
                **metadata,
            )
        )

    if truncated:
        findings.append(
            _finding(
                FindingCode.SCHEMA_INVALID,
                Severity.ERROR,
                f"({len(selected) - _MAX_SCHEMA_FINDINGS_PER_EVENT} further schema problems "
                "on this event are not listed; the engine's full report would repeat them "
                "once per candidate payload type.)",
                index=index,
                event_type=event_type,
                suppressed=len(selected) - _MAX_SCHEMA_FINDINGS_PER_EVENT,
            )
        )

    hint = _flattened_hint(row)
    if hint is not None:
        findings.append(
            _finding(
                FindingCode.FLATTENED_EVENT,
                Severity.ERROR,
                hint,
                index=index,
                event_type=event_type,
                stray_fields=sorted(set(_PAYLOAD_FIELD_NAMES).intersection(row)),
            )
        )
    return findings


def validate_dataset(
    rows: Iterable[dict[str, Any]], *, submitted: int | None = None
) -> ValidationReport:
    """Check a dataset and return what the engine may have, plus everything that is wrong.

    Args:
        rows: Event-shaped dicts to validate. Only the dicts are given, so items the
            parser already rejected as non-objects are invisible here; ``submitted``
            carries the original count so the report can still account for all of them.
        submitted: How many event positions the caller presented. Defaults to
            ``len(rows)``, which is correct when the caller has no better number.

    The order of checks matters and is not arbitrary: an event that fails schema validation
    cannot contribute a timestamp, a session boundary, or a response time, so the per-event
    checks run first and only the survivors take part in the set-level checks. This keeps a
    finding about a *dataset* from being reported against an event that is not in the
    dataset.
    """
    rows = list(rows)
    presented = len(rows) if submitted is None else submitted
    findings: list[Finding] = []

    if not rows:
        findings.append(
            _finding(
                FindingCode.EMPTY_DATASET,
                Severity.ERROR,
                f"No events reached per-event validation ({presented} presented). Check the "
                "findings above this one: a root-shape problem rejects the collection before "
                "any event is examined, and this is the consequence of that, not its cause.",
            )
        )
        return ValidationReport(events=(), findings=tuple(findings), submitted=presented)

    accepted: list[EventEnvelope] = []
    seen_ids: dict[str, int] = {}

    # The engine's own batch validator is the authority on whether an event is valid. This
    # module does not decide validity, does not re-implement a schema, and does not soften
    # one: it hands the rows to `EventIngestor` and reports what came back. The two options
    # below are deliberate and neither weakens a rule:
    #
    # * `strict_mode=False` collects errors instead of raising, so one bad event does not
    #   discard the diagnosis of the others. This is the engine's own non-strict setting.
    # * `reject_duplicates=False` because this module's documented discipline is to *report*
    #   a duplicate, not to drop it (see `_finding(DUPLICATE_EVENT_ID)` below). The
    #   ingestor's duplicate handling silently removes the second copy, which would make a
    #   dataset look smaller than the caller submitted without saying so. Duplicate
    #   detection is therefore performed below, as a warning that keeps both events — the
    #   check still happens, it just reports rather than removes.
    ingestor = EventIngestor(EventIngestorConfig(reject_duplicates=False, strict_mode=False))
    ingested = ingestor.ingest(rows)

    if len(rows) > ingestor.config.max_batch_size:
        findings.append(
            _finding(
                FindingCode.BATCH_SIZE_EXCEEDED,
                Severity.WARNING,
                f"{len(rows)} events were submitted; the engine's ingestor accepts at most "
                f"{ingestor.config.max_batch_size} per batch. The excess "
                f"{len(rows) - ingestor.config.max_batch_size} were not analysed. Raise the "
                "ingestor's max_batch_size rather than assuming they passed.",
                submitted=len(rows),
                analysed=ingestor.config.max_batch_size,
            )
        )

    # Consume accepted envelopes in submission order. The ingestor appends only events that
    # validated, so advancing this iterator once per row that validates keeps the two in
    # step; a row the engine rejected advances nothing, exactly as in the ingestor.
    accepted_stream = iter(ingested.accepted)

    for index, row in enumerate(rows):
        event_type = row.get("event_type")
        if event_type is None:
            findings.append(
                _finding(
                    FindingCode.MISSING_FIELD,
                    Severity.ERROR,
                    "Event has no 'event_type'.",
                    index=index,
                    field="event_type",
                )
            )
            continue
        if not isinstance(event_type, str) or event_type not in _known_event_types():
            findings.append(
                _finding(
                    FindingCode.UNKNOWN_EVENT_TYPE,
                    Severity.ERROR,
                    f"Unknown event_type {event_type!r}. The vocabulary is closed: an "
                    "unrecognised type is an error rather than a warning, so that every "
                    "downstream switch stays exhaustive.",
                    index=index,
                    event_id=str(row.get("event_id")) if row.get("event_id") else None,
                    event_type=str(event_type),
                )
            )
            continue

        parsed, timestamp_findings = _check_timestamps(row, index)
        findings.extend(timestamp_findings)

        try:
            event = EventEnvelope.model_validate(row)
        except ValidationError as error:
            # Same engine call the ingestor made; re-run here only to recover the complete
            # error list, because `validate_event` keeps just the first for its own log.
            findings.extend(_schema_findings(row, event_type, error, index))
            continue

        # The engine already accepted this one. Take its object rather than our second
        # instance, so the events that flow onward are the ingestor's own.
        if index >= ingestor.config.max_batch_size:
            # Beyond the ingestor's cap: it never looked at this row, so there is no
            # accepted envelope to take. Already reported once, above.
            continue
        event = next(accepted_stream)

        if event.event_id in seen_ids:
            findings.append(
                _finding(
                    FindingCode.DUPLICATE_EVENT_ID,
                    Severity.WARNING,
                    f"event_id {event.event_id!r} already appeared at index "
                    f"{seen_ids[event.event_id]}. Both are kept: a duplicate is reported, "
                    "not deduplicated, because which copy is the real one is not "
                    "something this lab can know.",
                    index=index,
                    event_id=event.event_id,
                    timestamp=event.timestamp.isoformat(),
                    first_seen_index=seen_ids[event.event_id],
                )
            )
        else:
            seen_ids[event.event_id] = index

        if (
            event.event_type is EventType.QUESTION_ANSWERED
            and isinstance(event.payload, QuestionAnsweredPayload)
            and event.payload.response_seconds > MAX_PLAUSIBLE_RESPONSE_SECONDS
        ):
            findings.append(
                _finding(
                    FindingCode.IMPLAUSIBLE_RESPONSE_TIME,
                    Severity.WARNING,
                    f"response_seconds={event.payload.response_seconds:.0f} exceeds "
                    f"{MAX_PLAUSIBLE_RESPONSE_SECONDS:.0f}s. Schema-valid, so no per-event "
                    "check refuses it, and it will widen the baseline spread enough to "
                    "hide a real change. Reported, not clamped.",
                    index=index,
                    event_id=event.event_id,
                    timestamp=event.timestamp.isoformat(),
                    response_seconds=event.payload.response_seconds,
                )
            )

        if parsed is not None and event.timestamp != parsed:
            # Defensive: the two paths read the same field, so a divergence would mean the
            # engine's parser and ours disagreed. Surfacing it beats silently picking one.
            findings.append(
                _finding(
                    FindingCode.INVALID_TIMESTAMP,
                    Severity.ERROR,
                    "Timestamp parsed differently by the lab and by the engine.",
                    index=index,
                    event_id=event.event_id,
                )
            )
            continue

        accepted.append(event)

    if len(accepted) < MIN_DATASET_EVENTS:
        findings.append(
            _finding(
                FindingCode.TOO_FEW_EVENTS,
                Severity.ERROR,
                f"{len(accepted)} event(s) survived validation; at least "
                f"{MIN_DATASET_EVENTS} are needed before any layer can be shown. This is "
                "the INSUFFICIENT DATA case, and it is reported rather than padded.",
                event_count=len(accepted),
            )
        )

    findings.extend(_dataset_findings(accepted))
    return ValidationReport(events=tuple(accepted), findings=tuple(findings), submitted=presented)


def _dataset_findings(events: Sequence[EventEnvelope]) -> list[Finding]:
    """Checks that need more than one event to mean anything."""
    findings: list[Finding] = []
    if not events:
        return findings

    learners = {event.learner_id for event in events}
    if len(learners) > 1:
        findings.append(
            _finding(
                FindingCode.MULTIPLE_LEARNERS,
                Severity.WARNING,
                f"{len(learners)} distinct learner_id values in one dataset. The engine "
                "baselines per learner, so mixing them would attribute one learner's "
                "behaviour to another's baseline.",
                learners=sorted(learners),
            )
        )

    sessions = {event.session_id for event in events}
    if len(sessions) > 1:
        findings.append(
            _finding(
                FindingCode.MULTIPLE_SESSIONS,
                Severity.WARNING,
                f"{len(sessions)} distinct session_id values in one dataset. Each session "
                "is analysed separately; the trace below covers the first by timestamp.",
                sessions=sorted(sessions),
            )
        )

    ordered = sorted(events, key=lambda e: e.timestamp)
    if [e.event_id for e in events] != [e.event_id for e in ordered]:
        findings.append(
            _finding(
                FindingCode.OUT_OF_ORDER,
                Severity.WARNING,
                "Events are not in timestamp order. They are sorted for analysis; the "
                "engine's own context layer tolerates unordered input, but the trace is "
                "easier to read in order.",
            )
        )

    start, end = _session_bounds(ordered)
    if start is None:
        findings.append(
            _finding(
                FindingCode.MISSING_SESSION_START,
                Severity.WARNING,
                "No session_started event. The context engine will infer the start from "
                "the first event and mark it inferred; nothing is invented silently.",
            )
        )
    if end is None:
        findings.append(
            _finding(
                FindingCode.MISSING_SESSION_END,
                Severity.WARNING,
                "No session_ended event. The session is treated as still open, so the "
                "outcome layer's after-window has no closing boundary.",
            )
        )

    if start is not None:
        for event in ordered:
            if event.event_type is EventType.SESSION_STARTED:
                continue
            if event.timestamp < start:
                findings.append(
                    _finding(
                        FindingCode.OUTSIDE_SESSION_BOUNDS,
                        Severity.ERROR,
                        f"Event at {event.timestamp.isoformat()} precedes the declared "
                        f"session start {start.isoformat()}.",
                        event_id=event.event_id,
                        timestamp=event.timestamp.isoformat(),
                        session_start=start.isoformat(),
                    )
                )
    if end is not None:
        for event in ordered:
            if event.event_type is not EventType.SESSION_ENDED and event.timestamp > end:
                findings.append(
                    _finding(
                        FindingCode.OUTSIDE_SESSION_BOUNDS,
                        Severity.ERROR,
                        f"Event at {event.timestamp.isoformat()} follows the declared "
                        f"session end {end.isoformat()}.",
                        event_id=event.event_id,
                        timestamp=event.timestamp.isoformat(),
                        session_end=end.isoformat(),
                    )
                )
    return findings


def _known_event_types() -> frozenset[str]:
    """The engine's closed vocabulary, read from the engine rather than restated here."""
    return frozenset(member.value for member in EventType)


def dataset_summary(events: Sequence[EventEnvelope]) -> dict[str, Any]:
    """A small factual header for the UI: counts, span, and who this is about.

    Every value is derived from the events. Nothing is defaulted: a dataset with one
    learner reports one learner, and a dataset with no answers reports zero accuracy
    rather than 0.0.
    """
    if not events:
        return {
            "event_count": 0,
            "learner_ids": [],
            "session_ids": [],
            "first_timestamp": None,
            "last_timestamp": None,
            "span_seconds": None,
            "answered_count": 0,
            "correct_count": None,
            "accuracy": None,
            "mean_response_seconds": None,
            "data_origin": None,
        }

    ordered = sorted(events, key=lambda e: e.timestamp)
    answers = [e for e in ordered if e.event_type is EventType.QUESTION_ANSWERED]
    typed_answers = [
        e.payload
        for e in ordered
        if e.event_type is EventType.QUESTION_ANSWERED
        and isinstance(e.payload, QuestionAnsweredPayload)
    ]
    responses = [p.response_seconds for p in typed_answers]
    correct = [p.correct for p in typed_answers]
    origins = {event.origin for event in ordered}

    if origins == {DataOrigin.SYNTHETIC}:
        origin_label = DataOrigin.SYNTHETIC.value
    elif origins == {DataOrigin.REAL}:
        origin_label = DataOrigin.REAL.value
    elif DataOrigin.REAL in origins and DataOrigin.SYNTHETIC in origins:
        origin_label = "mixed"
    elif origins:
        origin_label = DataOrigin.REAL.value
    else:
        origin_label = "unknown"

    return {
        "event_count": len(ordered),
        "learner_ids": sorted({e.learner_id for e in ordered}),
        "session_ids": sorted({e.session_id for e in ordered}),
        "first_timestamp": ordered[0].timestamp.isoformat(),
        "last_timestamp": ordered[-1].timestamp.isoformat(),
        "span_seconds": (ordered[-1].timestamp - ordered[0].timestamp).total_seconds(),
        "answered_count": len(answers),
        "correct_count": sum(correct) if answers else None,
        "accuracy": (sum(correct) / len(correct)) if answers else None,
        "mean_response_seconds": (sum(responses) / len(responses)) if responses else None,
        "data_origin": origin_label,
    }
