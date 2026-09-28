"""Shared builders for the outcome tests (Phase 11).

The outcome engine reads two windows either side of a delivery, so a test that assembled a
whole session to check one measure would be testing seven measures at once. These builders
therefore take the *shape of the evidence* as an argument: how many activity events fall in
each window, how many questions are answered, whether a response was recorded, and whether
a temporal state exists. A test states the case it is about and nothing else.

**Every default here is synthetic.** The engine derives a record's origin from its events,
and a slice that is accidentally real would silently mark a whole record as a measurement
of a person. Building synthetic by default means a test that forgets to say otherwise gets
the safe label rather than the dangerous one, and the tests that care about a real label
have to write it out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

from focus_engine.configuration.thresholds import OutcomeSettings
from focus_engine.events.types import (
    ContentClosedPayload,
    ContentOpenedPayload,
    EventEnvelope,
    EventType,
    InactivityStartedPayload,
    InteractionPayload,
    InterventionCompletedPayload,
    InterventionStartedPayload,
    PageChangedPayload,
    QuestionAnsweredPayload,
    QuestionStartedPayload,
    SessionEndedPayload,
    VideoStartedPayload,
)
from focus_engine.outcomes.engine import OutcomeEngine
from focus_engine.policy.history import DeliveredIntervention, InterventionHistory
from focus_engine.schemas.primitives import DataOrigin, Provenance, SyntheticDataStamp
from focus_engine.temporal.models import TemporalState, TrendDirection
from focus_engine.utils.clock import FixedClock

#: A fixed instant, so a failure message reads as a date rather than as "now".
START = datetime(2026, 3, 16, 14, 0, tzinfo=UTC)

#: The instant every scenario delivers its intervention. The before window reaches back
#: from here and the after window reaches forward.
DELIVERED_AT = START

LEARNER = "learner-0001"
SESSION = "session-0001"
INTERVENTION = "intervention-0001"

#: The delivery-side outcome labels, which are free strings the delivery component owns.
#: The policy layer classifies them into :class:`~focus_engine.policy.history.OutcomeClass`
#: values, and the scenarios emit the raw label so the classification is exercised rather
#: than bypassed.
ACCEPTED = "accepted"
IGNORED = "ignored"

#: Payload factories for the event types the measures read. Kept in one table so a test can
#: name a type and get a valid payload without restating the required fields, and so a new
#: field on a payload breaks these builders in one place rather than in forty.
#:
#: Every factory takes the same two arguments - a sequence number and keyword overrides - so
#: that :func:`event` can call any of them without special-casing the type. A factory that
#: needs neither prefixes the names it ignores with an underscore, which says so rather than
#: leaving a reader to work out whether the omission was an oversight.
_PAYLOADS: dict[EventType, Callable[..., object]] = {
    EventType.QUESTION_ANSWERED: lambda n, **kw: QuestionAnsweredPayload(
        question_id=f"question-{n:04d}",
        correct=kw.get("correct", True),
        response_seconds=kw.get("response_seconds", 20.0),
    ),
    EventType.QUESTION_STARTED: lambda n, **_kw: QuestionStartedPayload(
        question_id=f"question-{n:04d}"
    ),
    EventType.INTERACTION: lambda _n, **kw: InteractionPayload(
        interaction_type=kw.get("interaction_type", "click")
    ),
    EventType.PAGE_CHANGED: lambda _n, **kw: PageChangedPayload(to_page=kw.get("to_page", "/next")),
    EventType.VIDEO_STARTED: lambda n, **_kw: VideoStartedPayload(video_id=f"video-{n:04d}"),
    EventType.CONTENT_OPENED: lambda n, **kw: ContentOpenedPayload(
        content_id=kw.get("content_id", f"content-{n:04d}")
    ),
    EventType.CONTENT_CLOSED: lambda n, **kw: ContentClosedPayload(
        content_id=kw.get("content_id", f"content-{n:04d}"),
        time_open_seconds=kw.get("time_open_seconds", 120.0),
    ),
    EventType.SESSION_ENDED: lambda _n, **kw: SessionEndedPayload(
        duration_seconds=kw.get("duration_seconds", 600.0), reason=kw.get("reason", "logout")
    ),
    EventType.INACTIVITY_STARTED: lambda _n, **kw: InactivityStartedPayload(
        last_interaction_type=kw.get("last_interaction_type", "click"),
    ),
}

#: Offsets in seconds used by the standard "learner was working, then was prompted, then
#: carried on" scenario, relative to ``DELIVERED_AT``. Every "before" offset is strictly
#: inside the default 30-minute before window and every "after" offset strictly inside the
#: 15-minute after window, spaced far enough apart that a boundary mistake of a few seconds
#: cannot move an event across a window edge. The boundaries themselves are tested
#: explicitly rather than by relying on these being awkward.
EARLY_BEFORE_TS = -60.0
MID_BEFORE_TS = -300.0
FAR_BEFORE_TS = -600.0
LONG_BEFORE_TS = -1200.0
DEEP_BEFORE_TS = -1750.0
IMMEDIATE_TS = 30.0
EARLY_AFTER_TS = 120.0
MID_AFTER_TS = 300.0
LATE_AFTER_TS = 600.0
BEYOND_TS = 1800.0
LONG_BEYOND_TS = 5400.0


def _stamp() -> SyntheticDataStamp:
    """A synthetic stamp for an event.

    Returns:
        A stamp marking the event as simulator output, carrying the generator and seed the
        stamp schema requires so the slice can be regenerated exactly.
    """
    return SyntheticDataStamp(generator="focus_engine.outcomes.test-scenarios", seed=11)


def event(
    event_type: EventType,
    offset_seconds: float,
    *,
    session_id: str = SESSION,
    learner_id: str = LEARNER,
    origin: DataOrigin = DataOrigin.SYNTHETIC,
    ordinal: int = 1,
    **payload_kwargs: object,
) -> EventEnvelope:
    """Build one event at a fixed offset from the delivery instant.

    Args:
        event_type: Which event to build.
        offset_seconds: Seconds relative to ``DELIVERED_AT``. Negative is before.
        session_id: The session the event belongs to.
        learner_id: The learner the event belongs to.
        origin: Whether the event is real or synthetic.
        ordinal: An index used to build distinct content and question identifiers.
        payload_kwargs: Per-type field overrides.

    Returns:
        The envelope.
    """
    factory = _PAYLOADS.get(event_type)
    payload = (
        factory(ordinal, **payload_kwargs)
        if factory is not None
        else _intervention_payload(event_type, ordinal, payload_kwargs)
    )
    return EventEnvelope(
        event_id=f"event-{abs(int(offset_seconds)):06d}-{event_type.value}-{ordinal:03d}",
        learner_id=learner_id,
        session_id=session_id,
        timestamp=DELIVERED_AT + timedelta(seconds=offset_seconds),
        event_type=event_type,
        payload=payload,
        origin=origin,
        provenance=Provenance.OBSERVED if origin is DataOrigin.REAL else Provenance.SYNTHETIC_LABEL,
        synthetic_stamp=None if origin is DataOrigin.REAL else _stamp(),
    )


def _number(overrides: dict[str, object], key: str, default: float) -> float:
    """Read a numeric override, rejecting anything the payload would have to coerce.

    Args:
        overrides: Field overrides.
        key: The field to read.
        default: The value to use when the field was not given.

    Returns:
        The override as a float.

    Raises:
        TypeError: If the override is not a number, which would otherwise reach the payload
            as a string and fail there with a message about the payload rather than about
            the override that caused it.
    """
    value = overrides.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number, got {type(value).__name__}")
    return float(value)


def _intervention_payload(
    event_type: EventType, ordinal: int, overrides: dict[str, object]
) -> object:
    """Build an intervention payload.

    Args:
        event_type: The event type, which must be an intervention type.
        ordinal: Unused for intervention payloads, which key off the intervention id.
        overrides: Field overrides.

    Returns:
        The payload.

    Raises:
        AssertionError: If the type is not an intervention type, which would mean the
            builder was asked for something it does not know how to construct.
    """
    if event_type is EventType.INTERVENTION_STARTED:
        return InterventionStartedPayload(
            intervention_id=str(overrides.get("intervention_id", INTERVENTION)),
            intervention_type=str(overrides.get("intervention_type", "recap")),
            trigger_probability=_number(overrides, "trigger_probability", 0.82),
            trigger_state=str(overrides.get("trigger_state", "declining_engagement")),
        )
    if event_type is EventType.INTERVENTION_COMPLETED:
        return InterventionCompletedPayload(
            intervention_id=str(overrides.get("intervention_id", INTERVENTION)),
            outcome=str(overrides.get("outcome", ACCEPTED)),
            response_seconds=overrides.get("response_seconds"),
        )
    raise AssertionError(f"{event_type} has no payload builder")


def delivery_event(
    *,
    intervention_id: str = INTERVENTION,
    intervention_type: str = "recap",
    offset_seconds: float = 0.0,
    origin: DataOrigin = DataOrigin.SYNTHETIC,
) -> EventEnvelope:
    """Build the ``intervention_started`` event for a delivery.

    Args:
        intervention_id: The delivery's identifier.
        intervention_type: The controlled-vocabulary type label.
        offset_seconds: Seconds relative to ``DELIVERED_AT``.
        origin: Whether the event is real or synthetic.

    Returns:
        The envelope.
    """
    return event(
        EventType.INTERVENTION_STARTED,
        offset_seconds,
        intervention_id=intervention_id,
        intervention_type=intervention_type,
        origin=origin,
    )


def completion_event(
    *,
    intervention_id: str = INTERVENTION,
    outcome_label: str = ACCEPTED,
    response_seconds: float | None = 8.0,
    offset_seconds: float = IMMEDIATE_TS,
    origin: DataOrigin = DataOrigin.SYNTHETIC,
) -> EventEnvelope:
    """Build the ``intervention_completed`` event for a delivery.

    Args:
        intervention_id: The delivery's identifier.
        outcome_label: The raw delivery-side outcome label.
        response_seconds: The response latency, or ``None`` for no recorded latency.
        offset_seconds: Seconds relative to ``DELIVERED_AT``.
        origin: Whether the event is real or synthetic.

    Returns:
        The envelope.
    """
    return event(
        EventType.INTERVENTION_COMPLETED,
        offset_seconds,
        intervention_id=intervention_id,
        outcome=outcome_label,
        response_seconds=response_seconds,
        origin=origin,
    )


def busy_before(
    count: int = 6, *, correct: int = 3, response_seconds: float = 20.0
) -> list[EventEnvelope]:
    """Build an active, answering learner before the delivery.

    Args:
        count: How many questions to answer. Each answer carries a ``question_started`` and
            an ``interaction`` around it, so the window holds activity as well as answers
            and the rate measures are not reading the answers twice.
        correct: How many of the *most recent* answers are correct. Applied to the recent
            end rather than the early end because a learner who is struggling reads as
            their earlier answers being wrong, and because a test that wants a flat 0.5 or
            a flat 1.0 baseline can say so.
        response_seconds: The response time for every answer.

    Returns:
        The events in chronological order.

    Note:
        Offsets are taken from the end of the ladder so that reducing ``count`` removes the
        *oldest* evidence and leaves the recent evidence in place. A test that varies the
        count to move a window below the sample floor is then removing old data, which is
        what a real thin window looks like, rather than truncating the recent history and
        leaving a gap where the learner was working.
    """
    ladder = (DEEP_BEFORE_TS, LONG_BEFORE_TS, FAR_BEFORE_TS, MID_BEFORE_TS, EARLY_BEFORE_TS)
    chosen = ladder[-count:] if count <= len(ladder) else ladder
    first_correct = len(chosen) - correct
    events: list[EventEnvelope] = []
    for index, offset in enumerate(chosen, start=1):
        events.append(
            event(
                EventType.QUESTION_ANSWERED,
                offset,
                ordinal=index,
                correct=index > first_correct,
                response_seconds=response_seconds,
            )
        )
        events.append(event(EventType.QUESTION_STARTED, offset - 20.0, ordinal=index))
        events.append(event(EventType.INTERACTION, offset - 35.0, ordinal=index))
    return events


def busy_after(count: int = 3, *, correct: int = 3) -> list[EventEnvelope]:
    """Build an active, answering learner after the delivery.

    Args:
        count: How many questions to answer after the prompt.
        correct: How many of the *most recent* answers are correct.

    Returns:
        The events in chronological order.

    Note:
        Answers are placed from the far end of the ladder inward so that reducing ``count``
        removes the *latest* evidence, which is the evidence furthest from the prompt. A
        test thinning the after window is then simulating a learner who was only briefly
        observed, not one who went quiet straight after being prompted.
    """
    ladder = (LATE_AFTER_TS, MID_AFTER_TS, EARLY_AFTER_TS, IMMEDIATE_TS + 20.0)
    chosen = ladder[-count:] if count <= len(ladder) else ladder
    first_correct = len(chosen) - correct
    events: list[EventEnvelope] = []
    for index, offset in enumerate(chosen, start=1):
        events.append(
            event(
                EventType.QUESTION_ANSWERED,
                offset,
                ordinal=100 + index,
                correct=index > first_correct,
                response_seconds=9.0,
            )
        )
        events.append(event(EventType.QUESTION_STARTED, offset - 20.0, ordinal=100 + index))
        events.append(event(EventType.INTERACTION, offset - 35.0, ordinal=100 + index))
    return events


def history(
    *,
    learner_id: str = LEARNER,
    session_id: str = SESSION,
    intervention_id: str = INTERVENTION,
    intervention_type: str = "recap",
    delivered_at: datetime = DELIVERED_AT,
    outcome_label: str | None = ACCEPTED,
) -> InterventionHistory:
    """Build a history containing one delivery.

    Args:
        learner_id: The learner.
        session_id: The session the delivery happened in.
        intervention_id: The delivery's identifier.
        intervention_type: The controlled-vocabulary type label.
        delivered_at: The delivery instant.
        outcome_label: The raw delivery-side outcome label, or ``None`` for a delivery that
            was never completed.

    Returns:
        The history.
    """
    return InterventionHistory(
        learner_id=learner_id,
        interventions=(
            DeliveredIntervention(
                intervention_id=intervention_id,
                session_id=session_id,
                intervention_type=intervention_type,
                delivered_at=delivered_at,
                outcome=outcome_label,
                responded_at=(
                    delivered_at + timedelta(seconds=8.0) if outcome_label is not None else None
                ),
            ),
        ),
    )


def temporal_state(
    direction: TrendDirection,
    *,
    observations: int = 3,
    reference_time: datetime | None = None,
    learner_id: str = LEARNER,
) -> TemporalState:
    """Build a stand-in temporal state carrying a direction for the trajectory measure.

    The outcome layer's trajectory measure is a *reader* of the temporal layer's verdict: it
    maps a direction and reports the sample count, and it does no temporal reasoning of its
    own. Building that verdict by driving real observations through the temporal engine would
    mean testing the temporal layer again from inside an outcome test, and any failure would
    be ambiguous between the two. So the verdict is stated directly here and the real state
    machinery is tested where it lives.

    Args:
        direction: The temporal layer's verdict.
        observations: The measurable observation count to report alongside it.
        reference_time: When the state speaks from. Defaults to the end of the after window,
            so the state is inside the window by default.
        learner_id: The learner the state describes.

    Returns:
        A state exposing exactly the attributes the trajectory measure reads.
    """
    return cast(
        TemporalState,
        SimpleNamespace(
            learner_id=learner_id,
            reference_time=(
                reference_time
                if reference_time is not None
                else DELIVERED_AT + timedelta(minutes=15)
            ),
            direction=direction,
            evidence=SimpleNamespace(observations=observations),
        ),
    )


def settings(**overrides: float) -> OutcomeSettings:
    """Build outcome settings with test-friendly overrides.

    Args:
        overrides: Field values to override.

    Returns:
        The settings.
    """
    return OutcomeSettings(**overrides)  # type: ignore[arg-type]


def engine(
    *,
    clock: FixedClock | None = None,
    **overrides: float,
) -> OutcomeEngine:
    """Build an engine on a fixed clock.

    Args:
        clock: The clock. Defaults to one fixed just after the delivery, so a record's
            ``computed_at`` is a constant rather than the moment the test happened to run.
        overrides: Measurement-setting overrides.

    Returns:
        The engine.
    """
    return OutcomeEngine(
        settings=OutcomeSettings(**overrides),  # type: ignore[arg-type]
        clock=clock if clock is not None else FixedClock(DELIVERED_AT + timedelta(hours=2)),
    )


def sorted_events(events: Sequence[EventEnvelope]) -> tuple[EventEnvelope, ...]:
    """Order a slice chronologically, breaking ties by identifier.

    Args:
        events: The events to order.

    Returns:
        The ordered slice. Ordering is applied by the builders rather than relied on from
        the caller, so a test that appends its events out of order still produces the same
        record as one that appends them in order.
    """
    return tuple(sorted(events, key=lambda item: (item.timestamp, item.event_id)))
