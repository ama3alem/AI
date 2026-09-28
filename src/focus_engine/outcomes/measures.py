"""The seven readings, one function each.

This module holds the reading logic as free functions over a fixed event slice. It is kept
separate from :mod:`focus_engine.outcomes.engine` for the same reason the policy layer keeps
its catalogue separate from its engine: the arithmetic that turns events into a measurement
is the part that is hardest to review and the part most likely to be wrong, so it is
isolated from the part that decides which windows to read and how to assemble the record.

Every function here is a pure function of its arguments. No clock, no configuration
lookup, no state carried between calls. That is what lets the determinism test replay a
whole record twice and compare, and it is why these functions can be reasoned about
individually rather than as steps inside a long method.

**The absence rule, applied uniformly.** A function that cannot read both sides of a
comparison returns a measurement with :attr:`MeasurementStatus.INSUFFICIENT_DATA` and
:attr:`OutcomeDirection.NOT_ASSESSED` rather than a default, a zero, or a deterioration.
The model then refuses to construct a direction on an unmeasured reading, so the rule
cannot be bypassed by arithmetic. Every reason string names the missing evidence and, where
the temptation is strongest, says in as many words that the absence is not evidence of
failure — because the reader of a record is exactly the person who will otherwise supply
the missing-event interpretation themselves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from focus_engine.events.types import (
    ContentClosedPayload,
    ContentOpenedPayload,
    EventEnvelope,
    EventType,
    InterventionCompletedPayload,
    QuestionAnsweredPayload,
)
from focus_engine.outcomes.models import (
    Measurement,
    MeasurementStatus,
    OutcomeDirection,
    OutcomeMeasure,
    OutcomeWindow,
)
from focus_engine.policy.history import OutcomeClass
from focus_engine.schemas.primitives import DataOrigin
from focus_engine.temporal.models import TemporalState, TrendDirection

__all__ = [
    "ACTIVITY_EVENTS",
    "direction_of",
    "measure_accuracy",
    "measure_continued_activity",
    "measure_immediate_interaction",
    "measure_response_latency",
    "measure_session_continuation",
    "measure_subsequent_trajectory",
    "measure_task_persistence",
]

#: Events that evidence the learner doing something.
#:
#: Two exclusions are deliberate and worth stating. ``INACTIVITY_STARTED`` is absent because
#: it evidences an *absence* of interaction, and folding it into an activity count would
#: let one event move a measure in the direction that its own name contradicts. The two
#: intervention events are absent because the intervention is the system's doing rather
#: than the learner's: counting the prompt as evidence that the learner resumed would be
#: the one fabrication this layer exists to prevent, because the learner did not click, the
#: system clicked for them.
ACTIVITY_EVENTS: Final[frozenset[EventType]] = frozenset(
    {
        EventType.CONTENT_OPENED,
        EventType.INTERACTION,
        EventType.PAGE_CHANGED,
        EventType.QUESTION_ANSWERED,
        EventType.QUESTION_REVIEWED,
        EventType.QUESTION_SKIPPED,
        EventType.QUESTION_STARTED,
        EventType.VIDEO_RESUMED,
        EventType.VIDEO_STARTED,
    }
)

#: Trajectory direction codes, oriented so that *falling is favourable*.
#:
#: A signed code is used rather than a raw standardised deviation because the sign of a
#: standardised deviation is defined per dimension, and reusing one of those signs here
#: would let the outcome layer re-derive a meaning the temporal layer never promised. This
#: table is the outcome layer's own reading of the temporal vocabulary, and it is inverted
#: relative to the usual "higher means more deviation" convention on purpose: a rising code
#: means the deviation is growing, and a growing deviation is the unfavourable direction for
#: every measure in this layer.
_DIRECTION_CODES: Final[dict[TrendDirection, float]] = {
    TrendDirection.WORSENING: 1.0,
    TrendDirection.STABLE: 0.0,
    TrendDirection.IMPROVING: -1.0,
}

_TRAJECTORY_CODES: Final[dict[float, OutcomeDirection]] = {
    -1.0: OutcomeDirection.IMPROVED,
    0.0: OutcomeDirection.NO_CHANGE,
    1.0: OutcomeDirection.DETERIORATED,
}

#: Relative slack allowed when testing a difference against a tolerance.
#:
#: The comparison is *inclusive* by definition — a difference at exactly the tolerance is no
#: change — and in binary floating point that boundary is not a number. ``1.05 - 1.0`` is
#: ``0.050000000000000044``, which is greater than ``0.05``, so a plain ``<=`` reports a
#: change for a difference that is mathematically equal to the tolerance and silently shifts
#: the boundary by one representable step. A caller configuring ``0.05`` would get a verdict
#: that disagrees with the rule it was given.
#:
#: The slack is set three orders of magnitude above double-precision epsilon (~2.2e-16) so it
#: absorbs representation error, and far below any difference a caller could mean: a
#: difference of one part in ``1e12`` is not a behavioural signal, and widening the band to
#: admit one would be an invented tolerance wearing a rounding guard's clothes.
_BOUNDARY_RELATIVE_SLACK: Final[float] = 1e-12


def _within_tolerance(difference: float, threshold: float) -> bool:
    """Whether a difference sits at or below its threshold, inclusively.

    Args:
        difference: The absolute before/after difference.
        threshold: The largest difference still counted as no change.

    Returns:
        ``True`` when the difference does not exceed the threshold, allowing only for the
        representation error of the two floats being compared.
    """
    return difference <= threshold or math.isclose(
        difference, threshold, rel_tol=_BOUNDARY_RELATIVE_SLACK
    )


def direction_of(
    before: float,
    after: float,
    *,
    lower_is_better: bool,
    tolerance: float,
) -> OutcomeDirection:
    """Classify a before/after difference, once, for every measure.

    No measure computes its own comparison. A per-measure tolerance check would let one of
    the seven get the inequality wrong, and a reader auditing the layer would have to
    compare seven implementations to find which one had.

    Args:
        before: The value read from the before window.
        after: The value read from the after window.
        lower_is_better: Whether a falling value is the favourable direction.
        tolerance: Smallest relative difference counted as a change, measured against
            ``max(|before|, 1.0)`` so the test stays defined when ``before`` is zero.

    Returns:
        The direction. A difference at or below the tolerance is
        :attr:`~focus_engine.outcomes.models.OutcomeDirection.NO_CHANGE`, which is a
        finding of stability rather than a default for having nothing to say.
    """
    if _within_tolerance(abs(after - before), tolerance * max(abs(before), 1.0)):
        return OutcomeDirection.NO_CHANGE
    fell = after < before
    return OutcomeDirection.IMPROVED if (fell is lower_is_better) else OutcomeDirection.DETERIORATED


def _in_window(
    events: Sequence[EventEnvelope], window: OutcomeWindow, kinds: frozenset[EventType]
) -> tuple[EventEnvelope, ...]:
    """The events of the given kinds inside a window.

    Args:
        events: Candidate events.
        window: The window to read.
        kinds: Event types to keep.

    Returns:
        A tuple of matching events, in the order supplied. Input order is preserved rather
        than re-sorted, because a record is a function of the slice it was handed and
        re-ordering would make the result depend on a sort the caller did not ask for.
    """
    return tuple(
        event for event in events if event.event_type in kinds and window.contains(event.timestamp)
    )


def _activity_rate(events: Sequence[EventEnvelope], window: OutcomeWindow) -> tuple[float, int]:
    """Activity events per minute in a window, with the supporting count.

    Args:
        events: Candidate events.
        window: The window to read.

    Returns:
        The rate and the count. A count of zero yields a rate of ``0.0``, which is why the
        count is returned alongside: ``0.0`` computed from no observations and ``0.0``
        computed from a learner who was observed and did nothing are different findings,
        and only the count distinguishes them.
    """
    count = len(_in_window(events, window, ACTIVITY_EVENTS))
    minutes = window.duration_seconds / 60.0
    return (count / minutes, count) if minutes > 0.0 else (0.0, count)


def _rate_measurement(
    measure: OutcomeMeasure,
    events: Sequence[EventEnvelope],
    before: OutcomeWindow,
    after: OutcomeWindow,
    min_samples: int,
    tolerance: float,
    origin: DataOrigin,
) -> Measurement:
    """Shared body of the two rate comparisons.

    Args:
        measure: Which measure this is.
        events: Candidate events.
        before: The before window.
        after: The after window, or the immediate sub-window.
        min_samples: Minimum events required in a window before it supports a reading.
        tolerance: Relative change below which the reading is no change.
        origin: Whether the events were real or synthetic.

    Returns:
        A measurement covering the two windows.
    """
    before_rate, before_count = _activity_rate(events, before)
    after_rate, after_count = _activity_rate(events, after)

    if before_count < min_samples:
        return Measurement(
            measure=measure,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="events_per_minute",
            after_samples=after_count,
            lower_is_better=False,
            reason=(
                f"the before window holds {before_count} activity event(s) and the measure "
                f"requires {min_samples}. A baseline built from fewer events is not a rate, "
                "it is a few clicks, and a comparison against it would read a difference "
                "that is not there."
            ),
            data_origin=origin,
        )

    if after_count < min_samples:
        return Measurement(
            measure=measure,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="events_per_minute",
            before_value=before_rate,
            before_samples=before_count,
            lower_is_better=False,
            reason=(
                f"the measured window holds {after_count} activity event(s) and the measure "
                f"requires {min_samples}. This is an absence of observation, not an observed "
                "absence of engagement: the learner may have left, the window may have closed "
                "before they acted, or the events may not have reached the store yet. No "
                "conclusion about the intervention follows from it."
            ),
            data_origin=origin,
        )

    return Measurement(
        measure=measure,
        status=MeasurementStatus.MEASURED,
        direction=direction_of(before_rate, after_rate, lower_is_better=False, tolerance=tolerance),
        unit="events_per_minute",
        before_value=before_rate,
        after_value=after_rate,
        before_samples=before_count,
        after_samples=after_count,
        lower_is_better=False,
        data_origin=origin,
    )


def measure_immediate_interaction(
    events: Sequence[EventEnvelope],
    before: OutcomeWindow,
    immediate: OutcomeWindow,
    *,
    min_samples: int,
    tolerance: float,
    origin: DataOrigin,
) -> Measurement:
    """Read whether interaction resumed promptly after the prompt.

    Args:
        events: Candidate events.
        before: The before window, supplying the comparison rate.
        immediate: The short window at the start of the after window.
        min_samples: Minimum activity events required in each window.
        tolerance: Relative change below which the reading is no change.
        origin: Whether the events were real or synthetic.

    Returns:
        A measurement over the immediate window, compared against the before rate.
    """
    return _rate_measurement(
        OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE,
        events,
        before,
        immediate,
        min_samples,
        tolerance,
        origin,
    )


def measure_continued_activity(
    events: Sequence[EventEnvelope],
    before: OutcomeWindow,
    after: OutcomeWindow,
    *,
    min_samples: int,
    tolerance: float,
    origin: DataOrigin,
) -> Measurement:
    """Read whether interaction persisted across the whole after window.

    Args:
        events: Candidate events.
        before: The before window, supplying the comparison rate.
        after: The full after window.
        min_samples: Minimum activity events required in each window.
        tolerance: Relative change below which the reading is no change.
        origin: Whether the events were real or synthetic.

    Returns:
        A measurement over the full after window, compared against the before rate.

    Note:
        A short window would fold a burst of resumed interaction into the same number as
        sustained engagement. Reading the same after window at two lengths is the only way
        to tell a learner who came back and stayed from one who came back once.
    """
    return _rate_measurement(
        OutcomeMeasure.CONTINUED_ACTIVITY,
        events,
        before,
        after,
        min_samples,
        tolerance,
        origin,
    )


def measure_accuracy(
    events: Sequence[EventEnvelope],
    before: OutcomeWindow,
    after: OutcomeWindow,
    *,
    tolerance: float,
    origin: DataOrigin,
) -> Measurement:
    """Read the share of questions answered correctly, before against after.

    Args:
        events: Candidate events.
        before: The before window.
        after: The after window.
        tolerance: Relative change below which the reading is no change.
        origin: Whether the events were real or synthetic.

    Returns:
        A measurement of accuracy as a proportion.
    """
    answered = _in_window(events, before, frozenset({EventType.QUESTION_ANSWERED}))
    resolved = _in_window(events, after, frozenset({EventType.QUESTION_ANSWERED}))
    before_correct = sum(
        1
        for event in answered
        if isinstance(event.payload, QuestionAnsweredPayload) and event.payload.correct
    )
    after_correct = sum(
        1
        for event in resolved
        if isinstance(event.payload, QuestionAnsweredPayload) and event.payload.correct
    )

    if not answered:
        return Measurement(
            measure=OutcomeMeasure.ACCURACY,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="fraction",
            after_samples=len(resolved),
            lower_is_better=False,
            reason=(
                "no question was answered in the before window, so the learner has no "
                "measurable standard to compare against. A post-delivery score on its own is "
                "a score, not an outcome."
            ),
            data_origin=origin,
        )
    if not resolved:
        return Measurement(
            measure=OutcomeMeasure.ACCURACY,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="fraction",
            before_value=before_correct / len(answered),
            before_samples=len(answered),
            lower_is_better=False,
            reason=(
                f"no question was answered in the after window, while {len(answered)} were "
                "answered before the prompt. The learner not answering is not the same as "
                "the learner answering badly, and reporting an absent answer as a fall in "
                "accuracy would turn a gap in the log into a claim about a person."
            ),
            data_origin=origin,
        )

    before_value = before_correct / len(answered)
    after_value = after_correct / len(resolved)
    return Measurement(
        measure=OutcomeMeasure.ACCURACY,
        status=MeasurementStatus.MEASURED,
        direction=direction_of(
            before_value, after_value, lower_is_better=False, tolerance=tolerance
        ),
        unit="fraction",
        before_value=before_value,
        after_value=after_value,
        before_samples=len(answered),
        after_samples=len(resolved),
        lower_is_better=False,
        data_origin=origin,
    )


def measure_response_latency(
    events: Sequence[EventEnvelope],
    before: OutcomeWindow,
    delivery: InterventionCompletedPayload | None,
    *,
    response_class: OutcomeClass,
    tolerance: float,
    origin: DataOrigin,
) -> Measurement:
    """Read how long the learner took to engage, against their own earlier speed.

    Args:
        events: Candidate events.
        before: The before window, supplying the learner's own mean latency.
        delivery: The recorded completion for this delivery, or ``None`` when the delivery
            was never completed.
        response_class: How the delivery's completion was classified. Read rather than
            re-derived, so the measure and the record cannot disagree about whether the
            learner engaged.
        tolerance: Relative change below which the reading is no change.
        origin: Whether the events were real or synthetic.

    Returns:
        A measurement of the learner's own baseline latency against the observed one.

    Note:
        A delivery with no response produces a latency of nothing rather than a latency of
        zero and a direction of improvement. The learner declining to engage is reported by
        the policy layer's response class, and this layer's job is to say only that no
        latency can be read.

        A *recognised* non-response produces the same absence even when the completion
        carries a response time, and this is the case the measure is most tempted to get
        wrong. The delivery component can attach a time to a delivery the learner dismissed;
        the time is real, and it is not a time they took to engage. Reading it as one would
        report a learner who ignored the prompt as a learner who engaged faster than usual,
        which is the single most flattering available misreading of a dismissal, and it
        would be invisible in the output — the numbers would look like an improvement.
    """
    if response_class is OutcomeClass.NON_RESPONSE:
        return Measurement(
            measure=OutcomeMeasure.RESPONSE_LATENCY,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="seconds",
            after_samples=0 if delivery is None else 1,
            lower_is_better=True,
            reason=(
                "the delivery was recorded as a non-response, so the learner did not engage "
                "with the prompt and there is no response time to read. A time recorded "
                "against a delivery the learner declined is not a time they took to engage, "
                "and reading it as one would report a learner who ignored the prompt as a "
                "learner who answered it faster than usual."
            ),
            data_origin=origin,
        )

    answered = _in_window(events, before, frozenset({EventType.QUESTION_ANSWERED}))
    latencies = [
        event.payload.response_seconds
        for event in answered
        if isinstance(event.payload, QuestionAnsweredPayload)
    ]

    if not latencies:
        return Measurement(
            measure=OutcomeMeasure.RESPONSE_LATENCY,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="seconds",
            after_samples=0 if delivery is None else 1,
            lower_is_better=True,
            reason=(
                "no question was answered in the before window, so the learner has no "
                "measurable response latency to compare against. A single post-delivery "
                "latency says how long they took once, not whether that was fast for them."
            ),
            data_origin=origin,
        )

    before_value = sum(latencies) / len(latencies)
    if delivery is None or delivery.response_seconds is None:
        return Measurement(
            measure=OutcomeMeasure.RESPONSE_LATENCY,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="seconds",
            before_value=before_value,
            before_samples=len(latencies),
            lower_is_better=True,
            reason=(
                "the delivery carries no response time, so none could be read. An "
                "unobserved response time is not a slow one, and this measure will not "
                "assume the learner was unresponsive."
            ),
            data_origin=origin,
        )

    after_value = delivery.response_seconds
    return Measurement(
        measure=OutcomeMeasure.RESPONSE_LATENCY,
        status=MeasurementStatus.MEASURED,
        direction=direction_of(
            before_value, after_value, lower_is_better=True, tolerance=tolerance
        ),
        unit="seconds",
        before_value=before_value,
        after_value=after_value,
        before_samples=len(latencies),
        after_samples=1,
        lower_is_better=True,
        data_origin=origin,
    )


def measure_task_persistence(
    events: Sequence[EventEnvelope],
    after: OutcomeWindow,
    *,
    origin: DataOrigin,
) -> Measurement:
    """Read whether the learner finished what they were already doing.

    The item tracked is the last one opened at or before the delivery instant. The measure
    then asks what happened to it: closed inside the after window, or displaced by a
    different item.

    Args:
        events: Candidate events.
        after: The after window, whose start is the delivery instant.
        origin: Whether the events were real or synthetic.

    Returns:
        A measurement of whether the tracked task was completed or set aside.

    Note:
        There is no before window here, and the omission is deliberate rather than an
        oversight. The tracked item is the last one opened at or before the delivery
        instant, which reaches further back than any configured before window: a task opened
        twenty minutes before delivery is still the task that was open when the prompt
        arrived, and a window that failed to see it would report that nothing was
        interrupted for a learner who was interrupted mid-item. Reading the item from the
        whole prior stream keeps the measure's question - was something open at the
        instant - separate from how recent the configured window happens to be.

    Note:
        Displacement is counted from *both* directions in which the learner's attention can
        have gone elsewhere: a different item opened inside the window, and a different item
        closed inside it. Opening one is the obvious case, but closing one is equally evidence
        that the window's work happened elsewhere, and a rule that watched only for opens
        would report a learner who spent the window finishing an earlier item as still on the
        task they were interrupted on. Watching closes as well is what makes the tracked item
        the one that matters rather than the one that happened to be opened last.

        Nothing observable inside the window means nothing happened, not that the learner
        was interrupted. Reporting a closed window as abandonment is the specific version of
        "missing events prove failure" that this measure is most tempted by, and it is why
        the unresolved case is insufficient data rather than a deterioration.
    """
    opened = [
        event.payload.content_id
        for event in events
        if event.event_type is EventType.CONTENT_OPENED
        and isinstance(event.payload, ContentOpenedPayload)
        and event.timestamp < after.start
    ]
    if not opened:
        return Measurement(
            measure=OutcomeMeasure.TASK_PERSISTENCE,
            status=MeasurementStatus.NOT_APPLICABLE,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="boolean",
            lower_is_better=False,
            reason=(
                "no content item was open when the prompt was delivered, so there is no task "
                "for the learner to have persisted on. The delivery interrupted nothing."
            ),
            data_origin=origin,
        )

    tracked = opened[-1]
    closed = {
        event.payload.content_id
        for event in _in_window(events, after, frozenset({EventType.CONTENT_CLOSED}))
        if isinstance(event.payload, ContentClosedPayload)
    }
    reopened = {
        event.payload.content_id
        for event in _in_window(events, after, frozenset({EventType.CONTENT_OPENED}))
        if isinstance(event.payload, ContentOpenedPayload)
    }
    moved_to = (closed | reopened) - {tracked}
    finished = tracked in closed
    replaced = bool(moved_to)

    if finished and not replaced:
        return Measurement(
            measure=OutcomeMeasure.TASK_PERSISTENCE,
            status=MeasurementStatus.MEASURED,
            direction=OutcomeDirection.IMPROVED,
            unit="boolean",
            after_value=1.0,
            after_samples=1,
            lower_is_better=False,
            data_origin=origin,
        )
    if replaced and not finished:
        return Measurement(
            measure=OutcomeMeasure.TASK_PERSISTENCE,
            status=MeasurementStatus.MEASURED,
            direction=OutcomeDirection.DETERIORATED,
            unit="boolean",
            after_value=0.0,
            after_samples=1,
            lower_is_better=False,
            data_origin=origin,
        )
    if finished and replaced:
        return Measurement(
            measure=OutcomeMeasure.TASK_PERSISTENCE,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="boolean",
            after_samples=2,
            lower_is_better=False,
            reason=(
                f"the tracked task ({tracked}) was closed and a different item was opened or "
                "completed inside the same window. The record cannot distinguish finishing "
                "the task and moving on from closing it and giving up, and a closed item read "
                "as a completed one would be the single most flattering available misreading. "
                "The measure reports nothing rather than choosing between them."
            ),
            data_origin=origin,
        )

    return Measurement(
        measure=OutcomeMeasure.TASK_PERSISTENCE,
        status=MeasurementStatus.INSUFFICIENT_DATA,
        direction=OutcomeDirection.NOT_ASSESSED,
        unit="boolean",
        after_samples=0,
        lower_is_better=False,
        reason=(
            f"the item open at delivery ({tracked}) was neither closed nor replaced inside "
            "the after window. The learner may still be working on it, and this measure will "
            "not report an unresolved task as an abandoned one."
        ),
        data_origin=origin,
    )


def measure_session_continuation(
    session_id: str,
    events: Sequence[EventEnvelope],
    before: OutcomeWindow,
    after: OutcomeWindow,
    *,
    origin: DataOrigin,
) -> Measurement:
    """Read whether the learner's session outlived the prompt.

    Args:
        session_id: The session the delivery happened in.
        events: Candidate events.
        before: The before window.
        after: The after window.
        origin: Whether the events were real or synthetic.

    Returns:
        A measurement of whether the session was still running at the end of the window.

    Note:
        An observed session end is recorded as :attr:`OutcomeDirection.DETERIORATED` in the
        literal sense that the session did not continue, and nothing more. A fifteen-minute
        window catches an ordinary learner who finished their work as readily as one who
        walked away, the engine has no evidence to tell the two apart, and the measure
        exists to record which happened rather than to argue that the prompt caused it.
    """
    ended_before = any(
        event.event_type is EventType.SESSION_ENDED
        and event.session_id == session_id
        and before.contains(event.timestamp)
        for event in events
    )
    if ended_before:
        return Measurement(
            measure=OutcomeMeasure.SESSION_CONTINUATION,
            status=MeasurementStatus.NOT_APPLICABLE,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="boolean",
            lower_is_better=False,
            reason=(
                "the session had already ended before the prompt was delivered, so a session "
                "that survived the prompt is not a thing that can have happened."
            ),
            data_origin=origin,
        )

    ended_after = any(
        event.event_type is EventType.SESSION_ENDED
        and event.session_id == session_id
        and after.contains(event.timestamp)
        for event in events
    )
    if ended_after:
        return Measurement(
            measure=OutcomeMeasure.SESSION_CONTINUATION,
            status=MeasurementStatus.MEASURED,
            direction=OutcomeDirection.DETERIORATED,
            unit="boolean",
            after_value=1.0,
            after_samples=1,
            lower_is_better=False,
            reason=(
                "a session end was observed inside the after window. The session did not "
                "outlive the prompt, which is what this measure records; whether the prompt "
                "is why is not something the events can answer."
            ),
            data_origin=origin,
        )

    still_active = any(
        event.session_id == session_id and event.timestamp >= after.end for event in events
    )
    if still_active:
        return Measurement(
            measure=OutcomeMeasure.SESSION_CONTINUATION,
            status=MeasurementStatus.MEASURED,
            direction=OutcomeDirection.NO_CHANGE,
            unit="boolean",
            after_value=0.0,
            after_samples=1,
            lower_is_better=False,
            data_origin=origin,
        )

    return Measurement(
        measure=OutcomeMeasure.SESSION_CONTINUATION,
        status=MeasurementStatus.INSUFFICIENT_DATA,
        direction=OutcomeDirection.NOT_ASSESSED,
        unit="boolean",
        lower_is_better=False,
        reason=(
            "the session produced no end event inside the after window and no activity after "
            "it. That is what a session that was still open looks like, and it is also what a "
            "log that simply stopped collecting looks like, and this measure cannot tell the "
            "two apart. It will not report a continuing session as an ended one."
        ),
        data_origin=origin,
    )


def measure_subsequent_trajectory(
    state: TemporalState | None,
    after: OutcomeWindow,
    *,
    origin: DataOrigin,
) -> Measurement:
    """Read where the learner's deviation was heading, using the temporal layer's verdict.

    Args:
        state: The temporal state as of the end of the after window, or ``None``.
        after: The after window, used to confirm the state speaks to this window.
        origin: Whether the state was real or synthetic.

    Returns:
        A measurement carrying the temporal layer's own direction.

    Note:
        This function reads a verdict and maps it. It does not count runs, does not decide
        whether a trajectory is sustained, and does not reconstruct a direction from raw
        deviations: the persistence rule and the anomaly-versus-trajectory distinction
        belong to the temporal layer, and a second copy of either would be free to disagree
        with the first, which is exactly the failure a single owner prevents.
    """
    if state is None:
        return Measurement(
            measure=OutcomeMeasure.SUBSEQUENT_TRAJECTORY,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="direction_code",
            lower_is_better=True,
            reason=(
                "no temporal state was supplied for the end of the after window. The "
                "trajectory is the temporal layer's finding to make, and this layer will not "
                "manufacture one in its absence."
            ),
            data_origin=origin,
        )

    if state.reference_time < after.start:
        return Measurement(
            measure=OutcomeMeasure.SUBSEQUENT_TRAJECTORY,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="direction_code",
            lower_is_better=True,
            reason=(
                f"the supplied state is dated {state.reference_time.isoformat()}, before the "
                f"after window opened at {after.start.isoformat()}, so it describes the "
                "learner before the prompt rather than after it."
            ),
            data_origin=origin,
        )

    samples = state.evidence.observations
    if state.direction is TrendDirection.INSUFFICIENT_DATA or samples < 1:
        return Measurement(
            measure=OutcomeMeasure.SUBSEQUENT_TRAJECTORY,
            status=MeasurementStatus.INSUFFICIENT_DATA,
            direction=OutcomeDirection.NOT_ASSESSED,
            unit="direction_code",
            after_samples=samples,
            lower_is_better=True,
            reason=(
                f"the temporal layer reports {state.direction.value} from {samples} "
                "observation(s). Its own absence vocabulary carries over unchanged: an "
                "unestablished direction is not a stable one."
            ),
            data_origin=origin,
        )

    code = _DIRECTION_CODES[state.direction]
    return Measurement(
        measure=OutcomeMeasure.SUBSEQUENT_TRAJECTORY,
        status=MeasurementStatus.MEASURED,
        direction=_TRAJECTORY_CODES[code],
        unit="direction_code",
        after_value=code,
        after_samples=samples,
        lower_is_better=True,
        data_origin=origin,
    )
