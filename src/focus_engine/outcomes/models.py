"""The vocabulary of a measured intervention outcome.

The policy layer answers *should something be delivered*. This module holds the vocabulary
for the entirely different question that comes after: *what happened next, and what may be
said about it from the events that exist*.

The distinction the whole layer rests on is the one the ROADMAP states for Phase 11 — an
outcome is an **observed** measurement, never a causal claim. A learner who answers the
next question correctly after a recap is not thereby a learner the recap helped. The events
after the prompt would very likely have looked the same without it. Naming a measurement
"improvement" is already a small causal commitment, which is why every measurement in this
module carries the *status* of the reading next to its *direction*: a record can say
"measured, no change" and "could not be measured" in the same field, and a consumer that
reads only the direction would otherwise treat the two as one fact.

Four properties are enforced in types rather than in review comments.

**Absence is a first-class answer, and it is never a negative one.** A measure that could
not be read carries :attr:`MeasurementStatus.INSUFFICIENT_DATA` and
:attr:`OutcomeDirection.NOT_ASSESSED`, together with a reason. It cannot be constructed as
:attr:`OutcomeDirection.DETERIORATED`. This is the single most important rule in the module:
an empty event stream after a delivery is *not* evidence that the learner disengaged, and a
layer that reports "no activity followed the prompt" as a deteriorated outcome would
manufacture a finding about a human from a gap in a log. Every window that yields too few
observations is therefore reported as an absence with a stated reason, and the reason text
names the gap rather than the consequence.

**"No change" and "worse" are different findings, and neither is "better".** A measured
before/after pair is only called a change once it exceeds a configured relative tolerance,
so ordinary sampling variation cannot accumulate into a claimed pattern. Below the
tolerance the reading is :attr:`OutcomeDirection.NO_CHANGE`, which is a positive claim that
something was measured and did not move. It is deliberately not :attr:`OutcomeDirection.
NOT_ASSESSED`, and the two must never be conflated: one is evidence of stability, the other
is an absence of evidence.

**A measurement is a before/after pair or it is nothing.** :class:`Measurement` refuses to
exist unless it names its own unit, and refuses a ``MEASURED`` status that carries no value
or no supporting sample count. A rate of ``0.0`` computed from zero observations is
arithmetically identical to a rate of ``0.0`` computed from a learner who was observed and
did nothing, and only the sample count distinguishes them — so the count is a required part
of the record rather than an optional annotation.

**Direction respects what "up" means for the quantity.** Response latency and activity rate
are not both better when larger. Every :class:`Measurement` records
:attr:`Measurement.lower_is_better`, so a consumer reading only the direction cannot
silently invert one measure relative to the others. Without that field, a latency that
halved and a rate that halved would both be described the same way, and one of the two
descriptions would be backwards.

The module holds vocabulary and structure only. Derivation lives in
:mod:`focus_engine.outcomes.engine`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.policy.history import OutcomeClass
from focus_engine.schemas.primitives import DataOrigin, Provenance, Timestamp, non_empty_text
from focus_engine.schemas.versioning import OutcomeVersion, PolicyVersion

__all__ = [
    "OUTCOME_MEASURES",
    "OUTCOME_V1",
    "Measurement",
    "MeasurementStatus",
    "OutcomeDirection",
    "OutcomeMeasure",
    "OutcomeRecord",
    "OutcomeWindow",
    "WindowKind",
]


OUTCOME_V1: Final[OutcomeVersion] = "OUTCOME_V1"
"""The initial outcome definition set.

Freeze the measure set, the direction vocabulary, and the window geometry together, so
that a change to any of them mints ``OUTCOME_V2`` rather than silently reinterpreting every
outcome already recorded. The window widths are part of this freeze for a reason the other
version aliases do not share: a narrower after window measures a shorter stretch of the
learner's behaviour, and can report a deterioration where a wider one reports no change. A
stored outcome read under different widths would therefore be a record of a measurement
that was never made.
"""


class WindowKind(StrEnum):
    """Which side of the delivery instant a window lies on."""

    BEFORE = "before"
    """The comparison window immediately preceding delivery."""

    AFTER = "after"
    """The measurement window following delivery."""


class OutcomeMeasure(StrEnum):
    """One thing an outcome record measures.

    The set is fixed and the tuple :data:`OUTCOME_MEASURES` is canonical, so a record can be
    required to cover every measure exactly once and a consumer's switch over measures can
    be checked for exhaustiveness. Each member names an *observable* property of the event
    stream. None of them names a mechanism, because no observational measure in this layer
    can distinguish "the intervention caused this" from "this would have happened anyway",
    and a vocabulary containing causal words would invite a reader to supply the causal
    reading the data cannot support.
    """

    IMMEDIATE_INTERACTION_CHANGE = "immediate_interaction_change"
    """Whether interaction resumed promptly, compared against the preceding rate.

    A rate comparison over a short window at the very start of the after window. It
    separates a burst of resumed interaction from sustained engagement, which a single
    after-window rate cannot tell apart: a learner who clicks once and stops improves this
    measure and worsens the continued-activity one.
    """

    RESPONSE_LATENCY = "response_latency"
    """How long the learner took to respond, compared against their own earlier latency.

    Read from the recorded response time against the mean response time of questions
    answered before the delivery. A learner who responds faster than they usually do is
    recorded as faster; the measure does not claim the prompt caused it.
    """

    ACCURACY = "accuracy"
    """The share of questions answered correctly, before against after.

    The one measure whose unit is a proportion in ``[0.0, 1.0]``. Reported only when both
    windows contain at least one answered question, because an accuracy computed from a
    single answer is that answer.
    """

    CONTINUED_ACTIVITY = "continued_activity"
    """Whether interaction continued across the whole after window.

    A rate over the full after window, so it answers "did engagement persist" where
    :attr:`IMMEDIATE_INTERACTION_CHANGE` answers "did it resume".
    """

    TASK_PERSISTENCE = "task_persistence"
    """Whether the learner stayed with the work they were already doing.

    The share of items opened after delivery that were already open before it. One-sided by
    nature: continuity is a property of the after window measured against the before
    window's contents, and there is no meaningful "before" figure for it to pair with.
    """

    SESSION_CONTINUATION = "session_continuation"
    """Whether the session survived the delivery.

    Read from the presence of a session-end event *after* the prompt. Its absence is never
    read as continuation *or* as termination: a session with no end event is reported as
    insufficient data, because a log that stops is not a learner who left.
    """

    SUBSEQUENT_TRAJECTORY = "subsequent_trajectory"
    """The direction the learner's deviation was already heading after the prompt.

    Reused from the temporal layer rather than recomputed, so the outcome layer inherits
    the persistence rule, the anomaly-versus-trajectory distinction, and the standing of
    the inference without owning a second implementation of any of them. One-sided: it
    reports where the learner was heading, and there is no direction figure to pair it
    with at the delivery instant.
    """


#: Canonical measure order. The immediate measures come first because they are the
#: narrowest in time, and the trajectory last because it is the slowest to resolve.
OUTCOME_MEASURES: Final[tuple[OutcomeMeasure, ...]] = (
    OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE,
    OutcomeMeasure.RESPONSE_LATENCY,
    OutcomeMeasure.ACCURACY,
    OutcomeMeasure.CONTINUED_ACTIVITY,
    OutcomeMeasure.TASK_PERSISTENCE,
    OutcomeMeasure.SESSION_CONTINUATION,
    OutcomeMeasure.SUBSEQUENT_TRAJECTORY,
)


class MeasurementStatus(StrEnum):
    """Whether a measure could be read at all.

    Carried beside every reading, because "no change" and "no reading" are different
    findings and a field that held only a direction would collapse them.
    """

    MEASURED = "measured"
    """Both sides of the comparison were available and the reading is quantified."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Not enough events in a window to support a reading.

    The same label the baseline, temporal, and uncertainty layers use for this condition,
    reused rather than reminted so that one word means one thing across the engine. It is
    never a negative finding: too few observations cannot support a claim in either
    direction.
    """

    NOT_APPLICABLE = "not_applicable"
    """The measure has no meaning for this delivery.

    A distinct value rather than another form of insufficiency, because "this cannot apply"
    and "this could not be read" call for different responses. A session that had already
    ended before the prompt was delivered cannot be said to have continued after it, and no
    amount of additional logging would change that.
    """


class OutcomeDirection(StrEnum):
    """Which way a measurement moved, relative to the learner's own earlier reading.

    Named for the absence of an optimistic default. There is deliberately no bare
    ``IMPROVED``-shaped shortcut and no neutral ``UNCHANGED``: the four members separate
    the three findings a before/after comparison can produce from the one that it cannot.
    """

    IMPROVED = "improved"
    """Measured, and moved in the favourable direction by more than the tolerance."""

    NO_CHANGE = "no_change"
    """Measured, and moved by less than the tolerance.

    A positive claim of stability, and therefore a *finding* rather than a default. It is
    kept separate from :attr:`NOT_ASSESSED` because "we looked and nothing moved" is a
    legitimate result that should not be reported as ignorance, and because a system that
    conflated the two would have no way to state that a stable learner is stable.
    """

    DETERIORATED = "deteriorated"
    """Measured, and moved in the unfavourable direction by more than the tolerance."""

    NOT_ASSESSED = "not_assessed"
    """No reading was produced.

    Only ever reachable alongside :attr:`MeasurementStatus.INSUFFICIENT_DATA` or
    :attr:`MeasurementStatus.NOT_APPLICABLE`, and a validator enforces the pairing. The
    prohibition this encodes is the important one: an absent measurement cannot be
    constructed as a deterioration, so no code path can report missing events as a
    learner who disengaged.
    """


class OutcomeWindow(BaseModel):
    """A half-open instant range that a measurement reads from.

    Half-open in the sense that matters for leakage: the before window ends *at* the
    delivery instant and excludes it, and the after window begins at the delivery instant
    and includes it. A single boundary rule, applied once, means no event can be read as
    evidence both before and after the intervention that produced it. An inclusive end on
    the before window would place the delivery itself in the baseline, and the baseline
    would then contain the thing it is supposed to be a comparison for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: WindowKind
    start: Timestamp
    end: Timestamp
    duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_window(self) -> OutcomeWindow:
        """Check the window is a forward interval consistent with its stated duration.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the end precedes the start, or the duration disagrees with the
                interval. The duration is a convenience field for a consumer, and a stale
                copy of it would misreport the width of the measurement a record claims
                to have taken.
        """
        if self.end < self.start:
            raise ValueError(
                f"{self.kind.value} window ends at {self.end.isoformat()}, before it starts "
                f"at {self.start.isoformat()}"
            )
        actual = (self.end - self.start).total_seconds()
        if abs(actual - self.duration_seconds) > 1e-6:
            raise ValueError(
                f"{self.kind.value} window spans {actual:g} seconds but records a duration "
                f"of {self.duration_seconds:g}; the recorded width must match the interval, "
                "or a reader cannot tell how wide the measurement actually was"
            )
        return self

    def contains(self, instant: datetime) -> bool:
        """Whether an instant lies inside this window.

        The rule is the one stated on the class: the start is inclusive and the end is
        exclusive, so the two windows either side of a delivery share a boundary instant
        that belongs to exactly one of them.

        Args:
            instant: The instant to test.

        Returns:
            ``True`` when ``start <= instant < end``.
        """
        return self.start <= instant < self.end


class Measurement(BaseModel):
    """One measure, read across the two windows.

    The shape is deliberately uniform across all seven measures: unit, status, direction,
    the two values, the two sample counts, and the standing of the data. A consumer can
    therefore read any measure with the same code, and a heterogeneous set of record
    types — one per measure, each with its own idea of what to do about a missing value —
    cannot exist.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    measure: OutcomeMeasure
    status: MeasurementStatus
    direction: OutcomeDirection
    unit: non_empty_text
    """What the numbers are in, e.g. ``events_per_minute``, ``seconds``, ``fraction``.

    Required because a value without a unit is unreadable across measures. A latency of
    ``12.0`` and an activity rate of ``12.0`` are unrelated quantities, and a record that
    carried the figure alone would invite a consumer to compare them.
    """

    before_value: float | None = None
    after_value: float | None = None
    before_samples: int = Field(default=0, ge=0)
    after_samples: int = Field(default=0, ge=0)
    lower_is_better: bool = False
    """Whether a smaller value is the favourable direction for this measure.

    Recorded per measure rather than as a lookup table, so a consumer reading a single
    measurement cannot misinterpret the direction without also reading the flag that
    defines it.
    """

    reason: str | None = None
    """Why the measure is not measured, or what the reading must not be taken to mean.

    Required exactly when the measure is not measured. Optional when it is, and the latitude
    is deliberate: some measured findings carry a caveat that a reader would otherwise supply
    themselves, in the unfavourable direction. A session that ended inside the after window is
    measured as a session that did not continue, and the events cannot say whether the prompt
    is why — a reading that stated the direction but left the causal reading to the reader
    would be handing over the one conclusion this layer exists not to draw. A caveat is
    therefore permitted on a measured reading, and required on none.
    """

    data_origin: DataOrigin
    """Whether the events behind this measurement were real or synthetic.

    Required rather than defaulted. A measurement derived from simulator output that
    defaulted to ``REAL`` would report the simulator's assumptions as measurements of a
    person, and the default is exactly how such an error would reach production unnoticed.
    """

    @model_validator(mode="after")
    def _validate_measurement(self) -> Measurement:
        """Check the reading, the status, the direction, and the samples agree.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If a measured reading has no value or no supporting sample, if an
                unmeasured reading carries a direction or omits its reason, if a value is
                present without a supporting sample count, or if a measured reading claims
                a sample count with no value behind it.

        Note:
            The reason is required when a measure is *not* measured and permitted when it is.
            The asymmetry is the point: an unexplained absence is a silent gap, so a reason is
            mandatory there, while a measured finding sometimes needs to say what it does not
            establish (a session end is not evidence the prompt ended the session). Refusing a
            caveat on a measured reading would force that distinction to be dropped, and
            dropping it is how "the learner stopped because of the prompt" gets read into a
            record that says only "the session ended".

            The sample-count-without-value pairing is enforced only for measured readings.
            An unmeasured reading is allowed to report a count without a value, because that
            is how it says *how far short of the evidence it fell*: "the before window holds
            two activity events and the measure requires five" is the useful fact, and the
            mandatory reason carries the explanation that makes it readable. Forbidding it
            would push the count out of the record and leave a reader with a reason string
            and no numbers. A measured reading gets no such latitude, because a count with no
            value there is a reading whose zero has been lost.
        """
        measured = self.status is MeasurementStatus.MEASURED
        unassessed = self.direction is OutcomeDirection.NOT_ASSESSED

        if measured and unassessed:
            raise ValueError(
                f"measure {self.measure.value} is measured but its direction is "
                f"{self.direction.value}; a measurement with no direction asserts that "
                "nothing changed, which is not the same as having read a value"
            )
        if not measured and not unassessed:
            raise ValueError(
                f"measure {self.measure.value} is {self.status.value!r} but reports "
                f"direction {self.direction.value!r}. Only a measured comparison can have a "
                "direction. In particular a measure with insufficient data can never be "
                "reported as deteriorated: too few observations cannot support a claim in "
                "either direction, and a log that stops is not a learner who disengaged."
            )
        if not measured and self.reason is None:
            raise ValueError(
                f"measure {self.measure.value} is {self.status.value!r} and states no reason. "
                "A reading that reports nothing must say what is missing, because the reason "
                "is the only channel through which a reader learns that a number is absent"
            )
        if self.reason is not None and not self.reason.strip():
            raise ValueError(
                f"measure {self.measure.value} carries an empty reason; an unexplained "
                "absence is indistinguishable from a bug"
            )

        if measured:
            if self.after_value is None:
                raise ValueError(
                    f"measure {self.measure.value} is measured but reports no after value"
                )
            if self.after_samples < 1:
                raise ValueError(
                    f"measure {self.measure.value} reports an after value from "
                    f"{self.after_samples} sample(s); a value with no observation behind it "
                    "is arithmetic, not a measurement"
                )
            if self.before_value is None and self.before_samples > 0:
                raise ValueError(
                    f"measure {self.measure.value} reports {self.before_samples} before "
                    "sample(s) but no before value"
                )

        for label, value, samples in (
            ("before", self.before_value, self.before_samples),
            ("after", self.after_value, self.after_samples),
        ):
            if value is not None and samples < 1:
                raise ValueError(
                    f"measure {self.measure.value} reports a {label} value from "
                    f"{samples} sample(s); a value with no observation behind it is "
                    "arithmetic, not a measurement"
                )
            if measured and value is None and samples > 0:
                raise ValueError(
                    f"measure {self.measure.value} reports {samples} {label} sample(s) but "
                    f"no {label} value. The sample count is what distinguishes a measured zero "
                    "from an absence, so it cannot stand on its own."
                )
        return self

    @property
    def is_measured(self) -> bool:
        """Whether this measure produced a reading."""
        return self.status is MeasurementStatus.MEASURED

    @property
    def change(self) -> float | None:
        """The signed after-minus-before difference.

        Returns:
            ``None`` when either side is absent, so a one-sided measure cannot report a
            change against nothing. The sign alone carries no judgement: whether a
            positive value is favourable is :attr:`lower_is_better`.
        """
        if self.before_value is None or self.after_value is None:
            return None
        return self.after_value - self.before_value

    def describe(self) -> str:
        """Render the measurement as a short human-readable line.

        Returns:
            A one-line description naming the measure, the status, the direction, and the
            two values in their stated unit. Punctuation is kept to ASCII because this
            string is a rendering, and a rendering that raises ``UnicodeEncodeError`` in a
            log stream is a rendering that fails at the moment it is most needed.
        """
        marker = "lower is better" if self.lower_is_better else "higher is better"
        if not self.is_measured:
            return f"{self.measure.value}: {self.status.value} [{self.direction.value}] - {self.reason}"
        before = "none" if self.before_value is None else f"{self.before_value:g}"
        after = "none" if self.after_value is None else f"{self.after_value:g}"
        return (
            f"{self.measure.value}: {self.direction.value} "
            f"({before} -> {after} {self.unit}, {self.before_samples}->"
            f"{self.after_samples} samples, {marker})"
        )


class OutcomeRecord(BaseModel):
    """Everything measured about one delivered intervention, and nothing claimed about it.

    The record is per delivery, not per learner or per type. Aggregating first would lose
    the pairing with the event that produced it, and a measure of "what usually happens
    after a recap" is a different and much weaker claim than a measure of what happened
    after *this* recap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome_version: OutcomeVersion
    intervention_id: non_empty_text
    learner_id: non_empty_text
    session_id: non_empty_text
    intervention_type: non_empty_text
    """The controlled-vocabulary type label the policy selected, carried through so an
    outcome can be grouped by type without re-joining the policy's own records."""

    delivered_at: Timestamp
    """The delivery instant, which anchors both windows.

    The after window is anchored here and not at the response, deliberately. A
    response-anchored window would measure a different interval for a learner who engaged
    than for one who did not, so the two deliveries would not be comparable; and a learner
    who never responded would have no after window at all, which is precisely the
    delivery whose outcome most needs recording. Anchoring both windows to delivery also
    means the window is knowable from the delivery record alone, before any later event has
    occurred.
    """

    before_window: OutcomeWindow
    after_window: OutcomeWindow
    measurements: tuple[Measurement, ...]
    response_class: OutcomeClass
    """How the delivery's own reported completion was classified.

    Carried from the policy layer's vocabulary rather than redefined: whether the learner
    engaged with the prompt at all is a fact about the delivery, and the policy already
    classifies it. This record adds what happened *afterwards*, and keeping the two apart
    is what stops "the learner did not respond to the prompt" from being read as "the
    prompt did not work" — the first is an observation, the second is a causal claim, and
    only one of them is supported by the events.
    """

    policy_version: PolicyVersion
    """The policy definition in force when the intervention was selected.

    Carried so an outcome can be read against the rules that chose the intervention. An
    outcome measured against a policy nobody can reconstruct is an outcome nobody can
    interpret.
    """

    data_origin: DataOrigin
    """Whether the events behind this record were real or synthetic."""

    provenance: Provenance
    """How this record came to exist.

    Constrained to :attr:`Provenance.OBSERVED` by the record's validator. A record that
    carries its own provenance as a free choice is a record in which "this is a causal
    finding" is one field assignment away, and the Phase 11 line between an observed
    measurement and a causal claim is exactly the line that must not be crossable by
    setting a value. Stating the vocabulary and then refusing the other members is the
    only form of this constraint that cannot be bypassed.
    """

    settings_fingerprint: str
    """Digest of the measurement configuration, so a record cannot later be read as though
    it had been produced under window widths it never saw."""

    computed_at: Timestamp

    @model_validator(mode="after")
    def _validate_record(self) -> OutcomeRecord:
        """Check the record covers every measure once, in a consistent frame.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If a measure is missing or repeated, if the windows are not one
                before and one after, if the windows do not meet at the delivery instant,
                or if any measurement's recorded origin contradicts the record's own.
        """
        covered = [item.measure for item in self.measurements]
        if len(set(covered)) != len(covered):
            duplicates = sorted({item.value for item in covered if covered.count(item) > 1})
            raise ValueError(f"outcome record repeats measures: {duplicates}")
        if (
            tuple(sorted(covered, key=lambda item: OUTCOME_MEASURES.index(item)))
            != OUTCOME_MEASURES
        ):
            missing = sorted(
                {item.value for item in OUTCOME_MEASURES} - {item.value for item in covered}
            )
            raise ValueError(
                f"outcome record does not cover every measure; missing {missing}. A record "
                "with a hole in it is indistinguishable from one where the measure was "
                "considered and found to be pure noise, so the coverage is required."
            )

        windows = {self.before_window.kind, self.after_window.kind}
        if windows != {WindowKind.BEFORE, WindowKind.AFTER}:
            raise ValueError(
                f"outcome record must carry one before and one after window; got "
                f"{sorted(kind.value for kind in windows)}"
            )
        if self.before_window.end != self.after_window.start:
            raise ValueError(
                f"the before window ends at {self.before_window.end.isoformat()} and the after "
                f"window starts at {self.after_window.start.isoformat()}. Both are anchored to "
                "the delivery instant, so they must meet exactly there; a gap would silently "
                "drop events and an overlap would let one event be evidence on both sides of "
                "the intervention."
            )
        if self.after_window.start != self.delivered_at:
            raise ValueError(
                f"the after window starts at {self.after_window.start.isoformat()} but the "
                f"intervention was delivered at {self.delivered_at.isoformat()}; the after "
                "window is anchored to delivery and to nothing else"
            )

        synthetic = {
            item.data_origin
            for item in self.measurements
            if item.data_origin is DataOrigin.SYNTHETIC
        }
        if synthetic and self.data_origin is not DataOrigin.SYNTHETIC:
            raise ValueError(
                "outcome record reports real data while "
                f"{len(synthetic)} of its measurements report synthetic origins. A record "
                "that hides synthetic evidence behind a real marker is the one error this "
                "project's provenance rules exist to prevent."
            )
        if self.provenance is not Provenance.OBSERVED:
            raise ValueError(
                f"outcome record carries provenance {self.provenance.value!r}, but a "
                "before/after measurement of an event stream is an observation and nothing "
                "else. A record marked as anything but observed is a causal claim wearing "
                "the vocabulary of a measurement."
            )
        return self

    def measurement(self, measure: OutcomeMeasure) -> Measurement:
        """Return one measure by name.

        Args:
            measure: The measure to look up.

        Returns:
            Its measurement.

        Raises:
            KeyError: If the record does not carry the measure. The record's validator
                requires full coverage, so this is unreachable for a constructed record and
                exists to keep the failure explicit rather than to be handled.
        """
        for item in self.measurements:
            if item.measure is measure:
                return item
        raise KeyError(measure)

    @property
    def measured(self) -> tuple[Measurement, ...]:
        """The measures that produced a reading, in canonical order."""
        return tuple(item for item in self.measurements if item.is_measured)

    @property
    def is_complete(self) -> bool:
        """Whether every measure produced a reading.

        A rare condition, and the record says so rather than leaving a consumer to infer
        completeness from a count. Most deliveries will have at least one unmeasured
        measure, because a prompt delivered seconds before a session ends has no room for
        a trajectory reading and that is a fact about the data rather than a defect.
        """
        return len(self.measured) == len(OUTCOME_MEASURES)

    @property
    def unmeasured(self) -> tuple[Measurement, ...]:
        """The measures that produced no reading, in canonical order."""
        return tuple(item for item in self.measurements if not item.is_measured)

    def directions(self) -> dict[OutcomeMeasure, OutcomeDirection]:
        """The direction of every measure.

        Returns:
            A mapping from measure to direction, in canonical order. A consumer that wants
            a single verdict still has to say how it combines them; the layer deliberately
            does not reduce seven measurements to one score, because no such reduction is
            defensible without knowing which measures matter for which learner and state,
            and that weighting is a policy decision belonging to Phase 12.
        """
        return {item.measure: item.direction for item in self.measurements}

    @property
    def improved(self) -> tuple[OutcomeMeasure, ...]:
        """The measures that moved favourably."""
        return tuple(
            item.measure
            for item in self.measurements
            if item.direction is OutcomeDirection.IMPROVED
        )

    @property
    def deteriorated(self) -> tuple[OutcomeMeasure, ...]:
        """The measures that moved unfavourably."""
        return tuple(
            item.measure
            for item in self.measurements
            if item.direction is OutcomeDirection.DETERIORATED
        )

    @property
    def is_synthetic_only(self) -> bool:
        """Whether every measurement rests on synthetic data."""
        return all(item.data_origin is DataOrigin.SYNTHETIC for item in self.measurements)

    def describe(self) -> str:
        """Render the record as a short human-readable block.

        Returns:
            A multi-line summary naming the delivery, both windows, and every measure.
        """
        header = (
            f"{self.intervention_type} ({self.intervention_id}) for {self.learner_id}, "
            f"delivered {self.delivered_at.isoformat()}"
        )
        windows = (
            f"  before {self.before_window.start.isoformat()} .. "
            f"{self.before_window.end.isoformat()}"
            f"  after {self.after_window.start.isoformat()} .. "
            f"{self.after_window.end.isoformat()}"
        )
        return "\n".join([header, windows, *(f"  {item.describe()}" for item in self.measurements)])

    def to_summary_dict(self) -> dict[str, object]:
        """Reduce the record to flat, countable fields.

        Returns:
            A mapping with no nested containers, suitable for a counter or a metrics frame.
            The measure readings are counted rather than embedded, and each direction is
            reported as a scalar, so a summary of thousands of outcomes stays small and
            stays countable. A summary that omitted the unmeasured count would let a
            corpus of absent measurements present as a corpus of stable learners.
        """
        return {
            "outcome_version": self.outcome_version,
            "intervention_id": self.intervention_id,
            "learner_id": self.learner_id,
            "session_id": self.session_id,
            "intervention_type": self.intervention_type,
            "delivered_at": self.delivered_at.isoformat(),
            "response_class": self.response_class.value,
            "policy_version": self.policy_version,
            "measures_total": len(self.measurements),
            "measures_measured": len(self.measured),
            "measures_improved": len(self.improved),
            "measures_deteriorated": len(self.deteriorated),
            "before_seconds": self.before_window.duration_seconds,
            "after_seconds": self.after_window.duration_seconds,
            "settings_fingerprint": self.settings_fingerprint,
            "data_origin": self.data_origin.value,
        }
