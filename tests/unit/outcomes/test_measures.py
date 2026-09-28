"""The readings built on rates, shares, and latencies, and what they do without evidence.

Every measure here obeys one rule: a comparison that cannot be made reports nothing, and
says why, in a sentence that names the missing evidence. The rule exists because the failure
it prevents is not a crash. A missing event is far more easily rendered as a zero, a
deterioration, or a no-change, and each of those renders as a fact about a person rather than
a gap in a log.

So the tests below come in two families, and the second is the one that matters. The happy
paths confirm each measure reads the right thing from evidence that is present; a reader who
only saw those would conclude the layer is confident. The absence paths confirm each measure
declines to conclude when evidence is missing. The absence tests also check the reason
wording, because the reason string is the only channel through which a reader learns that a
number is missing. A reason that says "no data" leaves the reader to supply the missing-event
interpretation themselves, and they will, and it will read as a fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from focus_engine.events.types import EventEnvelope, EventType
from focus_engine.outcomes.measures import direction_of
from focus_engine.outcomes.models import (
    Measurement,
    OutcomeDirection,
    OutcomeMeasure,
    OutcomeRecord,
)
from tests.unit.outcomes import scenarios as s

pytestmark = pytest.mark.unit


def measured(record: OutcomeRecord, measure: OutcomeMeasure) -> Measurement:
    """Pull one measurement out of a record.

    Args:
        record: The record to read.
        measure: Which measurement to return.

    Returns:
        The measurement.

    Raises:
        AssertionError: If the record does not cover the measure, which the model forbids
            and so should never happen.
    """
    for measurement in record.measurements:
        if measurement.measure is measure:
            return measurement
    raise AssertionError(f"{measure.value} is missing from the record")


def reason_of(reading: Measurement) -> str:
    """Read a measurement's reason as text.

    Every measurement that withheld a reading has to say why, so the reason is not optional
    on those and a test that asserts on its wording should not be satisfiable by a reading
    that has none. Returning the narrowed text rather than asserting and continuing keeps
    the assertion about the wording rather than about the absence of a wording.

    Args:
        reading: The measurement to read.

    Returns:
        The reason.

    Raises:
        AssertionError: If the measurement carries no reason, which is a defect in the
            measure rather than in the test.
    """
    assert reading.reason is not None, (
        f"{reading.measure.value} withheld a reading without a reason"
    )
    return reading.reason


def run(events: Sequence[EventEnvelope], **overrides: object) -> OutcomeRecord:
    """Measure the standard delivery against a slice of events.

    Args:
        events: The event slice. The delivery event is appended, since a measurement of a
            delivery that was never recorded would be refused before it started.
        overrides: Measurement-setting overrides, passed through to the scenario engine.

    Returns:
        The record.
    """
    engine = s.engine(**cast("dict[str, Any]", overrides))
    return engine.measure(
        s.history(), s.INTERVENTION, s.sorted_events([*events, s.delivery_event()])
    )


class TestImmediateInteractionChange:
    """Did the learner resume interacting promptly after the prompt?"""

    def test_a_burst_right_after_the_prompt_is_read_as_improved(self) -> None:
        """Dense early activity is compared against the learner's own baseline rate."""
        reading = measured(
            run([*s.busy_before(6), *s.busy_after(3)]),
            OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE,
        )

        assert reading.status.value == "measured"
        assert reading.direction is OutcomeDirection.IMPROVED
        assert reading.lower_is_better is False
        assert reading.before_value is not None
        assert reading.after_value is not None
        assert reading.after_value > reading.before_value

    def test_the_immediate_window_reads_less_evidence_than_the_continued_one(self) -> None:
        """The two rate measures read different evidence rather than repeating one.

        A learner who came back once and a learner who came back and stayed produce the same
        number in a short window, so the short window is only interesting if the long one
        exists alongside it.
        """
        record = run([*s.busy_before(6), *s.busy_after(3)])
        immediate = measured(record, OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE)
        continued = measured(record, OutcomeMeasure.CONTINUED_ACTIVITY)

        assert immediate.after_samples < continued.after_samples

    def test_a_thin_baseline_yields_no_reading(self) -> None:
        """Too few baseline events means there is no rate to compare against."""
        reading = measured(
            run([*s.busy_before(1), *s.busy_after(3)]),
            OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE,
        )

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert reading.after_value is None

    def test_a_thin_baseline_says_a_rate_needs_more_than_a_click(self) -> None:
        """The reason rejects the specific misreading a reader would otherwise supply."""
        reading = measured(
            run([*s.busy_before(1), *s.busy_after(3)]),
            OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE,
        )

        assert "not a rate" in reason_of(reading)

    def test_silence_after_the_prompt_is_not_reported_as_deterioration(self) -> None:
        """No activity after the prompt produces no direction at all.

        This is the most consequential case in the layer. A learner who left, a window that
        closed early, and an event pipeline that has not flushed all look identical from here,
        and rendering any of them as "the intervention did not help" is a claim about a person
        drawn from a gap in a log.
        """
        reading = measured(run(s.busy_before(6)), OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE)

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert reading.before_value is not None
        assert "absence of observation" in reason_of(reading)

    def test_silence_names_the_things_it_could_mean(self) -> None:
        """The reason distinguishes an unobserved learner from an unobserved log."""
        reading = measured(run(s.busy_before(6)), OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE)

        assert "may have left" in reason_of(reading)
        assert "not have reached the store yet" in reason_of(reading)

    def test_a_withheld_reading_makes_the_record_incomplete(self) -> None:
        """A consumer can tell the difference between a report and a non-report."""
        record = run(s.busy_before(6))

        assert record.is_complete is False

    def test_the_prompt_itself_does_not_count_as_the_learner_resuming(self) -> None:
        """The intervention events are excluded from the activity count.

        The learner did not click; the system clicked for them. Counting the prompt as evidence
        of resumption is the one fabrication this layer exists to prevent, and it would be
        invisible in the output, because the count would simply be one higher.
        """
        with_prompt = measured(
            run([*s.busy_before(6), *s.busy_after(3), s.completion_event()]),
            OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE,
        )
        without_prompt = measured(
            run([*s.busy_before(6), *s.busy_after(3)]),
            OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE,
        )

        assert with_prompt.after_samples == without_prompt.after_samples

    def test_an_inactivity_event_is_not_counted_as_activity(self) -> None:
        """An inactivity event evidences an absence, so it cannot evidence engagement.

        Folding it into an activity count would let one event move a measure in the direction
        its own name contradicts.
        """
        record = run(
            [
                *s.busy_before(6),
                *s.busy_after(3),
                s.event(EventType.INACTIVITY_STARTED, s.IMMEDIATE_TS),
            ]
        )
        reading = measured(record, OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE)

        assert reading.status.value == "measured"
        assert reading.after_samples is not None


class TestContinuedActivity:
    """Did interaction persist across the whole after window?"""

    def test_sustained_activity_is_read_as_improved(self) -> None:
        """Activity across the full window is a favourable reading."""
        reading = measured(
            run([*s.busy_before(6), *s.busy_after(3)]), OutcomeMeasure.CONTINUED_ACTIVITY
        )

        assert reading.status.value == "measured"
        assert reading.direction is OutcomeDirection.IMPROVED

    def test_a_brief_return_is_flat_over_a_long_window(self) -> None:
        """A short burst of activity does not read as sustained engagement.

        The learner answers twice and stops, which is the brief return in its smallest
        measurable form: one question is three activity events, and a floor of four keeps a
        single answer from becoming a rate at all. Those two answers are a 2.5 per minute
        burst against a 0.4 per minute baseline, and a clear improvement in the immediate
        window. Spread across the whole fifteen minutes they come to 0.4, which is exactly
        the learner's own pace before the prompt: a wash. The same burst read as persistence
        would be the one flattering reading available here, and the difference between the
        two readings is the reason both windows exist.
        """
        record = run([*s.busy_before(4), *s.busy_after(2)])

        assert measured(record, OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE).direction is (
            OutcomeDirection.IMPROVED
        )
        assert measured(record, OutcomeMeasure.CONTINUED_ACTIVITY).direction is (
            OutcomeDirection.NO_CHANGE
        )

    def test_no_change_is_a_finding_rather_than_a_default(self) -> None:
        """A flat reading is measured, carries both values, and states no reason.

        A no-change with a reason attached would be indistinguishable from a failure to
        measure, and the model refuses that ambiguity outright.
        """
        reading = measured(
            run([*s.busy_before(4), *s.busy_after(2)]), OutcomeMeasure.CONTINUED_ACTIVITY
        )

        assert reading.status.value == "measured"
        assert reading.reason is None
        assert reading.before_value is not None
        assert reading.after_value is not None

    def test_a_rate_is_reported_per_minute_with_its_counts(self) -> None:
        """The unit and both counts travel with the value.

        A rate without its count is unreadable: 0.5 events per minute over thirty minutes and
        over three minutes are different amounts of evidence.
        """
        reading = measured(
            run([*s.busy_before(6), *s.busy_after(3)]), OutcomeMeasure.CONTINUED_ACTIVITY
        )

        assert reading.unit == "events_per_minute"
        assert reading.before_samples is not None and reading.before_samples > 0
        assert reading.after_samples is not None and reading.after_samples > 0

    def test_a_silent_after_window_yields_no_reading(self) -> None:
        """The continued measure reports nothing rather than reporting a fall to zero."""
        reading = measured(run(s.busy_before(6)), OutcomeMeasure.CONTINUED_ACTIVITY)

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert "not an observed absence of engagement" in reason_of(reading)


class TestDirectionClassification:
    """The single shared before/after comparison."""

    @pytest.mark.parametrize(
        ("before", "after", "lower_is_better", "expected"),
        [
            (1.0, 2.0, False, OutcomeDirection.IMPROVED),
            (2.0, 1.0, False, OutcomeDirection.DETERIORATED),
            (1.0, 0.5, True, OutcomeDirection.IMPROVED),
            (0.5, 1.0, True, OutcomeDirection.DETERIORATED),
            (1.0, 1.0, False, OutcomeDirection.NO_CHANGE),
        ],
    )
    def test_the_favourable_direction_depends_on_the_measure(
        self,
        before: float,
        after: float,
        lower_is_better: bool,
        expected: OutcomeDirection,
    ) -> None:
        """A fall is good for latency and bad for accuracy.

        Pinned as a table because the two conventions sit next to each other in the same
        field, and a single mixed-up flag would invert one measure with no visible symptom.
        """
        assert (
            direction_of(before, after, lower_is_better=lower_is_better, tolerance=0.0) is expected
        )

    @pytest.mark.parametrize(
        ("delta", "expected"),
        [
            (0.049, OutcomeDirection.NO_CHANGE),
            (0.05, OutcomeDirection.NO_CHANGE),
            (0.051, OutcomeDirection.IMPROVED),
        ],
    )
    def test_tolerance_is_inclusive_at_the_boundary(
        self, delta: float, expected: OutcomeDirection
    ) -> None:
        """A difference at exactly the tolerance is no change, not a change.

        Pinned at three points around the boundary so that changing ``<=`` to ``<`` fails a
        test rather than shifting one measure's verdict silently.
        """
        result = direction_of(1.0, 1.0 + delta, lower_is_better=False, tolerance=0.05)

        assert result is expected

    def test_tolerance_is_measured_against_a_floor_of_one(self) -> None:
        """A near-zero baseline does not make any change look enormous.

        At a baseline of 0.01, moving to 0.06 is a sixfold change in relative terms and means
        almost nothing in absolute terms. Scaling the tolerance by ``max(|before|, 1)`` keeps
        the test meaningful there instead of making every near-empty window read as a
        dramatic change: the difference is 0.05, exactly the floored threshold, and it is
        the floor alone that keeps it a no-change. Against the unfloored 0.0005 the same
        numbers would read as a dramatic rise.
        """
        result = direction_of(0.01, 0.06, lower_is_better=False, tolerance=0.05)

        assert result is OutcomeDirection.NO_CHANGE

    def test_a_tolerance_of_zero_makes_every_difference_a_change(self) -> None:
        """Zero tolerance is the degenerate but legal case.

        Worth pinning because the engine is free to be configured that way, and a strict
        inequality silently used instead would turn "report every difference" into "report
        only the non-trivial ones".
        """
        result = direction_of(1.0, 1.0000001, lower_is_better=False, tolerance=0.0)

        assert result is OutcomeDirection.IMPROVED


class TestAccuracy:
    """Did the share of correct answers change?"""

    def test_a_rising_share_is_read_as_improved(self) -> None:
        """More correct answers out of the same measure is favourable."""
        reading = measured(
            run([*s.busy_before(6, correct=3), *s.busy_after(3, correct=3)]),
            OutcomeMeasure.ACCURACY,
        )

        assert reading.status.value == "measured"
        assert reading.direction is OutcomeDirection.IMPROVED
        assert reading.after_value == pytest.approx(1.0)

    def test_the_reading_is_a_share_with_its_counts(self) -> None:
        """Accuracy is a fraction, and the answer counts travel with it."""
        reading = measured(
            run([*s.busy_before(6, correct=3), *s.busy_after(3, correct=3)]),
            OutcomeMeasure.ACCURACY,
        )

        assert reading.unit == "fraction"
        assert reading.before_value == pytest.approx(0.6)
        assert reading.before_samples == 5
        assert reading.after_samples == 3

    def test_a_falling_share_is_read_as_deterioration(self) -> None:
        """Fewer correct answers is a real, reportable finding."""
        reading = measured(
            run([*s.busy_before(6, correct=5), *s.busy_after(3, correct=0)]),
            OutcomeMeasure.ACCURACY,
        )

        assert reading.direction is OutcomeDirection.DETERIORATED

    def test_a_flat_share_reads_as_no_change(self) -> None:
        """Unchanged accuracy is stability, and says so."""
        reading = measured(
            run([*s.busy_before(6, correct=5), *s.busy_after(3, correct=3)]),
            OutcomeMeasure.ACCURACY,
        )

        assert reading.direction is OutcomeDirection.NO_CHANGE

    def test_no_baseline_answers_yields_no_reading(self) -> None:
        """A post-delivery score alone is a score, not an outcome.

        A learner who answers everything correctly after a recap, having answered nothing
        before it, has improved nothing that was measured.
        """
        reading = measured(run(s.busy_after(3)), OutcomeMeasure.ACCURACY)

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert "a score, not an outcome" in reason_of(reading)

    def test_not_answering_is_not_answering_badly(self) -> None:
        """No answers after the prompt is not a fall in accuracy.

        This is the accuracy twin of the silence case. Reporting a dropped answer as a wrong
        answer would convert a missing event into a claim about a person's ability, and it
        would do so in the direction that looks like a real result.
        """
        reading = measured(run(s.busy_before(6, correct=5)), OutcomeMeasure.ACCURACY)

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert "not the same as" in reason_of(reading)

    def test_a_failed_comparison_still_carries_the_known_baseline(self) -> None:
        """The baseline survives into the record even when the comparison fails.

        A consumer reading only the numbers should still see what the learner's accuracy
        actually was, rather than inferring it from a reason string.
        """
        reading = measured(run(s.busy_before(6, correct=5)), OutcomeMeasure.ACCURACY)

        assert reading.before_value == pytest.approx(1.0)
        assert reading.before_samples == 5


class TestResponseLatency:
    """Did the learner engage faster than they usually do?"""

    def test_a_faster_response_is_read_as_improved(self) -> None:
        """The learner's own earlier speed is the comparison, not a population norm.

        A fixed threshold would call a fast learner slow and a slow learner fast, and the
        measure would then be describing the cohort rather than this learner.
        """
        reading = measured(
            run(
                [
                    *s.busy_before(6, response_seconds=20.0),
                    s.completion_event(response_seconds=8.0),
                ]
            ),
            OutcomeMeasure.RESPONSE_LATENCY,
        )

        assert reading.status.value == "measured"
        assert reading.direction is OutcomeDirection.IMPROVED
        assert reading.lower_is_better is True
        assert reading.before_value == pytest.approx(20.0)
        assert reading.after_value == pytest.approx(8.0)

    def test_a_slower_response_is_read_as_deterioration(self) -> None:
        """Taking longer than usual is a reportable finding."""
        reading = measured(
            run(
                [
                    *s.busy_before(6, response_seconds=20.0),
                    s.completion_event(response_seconds=45.0),
                ]
            ),
            OutcomeMeasure.RESPONSE_LATENCY,
        )

        assert reading.direction is OutcomeDirection.DETERIORATED

    def test_the_baseline_is_the_learners_own_mean(self) -> None:
        """Varying latencies average rather than being read as a single sample."""
        events = s.sorted_events(
            [
                s.event(
                    EventType.QUESTION_ANSWERED,
                    s.DEEP_BEFORE_TS,
                    correct=True,
                    response_seconds=10.0,
                ),
                s.event(
                    EventType.QUESTION_ANSWERED,
                    s.EARLY_BEFORE_TS,
                    correct=True,
                    response_seconds=30.0,
                ),
                s.delivery_event(),
                s.completion_event(response_seconds=20.0),
            ]
        )
        reading = measured(
            s.engine().measure(s.history(), s.INTERVENTION, events),
            OutcomeMeasure.RESPONSE_LATENCY,
        )

        assert reading.before_value == pytest.approx(20.0)
        assert reading.before_samples == 2

    def test_no_response_is_not_a_response_of_zero_seconds(self) -> None:
        """A delivery carrying no response time produces no latency reading.

        A zero-second latency with an improved direction is the most flattering available
        misreading of a learner who did nothing at all.
        """
        reading = measured(
            run([*s.busy_before(6), s.completion_event(response_seconds=None)]),
            OutcomeMeasure.RESPONSE_LATENCY,
        )

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert reading.after_value is None
        assert "not a slow one" in reason_of(reading)

    def test_an_ignored_delivery_still_records_the_non_response(self) -> None:
        """The non-response is reported, by the layer that owns that fact.

        The measure says nothing and the record as a whole still says the learner did not
        engage. Losing the finding because the arithmetic declined to proceed would be the
        opposite error to inventing one.
        """
        record = run([*s.busy_before(6), s.completion_event(outcome_label=s.IGNORED)])
        reading = measured(record, OutcomeMeasure.RESPONSE_LATENCY)

        assert record.response_class.value == "non_response"
        assert reading.status.value == "insufficient_data"

    def test_no_baseline_latency_yields_no_reading(self) -> None:
        """A single response time says how long they took once, not whether it was fast."""
        reading = measured(
            run([s.completion_event(response_seconds=8.0)]), OutcomeMeasure.RESPONSE_LATENCY
        )

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert "not whether that was fast for them" in reason_of(reading)

    def test_a_failed_comparison_still_carries_the_known_baseline(self) -> None:
        """The learner's normal speed survives into the record."""
        reading = measured(
            run([*s.busy_before(6), s.completion_event(response_seconds=None)]),
            OutcomeMeasure.RESPONSE_LATENCY,
        )

        assert reading.before_value == pytest.approx(20.0)
        assert reading.before_samples == 5
