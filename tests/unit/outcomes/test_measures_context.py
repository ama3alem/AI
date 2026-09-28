"""The readings that follow a single item, a session, and the temporal layer's verdict.

These three measures sit apart from the rate and share measures in one respect that matters:
each of them is one absence away from inventing a fact. A rate can go quiet; an item can be
declared abandoned; a session can be declared over; a trajectory can be declared stable. In
every case the missing evidence is the *entire* finding, and a default would read as a
result.

The tests below are therefore weighted towards the unmeasured paths, and each one asserts on
the reason wording, because that string is the only channel a reader has to learn that the
number is missing.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from focus_engine.events.types import EventEnvelope, EventType
from focus_engine.outcomes.models import (
    OutcomeDirection,
    OutcomeMeasure,
    OutcomeRecord,
)
from focus_engine.temporal.models import TrendDirection
from tests.unit.outcomes import scenarios as s
from tests.unit.outcomes.test_measures import measured, reason_of, run

pytestmark = pytest.mark.unit


def _opened(content_id: str, offset: float) -> EventEnvelope:
    """Build a content-opened event.

    Args:
        content_id: The item's identifier.
        offset: Seconds relative to the delivery instant.

    Returns:
        The event.
    """
    return s.event(EventType.CONTENT_OPENED, offset, content_id=content_id)


def _closed(content_id: str, offset: float) -> EventEnvelope:
    """Build a content-closed event.

    Args:
        content_id: The item's identifier.
        offset: Seconds relative to the delivery instant.

    Returns:
        The event.
    """
    return s.event(EventType.CONTENT_CLOSED, offset, content_id=content_id)


class TestTaskPersistence:
    """Did the learner finish what they were already doing?"""

    def test_closing_the_tracked_item_is_read_as_persistence(self) -> None:
        """A task completed inside the window is the favourable reading."""
        reading = measured(
            run(
                [
                    _opened("content-0001", s.MID_BEFORE_TS),
                    *s.busy_after(3),
                    _closed("content-0001", s.EARLY_AFTER_TS),
                ]
            ),
            OutcomeMeasure.TASK_PERSISTENCE,
        )

        assert reading.status.value == "measured"
        assert reading.direction is OutcomeDirection.IMPROVED
        assert reading.after_value == pytest.approx(1.0)

    def test_moving_to_a_different_item_is_read_as_deterioration(self) -> None:
        """The tracked task being replaced is the unfavourable reading."""
        reading = measured(
            run(
                [
                    _opened("content-0001", s.MID_BEFORE_TS),
                    *s.busy_after(3),
                    _opened("content-9999", s.EARLY_AFTER_TS),
                ]
            ),
            OutcomeMeasure.TASK_PERSISTENCE,
        )

        assert reading.status.value == "measured"
        assert reading.direction is OutcomeDirection.DETERIORATED
        assert reading.after_value == pytest.approx(0.0)

    def test_reopening_the_same_item_is_not_replacing_it(self) -> None:
        """A reopen of the tracked item is a visit, not a switch away from it.

        Treating a reopen as abandonment would report a learner who went back to their work as
        a learner who gave it up, which is the whole class of error this measure guards
        against and the easiest one to commit by accident.
        """
        reading = measured(
            run(
                [
                    _opened("content-0001", s.MID_BEFORE_TS),
                    *s.busy_after(3),
                    _opened("content-0001", s.EARLY_AFTER_TS),
                ]
            ),
            OutcomeMeasure.TASK_PERSISTENCE,
        )

        assert reading.status.value == "insufficient_data"
        assert "neither closed nor replaced" in reason_of(reading)

    def test_an_unresolved_task_is_not_an_abandoned_one(self) -> None:
        """Nothing happening to the item means nothing, not failure.

        The learner may simply still be working on it, and the window closing is not evidence
        that they stopped. This is the specific version of "missing events prove failure" that
        the measure is most tempted by, because a window boundary always looks like an ending.
        """
        reading = measured(
            run([_opened("content-0001", s.MID_BEFORE_TS), *s.busy_after(3)]),
            OutcomeMeasure.TASK_PERSISTENCE,
        )

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert "may still be working on it" in reason_of(reading)

    def test_finishing_and_moving_on_is_reported_as_undecidable(self) -> None:
        """A closed item plus a new one is genuinely ambiguous, and says so.

        Both readings are available from the record and they have opposite signs, so choosing
        between them would be a guess dressed as a measurement. The reason names both
        possibilities rather than picking the flattering one, which is what a reader would
        otherwise have done unaided.
        """
        reading = measured(
            run(
                [
                    _opened("content-0001", s.MID_BEFORE_TS),
                    *s.busy_after(3),
                    _closed("content-0001", s.EARLY_AFTER_TS),
                    _opened("content-9999", s.MID_AFTER_TS),
                ]
            ),
            OutcomeMeasure.TASK_PERSISTENCE,
        )

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert reading.after_value is None
        assert "cannot distinguish" in reason_of(reading)

    def test_a_closed_item_read_as_completed_would_be_the_flattering_misreading(self) -> None:
        """The reason says why the ambiguous case is not resolved favourably.

        Worth pinning as wording because the temptation is specifically to treat a closed
        item as a finished one. The measure's own note on the ambiguous case says the same
        thing; the test is here so that shortening the reason cannot quietly lose it.
        """
        reading = measured(
            run(
                [
                    _opened("content-0001", s.MID_BEFORE_TS),
                    *s.busy_after(3),
                    _closed("content-0001", s.EARLY_AFTER_TS),
                    _opened("content-9999", s.MID_AFTER_TS),
                ]
            ),
            OutcomeMeasure.TASK_PERSISTENCE,
        )

        assert "most flattering available misreading" in reason_of(reading)

    def test_no_open_item_is_not_applicable_rather_than_unmeasured(self) -> None:
        """A prompt that interrupted nothing has no task to have persisted on.

        This is a different finding from missing evidence: the question does not arise,
        rather than arising and going unanswered. A reader who saw "insufficient data" here
        would go looking for the log that was too thin, and there is nothing wrong with it.
        """
        reading = measured(run(s.busy_after(3)), OutcomeMeasure.TASK_PERSISTENCE)

        assert reading.status.value == "not_applicable"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert "interrupted nothing" in reason_of(reading)

    def test_the_tracked_item_is_the_last_one_open_at_delivery(self) -> None:
        """The most recent item is the one the measure follows.

        A measure that tracked the first item opened in a thirty-minute window would be
        reporting on something the learner finished long before the prompt, and would then
        read their leaving it as abandoning the task they were actually on.
        """
        reading = measured(
            run(
                [
                    _opened("content-0001", s.DEEP_BEFORE_TS),
                    _opened("content-0002", s.EARLY_BEFORE_TS),
                    *s.busy_after(3),
                    _closed("content-0001", s.EARLY_AFTER_TS),
                ]
            ),
            OutcomeMeasure.TASK_PERSISTENCE,
        )

        assert reading.status.value == "measured"
        assert reading.direction is OutcomeDirection.DETERIORATED

    def test_an_item_opened_after_the_prompt_is_not_the_tracked_one(self) -> None:
        """An item opened inside the after window cannot be what was interrupted.

        The tracked item is defined at or before the delivery, so an item opened afterwards is
        a consequence of the window rather than a precondition of it. Including it would let
        the measure call its own output a persistence.
        """
        reading = measured(
            run([_opened("content-0002", s.EARLY_AFTER_TS), *s.busy_after(3)]),
            OutcomeMeasure.TASK_PERSISTENCE,
        )

        assert reading.status.value == "not_applicable"

    def test_a_closure_outside_the_window_does_not_count(self) -> None:
        """Closing the item after the window is not a finding about the window.

        The after window is the evidence. A closure after it would let the measure claim
        persistence from something that happened after the question was asked.
        """
        reading = measured(
            run(
                [
                    _opened("content-0001", s.MID_BEFORE_TS),
                    *s.busy_after(3),
                    _closed("content-0001", s.LATE_AFTER_TS + 600.0),
                ]
            ),
            OutcomeMeasure.TASK_PERSISTENCE,
        )

        assert reading.status.value == "insufficient_data"
        assert "neither closed nor replaced" in reason_of(reading)


class TestSessionContinuation:
    """Did the session outlive the prompt?"""

    def test_a_session_ending_in_the_window_is_measured_as_deterioration(self) -> None:
        """An observed session end is recorded, and only as the session not continuing."""
        reading = measured(
            run(
                [
                    *s.busy_before(6),
                    *s.busy_after(3),
                    s.event(EventType.SESSION_ENDED, s.EARLY_AFTER_TS),
                ]
            ),
            OutcomeMeasure.SESSION_CONTINUATION,
        )

        assert reading.status.value == "measured"
        assert reading.direction is OutcomeDirection.DETERIORATED
        assert reading.after_value == pytest.approx(1.0)

    def test_a_session_end_does_not_claim_the_prompt_caused_it(self) -> None:
        """The reason declines the causal reading the reader would otherwise supply.

        A fifteen-minute window catches a learner who finished their work as readily as one
        who walked away, and an events log cannot tell the two apart. The direction is
        therefore the literal one, with the reasoning left open.
        """
        reading = measured(
            run(
                [
                    *s.busy_before(6),
                    *s.busy_after(3),
                    s.event(EventType.SESSION_ENDED, s.EARLY_AFTER_TS),
                ]
            ),
            OutcomeMeasure.SESSION_CONTINUATION,
        )

        assert "not something the events can answer" in reason_of(reading)

    def test_activity_past_the_window_measures_as_the_session_continuing(self) -> None:
        """Activity beyond the window is positive evidence the session was still open.

        This is the one case where the measure can affirm continuity rather than merely decline
        to deny it, and it is worth having: without it, every session would read as unknown
        and the measure would never contribute a finding.
        """
        reading = measured(
            run(
                [
                    *s.busy_before(6),
                    *s.busy_after(3),
                    s.event(EventType.INTERACTION, s.LONG_BEYOND_TS),
                ]
            ),
            OutcomeMeasure.SESSION_CONTINUATION,
        )

        assert reading.status.value == "measured"
        assert reading.direction is OutcomeDirection.NO_CHANGE
        assert reading.after_value == pytest.approx(0.0)

    def test_a_silent_session_is_not_reported_as_an_ended_one(self) -> None:
        """No end event and no later activity yields no reading at all.

        This is the case the measure's existence is really about. A log that simply stopped
        collecting looks exactly like a session that was still open, and rendering the second
        as the first would report a session ending that never happened.
        """
        reading = measured(
            run([*s.busy_before(6), *s.busy_after(3)]), OutcomeMeasure.SESSION_CONTINUATION
        )

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert "cannot tell the two apart" in reason_of(reading)

    def test_a_session_that_had_already_ended_is_not_applicable(self) -> None:
        """A prompt delivered after the session ended has nothing to survive."""
        reading = measured(
            run(
                [
                    *s.busy_before(6),
                    s.event(EventType.SESSION_ENDED, s.EARLY_BEFORE_TS),
                    *s.busy_after(3),
                ]
            ),
            OutcomeMeasure.SESSION_CONTINUATION,
        )

        assert reading.status.value == "not_applicable"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED

    def test_another_sessions_end_does_not_end_this_sessions_continuation(self) -> None:
        """A session end is only about the session it happened in.

        A learner with two courses open in two tabs would otherwise have their active session
        marked ended by a neighbouring tab closing, which would then be read as a prompt that
        ended their work.
        """
        reading = measured(
            run(
                [
                    *s.busy_before(6),
                    *s.busy_after(3),
                    s.event(EventType.SESSION_ENDED, s.EARLY_AFTER_TS, session_id="session-9999"),
                ]
            ),
            OutcomeMeasure.SESSION_CONTINUATION,
        )

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED


class TestSubsequentTrajectory:
    """Where was the learner heading, according to the temporal layer?"""

    def test_no_state_yields_no_trajectory(self) -> None:
        """The measure reports nothing rather than inventing a direction."""
        reading = measured(run([*s.busy_before(6)]), OutcomeMeasure.SUBSEQUENT_TRAJECTORY)

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert "will not manufacture" in reason_of(reading)

    @pytest.mark.parametrize(
        ("verdict", "expected", "code"),
        [
            (TrendDirection.WORSENING, OutcomeDirection.DETERIORATED, 1.0),
            (TrendDirection.STABLE, OutcomeDirection.NO_CHANGE, 0.0),
            (TrendDirection.IMPROVING, OutcomeDirection.IMPROVED, -1.0),
        ],
    )
    def test_the_temporal_verdict_is_mapped_across_unchanged(
        self,
        verdict: TrendDirection,
        expected: OutcomeDirection,
        code: float,
    ) -> None:
        """A rising deviation is unfavourable and a shrinking one is favourable.

        Pinned as a table because the mapping inverts the sign convention of the temporal
        layer. The temporal layer codes a rising standardisation as more deviation, which for
        some dimensions is worse and for others is better; the outcome layer must not inherit
        that ambiguity, and it must not guess it either.
        """
        record = s.engine().measure(
            s.history(),
            s.INTERVENTION,
            s.sorted_events([*s.busy_before(6), s.delivery_event()]),
            state=s.temporal_state(verdict),
        )
        reading = measured(record, OutcomeMeasure.SUBSEQUENT_TRAJECTORY)

        assert reading.status.value == "measured"
        assert reading.direction is expected
        assert reading.after_value == pytest.approx(code)

    def test_the_temporal_layers_own_absence_carries_over_unchanged(self) -> None:
        """Its insufficient-data verdict is not reinterpreted as stability.

        This is the specific translation error the measure could make, and it would report calm
        where there is only silence: an unestablished direction read as a stable one is a
        finding invented from the absence of one.
        """
        record = s.engine().measure(
            s.history(),
            s.INTERVENTION,
            s.sorted_events([*s.busy_before(6), s.delivery_event()]),
            state=s.temporal_state(TrendDirection.INSUFFICIENT_DATA, observations=1),
        )
        reading = measured(record, OutcomeMeasure.SUBSEQUENT_TRAJECTORY)

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED
        assert "not a stable one" in reason_of(reading)

    def test_a_state_with_no_observations_yields_no_reading(self) -> None:
        """A verdict with nothing behind it is not a verdict."""
        record = s.engine().measure(
            s.history(),
            s.INTERVENTION,
            s.sorted_events([*s.busy_before(6), s.delivery_event()]),
            state=s.temporal_state(TrendDirection.STABLE, observations=0),
        )
        reading = measured(record, OutcomeMeasure.SUBSEQUENT_TRAJECTORY)

        assert reading.status.value == "insufficient_data"
        assert reading.direction is OutcomeDirection.NOT_ASSESSED

    def test_a_state_predating_the_window_is_refused(self) -> None:
        """A state describing the learner before the prompt cannot describe the trajectory."""
        record = s.engine().measure(
            s.history(),
            s.INTERVENTION,
            s.sorted_events([*s.busy_before(6), s.delivery_event()]),
            state=s.temporal_state(
                TrendDirection.WORSENING,
                reference_time=s.DELIVERED_AT - timedelta(minutes=1),
            ),
        )
        reading = measured(record, OutcomeMeasure.SUBSEQUENT_TRAJECTORY)

        assert reading.status.value == "insufficient_data"
        assert "before the prompt" in reason_of(reading)

    def test_the_measure_carries_the_observation_count(self) -> None:
        """The temporal layer's evidence count travels with the verdict.

        A trajectory established from three observations and one from fifty are different
        strengths of claim, and the record should let a reader see which it is holding.
        """
        record = s.engine().measure(
            s.history(),
            s.INTERVENTION,
            s.sorted_events([*s.busy_before(6), s.delivery_event()]),
            state=s.temporal_state(TrendDirection.WORSENING, observations=7),
        )
        reading = measured(record, OutcomeMeasure.SUBSEQUENT_TRAJECTORY)

        assert reading.after_samples == 7


class TestRecordLevelReporting:
    """What the record as a whole says about the strength of its own report."""

    def test_a_fully_measured_record_is_complete(self) -> None:
        """Seven measured readings make a complete record.

        The scenario is built to supply everything all seven measures need, which is worth
        doing once: it confirms the seven measures are jointly satisfiable and that the model
        does not refuse a record for a reason no test had anticipated.
        """
        events = [
            *s.busy_before(6),
            _opened("content-0001", s.MID_BEFORE_TS),
            *s.busy_after(3),
            _closed("content-0001", s.EARLY_AFTER_TS),
            s.event(EventType.INTERACTION, s.LONG_BEYOND_TS),
            s.completion_event(response_seconds=8.0),
        ]
        record: OutcomeRecord = s.engine().measure(
            s.history(),
            s.INTERVENTION,
            s.sorted_events([*events, s.delivery_event()]),
            state=s.temporal_state(TrendDirection.STABLE),
        )

        assert [m for m in record.measurements if m.status.value == "measured"]
        assert record.is_complete is True

    def test_one_withheld_reading_makes_the_whole_record_incomplete(self) -> None:
        """Completeness is all-or-nothing across the seven.

        A record marked complete while one measure reports nothing would let a consumer read
        it as a full report. The record has to declare the gap so the reader can decide
        whether the six measured readings are worth having.
        """
        events = [
            *s.busy_before(6),
            _opened("content-0001", s.MID_BEFORE_TS),
            *s.busy_after(3),
            _closed("content-0001", s.EARLY_AFTER_TS),
            s.event(EventType.INTERACTION, s.LONG_BEYOND_TS),
            s.completion_event(response_seconds=8.0),
        ]
        record = s.engine().measure(
            s.history(),
            s.INTERVENTION,
            s.sorted_events([*events, s.delivery_event()]),
        )

        assert record.is_complete is False

    def test_every_measure_carries_a_reason_when_it_reports_nothing(self) -> None:
        """A withheld reading always explains itself.

        The record is the only channel a consumer has, so a non-measured reading with a null
        reason would be a silent gap in a document whose entire purpose is to say what it does
        and does not know.
        """
        record = run(s.busy_before(6))

        for reading in record.measurements:
            if not reading.is_measured:
                assert reading.reason, (
                    f"{reading.measure.value} reports {reading.status.value} with no reason"
                )

    def test_every_measured_reading_states_no_reason(self) -> None:
        """A measurement does not apologise for having been made."""
        record = run([*s.busy_before(6), *s.busy_after(3)])

        for reading in record.measurements:
            if reading.is_measured:
                assert reading.reason is None, (
                    f"{reading.measure.value} is measured but states a reason"
                )
