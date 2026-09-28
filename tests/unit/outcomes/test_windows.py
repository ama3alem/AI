"""Window geometry for the outcome layer (Phase 11).

The windows are where a before/after measurement is decided to be honest or not, and the
whole claim rests on two properties that are cheap to state and easy to lose:

* **The two windows share exactly one boundary, and it is the delivery instant.** Nothing
  can be evidence both before and after the intervention that produced it.
* **Each window is half-open.** The delivery instant itself belongs to the after window,
  because the before window is a description of the learner *as they were* and the delivery
  is not part of it. An inclusive end on the before window would put the prompt inside its
  own baseline, and the baseline would then contain the thing it is a comparison for.

The tests here are about the geometry, not about any measure. A rate or an accuracy can be
correctly computed over a window that is itself the wrong shape, which is why these are
tested separately from the measures that read them.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from focus_engine.events.types import EventType
from focus_engine.outcomes.engine import OutcomeEngine, OutcomeError
from focus_engine.outcomes.models import (
    OUTCOME_MEASURES,
    OUTCOME_V1,
    Measurement,
    MeasurementStatus,
    OutcomeDirection,
    OutcomeRecord,
    OutcomeWindow,
    WindowKind,
)
from focus_engine.schemas.primitives import DataOrigin, Provenance
from focus_engine.temporal.engine import TemporalEngine
from tests.unit.outcomes import scenarios as s

pytestmark = pytest.mark.unit


class TestWindowGeometry:
    """The shape of the two windows and the instant between them."""

    def test_the_before_window_ends_at_the_delivery_instant(self) -> None:
        """The before window's exclusive end is the delivery instant."""
        window = s.engine().before_window(s.DELIVERED_AT)

        assert window.end == s.DELIVERED_AT

    def test_the_after_window_begins_at_the_delivery_instant(self) -> None:
        """The after window's inclusive start is the delivery instant."""
        window = s.engine().after_window(s.DELIVERED_AT)

        assert window.start == s.DELIVERED_AT

    def test_the_windows_meet_at_exactly_one_instant(self) -> None:
        """The before end and the after start are the same instant."""
        engine = s.engine()
        before = engine.before_window(s.DELIVERED_AT)
        after = engine.after_window(s.DELIVERED_AT)

        assert before.end == after.start

    def test_the_windows_do_not_overlap(self) -> None:
        """No instant is inside both windows."""
        engine = s.engine()
        before = engine.before_window(s.DELIVERED_AT)
        after = engine.after_window(s.DELIVERED_AT)
        probes = [
            s.DELIVERED_AT - timedelta(minutes=45),
            s.DELIVERED_AT - timedelta(minutes=1),
            s.DELIVERED_AT,
            s.DELIVERED_AT + timedelta(minutes=1),
            s.DELIVERED_AT + timedelta(minutes=45),
        ]

        for instant in probes:
            assert not (before.contains(instant) and after.contains(instant)), (
                f"{instant.isoformat()} is inside both windows"
            )

    def test_the_delivery_instant_belongs_to_the_after_window_only(self) -> None:
        """The prompt is in the measurement window and not in the baseline."""
        engine = s.engine()
        before = engine.before_window(s.DELIVERED_AT)
        after = engine.after_window(s.DELIVERED_AT)

        assert after.contains(s.DELIVERED_AT)
        assert not before.contains(s.DELIVERED_AT)

    def test_each_window_is_half_open_at_its_own_start(self) -> None:
        """A window includes its own start and excludes its own end."""
        window = s.engine().after_window(s.DELIVERED_AT)

        assert window.contains(window.start)
        assert not window.contains(window.end)

    def test_the_window_widths_come_from_the_settings(self) -> None:
        """Both widths follow the configuration rather than a constant."""
        engine = s.engine(before_window_minutes=10.0, after_window_minutes=4.0)
        before = engine.before_window(s.DELIVERED_AT)
        after = engine.after_window(s.DELIVERED_AT)

        assert before.duration_seconds == 600.0
        assert after.duration_seconds == 240.0

    def test_the_immediate_window_is_a_prefix_of_the_after_window(self) -> None:
        """The immediate measure reads a strict prefix of what the after measure reads."""
        engine = s.engine(immediate_window_minutes=2.0, after_window_minutes=15.0)
        immediate = engine.immediate_window(s.DELIVERED_AT)
        after = engine.after_window(s.DELIVERED_AT)

        assert immediate.start == after.start
        assert immediate.end < after.end

    def test_an_immediate_window_equal_to_the_after_window_is_allowed(self) -> None:
        """A configuration where the two windows coincide is legal, if degenerate.

        Worth pinning because the settings validator permits equality. The immediate and
        continued measures then read the same evidence and must agree, which is a fact a
        future change to the validator should have to think about rather than discover.
        """
        engine = s.engine(immediate_window_minutes=15.0, after_window_minutes=15.0)

        assert (
            engine.immediate_window(s.DELIVERED_AT).end == engine.after_window(s.DELIVERED_AT).end
        )


class TestWindowValidation:
    """A window refuses to misdescribe itself."""

    def test_a_window_ending_before_it_starts_is_refused(self) -> None:
        """An inverted interval is not a window."""
        with pytest.raises(ValueError, match="before it starts"):
            OutcomeWindow(
                kind=WindowKind.BEFORE,
                start=s.DELIVERED_AT,
                end=s.DELIVERED_AT - timedelta(minutes=1),
                duration_seconds=0.0,
            )

    def test_a_stale_recorded_duration_is_refused(self) -> None:
        """A duration that disagrees with the interval is a misdescription.

        The duration is what a reader uses to judge how wide the measurement was, so a
        stale copy would misinform without being obviously wrong to the process that wrote
        it.
        """
        with pytest.raises(ValueError, match="the recorded width must match"):
            OutcomeWindow(
                kind=WindowKind.AFTER,
                start=s.DELIVERED_AT,
                end=s.DELIVERED_AT + timedelta(minutes=15),
                duration_seconds=30.0,
            )

    def test_a_zero_width_window_is_permitted(self) -> None:
        """A degenerate window is representable; an inverted one is not.

        Nothing in the model forbids a zero-width window, and forbidding it would be a
        restriction on geometry that belongs to the settings validator instead.
        """
        window = OutcomeWindow(
            kind=WindowKind.AFTER,
            start=s.DELIVERED_AT,
            end=s.DELIVERED_AT,
            duration_seconds=0.0,
        )

        assert window.duration_seconds == 0.0
        assert not window.contains(s.DELIVERED_AT)

    def test_a_window_is_immutable(self) -> None:
        """A window cannot be edited after the fact."""
        window = s.engine().after_window(s.DELIVERED_AT)

        with pytest.raises(ValueError, match="frozen"):
            window.end = s.DELIVERED_AT + timedelta(days=1)  # type: ignore[misc]


class TestOutcomeRecordWindowAgreement:
    """A record cannot claim windows that do not line up with its delivery."""

    def _kwargs(self, **overrides: object) -> dict[str, object]:
        """Build the minimal constructor arguments for a record.

        Every measure is set to ``NOT_APPLICABLE`` so that the record is valid on every axis
        except the one a given test is perturbing, which keeps each test about exactly one
        disagreement.

        Args:
            overrides: Fields to replace.

        Returns:
            Keyword arguments for :class:`~focus_engine.outcomes.models.OutcomeRecord`.
        """
        engine = s.engine()
        measurements = tuple(
            Measurement(
                measure=measure,
                status=MeasurementStatus.NOT_APPLICABLE,
                direction=OutcomeDirection.NOT_ASSESSED,
                unit="test",
                reason="constructed for a window test",
                data_origin=DataOrigin.SYNTHETIC,
            )
            for measure in OUTCOME_MEASURES
        )
        base: dict[str, object] = {
            "outcome_version": OUTCOME_V1,
            "intervention_id": s.INTERVENTION,
            "learner_id": s.LEARNER,
            "session_id": s.SESSION,
            "intervention_type": "recap",
            "delivered_at": s.DELIVERED_AT,
            "before_window": engine.before_window(s.DELIVERED_AT),
            "after_window": engine.after_window(s.DELIVERED_AT),
            "measurements": measurements,
            "response_class": s.history().interventions[0].outcome_class,
            "policy_version": "POLICY_V1",
            "data_origin": DataOrigin.SYNTHETIC,
            "provenance": Provenance.OBSERVED,
            "settings_fingerprint": engine.settings_fingerprint(),
            "computed_at": s.DELIVERED_AT,
        }
        base.update(overrides)
        return base

    def test_a_record_with_misaligned_windows_is_refused(self) -> None:
        """A gap or an overlap between the windows is refused.

        A gap would silently drop the events in it and an overlap would let one event serve
        as evidence on both sides of the intervention. Either would make the record
        uninterpretable while still looking complete.
        """
        engine = s.engine()
        with pytest.raises(ValueError, match="they must meet exactly there"):
            OutcomeRecord(
                **self._kwargs(
                    after_window=engine.after_window(s.DELIVERED_AT + timedelta(minutes=1))
                )
            )

    def test_a_record_whose_after_window_is_not_anchored_to_delivery_is_refused(self) -> None:
        """A response-anchored after window is refused at the type.

        The engine only ever builds a delivery-anchored window, so this cannot arise from
        the engine. It is pinned at the model because the record is the thing a consumer
        reads, and an after window anchored to a response would compare two different
        intervals for a learner who engaged and one who did not.
        """
        engine = s.engine()
        with pytest.raises(ValueError, match="anchored to delivery and to nothing else"):
            OutcomeRecord(
                **self._kwargs(
                    after_window=engine.after_window(s.DELIVERED_AT + timedelta(minutes=5)),
                    before_window=engine.before_window(s.DELIVERED_AT + timedelta(minutes=5)),
                )
            )

    def test_a_record_missing_a_measure_is_refused(self) -> None:
        """Coverage of all seven measures is required."""
        measurements = self._kwargs()["measurements"]
        assert isinstance(measurements, tuple)
        with pytest.raises(ValueError, match="does not cover every measure"):
            OutcomeRecord(**self._kwargs(measurements=measurements[:-1]))
        assert len(OUTCOME_MEASURES) == 7

    def test_a_record_repeating_a_measure_is_refused(self) -> None:
        """A duplicated measure is refused, so coverage is exactly once."""
        measurements = self._kwargs()["measurements"]
        assert isinstance(measurements, tuple)
        doubled = (measurements[0], *measurements[:-1])
        assert len(doubled) == len(OUTCOME_MEASURES)

        with pytest.raises(ValueError, match="repeats measures"):
            OutcomeRecord(**self._kwargs(measurements=doubled))

    def test_a_record_claiming_a_measured_window_of_the_wrong_kind_is_refused(self) -> None:
        """Two before windows are refused.

        The window-kind check is kept separate from the alignment check because a record
        with two before windows fails for a different reason than one with a gap, and a
        single message covering both would leave a reader guessing which mistake was made.
        """
        engine = s.engine()
        with pytest.raises(ValueError, match="one before and one after"):
            OutcomeRecord(**self._kwargs(after_window=engine.before_window(s.DELIVERED_AT)))


class TestMeasureRequiresADelivery:
    """An outcome is measured against a delivery that happened."""

    def test_an_unknown_delivery_is_refused(self) -> None:
        """No record is produced for an intervention that was never delivered."""
        events = s.sorted_events([*s.busy_before(6), s.delivery_event()])

        with pytest.raises(OutcomeError, match="was not delivered|Recorded deliveries"):
            s.engine().measure(s.history(), "intervention-9999", events)

    def test_the_refusal_names_the_deliveries_that_do_exist(self) -> None:
        """The error lists the recorded deliveries.

        A caller that mistyped an identifier would otherwise have to guess which of several
        measurements it was asking about, and the guess would be silent.
        """
        events = s.sorted_events([*s.busy_before(6), s.delivery_event()])

        with pytest.raises(OutcomeError, match=s.INTERVENTION):
            s.engine().measure(s.history(), "intervention-9999", events)

    def test_events_from_another_learner_are_refused(self) -> None:
        """A foreign event is an error rather than a silently dropped event.

        Filtering instead would turn a routing bug upstream into a mysteriously thin
        measurement, and the bug would be found by whoever later wondered why the numbers
        were low.
        """
        events = s.sorted_events(
            [
                *s.busy_before(6),
                s.event(EventType.INTERACTION, s.EARLY_AFTER_TS, learner_id="learner-0002"),
                s.delivery_event(),
            ]
        )

        with pytest.raises(OutcomeError, match="belongs to learner"):
            s.engine().measure(s.history(), s.INTERVENTION, events)

    def test_a_temporal_state_for_another_learner_is_refused(self) -> None:
        """A trajectory reading about someone else is a record of the wrong person."""
        state = TemporalEngine().create_state("learner-0002", created_at=s.DELIVERED_AT)
        events = s.sorted_events([*s.busy_before(6), s.delivery_event()])

        with pytest.raises(OutcomeError, match="trajectory reading about someone else"):
            s.engine().measure(s.history(), s.INTERVENTION, events, state=state)

    def test_the_engine_constructs_with_no_arguments(self) -> None:
        """The defaults compose into a usable engine.

        Worth a test because the defaults live in three separate modules, and a change to
        any of them could leave the no-argument construction unusable in a way that only
        the serving path would discover.
        """
        engine = OutcomeEngine()

        assert engine.outcome_version == OUTCOME_V1
        assert engine.settings.min_samples_for_comparison >= 1
