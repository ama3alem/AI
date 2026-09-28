"""The model's own refusals, tested at the model rather than through the measures.

The layer's central claim is that it is *structurally* unable to say more than the evidence
supports: an unread measure cannot be constructed as a deterioration, a record cannot claim
a provenance other than observed, and a value cannot be reported without an observation
behind it. Those are validator rules, and a validator rule that is only ever reached by
accident - because some measure happened to violate it - is a rule whose removal would go
unnoticed until the day a measure did violate it in production.

So each refusal here is provoked directly, with a hand-built :class:`Measurement` or
:class:`OutcomeRecord`, and the cases chosen are the ones no measure in the suite is likely
to produce on its own: a value with no sample count, a sample count with no value, an empty
reason, a measured reading that carries a caveat. That last one is a permission rather than
a refusal, and it is pinned here because it is the one that was missing: a session that ended
inside the after window is measured, and the reading needs to say the events do not show the
prompt is why.
"""

from __future__ import annotations

import pytest

import tests.unit.outcomes.scenarios as s
from focus_engine.outcomes import (
    OUTCOME_MEASURES,
    OUTCOME_V1,
    Measurement,
    MeasurementStatus,
    OutcomeDirection,
    OutcomeMeasure,
    OutcomeRecord,
)
from focus_engine.outcomes.models import WindowKind
from focus_engine.policy.history import OutcomeClass
from focus_engine.schemas.primitives import DataOrigin, Provenance

pytestmark = pytest.mark.unit


def _measured(**overrides: object) -> Measurement:
    """Build a valid measured reading.

    Args:
        overrides: Fields to replace.

    Returns:
        A measurement that passes every validator.
    """
    base: dict[str, object] = {
        "measure": OutcomeMeasure.CONTINUED_ACTIVITY,
        "status": MeasurementStatus.MEASURED,
        "direction": OutcomeDirection.NO_CHANGE,
        "unit": "events_per_minute",
        "before_value": 0.5,
        "after_value": 0.5,
        "before_samples": 15,
        "after_samples": 8,
        "data_origin": DataOrigin.SYNTHETIC,
    }
    base.update(overrides)
    return Measurement(**base)


def _unmeasured(**overrides: object) -> Measurement:
    """Build a valid unmeasured reading.

    Args:
        overrides: Fields to replace.

    Returns:
        A measurement that passes every validator.
    """
    base: dict[str, object] = {
        "measure": OutcomeMeasure.CONTINUED_ACTIVITY,
        "status": MeasurementStatus.INSUFFICIENT_DATA,
        "direction": OutcomeDirection.NOT_ASSESSED,
        "unit": "events_per_minute",
        "reason": "constructed for a model test",
        "data_origin": DataOrigin.SYNTHETIC,
    }
    base.update(overrides)
    return Measurement(**base)


class TestMeasurementRefusals:
    """A reading cannot misdescribe itself."""

    def test_an_absent_reading_cannot_carry_a_direction(self) -> None:
        """Insufficient data with a direction is the forbidden construction.

        This is the single most consequential refusal in the layer: it is what makes "a log
        that stops" structurally unable to become "a learner who disengaged".
        """
        with pytest.raises(ValueError, match="Only a measured comparison can have a direction"):
            _unmeasured(direction=OutcomeDirection.DETERIORATED, reason="nothing observed")

    def test_a_measured_reading_with_no_direction_is_refused(self) -> None:
        """A measurement that read a value but reports no direction is refused.

        ``NOT_ASSESSED`` on a measured reading would assert that nothing changed, which is a
        different claim from having read a value.
        """
        with pytest.raises(ValueError, match="is measured but its direction is"):
            _measured(direction=OutcomeDirection.NOT_ASSESSED)

    def test_an_absent_reading_without_a_reason_is_refused(self) -> None:
        """Silence about why a number is missing is refused.

        The reason is the only channel through which a reader learns that a number is
        absent, so an unexplained absence is a silent gap.
        """
        with pytest.raises(ValueError, match="states no reason"):
            _unmeasured(reason=None)

    def test_an_empty_reason_is_refused(self) -> None:
        """Whitespace is not a reason.

        An empty string would satisfy ``reason is not None`` while telling a reader nothing,
        which is the same silent gap wearing a disguise.
        """
        with pytest.raises(ValueError, match="empty reason"):
            _unmeasured(reason="   ")

    def test_a_measured_reading_with_no_value_is_refused(self) -> None:
        """``MEASURED`` with no after value is a contradiction."""
        with pytest.raises(ValueError, match="is measured but reports no after value"):
            _measured(after_value=None, after_samples=8)

    def test_a_value_with_no_sample_count_is_refused(self) -> None:
        """A number with no observation behind it is arithmetic, not a measurement."""
        with pytest.raises(ValueError, match="arithmetic, not a measurement"):
            _measured(after_samples=0)

    def test_a_measured_sample_count_with_no_value_is_refused(self) -> None:
        """A count standing alone is refused on a measured reading.

        The count is what distinguishes a measured zero from an absence, so it cannot stand
        without the value it counts.
        """
        with pytest.raises(ValueError, match="but no before value"):
            _measured(before_value=None, before_samples=15)

    def test_an_unmeasured_count_without_a_value_is_permitted(self) -> None:
        """The one latitude the model allows, and it is deliberate.

        "The before window holds two activity events and the measure requires five" is the
        useful fact, and refusing the count would push it out of the record and leave a reader
        with a reason string and no numbers.
        """
        reading = _unmeasured(before_samples=2)

        assert reading.before_samples == 2
        assert reading.before_value is None


class TestMeasurementPermissions:
    """A measured reading may say what it does not establish."""

    def test_a_measured_reading_may_carry_a_caveat(self) -> None:
        """A reason on a measured reading is permitted, and is the rule in force.

        Some measured findings need to refuse a conclusion a reader would otherwise supply
        themselves. A session that ended inside the after window is measured as a session
        that did not continue; the events cannot say whether the prompt is why. A model that
        forbade the caveat would force that distinction to be dropped.
        """
        reading = _measured(
            status=MeasurementStatus.MEASURED,
            direction=OutcomeDirection.DETERIORATED,
            after_value=0.0,
            reason="a session end event was recorded; the events do not show the prompt caused it",
        )

        assert reading.is_measured
        assert reading.reason is not None
        assert "do not show" in reading.reason

    def test_a_measured_reading_need_not_carry_a_reason(self) -> None:
        """The caveat is permitted, not required.

        Requiring it would put a sentence on every reading and train a reader to skip it.
        """
        assert _measured().reason is None


class TestMeasurementAccessors:
    """The convenience surface a consumer reads instead of the fields."""

    def test_change_is_the_signed_difference(self) -> None:
        """The sign carries no judgement; ``lower_is_better`` carries the judgement."""
        assert _measured(before_value=0.4, after_value=0.6).change == pytest.approx(0.2)

    def test_change_is_absent_when_one_side_is_absent(self) -> None:
        """A one-sided measure cannot report a change against nothing.

        Task persistence, session continuation, and the trajectory reading are all
        one-sided by nature, and a change computed against a missing side would be a number
        with no meaning.
        """
        assert (
            _unmeasured(
                before_value=None, before_samples=0, after_value=1.0, after_samples=2
            ).change
            is None
        )

    def test_describe_names_the_measure_and_both_values(self) -> None:
        """The rendering is what a log line will carry, so it names the unit and the counts."""
        text = _measured().describe()

        assert "continued_activity" in text
        assert "events_per_minute" in text
        assert "no_change" in text

    def test_describe_of_an_absent_reading_carries_the_reason(self) -> None:
        """An unmeasured line must say why, because the line is often all a reader gets."""
        assert "constructed for a model test" in _unmeasured().describe()

    def test_describe_stays_ascii(self) -> None:
        """A rendering that raises on a log stream fails when it is most needed."""
        assert _measured().describe().isascii()


class TestRecordRefusals:
    """A record cannot claim more than its own contents support."""

    def _kwargs(self, **overrides: object) -> dict[str, object]:
        """Build the minimal constructor arguments for a valid record.

        Args:
            overrides: Fields to replace.

        Returns:
            Keyword arguments for :class:`~focus_engine.outcomes.models.OutcomeRecord`.
        """
        engine = s.engine()
        measurements = tuple(
            _unmeasured(measure=measure, reason="constructed for a record test")
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
            "response_class": OutcomeClass.RESPONSE,
            "policy_version": "POLICY_V1",
            "data_origin": DataOrigin.SYNTHETIC,
            "provenance": Provenance.OBSERVED,
            "settings_fingerprint": engine.settings_fingerprint(),
            "computed_at": s.DELIVERED_AT,
        }
        base.update(overrides)
        return base

    def test_a_record_is_constructible_when_everything_agrees(self) -> None:
        """The baseline case, so each refusal below is about one disagreement only."""
        record = OutcomeRecord(**self._kwargs())

        assert record.is_complete is False
        assert len(record.unmeasured) == len(OUTCOME_MEASURES)

    def test_a_record_claiming_a_provenance_other_than_observed_is_refused(self) -> None:
        """The record is a measurement, and the type will not let it be a causal claim.

        Stating the vocabulary and then refusing the other members is the only form of this
        constraint that cannot be bypassed by setting a field.
        """
        with pytest.raises(ValueError, match="a causal claim wearing"):
            OutcomeRecord(**self._kwargs(provenance=Provenance.PROXY_LABEL))

    def test_a_record_hiding_synthetic_measurements_behind_a_real_origin_is_refused(self) -> None:
        """Provenance is not a label a caller may set; it is derived from the evidence.

        This is the one error the project's provenance rules exist to prevent, so it is
        refused at the type rather than left to review.
        """
        with pytest.raises(ValueError, match="provenance rules exist to prevent"):
            OutcomeRecord(**self._kwargs(data_origin=DataOrigin.REAL))

    def test_a_record_with_a_window_of_the_wrong_kind_is_refused(self) -> None:
        """Two before windows, or two after, would misdescribe the measurement."""
        engine = s.engine()
        with pytest.raises(ValueError, match="one before and one after window"):
            OutcomeRecord(**self._kwargs(before_window=engine.after_window(s.DELIVERED_AT)))

    def test_a_record_lookup_of_an_absent_measure_raises(self) -> None:
        """The lookup is total for a constructed record, and says so if it is ever not.

        A ``KeyError`` rather than a silent ``None``, so a consumer cannot mistake a missing
        measure for an absent measurement.
        """
        record = OutcomeRecord(**self._kwargs())

        assert record.measurement(OutcomeMeasure.ACCURACY) is record.measurements[2]
        with pytest.raises(KeyError):
            record.measurement("not_a_measure")  # type: ignore[arg-type]

    def test_the_summary_counts_the_unmeasured(self) -> None:
        """A summary that omitted the unmeasured count would let absent readings read as stable.

        The summary is what reaches a metrics frame, so the count of measurements that were
        never read has to travel with it.
        """
        summary = OutcomeRecord(**self._kwargs()).to_summary_dict()

        assert summary["measures_total"] == len(OUTCOME_MEASURES)
        assert summary["measures_measured"] == 0
        assert summary["measures_deteriorated"] == 0


class TestMeasureSetIsFrozen:
    """The measure set is versioned, so it is checked as a contract."""

    def test_the_canonical_order_puts_the_narrowest_measure_first(self) -> None:
        """The order is part of the frozen definition set, not an implementation detail.

        A consumer's switch over measures is written against this order, so reordering it
        would silently change what a record reads as first.
        """
        assert OUTCOME_MEASURES[0] is OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE
        assert OUTCOME_MEASURES[-1] is OutcomeMeasure.SUBSEQUENT_TRAJECTORY
        assert len(set(OUTCOME_MEASURES)) == len(OUTCOME_MEASURES)

    def test_the_window_kinds_are_the_two_sides_of_a_delivery(self) -> None:
        """A third window kind would need a version, so the vocabulary is checked here."""
        assert {kind.value for kind in WindowKind} == {"before", "after"}
