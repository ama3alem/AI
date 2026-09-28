"""Integration tests for feedback accumulation from measured outcome records."""

from __future__ import annotations

from focus_engine.feedback.engine import FeedbackEngine
from focus_engine.feedback.models import ProfileMaturity
from tests.unit.outcomes import scenarios as outcomes
from tests.unit.outcomes.test_measures import run


def test_outcome_records_accumulate_into_a_gated_response_profile() -> None:
    """Measured outcomes become observations and mature only at the configured threshold."""
    engine = FeedbackEngine(min_observations=2)
    record = run([*outcomes.busy_before(6), *outcomes.busy_after(3)])

    first = engine.record_from_outcome(record, candidate_id="take_break")
    assert first.learner_id == record.learner_id
    assert first.candidate_id == "take_break"
    assert first.maturity == ProfileMaturity.COLLECTING
    assert first.effective_success_rate is None

    second = engine.record_from_outcome(record, candidate_id="take_break")
    assert second.observation_count == 2
    assert second.maturity == ProfileMaturity.ACTIVE
    assert second.effective_success_rate is None or 0.0 <= second.effective_success_rate <= 1.0
    assert second.data_origin == record.data_origin


def test_unassessable_outcome_is_recorded_without_a_fabricated_success_rate() -> None:
    """A record without a trajectory reading is retained as unassessable, not negative."""
    engine = FeedbackEngine(min_observations=1)
    record = run([])

    profile = engine.record_from_outcome(record, candidate_id="take_break")
    assert profile.maturity == ProfileMaturity.ACTIVE
    assert profile.unassessable_count == 1
    assert profile.positive_count == 0
    assert profile.negative_count == 0
    assert profile.effective_success_rate is None
