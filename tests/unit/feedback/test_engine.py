"""Unit tests for FeedbackEngine accumulation and candidate adjustment."""

from __future__ import annotations

from focus_engine.feedback.engine import FeedbackEngine
from focus_engine.feedback.models import (
    InterventionResponseObservation,
    ProfileMaturity,
    ResponseOutcomeDirection,
)
from focus_engine.policy.models import InterventionCandidate
from focus_engine.schemas.primitives import ConfidenceLevel


def test_engine_profile_accumulation_under_threshold() -> None:
    """Engine accumulates observations in collecting status until threshold."""
    engine = FeedbackEngine(min_observations=3)
    learner_id = "test-learner-1001"
    candidate_id = "take_break"

    obs1 = InterventionResponseObservation(
        learner_id=learner_id,
        candidate_id=candidate_id,
        session_id="session-1",
        direction=ResponseOutcomeDirection.POSITIVE,
    )
    p1 = engine.record_observation(obs1)
    assert p1.observation_count == 1
    assert p1.positive_count == 1
    assert p1.maturity == ProfileMaturity.COLLECTING
    assert p1.effective_success_rate is None

    obs2 = InterventionResponseObservation(
        learner_id=learner_id,
        candidate_id=candidate_id,
        session_id="session-2",
        direction=ResponseOutcomeDirection.NEGATIVE,
    )
    p2 = engine.record_observation(obs2)
    assert p2.observation_count == 2
    assert p2.positive_count == 1
    assert p2.negative_count == 1
    assert p2.maturity == ProfileMaturity.COLLECTING
    assert p2.effective_success_rate is None


def test_engine_profile_maturation_at_threshold() -> None:
    """Engine transitions profile to ACTIVE when reaching min_observations."""
    engine = FeedbackEngine(min_observations=3)
    learner_id = "test-learner-1001"
    candidate_id = "take_break"

    for i in range(2):
        engine.record_observation(
            InterventionResponseObservation(
                learner_id=learner_id,
                candidate_id=candidate_id,
                session_id=f"session-{i}",
                direction=ResponseOutcomeDirection.POSITIVE,
            )
        )

    # 3rd observation hits the threshold
    p3 = engine.record_observation(
        InterventionResponseObservation(
            learner_id=learner_id,
            candidate_id=candidate_id,
            session_id="session-3",
            direction=ResponseOutcomeDirection.NEUTRAL,
        )
    )
    assert p3.observation_count == 3
    assert p3.positive_count == 2
    assert p3.neutral_count == 1
    assert p3.maturity == ProfileMaturity.ACTIVE
    assert p3.effective_success_rate == round(2 / 3, 4)


def test_candidate_reordering_only_by_active_profiles() -> None:
    """Collecting profiles do not perturb candidate order; active high-success candidates rise."""
    engine = FeedbackEngine(min_observations=3)
    learner_id = "test-learner-1001"

    cand_a = InterventionCandidate(
        intervention_type="cand_a",
        priority=0,
        min_confidence=ConfidenceLevel.LOW,
    )
    cand_b = InterventionCandidate(
        intervention_type="cand_b",
        priority=1,
        min_confidence=ConfidenceLevel.LOW,
    )

    # Initially, order is preserved (catalogue order)
    adjusted_0 = engine.select_adjusted_candidates(learner_id, [cand_a, cand_b])
    assert [c.intervention_type for c in adjusted_0] == ["cand_a", "cand_b"]

    # Record 2 observations for cand_b (still COLLECTING) -> order still preserved
    for i in range(2):
        engine.record_observation(
            InterventionResponseObservation(
                learner_id=learner_id,
                candidate_id="cand_b",
                session_id=f"session-{i}",
                direction=ResponseOutcomeDirection.POSITIVE,
            )
        )
    adjusted_1 = engine.select_adjusted_candidates(learner_id, [cand_a, cand_b])
    assert [c.intervention_type for c in adjusted_1] == ["cand_a", "cand_b"]

    # 3rd positive observation matures cand_b to active rate 1.0, moving it forward.
    engine.record_observation(
        InterventionResponseObservation(
            learner_id=learner_id,
            candidate_id="cand_b",
            session_id="session-3",
            direction=ResponseOutcomeDirection.POSITIVE,
        )
    )
    adjusted_2 = engine.select_adjusted_candidates(learner_id, [cand_a, cand_b])
    assert [c.intervention_type for c in adjusted_2] == ["cand_b", "cand_a"]
