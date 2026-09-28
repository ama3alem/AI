"""Engine for accumulating intervention feedback and managing response profiles.

This engine maintains response profiles for learners across intervention candidates.
Profiles track empirical directional outcomes and remain strictly observational.
Automated direct retraining of models from feedback is structurally prohibited.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Final

from focus_engine.feedback.models import (
    DEFAULT_MIN_RESPONSE_OBSERVATIONS,
    FEEDBACK_V1,
    InterventionResponseObservation,
    InterventionResponseProfile,
    ProfileMaturity,
    ResponseOutcomeDirection,
)
from focus_engine.outcomes.models import OutcomeDirection, OutcomeMeasure, OutcomeRecord
from focus_engine.policy.models import InterventionCandidate
from focus_engine.schemas.primitives import DataOrigin, LearnerId

__all__ = ["FeedbackEngine"]


def _map_outcome_direction(direction: OutcomeDirection) -> ResponseOutcomeDirection:
    """Map an OutcomeRecord's primary direction to a ResponseOutcomeDirection."""
    if direction == OutcomeDirection.IMPROVED:
        return ResponseOutcomeDirection.POSITIVE
    if direction == OutcomeDirection.NO_CHANGE:
        return ResponseOutcomeDirection.NEUTRAL
    if direction == OutcomeDirection.DETERIORATED:
        return ResponseOutcomeDirection.NEGATIVE
    return ResponseOutcomeDirection.NOT_ASSESSABLE


class FeedbackEngine:
    """Manages intervention response accumulation and profile synthesis.

    Attributes:
        min_observations: Threshold for profile maturation.
    """

    def __init__(
        self,
        min_observations: int = DEFAULT_MIN_RESPONSE_OBSERVATIONS,
    ) -> None:
        self.min_observations: Final[int] = min_observations
        self._observations: dict[tuple[str, str, str], list[InterventionResponseObservation]] = (
            defaultdict(list)
        )

    def record_observation(
        self,
        observation: InterventionResponseObservation,
    ) -> InterventionResponseProfile:
        """Record an observed intervention response and return the updated profile.

        Args:
            observation: The validated observation record.

        Returns:
            The newly recalculated InterventionResponseProfile.
        """
        key = (
            str(observation.learner_id),
            observation.candidate_id,
            observation.context_key,
        )
        self._observations[key].append(observation)
        return self._compute_profile(
            learner_id=observation.learner_id,
            candidate_id=observation.candidate_id,
            context_key=observation.context_key,
        )

    def record_from_outcome(
        self,
        outcome: OutcomeRecord,
        candidate_id: str,
        context_key: str = "global",
    ) -> InterventionResponseProfile:
        """Record an observation derived directly from an upstream OutcomeRecord.

        Args:
            outcome: The verified OutcomeRecord.
            candidate_id: The identifier of the delivered intervention candidate.
            context_key: Scoping context key.

        Returns:
            The newly recalculated InterventionResponseProfile.
        """
        primary_measure = outcome.measurement(OutcomeMeasure.SUBSEQUENT_TRAJECTORY)
        if primary_measure.direction is OutcomeDirection.NOT_ASSESSED:
            primary_measure = outcome.measurement(OutcomeMeasure.SESSION_CONTINUATION)
        direction = _map_outcome_direction(primary_measure.direction)

        obs = InterventionResponseObservation(
            learner_id=outcome.learner_id,
            candidate_id=candidate_id,
            session_id=outcome.session_id,
            context_key=context_key,
            observed_at=outcome.computed_at,
            direction=direction,
            data_origin=outcome.data_origin,
        )
        return self.record_observation(obs)

    def get_profile(
        self,
        learner_id: LearnerId | str,
        candidate_id: str,
        context_key: str = "global",
    ) -> InterventionResponseProfile:
        """Fetch the current response profile for a specific learner-candidate-context tuple."""
        return self._compute_profile(
            learner_id=str(learner_id),
            candidate_id=candidate_id,
            context_key=context_key,
        )

    def get_learner_profiles(
        self,
        learner_id: LearnerId | str,
    ) -> Sequence[InterventionResponseProfile]:
        """Fetch all response profiles recorded for a given learner."""
        target_learner = str(learner_id)
        profiles: list[InterventionResponseProfile] = []
        for lid, cid, ckey in self._observations:
            if lid == target_learner:
                profiles.append(self._compute_profile(lid, cid, ckey))
        return sorted(profiles, key=lambda p: (p.candidate_id, p.context_key))

    def _compute_profile(
        self,
        learner_id: str,
        candidate_id: str,
        context_key: str,
    ) -> InterventionResponseProfile:
        """Synthesize an immutable profile from the accumulated observations."""
        key = (learner_id, candidate_id, context_key)
        observations = self._observations.get(key, [])
        total = len(observations)

        pos = sum(1 for o in observations if o.direction == ResponseOutcomeDirection.POSITIVE)
        neu = sum(1 for o in observations if o.direction == ResponseOutcomeDirection.NEUTRAL)
        neg = sum(1 for o in observations if o.direction == ResponseOutcomeDirection.NEGATIVE)
        unassessable = sum(
            1
            for o in observations
            if o.direction
            in (
                ResponseOutcomeDirection.INSUFFICIENT_DATA,
                ResponseOutcomeDirection.NOT_ASSESSABLE,
            )
        )

        data_origin = (
            DataOrigin.REAL
            if any(o.data_origin == DataOrigin.REAL for o in observations)
            else DataOrigin.SYNTHETIC
        )

        if total < self.min_observations:
            return InterventionResponseProfile(
                learner_id=learner_id,
                candidate_id=candidate_id,
                context_key=context_key,
                feedback_version=FEEDBACK_V1,
                observation_count=total,
                positive_count=pos,
                neutral_count=neu,
                negative_count=neg,
                unassessable_count=unassessable,
                maturity=ProfileMaturity.COLLECTING,
                min_observations=self.min_observations,
                effective_success_rate=None,
                data_origin=data_origin,
            )

        assessable = pos + neu + neg
        rate = round(pos / assessable, 4) if assessable > 0 else None

        return InterventionResponseProfile(
            learner_id=learner_id,
            candidate_id=candidate_id,
            context_key=context_key,
            feedback_version=FEEDBACK_V1,
            observation_count=total,
            positive_count=pos,
            neutral_count=neu,
            negative_count=neg,
            unassessable_count=unassessable,
            maturity=ProfileMaturity.ACTIVE,
            min_observations=self.min_observations,
            effective_success_rate=rate,
            data_origin=data_origin,
        )

    def select_adjusted_candidates(
        self,
        learner_id: LearnerId | str,
        candidates: Sequence[InterventionCandidate],
        context_key: str = "global",
    ) -> Sequence[InterventionCandidate]:
        """Optionally adjust candidate rankings if active profiles exist.

        Gating rules:
        - If profile is in COLLECTING status, candidate priority is left untouched.
        - If profile is ACTIVE and has high success rate, candidate is preferred.
        - If profile is ACTIVE and has zero success rate / high negative rate, candidate is deprioritized.
        """
        if not candidates:
            return ()

        # Score each candidate: (has_active_bonus, success_rate, original_index)
        scored: list[tuple[float, int, InterventionCandidate]] = []
        for idx, cand in enumerate(candidates):
            profile = self.get_profile(learner_id, cand.intervention_type, context_key)
            if (
                profile.maturity == ProfileMaturity.ACTIVE
                and profile.effective_success_rate is not None
            ):
                # Active profile score
                scored.append((profile.effective_success_rate, -idx, cand))
            else:
                # Neutral default baseline (keeps catalogue relative ordering)
                scored.append((0.5, -idx, cand))

        # Sort descending by score, breaking ties by catalogue index
        sorted_candidates = [
            cand for _, _, cand in sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)
        ]
        return tuple(sorted_candidates)

    def retrain_model_from_feedback(self) -> None:
        """Structurally refuse direct automated model retraining from feedback streams.

        Raises:
            NotImplementedError: Always. Live streaming retraining violates safety governance.
        """
        raise NotImplementedError(
            "Direct automated retraining from live feedback is architecturally prohibited. "
            "Retraining must follow an offline, versioned, evaluated governance workflow."
        )
