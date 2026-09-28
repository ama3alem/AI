"""Data models and schemas for the feedback and response profiling architecture.

Every intervention delivery produces an observable before-and-after trajectory.
The feedback layer accumulates these observed trajectories per learner, candidate,
and context without making causal leaps or triggering automated model retraining.

Key properties:
- Observations are keyed by learner, intervention candidate, and context.
- Below ``min_response_observations``, profiles are in ``COLLECTING`` status and
  refuse to emit selection adjustment weights.
- All records preserve strict ``DataOrigin`` and ``Provenance`` stamps.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.schemas.primitives import (
    DataOrigin,
    LearnerId,
    Provenance,
    SessionId,
    UtcTimestamp,
    non_empty_text,
    utc_now,
)
from focus_engine.schemas.versioning import FeedbackVersion

__all__ = [
    "DEFAULT_MIN_RESPONSE_OBSERVATIONS",
    "FEEDBACK_V1",
    "InterventionResponseObservation",
    "InterventionResponseProfile",
    "ProfileMaturity",
    "ResponseOutcomeDirection",
]

#: The standard feedback definition version.
FEEDBACK_V1: Final[FeedbackVersion] = "FEEDBACK_V1"

#: Minimum number of observations required before a profile matures out of COLLECTING.
DEFAULT_MIN_RESPONSE_OBSERVATIONS: Final[int] = 5


class ResponseOutcomeDirection(StrEnum):
    """Direction of observed learner behavior following an intervention delivery."""

    POSITIVE = "positive"
    """Observable metrics showed stabilization or recovery."""

    NEUTRAL = "neutral"
    """Observable metrics showed no measurable change."""

    NEGATIVE = "negative"
    """Observable metrics continued to decline."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Too few events in the post-intervention window to assess."""

    NOT_ASSESSABLE = "not_assessable"
    """Evaluation window was truncated or invalid."""


class ProfileMaturity(StrEnum):
    """Maturity stage of an intervention response profile."""

    COLLECTING = "collecting"
    """Fewer than min_observations recorded. Gated from influencing selection."""

    ACTIVE = "active"
    """Sufficient observations recorded to supply empirical weighting."""


class InterventionResponseObservation(BaseModel):
    """An immutable record of one observed response to an intervention delivery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str = Field(default_factory=non_empty_text)
    learner_id: LearnerId
    candidate_id: str
    session_id: SessionId
    context_key: str = "global"
    observed_at: UtcTimestamp = Field(default_factory=utc_now)
    direction: ResponseOutcomeDirection
    data_origin: DataOrigin = DataOrigin.SYNTHETIC
    provenance: Provenance = Provenance.OBSERVED


class InterventionResponseProfile(BaseModel):
    """Accumulated empirical response statistics for a learner-candidate pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    learner_id: LearnerId
    candidate_id: str
    context_key: str = "global"
    feedback_version: FeedbackVersion = FEEDBACK_V1
    observation_count: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    unassessable_count: int = 0
    maturity: ProfileMaturity = ProfileMaturity.COLLECTING
    min_observations: int = DEFAULT_MIN_RESPONSE_OBSERVATIONS
    effective_success_rate: float | None = None
    data_origin: DataOrigin = DataOrigin.SYNTHETIC

    @model_validator(mode="after")
    def _validate_consistency(self) -> InterventionResponseProfile:
        """Enforce internal mathematical consistency and maturity gating."""
        total = (
            self.positive_count + self.neutral_count + self.negative_count + self.unassessable_count
        )
        if self.observation_count != total:
            raise ValueError(
                f"observation_count ({self.observation_count}) does not match sum of counts ({total})"
            )

        if self.observation_count < self.min_observations:
            if self.maturity != ProfileMaturity.COLLECTING:
                raise ValueError(
                    f"Profile with {self.observation_count} observations must be in COLLECTING maturity (min: {self.min_observations})"
                )
            if self.effective_success_rate is not None:
                raise ValueError(
                    "effective_success_rate must be None when profile is in COLLECTING maturity"
                )
        else:
            if self.maturity != ProfileMaturity.ACTIVE:
                raise ValueError(
                    f"Profile with {self.observation_count} observations must be in ACTIVE maturity (min: {self.min_observations})"
                )
            assessable = self.positive_count + self.neutral_count + self.negative_count
            if assessable > 0:
                expected_rate = round(self.positive_count / assessable, 4)
                if (
                    self.effective_success_rate is None
                    or abs(self.effective_success_rate - expected_rate) > 1e-3
                ):
                    raise ValueError(
                        f"effective_success_rate ({self.effective_success_rate}) inconsistent with counts (expected: {expected_rate})"
                    )
            else:
                if self.effective_success_rate is not None:
                    raise ValueError(
                        "effective_success_rate must be None when all observations are unassessable"
                    )

        return self
