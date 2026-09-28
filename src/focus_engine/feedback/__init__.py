"""Feedback and response profile architecture (Phase 12).

Accumulates observed before/after intervention trajectories per learner, candidate,
and context. Gated from candidate adjustment until minimum observations threshold is met.
Direct automated model retraining is strictly prohibited.
"""

from __future__ import annotations

from focus_engine.feedback.engine import FeedbackEngine
from focus_engine.feedback.models import (
    DEFAULT_MIN_RESPONSE_OBSERVATIONS,
    FEEDBACK_V1,
    InterventionResponseObservation,
    InterventionResponseProfile,
    ProfileMaturity,
    ResponseOutcomeDirection,
)

__all__ = [
    "DEFAULT_MIN_RESPONSE_OBSERVATIONS",
    "FEEDBACK_V1",
    "FeedbackEngine",
    "InterventionResponseObservation",
    "InterventionResponseProfile",
    "ProfileMaturity",
    "ResponseOutcomeDirection",
]
