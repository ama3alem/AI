"""The intervention policy layer.

The policy decides whether to act, under hard restraints, and does so deterministically.
It does not learn from live interactions, does not issue concrete intervention payloads,
and never calls the delivery side. That separation preserves the line between "should we
intervene?" (policy) and "did it help?" (outcome engine), and it keeps the system
auditable and testable.

The public surface is small on purpose. :class:`InterventionPolicy` is the composition point,
:class:`InterventionHistory` is the recorded evidence it reasons over, and
:class:`PolicyDecision` is the answer - including the case where the answer is to do
nothing, which is a real outcome rather than a failure to produce one.
"""

from __future__ import annotations

from focus_engine.policy.engine import DEFAULT_CATALOGUE, InterventionPolicy
from focus_engine.policy.history import (
    DeliveredIntervention,
    InterventionHistory,
    OutcomeClass,
    history_from_events,
)
from focus_engine.policy.models import (
    ACTIONABLE_STATES,
    CONFIDENCE_RANK,
    POLICY_V1,
    InterventionCandidate,
    NonActionReason,
    PolicyDecision,
    PolicyDecisionType,
    Restraint,
    RestraintKind,
    meets_confidence,
)

__all__ = [
    "ACTIONABLE_STATES",
    "CONFIDENCE_RANK",
    "DEFAULT_CATALOGUE",
    "POLICY_V1",
    "DeliveredIntervention",
    "InterventionCandidate",
    "InterventionHistory",
    "InterventionPolicy",
    "NonActionReason",
    "OutcomeClass",
    "PolicyDecision",
    "PolicyDecisionType",
    "Restraint",
    "RestraintKind",
    "history_from_events",
    "meets_confidence",
]
