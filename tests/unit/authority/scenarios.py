"""Shared builders for the authority tests.

The authority layer is a pure function of a context, a memory, a band, and a clock, so the
tests state each of those directly instead of replaying a session to reach them. A test
that had to synthesise a declining learner in order to check that an expired envelope is
refused is testing the simulator as much as the gate, and when it fails there is no way to
tell which one moved.

The builders here clear every restraint that is not under test. A test about human
availability should not have to earn a low-risk context first, and a test that has to is a
test that will quietly start testing something else as soon as a default changes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from focus_engine.authority import (
    AuthorityEngine,
    ContextDescriptor,
    ContextRisk,
    FeedbackType,
    HumanFeedback,
)
from focus_engine.configuration.thresholds import AuthoritySettings
from focus_engine.policy.models import (
    InterventionCandidate,
    NonActionReason,
    PolicyDecision,
    PolicyDecisionType,
)
from focus_engine.schemas.primitives import DataOrigin
from focus_engine.utils.clock import FixedClock

#: A fixed instant every test starts from, so failures read as dates rather than as "now".
START = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)

LEARNER = "learner-0001"
SESSION = "session-0001"

#: A low-risk, reversible, non-high-stakes context: the only shape that can grant
#: autonomous execution at all.
FREE = ContextDescriptor(
    context_type="free_practice",
    risk=ContextRisk.LOW,
    description="Unsupervised practice between lessons.",
)

#: Same learner-facing consequence, but the decision feeds a recorded mark.
GRADED = ContextDescriptor(
    context_type="graded_assessment",
    risk=ContextRisk.MEDIUM,
    is_high_stakes=True,
    description="A graded quiz whose result is recorded.",
)

#: Irreversible and high-stakes. Escalates on structure alone, whatever the history says.
EXAM = ContextDescriptor(
    context_type="exam_hall",
    risk=ContextRisk.HIGH,
    is_reversible=False,
    is_high_stakes=True,
    description="A proctored examination.",
)


def engine(
    at: datetime = START,
    settings: AuthoritySettings | None = None,
    *,
    origin: DataOrigin = DataOrigin.SYNTHETIC,
) -> AuthorityEngine:
    """Build an engine frozen at a known instant.

    Args:
        at: The instant the clock reads.
        settings: Authority thresholds, or ``None`` for the defaults.
        origin: Whether the evidence behind grants is real or synthetic.

    Returns:
        An :class:`~focus_engine.authority.AuthorityEngine` with an empty ledger.
    """
    return AuthorityEngine(
        AuthoritySettings() if settings is None else settings,
        clock=FixedClock(at),
        data_origin=origin,
    )


def intervene(action: str = "recap") -> PolicyDecision:
    """Build a policy decision that has already chosen to act.

    The authority layer does not re-derive this decision, so the builder states the
    outcome rather than arranging for a real one: a test about what the safety gate does
    with an action should not fail because the policy changed its mind about a cooldown.

    Args:
        action: The intervention type the policy selected.

    Returns:
        A :class:`~focus_engine.policy.models.PolicyDecision` of type ``INTERVENE``.
    """
    return PolicyDecision(
        decision=PolicyDecisionType.INTERVENE,
        selected=InterventionCandidate(intervention_type=action),
        learner_id=LEARNER,
        session_id=SESSION,
        detail="every restraint was clear",
    )


def decline() -> PolicyDecision:
    """Build a policy decision that declined to act.

    Returns:
        A :class:`~focus_engine.policy.models.PolicyDecision` of type
        ``NO_INTERVENTION`` carrying a non-action reason and no candidate.
    """
    return PolicyDecision(
        decision=PolicyDecisionType.NO_INTERVENTION,
        reason=NonActionReason.NOT_RESOLVED,
        learner_id=LEARNER,
        session_id=SESSION,
        detail="the band was too low to resolve anything",
    )


def dispute(
    *,
    feedback_id: str = "fb-0001",
    context_type: str = "free_practice",
    at: datetime = START,
    feedback_type: FeedbackType = FeedbackType.DISPUTED,
) -> HumanFeedback:
    """Build a piece of feedback from a named role.

    Args:
        feedback_id: The feedback's identifier.
        context_type: The context the feedback is about. Varying this is how a test shows
            that history does not leak between contexts.
        at: When the feedback was submitted.
        feedback_type: What the person did.

    Returns:
        The :class:`~focus_engine.authority.HumanFeedback`.
    """
    return HumanFeedback(
        feedback_id=feedback_id,
        decision_ref="decision-0001",
        feedback_type=feedback_type,
        actor_role="instructor",
        submitted_at=at,
        context_type=context_type,
        corrected_reading=(
            "the learner was recalibrating, not disengaging"
            if feedback_type is FeedbackType.CORRECTED
            else None
        ),
    )
