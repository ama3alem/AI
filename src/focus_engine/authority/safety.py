"""The safety gate: the last thing a decision passes through before it can happen.

The safety layer is deliberately narrow. It does not re-evaluate the probability or
re-examine the uncertainty; it takes the authority envelope, the policy decision, and a
single piece of structural evidence - whether a human is available - and answers one
question: should this specific action happen *right now*, in *this context*?

Three outcomes are possible:

- **ALLOWED**: The system may act autonomously. The safety layer has no objections.
- **DEFERRED**: The system would act but a human has not yet confirmed. The action is
  queued rather than executed, and the queue is visible so that a reviewer can see the
  backlog.
- **REFUSED**: The system will not act. The refusal is structural, not probabilistic:
  the action is either not permitted at this authority weight, or the envelope is
  expired, or a safety rule explicitly blocks it regardless of everything else.

The safety layer does not override. An envelope that says ``ALLOWED`` and an envelope
that says ``HUMAN_REQUIRED`` are equally final, and the safety layer does not
contradict either. What it does is enforce the structural rules that the authority
layer depends on but cannot self-enforce: expiry is checked here, because the authority
layer is the only place it *could* be checked and the one place a stale envelope is
most dangerous.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from focus_engine.authority.feedback import HumanAvailability
from focus_engine.authority.types import AuthorityEnvelope, AuthorityLevel, RestrictedAction
from focus_engine.policy.models import PolicyDecision, PolicyDecisionType

__all__ = ["FinalAction", "FinalActionOutcome", "SafetyGate"]

#: High-impact actions are always restricted regardless of context risk. The list is a
#: closed vocabulary so that adding a new high-impact action requires an explicit edit
#: here rather than an implicit discovery in a configuration file.
_HIGH_IMPACT_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        RestrictedAction.GRADE_MODIFICATION.value,
        RestrictedAction.HIGH_STAKES_RECOMMENDATION.value,
        RestrictedAction.EXTERNAL_COMMUNICATION.value,
        RestrictedAction.DATA_EXPORT.value,
        RestrictedAction.PERMANENT_RECORD_CHANGE.value,
        RestrictedAction.CREDENTIAL_REVOCATION.value,
    }
)


class FinalActionOutcome(StrEnum):
    """What the system will do with the intervention decision.

    Three, not two, because "do nothing autonomously" and "do nothing at all" are
    different situations and collapsing them loses the distinction between a system that
    is choosing restraint and a system that is blocked by its own safety rules.
    """

    ALLOWED = "allowed"
    """The system may execute the action autonomously."""

    DEFERRED = "deferred"
    """A human must confirm before this happens. The action is queued."""

    REFUSED = "refused"
    """The system will not act. A structural rule prevents it."""


@dataclass(frozen=True, slots=True)
class FinalAction:
    """The resolved status of an intervention decision.

    This is the end of the line: the object a downstream executor reads and either acts
    on, queues, or discards. It carries the full provenance chain so that a reviewer who
    sees a refusal can trace it back through the safety gate, the authority envelope, and
    the original policy decision.

    Attributes:
        outcome: Whether the action is cleared, queued, or dead.
        action: The action type under consideration, or ``None`` when the policy declined.
        intervention_type: The specific intervention selected by the policy, or ``None``.
        reason: A human-readable explanation for the outcome.
        authority_level: The envelope's level, carried forward.
        confidence_weight: The envelope's confidence weight, carried forward.
        envelope_expired: Whether the authority envelope was stale at the time of
            evaluation. This is the single most common reason for ``REFUSED`` and it is
            always a bug, because a stale envelope must not have been passed to the safety
            gate in the first place.
        requires_human_approval: ``True`` when ``DEFERRED``, carried forward for
            machine-readability.
        escalation_path: The concrete route to follow if this action is still desired.
        available_human: What was known about human availability.
    """

    outcome: FinalActionOutcome
    action: str | None
    intervention_type: str | None
    reason: str
    authority_level: AuthorityLevel | None
    confidence_weight: float | None
    envelope_expired: bool
    requires_human_approval: bool
    escalation_path: tuple[str, ...]
    available_human: HumanAvailability | None

    def to_summary_dict(self) -> dict[str, object]:
        """Render the action as flat, loggable fields.

        Returns:
            A dict suitable for ``json.dumps`` and for the stage log.
        """
        return {
            "outcome": self.outcome.value,
            "action": self.action,
            "intervention_type": self.intervention_type,
            "reason": self.reason,
            "authority_level": None if self.authority_level is None else self.authority_level.value,
            "confidence_weight": self.confidence_weight,
            "envelope_expired": self.envelope_expired,
            "requires_human_approval": self.requires_human_approval,
            "escalation_path": list(self.escalation_path),
            "available_human": None if self.available_human is None else self.available_human.value,
        }


class SafetyGate:
    """Evaluates a policy decision against the authority envelope and human availability.

    The gate is stateless. It reads the same inputs every time, which is what makes it
    testable: a test that provides an expired envelope will always receive ``REFUSED``,
    regardless of the policy's confidence or the system's history. That predictability is
    the property a safety gate is for.

    Args:
        high_impact_min_authority: Minimum authority weight at which a ``HIGH_IMPACT``
            action may be considered. Pulled from settings so the gate does not import a
            settings module; the engine that owns the gate supplies this.
    """

    __slots__ = ("_high_impact_min",)

    def __init__(self, *, high_impact_min_authority: float = 0.75) -> None:
        """Initialise the gate.

        Args:
            high_impact_min_authority: The authority floor for high-impact actions. Must be
                in ``(0, 1]``; an impossible floor wastes no computation because the gate
                is fast.

        Raises:
            ValueError: If ``high_impact_min_authority`` is not in ``(0, 1]``.
        """
        if not 0.0 < high_impact_min_authority <= 1.0:
            raise ValueError(
                f"high_impact_min_authority must be in (0, 1]; got {high_impact_min_authority!r}"
            )
        self._high_impact_min = high_impact_min_authority

    def evaluate(
        self,
        decision: PolicyDecision,
        envelope: AuthorityEnvelope,
        availability: HumanAvailability,
        *,
        as_of: datetime,
    ) -> FinalAction:
        """Evaluate whether the system may act on ``decision`` at ``as_of``.

        The order of checks matters. The structural checks (policy has no action, envelope
        is expired, action is not permitted) are applied first, before human availability,
        because a structural refusal is final regardless of who is in the building. Human
        availability is checked last because its meaning changes depending on whether the
        earlier checks already cleared the path.

        ``as_of`` is the decision's own instant, not the envelope's creation time. A grant
        that was valid when it was minted may be long dead by the time an executor reaches
        it, and reading the clock from the envelope would be exactly the stale-permission
        bug this gate exists to catch.

        Args:
            decision: The policy decision, which may or may not have selected an action.
            envelope: The authority envelope produced by the authority engine.
            availability: Whether a human who could approve is reachable.
            as_of: The instant the action would take effect. Used for every expiry test.

        Returns:
            A resolved :class:`FinalAction` with the outcome and full provenance.
        """
        # --- structural rules ---
        # The candidate is checked alongside the decision type rather than after it.
        # ``PolicyDecision`` guarantees an INTERVENE decision names a candidate, but a
        # type checker cannot see that guarantee, and the defensive branch costs nothing:
        # a decision that claims to act while naming nothing has no action to gate, and
        # refusing it is the only honest reading.
        selected = decision.selected
        if decision.decision is not PolicyDecisionType.INTERVENE or selected is None:
            return FinalAction(
                outcome=FinalActionOutcome.REFUSED,
                action=None,
                intervention_type=None,
                reason=(
                    "the policy declined to select an intervention; the safety gate does "
                    "not invent actions the policy chose not to make"
                ),
                authority_level=envelope.authority_level,
                confidence_weight=envelope.confidence_weight,
                envelope_expired=False,
                requires_human_approval=False,
                escalation_path=(),
                available_human=availability,
            )

        action = selected.intervention_type

        envelope_expired = envelope.is_expired(as_of)
        if envelope_expired:
            return FinalAction(
                outcome=FinalActionOutcome.REFUSED,
                action=action,
                intervention_type=action,
                reason=(
                    "the authority envelope is expired; a stale grant must not be acted on, "
                    "and the envelope must be recalculated rather than reused"
                ),
                authority_level=envelope.authority_level,
                confidence_weight=envelope.confidence_weight,
                envelope_expired=True,
                requires_human_approval=False,
                escalation_path=(),
                available_human=availability,
            )

        # A HUMAN_REQUIRED envelope is checked *before* the permitted list, because such an
        # envelope deliberately permits nothing and the permitted-list check would
        # therefore reject every action and report a dead end. The truth is the opposite:
        # the action is not refused, it is queued for a person who has not decided yet.
        # Conflating the two would report a pending escalation as a closed question and
        # leave a real request for a human looking like nothing was ever raised.
        if envelope.authority_level is AuthorityLevel.HUMAN_REQUIRED:
            if availability is HumanAvailability.AVAILABLE:
                reason = (
                    "the envelope requires human approval and a named approver is "
                    "available; the action is queued for their decision"
                )
            else:
                reason = (
                    "the envelope requires human approval but no approver is reachable; the "
                    "action stays queued and cannot be confirmed or refused until a person "
                    "responds. An unreachable human never converts into an approval."
                )
            return FinalAction(
                outcome=FinalActionOutcome.DEFERRED,
                action=action,
                intervention_type=action,
                reason=reason,
                authority_level=envelope.authority_level,
                confidence_weight=envelope.confidence_weight,
                envelope_expired=False,
                requires_human_approval=True,
                escalation_path=envelope.escalation_path,
                available_human=availability,
            )

        if not envelope.permits(action, at=as_of):
            return FinalAction(
                outcome=FinalActionOutcome.REFUSED,
                action=action,
                intervention_type=action,
                reason=(
                    f"the action {action!r} is not in the envelope's permitted list; the "
                    "authority layer withheld it, and the safety gate does not override"
                ),
                authority_level=envelope.authority_level,
                confidence_weight=envelope.confidence_weight,
                envelope_expired=False,
                requires_human_approval=False,
                escalation_path=envelope.escalation_path,
                available_human=availability,
            )

        is_high_impact = action in _HIGH_IMPACT_ACTIONS
        if is_high_impact and envelope.confidence_weight < self._high_impact_min:
            return FinalAction(
                outcome=FinalActionOutcome.REFUSED,
                action=action,
                intervention_type=action,
                reason=(
                    f"action {action!r} is high-impact and the confidence weight "
                    f"{envelope.confidence_weight:.3f} is below the required minimum "
                    f"{self._high_impact_min:.3f}"
                ),
                authority_level=envelope.authority_level,
                confidence_weight=envelope.confidence_weight,
                envelope_expired=False,
                requires_human_approval=False,
                escalation_path=envelope.escalation_path,
                available_human=availability,
            )

        # --- human availability ---
        if envelope.authority_level is AuthorityLevel.RESTRICTED:
            if availability is HumanAvailability.AVAILABLE:
                return FinalAction(
                    outcome=FinalActionOutcome.ALLOWED,
                    action=action,
                    intervention_type=action,
                    reason=(
                        "the envelope restricts autonomous action but a human is available "
                        "for confirmation; the action proceeds with approval"
                    ),
                    authority_level=envelope.authority_level,
                    confidence_weight=envelope.confidence_weight,
                    envelope_expired=False,
                    requires_human_approval=True,
                    escalation_path=envelope.escalation_path,
                    available_human=availability,
                )
            return FinalAction(
                outcome=FinalActionOutcome.DEFERRED,
                action=action,
                intervention_type=action,
                reason=(
                    "the envelope restricts autonomous action and no human is available to "
                    "confirm; the action is queued"
                ),
                authority_level=envelope.authority_level,
                confidence_weight=envelope.confidence_weight,
                envelope_expired=False,
                requires_human_approval=True,
                escalation_path=envelope.escalation_path,
                available_human=availability,
            )

        # ALLOWED envelope
        return FinalAction(
            outcome=FinalActionOutcome.ALLOWED,
            action=action,
            intervention_type=action,
            reason=(
                "the authority envelope permits autonomous action in this context and the "
                "safety gate has no structural objection"
            ),
            authority_level=envelope.authority_level,
            confidence_weight=envelope.confidence_weight,
            envelope_expired=False,
            requires_human_approval=False,
            escalation_path=(),
            available_human=availability,
        )
