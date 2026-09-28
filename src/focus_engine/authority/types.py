"""The authority vocabulary: what a decision is allowed to do, and where.

**Authority is not confidence.** The uncertainty engine answers "how much do we believe
this?"; authority answers "even if we believe it, who is permitted to act on it, in this
context, and for how long?". Conflating the two is the specific failure this layer exists
to prevent: a system that treats a confident prediction as permission will act hardest
exactly when it is most wrong, because confidence and error are least correlated at high
confidence. A learner can be *predicted* to disengage with high confidence and still be
someone the system has no business acting on automatically.

**Authority is contextual, never global.** The same learner, the same probability, and the
same model output yield different authority in a graded quiz than in free practice,
because the consequence of being wrong differs by orders of magnitude. A permission
granted in one context therefore does not travel to another: :class:`ContextDescriptor`
is part of the envelope's identity, and :class:`AuthorityEnvelope` refuses to be
constructed without one. This is why a single global "autonomy level" per user is not
expressible here, and why it should not be: the risky action and the harmless one are
frequently the same action in different rooms.

**The absence of a human grants nothing.** When no human is reachable the honest outcome
is *more* restriction, not less: an action that would have been queued for approval becomes
deferred, and one that was already permitted is unaffected. The inverse would mean a
system becomes more capable at 3am, which is the opposite of what unattended autonomy
should mean. :class:`HumanAvailability` is an input to the envelope precisely so that this
can be enforced rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from focus_engine.schemas.primitives import ConfidenceLevel, DataOrigin

__all__ = [
    "ActionPermission",
    "AuthorityEnvelope",
    "AuthorityLevel",
    "ContextDescriptor",
    "ContextRisk",
    "RestrictedAction",
    "WithheldAction",
    "confidence_ceiling",
]


class AuthorityLevel(StrEnum):
    """The permission class an envelope carries.

    Three levels rather than a boolean, because "do nothing" and "ask a person" are
    different operational states and collapsing them loses the distinction between a
    system that is choosing restraint and a system that is blocked.
    """

    ALLOWED = "allowed"
    """The proposed action set may be executed autonomously in this context."""

    RESTRICTED = "restricted"
    """Only :attr:`AuthorityEnvelope.permitted_actions` may run; the rest are withheld."""

    HUMAN_REQUIRED = "human_required"
    """Nothing runs autonomously. A named human must decide first."""


class ContextRisk(StrEnum):
    """How much harm an action in a context can cause.

    This is a property of the *context*, not of the learner, the model, or the action's
    label. A recap shown in free practice and a recap shown immediately before a graded
    submission are the same widget and different risks, which is why the risk travels with
    the context descriptor rather than with the intervention catalogue.
    """

    LOW = "low"
    """Reversible and private. Nothing the learner did is recorded or communicated."""

    MEDIUM = "medium"
    """Visible to the learner and mildly disruptive, but reversible and unrecorded."""

    HIGH = "high"
    """Affects a learner's record, standing, or access. Cannot be undone by the learner."""


class RestrictedAction(StrEnum):
    """Actions that are never taken autonomously, whatever the evidence says.

    The membership of this set is the operational meaning of "this system is not allowed
    to do that to a person without a person deciding". It is deliberately a closed
    vocabulary rather than a configurable list of strings, so that adding a capability
    cannot quietly add a permission: a new member is a visible edit to this file.
    """

    GRADE_MODIFICATION = "grade_modification"
    """Alter a recorded grade, score, or submission."""

    HIGH_STAKES_RECOMMENDATION = "high_stakes_recommendation"
    """Recommend progression, graduation, or removal from a course."""

    EXTERNAL_COMMUNICATION = "external_communication"
    """Contact a learner, guardian, employer, or any third party."""

    DATA_EXPORT = "data_export"
    """Move learner data outside the system it was collected in."""

    PERMANENT_RECORD_CHANGE = "permanent_record_change"
    """Write anything a learner cannot later contest or have corrected."""

    CREDENTIAL_REVOCATION = "credential_revocation"
    """Remove or suspend an access right or qualification."""


#: Actions that are permitted without further approval, in a low-risk context. The list is
#: an allow-list rather than a deny-list on purpose: an action nobody thought about is not
#: permitted, and adding one is a visible edit.
DEFAULT_PERMITTED_ACTIONS: frozenset[str] = frozenset(
    {
        "recap",
        "micro_question",
        "checkpoint_prompt",
        "focus_quiz",
        "hint",
        "pacing_adjustment",
        "resource_suggestion",
        "break_suggestion",
    }
)


class ActionPermission(StrEnum):
    """Whether a specific action type is cleared, withheld, or needs a named approver.

    This is the per-action resolution of an :class:`AuthorityLevel`, and the two are kept
    separate on purpose. The level answers "how much is this system trusted here?"; the
    permission answers "what may it do with that trust?". A system can be trusted
    (:attr:`AuthorityLevel.ALLOWED`) and still be forbidden from the one action that
    matters in the current context - a graded assessment permits hints and forbids
    anything that touches a mark.
    """

    PERMITTED = "permitted"
    """May be executed autonomously under the current envelope."""

    WITHHELD = "withheld"
    """Known and explicitly refused. The reason is recorded rather than implied."""

    NEEDS_APPROVAL = "needs_approval"
    """Deferred until a named role approves. Never executed autonomously."""


@dataclass(frozen=True, slots=True)
class WithheldAction:
    """A restricted action together with why it was refused and who could allow it.

    Recording the reason alongside the refusal is what makes a withheld action
    auditable: a reader who sees that ``grade_modification`` was refused can see that it
    was refused *because* the context is irreversible and high-stakes, and can see which
    role would have to change that. A bare list of forbidden actions is indistinguishable
    from a bug.

    Attributes:
        action: The restricted action that was withheld.
        reason: Why it was withheld, in terms a reviewer can act on.
        required_approval: The role that could authorise it, or ``None`` when no role can
            and the action is permanently out of scope for the system.
        escalation_path: Ordered steps to follow to request the approval.
    """

    action: RestrictedAction
    reason: str
    required_approval: str | None = None
    escalation_path: tuple[str, ...] = ()

    def to_summary_dict(self) -> dict[str, object]:
        """Render the withheld action as flat, loggable fields.

        Returns:
            A dict with the action name, the reason, the approver, and the path.
        """
        return {
            "action": self.action.value,
            "reason": self.reason,
            "required_approval": self.required_approval,
            "escalation_path": list(self.escalation_path),
        }


def confidence_ceiling(level: ConfidenceLevel | None) -> float:
    """Map an uncertainty band to the most authority weight it can justify.

    A low-confidence prediction cannot purchase a high authority weight, and this is the
    single place that relationship is expressed. Returning ``0.0`` for a missing band
    rather than a default is deliberate: an absent confidence is absent evidence, and a
    default here would quietly manufacture permission from a missing field.

    Args:
        level: The band the uncertainty engine reported, or ``None`` when it declined.

    Returns:
        A ceiling in ``[0.0, 1.0]``. ``0.0`` when the band is ``None``, ``UNKNOWN``, or
        ``INSUFFICIENT_DATA``.
    """
    if level is None:
        return 0.0
    return {
        ConfidenceLevel.LOW: 0.35,
        ConfidenceLevel.MEDIUM: 0.70,
        ConfidenceLevel.HIGH: 1.0,
    }.get(level, 0.0)


@dataclass(frozen=True, slots=True)
class ContextDescriptor:
    """Where a decision would take effect, and how much it could cost.

    Attributes:
        context_type: A stable, machine-readable name for the learning situation, such as
            ``"graded_assessment"`` or ``"free_practice"``. This is the *only* thing that
            distinguishes two contexts for the authority layer, and it is what makes
            authority contextual: two otherwise identical decisions with different
            ``context_type`` values do not share a permission.
        risk: The consequence class of acting here.
        is_reversible: Whether the learner can undo what was done. ``False`` forces
            escalation for every restricted action regardless of trust.
        is_high_stakes: Whether this context feeds a decision with consequences beyond the
            session, such as progression or standing.
        description: Human-readable text for the ledger. Never load-bearing.
    """

    context_type: str
    risk: ContextRisk = ContextRisk.LOW
    is_reversible: bool = True
    is_high_stakes: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        """Reject a context with no machine-readable identity.

        Raises:
            ValueError: If ``context_type`` is empty or not a slug. An unnamed context
                would silently collapse into every other unnamed context, which is how a
                per-context permission becomes a global one by accident.
        """
        if not self.context_type or not self.context_type.strip():
            raise ValueError("context_type is required; an unnamed context has no identity")
        if not self.context_type.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "context_type must be alphanumeric with underscores or hyphens; got "
                f"{self.context_type!r}"
            )

    def to_summary_dict(self) -> dict[str, str | bool]:
        """Render the descriptor as flat, loggable fields.

        Returns:
            A dict carrying the identity, the risk class, and the two structural flags.
        """
        return {
            "context_type": self.context_type,
            "risk": self.risk.value,
            "is_reversible": self.is_reversible,
            "is_high_stakes": self.is_high_stakes,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class AuthorityEnvelope:
    """A scoped, expiring grant of permission to act in one context.

    **The envelope is the unit of audit.** It answers, for one moment and one context:
    how much authority the system holds, what it may do with that authority, what it is
    explicitly forbidden from doing, what a human would have to approve, and when this
    grant stops being valid. Everything the safety layer needs to refuse an action is
    present in this one object, which is what allows the refusal to be recorded and
    replayed rather than re-derived from mutable state.

    Attributes:
        authority_level: The overall permission class. ``ALLOWED`` permits the proposed
            action set; ``RESTRICTED`` permits only the permitted subset and defers the
            rest; ``HUMAN_REQUIRED`` permits nothing autonomously.
        context: Where this grant applies. Part of the identity, not metadata.
        permitted_actions: Action types the system may execute now, without a human.
        restricted_actions: Action types that are known and explicitly withheld, each
            paired with the reason it was withheld and the role that could allow it.
            Present so that a reader can see what was
        confidence_weight: How much of the system's belief survives the authority layer.
            Never exceeds the ceiling implied by the uncertainty band, and is reduced -
            never increased - by history. This is the field a human reader should watch.
        restrictions: Human-readable reasons this envelope is narrower than the
            evidence alone would justify.
        required_approvals: The named approval roles needed before a restricted action
            may proceed. Never a bare "human in the loop": a role is answerable, a
            general human is not.
        escalation_path: The concrete route an escalation follows, as ordered steps.
        calculated_at: When this envelope was computed.
        expires_at: When it stops being valid. An expired envelope is not "probably still
            fine"; it is invalid, and the engine recalculates.
        basis: The identifiers of the inputs that produced it, so a reader can reconstruct
            the decision from the same artifacts.
        data_origin: Whether the evidence behind the grant was real or synthetic.
    """

    authority_level: AuthorityLevel
    context: ContextDescriptor
    permitted_actions: tuple[str, ...]
    restricted_actions: tuple[WithheldAction, ...]
    confidence_weight: float
    restrictions: tuple[str, ...]
    required_approvals: tuple[str, ...]
    escalation_path: tuple[str, ...]
    calculated_at: datetime
    expires_at: datetime
    basis: dict[str, str] = field(default_factory=dict)
    data_origin: DataOrigin = DataOrigin.SYNTHETIC

    def __post_init__(self) -> None:
        """Validate that the envelope is internally consistent.

        Raises:
            ValueError: If the weight escapes ``[0, 1]``, if the envelope is already born
                expired, if an ``ALLOWED`` envelope permits nothing, or if the same action
                appears in both the permitted and the withheld set. That last
                contradiction is checked here because it is otherwise a plausible-looking
                bug whose resolution would depend on which call site a reader happened to
                consult.
        """
        if not 0.0 <= self.confidence_weight <= 1.0:
            raise ValueError(
                f"confidence_weight must lie in [0, 1]; got {self.confidence_weight!r}"
            )
        if self.expires_at <= self.calculated_at:
            raise ValueError(
                "expires_at must be later than calculated_at; an envelope that is born "
                f"expired would refuse every action (calculated_at="
                f"{self.calculated_at.isoformat()}, expires_at={self.expires_at.isoformat()})"
            )
        if self.authority_level is AuthorityLevel.ALLOWED and not self.permitted_actions:
            raise ValueError(
                "an ALLOWED envelope must permit at least one action; an empty permit list "
                "reads as permission at every call site while authorising nothing"
            )
        permitted = set(self.permitted_actions)
        for withheld in self.restricted_actions:
            if withheld.action.value in permitted:
                raise ValueError(
                    f"action {withheld.action.value!r} is both permitted and withheld; the "
                    "envelope contradicts itself, and which reading won would depend on the "
                    "call site rather than on the envelope"
                )

    def permits(self, action: str, *, at: datetime) -> bool:
        """Report whether ``action`` may be executed autonomously at ``at``.

        The instant is a required argument rather than a default. An envelope is a
        time-limited grant, so "is this still valid?" has no meaning without a clock
        reading, and a method that quietly answered for its own creation time would
        report every envelope as live for ever. Requiring the caller to supply ``at``
        also makes the reading reproducible: the same envelope and the same instant
        always give the same answer, which is what lets a refusal be replayed rather than
        re-derived from mutable state.

        Args:
            action: The action type under consideration.
            at: The instant the decision would take effect.

        Returns:
            ``True`` only if the action is in :attr:`permitted_actions` *and* the envelope
            is at :attr:`AuthorityLevel.ALLOWED` and not expired at ``at``. A restricted
            envelope answers ``False`` here even for actions it also lists, because "this
            action is not forbidden" and "this action is authorised" are different
            questions.
        """
        if self.authority_level is not AuthorityLevel.ALLOWED:
            return False
        if self.is_expired(at):
            return False
        return action in self.permitted_actions

    def is_expired(self, at: datetime) -> bool:
        """Report whether this envelope is stale at ``at``.

        Args:
            at: The instant to test.

        Returns:
            ``True`` when ``at`` is at or after :attr:`expires_at`.
        """
        return at >= self.expires_at

    def to_summary_dict(self) -> dict[str, object]:
        """Render the envelope as flat, loggable fields.

        Returns:
            A dict carrying every field a ledger entry needs, with enums flattened to
            their string values and datetimes to ISO-8601.
        """
        return {
            "authority_level": self.authority_level.value,
            "context_type": self.context.context_type,
            "context_risk": self.context.risk.value,
            "permitted_actions": list(self.permitted_actions),
            "restricted_actions": [item.to_summary_dict() for item in self.restricted_actions],
            "confidence_weight": self.confidence_weight,
            "restrictions": list(self.restrictions),
            "required_approvals": list(self.required_approvals),
            "escalation_path": list(self.escalation_path),
            "calculated_at": self.calculated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "basis": dict(self.basis),
            "data_origin": self.data_origin.value,
        }
