"""The authority engine: composes context, memory, decay, feedback, and the safety gate.

This is the layer's public surface. Everything it does is a composition of the smaller
pieces, and the composition itself carries the architectural commitments that no single
component can enforce on its own.

**Authority is recomputed, never cached across a decision.** An envelope is a
point-in-time grant with a short expiry, and the engine holds no "current authority level"
field that a later step could read. Every call to :meth:`AuthorityEngine.authorize` reads
the clock, ages the memory, and produces a fresh envelope. A cached level would be a
number that silently drifts out of date, which is the one thing an authority figure must
never be.

**Escalation is monotonic in risk, and irreversible contexts escalate regardless of
trust.** A high-stakes context, an irreversible one, or a permanent-record change all
force human review no matter how much the system's history vouches for it. Trust earns
the *right to act in a low-risk room*; it never buys the right to write a mark.

**The ledger records every calculation, not just the outcomes that mattered.** A reader
who can see only the interventions that fired cannot audit the system, because the
interesting cases are the refusals. Every envelope, every feedback, and every decay
report is appended, so a reviewer can see what the system was permitted to do, what it
did, and what it declined.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from focus_engine.authority.decay import DecayEngine, DecayReport
from focus_engine.authority.feedback import (
    DisagreementRecord,
    FeedbackOutcome,
    FeedbackType,
    HumanAvailability,
    HumanFeedback,
)
from focus_engine.authority.ledger import AuthorityLedger, LedgerIntegrity
from focus_engine.authority.memory import (
    ContextMemory,
    InterventionOutcomeMemory,
    MemoryEntry,
)
from focus_engine.authority.safety import FinalAction, SafetyGate
from focus_engine.authority.types import (
    DEFAULT_PERMITTED_ACTIONS,
    ActionPermission,
    AuthorityEnvelope,
    AuthorityLevel,
    ContextDescriptor,
    ContextRisk,
    RestrictedAction,
    WithheldAction,
)
from focus_engine.configuration.thresholds import AuthoritySettings
from focus_engine.policy.models import PolicyDecision
from focus_engine.schemas.primitives import ConfidenceLevel, DataOrigin
from focus_engine.utils.clock import Clock

__all__ = [
    "ESCALATION_CATALOGUE",
    "AuthorityEngine",
    "AuthorityAssessment",
]

#: The named role and route for each restricted action. Escalation names a *role* rather
#: than a bare "human in the loop" because a role is answerable - one can ask who holds it,
#: whether they are rostered, and whether they declined - whereas a generic human is
#: nobody in particular and the request is unrouteable. The steps are ordered and concrete
#: so that an escalation can be followed rather than merely attempted.
ESCALATION_CATALOGUE: dict[str, tuple[str, tuple[str, ...]]] = {
    RestrictedAction.GRADE_MODIFICATION.value: (
        "course_coordinator",
        (
            "record the requested change with the assessment reference",
            "ask the course coordinator to confirm the intended mark",
            "apply the change through the gradebook, not through this system",
        ),
    ),
    RestrictedAction.HIGH_STAKES_RECOMMENDATION.value: (
        "academic_advisor",
        (
            "attach the supporting evidence and the current uncertainty band",
            "ask the academic advisor to review the learner's full record",
            "record the advisor's decision separately from this prediction",
        ),
    ),
    RestrictedAction.EXTERNAL_COMMUNICATION.value: (
        "student_services_officer",
        (
            "summarise what would be said and to whom",
            "ask student services to make contact through their own channel",
            "record that the system only proposed the contact",
        ),
    ),
    RestrictedAction.DATA_EXPORT.value: (
        "data_protection_officer",
        (
            "name the fields, the destination, and the lawful basis",
            "ask the data protection officer to approve the transfer",
            "export through the governed pipeline, not from this service",
        ),
    ),
    RestrictedAction.PERMANENT_RECORD_CHANGE.value: (
        "records_manager",
        (
            "state what would be written and that the learner could not contest it",
            "ask the records manager to authorise the change",
            "apply it through the records system of record",
        ),
    ),
    RestrictedAction.CREDENTIAL_REVOCATION.value: (
        "registrar",
        (
            "record the specific access right or qualification affected",
            "ask the registrar to confirm the revocation is warranted",
            "revoke through the identity provider, never from here",
        ),
    ),
}

#: Human-readable reasons, keyed by why a context escalated. Kept as data rather than
#: string literals at the call site so the ledger and the UI render the same words.
_IRREVERSIBLE_REASON = (
    "the context is irreversible: the learner cannot undo what happens here, so the "
    "decision is a person's to make regardless of how much the system's history vouches "
    "for it"
)
_HIGH_STAKES_REASON = (
    "the context feeds a decision with consequences beyond this session, so a system "
    "with no track record must not act on it alone"
)
_HIGH_RISK_REASON = (
    "the context is high risk: acting here can affect a learner's standing, record, or "
    "access, and only the permitted subset may proceed without a person"
)
_TRUSTED_REASON = (
    "the context is low risk and reversible, and the authority weight is above the "
    "autonomy floor, so the permitted actions may proceed without a person"
)
_DISTRUSTED_REASON = (
    "the authority weight has fallen to or below the autonomy floor, so the system has "
    "been contradicted often enough to stop acting alone even where it would otherwise "
    "be harmless"
)


@dataclass(frozen=True, slots=True)
class AuthorityAssessment:
    """The envelope together with everything that produced it.

    The envelope is the grant; this is the audit. A reviewer who receives only the
    envelope cannot tell whether a level of ``RESTRICTED`` came from a risky context or
    from a history of disputes, and those two cases call for completely different
    responses from whoever is reading. Keeping the decay report alongside the envelope
    makes the distinction visible instead of leaving it to be guessed.

    Attributes:
        envelope: The grant being issued.
        decay: The derivation of the confidence weight, term by term.
        memory_entries_counted: How many history entries contributed to the weight.
        permission: The per-action resolution of the envelope's level.
    """

    envelope: AuthorityEnvelope
    decay: DecayReport
    memory_entries_counted: int
    permission: dict[str, ActionPermission]

    def to_summary_dict(self) -> dict[str, object]:
        """Render the assessment as flat, loggable fields.

        Returns:
            A dict carrying the envelope, the decay derivation, and the per-action
            resolution.
        """
        return {
            "envelope": self.envelope.to_summary_dict(),
            "decay": self.decay.to_summary_dict(),
            "memory_entries_counted": self.memory_entries_counted,
            "permission": {name: value.value for name, value in self.permission.items()},
        }


class AuthorityEngine:
    """Issues scoped authority envelopes and records everything it does.

    Args:
        settings: Authority thresholds. Defaults to
            :class:`~focus_engine.configuration.thresholds.AuthoritySettings`.
        clock: The time source for decay, expiry, and ledger timestamps. Injected so the
            whole layer is reproducible under a
            :class:`~focus_engine.utils.clock.FixedClock`.
        data_origin: Whether the evidence behind grants is real or synthetic. The Lab uses
            :attr:`~focus_engine.schemas.primitives.DataOrigin.SYNTHETIC`; a production
            deployment would use ``REAL`` and the difference is carried on every envelope
            so a reviewer can tell a demonstration from a decision.

    Attributes:
        ledger: The append-only record. Exposed read-only by convention; the engine is the
            only writer.
    """

    __slots__ = ("_clock", "_decay", "_ledger", "_memories", "_origin", "_safety", "_settings")

    def __init__(
        self,
        settings: AuthoritySettings | None = None,
        *,
        clock: Clock,
        data_origin: DataOrigin = DataOrigin.SYNTHETIC,
    ) -> None:
        """Initialise the engine with an empty ledger and no memories.

        Args:
            settings: Authority thresholds.
            clock: The time source.
            data_origin: Whether evidence behind grants is real or synthetic.
        """
        self._settings = AuthoritySettings() if settings is None else settings
        self._clock = clock
        self._decay = DecayEngine(self._settings, clock=clock)
        self._safety = SafetyGate(
            high_impact_min_authority=self._settings.high_impact_min_authority
        )
        self._ledger = AuthorityLedger(max_entries=self._settings.max_ledger_entries)
        self._memories: dict[str, InterventionOutcomeMemory] = {}
        self._origin = data_origin

    @property
    def settings(self) -> AuthoritySettings:
        """Return the configured thresholds."""
        return self._settings

    @property
    def ledger(self) -> AuthorityLedger:
        """Return the append-only authority ledger."""
        return self._ledger

    def memory_for(self, learner_id: str, context_type: str) -> ContextMemory:
        """Return one learner's history in one context.

        Args:
            learner_id: The learner.
            context_type: The context to look up.

        Returns:
            The stored memory, or an empty one. An empty result is the correct reading of
            "no history", not a missing record.
        """
        store = self._memories.get(learner_id)
        if store is None:
            return ContextMemory(learner_id=learner_id, context_type=context_type)
        return store.for_context(context_type)

    def authorize(
        self,
        learner_id: str,
        context: ContextDescriptor,
        confidence: ConfidenceLevel | None,
        *,
        proposed_actions: tuple[str, ...] = (),
    ) -> AuthorityAssessment:
        """Issue a scoped authority envelope for one decision.

        This is the layer's read path. It never mutates memory and never reuses a previous
        envelope, so calling it twice with the same inputs at the same instant produces
        byte-identical output.

        Args:
            learner_id: The learner the decision concerns.
            context: Where the action would take effect, and how risky that is.
            confidence: The uncertainty band reported upstream. A band of ``None`` or
                ``UNKNOWN`` yields a zero ceiling, which means no autonomous action,
                because absent confidence is absent evidence.
            proposed_actions: The actions the policy is considering. Used to resolve
                per-action permissions; an action not proposed is still reported as
                withheld so a reader can see it was considered.

        Returns:
            An :class:`AuthorityAssessment` carrying the envelope, the decay derivation,
            and the per-action resolution.
        """
        now = self._clock.now()
        memory = self.memory_for(learner_id, context.context_type)
        decay = self._decay.evaluate(memory, context, confidence)

        level, level_restrictions = self._resolve_level(context, decay)
        withheld = self._withheld_actions(context)
        permitted = self._permitted_actions(context, level, level_restrictions)
        required_approvals = self._required_approvals(withheld)
        escalation = self._escalation_path(withheld, context)

        envelope = AuthorityEnvelope(
            authority_level=level,
            context=context,
            permitted_actions=permitted,
            restricted_actions=withheld,
            confidence_weight=decay.weight,
            restrictions=level_restrictions,
            required_approvals=required_approvals,
            escalation_path=escalation,
            calculated_at=now,
            expires_at=self._expiry_for(now),
            basis={
                "learner_id": learner_id,
                "context_type": context.context_type,
                "context_risk": context.risk.value,
                "confidence_level": "none" if confidence is None else confidence.value,
                "decay_binding": decay.binding,
                "memory_entries_counted": str(len(memory.entries)),
            },
            data_origin=self._origin,
        )

        permission = self._resolve_permissions(level, permitted, withheld, proposed_actions)
        assessment = AuthorityAssessment(
            envelope=envelope,
            decay=decay,
            memory_entries_counted=len(memory.entries),
            permission=permission,
        )

        self._ledger.append(
            event_type="authority_calculation",
            learner_id=learner_id,
            context_type=context.context_type,
            decision_ref=None,
            detail=assessment.to_summary_dict(),
            recorded_at=now,
        )
        return assessment

    def record_feedback(
        self,
        feedback: HumanFeedback,
        *,
        learner_id: str,
        action: str,
    ) -> tuple[FeedbackOutcome, str]:
        """Apply human feedback to memory and the ledger.

        This is a correction *about the authority layer*, not a training signal about the
        model. It lowers the weight in this context and records the disagreement. It never
        touches model parameters, and the return value says so explicitly: a caller that
        wanted a retrain has to go and do a versioned offline retrain, which is the only
        process allowed to learn from corrections.

        Args:
            feedback: The feedback as submitted.
            learner_id: The learner affected.
            action: The action type the decision concerned.

        Returns:
            A tuple of the :class:`~focus_engine.authority.feedback.FeedbackOutcome` and a
            human-readable explanation.
        """
        now = self._clock.now()
        payload = feedback.to_summary_dict()

        if not feedback.context_type.strip():
            outcome, detail = FeedbackOutcome.REJECTED, "the feedback named no context"
            self._ledger.append(
                event_type="feedback_rejected",
                learner_id=learner_id,
                context_type=None,
                decision_ref=feedback.decision_ref,
                detail={**payload, "rejection": detail},
                recorded_at=now,
            )
            return outcome, detail

        if not feedback.reduces_authority():
            # A confirmation is recorded and changes nothing. It is still written to the
            # ledger because "a human agreed and nothing happened" is a fact a reviewer
            # needs, and leaving it out would make silence ambiguous.
            self._ledger.append(
                event_type="feedback_received",
                learner_id=learner_id,
                context_type=feedback.context_type,
                decision_ref=feedback.decision_ref,
                detail={**payload, "authority_effect": "none", "retrained": False},
                recorded_at=now,
            )
            return (
                FeedbackOutcome.ACCEPTED,
                "the person confirmed the decision; recorded, with no change to authority "
                "and no model update",
            )

        entry = feedback.to_memory_entry(action)
        self._record_entry(learner_id, entry)

        if feedback.feedback_type is FeedbackType.OVERRIDE:
            detail = (
                "the person overrode the decision; authority in this context is reduced "
                "and the case is queued for review. No model update was performed."
            )
        else:
            detail = (
                "the disagreement is recorded and authority in this context is reduced. "
                "No model update was performed; retraining is a separate, versioned, "
                "offline process."
            )

        self._ledger.append(
            event_type="feedback_applied",
            learner_id=learner_id,
            context_type=feedback.context_type,
            decision_ref=feedback.decision_ref,
            detail={
                **payload,
                "action": action,
                "authority_effect": "reduced",
                "retrained": False,
                "memory_entry": {
                    "action": entry.action,
                    "context_type": entry.context_type,
                    "outcome": entry.outcome,
                    "weight": entry.weight,
                    "recorded_at": entry.recorded_at.isoformat(),
                },
                "detail": detail,
            },
            recorded_at=now,
        )
        return FeedbackOutcome.ACCEPTED, detail

    def record_disagreement(
        self,
        record: DisagreementRecord,
        *,
        learner_id: str,
    ) -> None:
        """Append a disagreement record for slower, aggregate review.

        Args:
            record: The disagreement to log.
            learner_id: The learner affected, for the ledger's index.

        Raises:
            ValueError: If ``record.learner_id`` disagrees with ``learner_id``. A
                disagreement filed under the wrong learner would aggregate into the wrong
                pattern and quietly corrupt the review it exists to support.
        """
        if record.learner_id != learner_id:
            raise ValueError(
                f"disagreement names learner {record.learner_id!r} but was filed under "
                f"{learner_id!r}"
            )
        self._ledger.append(
            event_type="aggregated_disagreement",
            learner_id=learner_id,
            context_type=record.context_type,
            decision_ref=record.feedback_id,
            detail=record.to_summary_dict(),
            recorded_at=self._clock.now(),
        )

    def record_outcome(
        self,
        learner_id: str,
        action: str,
        context_type: str,
        outcome: str,
        *,
        weight: float = 1.0,
    ) -> MemoryEntry:
        """Record a measured outcome so later decisions can be priced against it.

        Args:
            learner_id: The learner the intervention was delivered to.
            action: The action type delivered.
            context_type: The context it was delivered in. Part of the key: an outcome in
                one context never informs another.
            outcome: ``improved``, ``deteriorated``, ``no_response``, or any other string.
                Unrecognised values are penalised by the decay engine rather than ignored.
            weight: How much this observation counts, in ``[0, 1]``.

        Returns:
            The stored :class:`~focus_engine.authority.memory.MemoryEntry`.
        """
        entry = MemoryEntry(
            action=action,
            context_type=context_type,
            recorded_at=self._clock.now(),
            outcome=outcome,
            weight=weight,
        )
        self._record_entry(learner_id, entry)
        return entry

    def finalise(
        self,
        decision: PolicyDecision,
        envelope: AuthorityEnvelope,
        availability: HumanAvailability,
    ) -> FinalAction:
        """Pass a policy decision through the safety gate.

        The last step before anything may happen. A refusal here is final and is recorded
        to the ledger, because the refusals are what a reviewer most needs to see.

        Args:
            decision: The policy decision.
            envelope: The envelope issued by :meth:`authorize`.
            availability: Whether a human who could approve is reachable.

        Returns:
            The resolved :class:`~focus_engine.authority.safety.FinalAction`.
        """
        now = self._clock.now()
        final = self._safety.evaluate(decision, envelope, availability, as_of=now)
        learner_id = envelope.basis.get("learner_id", "unknown")
        self._ledger.append(
            event_type="safety_evaluated",
            learner_id=learner_id,
            context_type=envelope.context.context_type,
            decision_ref=decision.decision.value,
            detail={
                "final_action": final.to_summary_dict(),
                "policy_reason": None if decision.reason is None else decision.reason.value,
                "policy_selected": (
                    None if decision.selected is None else decision.selected.intervention_type
                ),
            },
            recorded_at=now,
        )
        return final

    def integrity(self) -> LedgerIntegrity:
        """Verify the authority ledger's hash chain.

        Returns:
            A :class:`~focus_engine.authority.ledger.LedgerIntegrity` report. The reader
            checks, not the writer: a writer that validated its own output would be
            auditing itself.
        """
        return self._ledger.integrity_ok()

    # --- internals ---

    def _resolve_level(
        self, context: ContextDescriptor, decay: DecayReport
    ) -> tuple[AuthorityLevel, tuple[str, ...]]:
        """Choose the authority level, and record why each escalation happened.

        The order is deliberate: the structural facts of the context are checked before
        the weight. An irreversible high-stakes context escalates to
        ``HUMAN_REQUIRED`` whether the system has a perfect record or a poor one, because
        the argument against autonomy there is about the action, not about the operator.

        Args:
            context: The context being assessed.
            decay: The computed weight, used only for the low-risk case.

        Returns:
            A tuple of the level and the human-readable restrictions that produced it.
        """
        restrictions: list[str] = []

        if not context.is_reversible:
            restrictions.append(_IRREVERSIBLE_REASON)
        if context.is_high_stakes:
            restrictions.append(_HIGH_STAKES_REASON)
        if context.risk is ContextRisk.HIGH:
            restrictions.append(_HIGH_RISK_REASON)

        if restrictions:
            return AuthorityLevel.HUMAN_REQUIRED, tuple(restrictions)

        if decay.weight < self._settings.autonomy_min_weight:
            return AuthorityLevel.RESTRICTED, (_DISTRUSTED_REASON,)

        if context.risk is ContextRisk.MEDIUM:
            return AuthorityLevel.RESTRICTED, (_HIGH_RISK_REASON,)

        return AuthorityLevel.ALLOWED, (_TRUSTED_REASON,)

    def _withheld_actions(self, context: ContextDescriptor) -> tuple[WithheldAction, ...]:
        """Build the explicit list of actions this envelope refuses.

        Every restricted action is withheld in every context. The *reasons* differ by
        context, which is the useful part: a grade modification is refused in free
        practice because it is not applicable there, and refused in a graded assessment
        because it is irreversible. A reader who can only see that the action was refused
        cannot tell those apart.

        Args:
            context: The context being assessed.

        Returns:
            One :class:`~focus_engine.authority.types.WithheldAction` per restricted action.
        """
        withheld: list[WithheldAction] = []
        for action in RestrictedAction:
            role, path = ESCALATION_CATALOGUE[action.value]
            if action is RestrictedAction.GRADE_MODIFICATION and not context.is_high_stakes:
                reason = (
                    "the action would alter a recorded mark and is never taken by this "
                    "system; escalation names the role that can, rather than offering a "
                    "self-service path to a grade change"
                )
            elif not context.is_reversible:
                reason = (
                    f"the context is irreversible, so {action.value!r} cannot be "
                    "undone by the learner and is not the system's to decide"
                )
            elif context.is_high_stakes:
                reason = (
                    f"the context is high-stakes, so {action.value!r} carries "
                    "consequences beyond this session and requires a named approver"
                )
            else:
                reason = (
                    f"{action.value!r} is outside what this system does autonomously in any context"
                )
            withheld.append(
                WithheldAction(
                    action=action,
                    reason=reason,
                    required_approval=role,
                    escalation_path=path,
                )
            )
        return tuple(withheld)

    def _permitted_actions(
        self,
        context: ContextDescriptor,
        level: AuthorityLevel,
        restrictions: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return the actions this envelope may execute without a person.

        Args:
            context: The context being assessed.
            level: The resolved level.
            restrictions: The recorded restrictions, used to explain an empty grant.

        Returns:
            A sorted tuple of permitted action types. Empty for
            ``HUMAN_REQUIRED``, and an empty tuple is itself informative: it means the
            grant exists to be escalated, not to be used.
        """
        if level is not AuthorityLevel.HUMAN_REQUIRED:
            return tuple(sorted(DEFAULT_PERMITTED_ACTIONS))
        del context, restrictions  # unused here, kept for signature symmetry
        return ()

    def _required_approvals(self, withheld: tuple[WithheldAction, ...]) -> tuple[str, ...]:
        """List the distinct roles that could authorise a withheld action.

        Args:
            withheld: The withheld actions.

        Returns:
            Unique role names, sorted, so the ledger is stable between runs.
        """
        return tuple(
            sorted({item.required_approval for item in withheld if item.required_approval})
        )

    def _escalation_path(
        self, withheld: tuple[WithheldAction, ...], context: ContextDescriptor
    ) -> tuple[str, ...]:
        """Assemble the ordered route to follow when a decision needs a person.

        Args:
            withheld: The withheld actions.
            context: The context being assessed.

        Returns:
            Ordered steps, deduplicated while preserving order. Empty for a context that
            required no escalation.
        """
        del context  # the per-action steps are already context-specific
        steps: list[str] = []
        for item in withheld:
            for step in item.escalation_path:
                if step not in steps:
                    steps.append(step)
        return tuple(steps)

    def _resolve_permissions(
        self,
        level: AuthorityLevel,
        permitted: tuple[str, ...],
        withheld: tuple[WithheldAction, ...],
        proposed: tuple[str, ...],
    ) -> dict[str, ActionPermission]:
        """Resolve each action to permitted, withheld, or needs-approval.

        Args:
            level: The resolved authority level.
            permitted: Actions the envelope clears.
            withheld: Actions the envelope refuses.
            proposed: Actions the policy is actually considering. Included so that an
                action nobody proposed is still reported, rather than vanishing from the
                audit.

        Returns:
            A mapping from action name to its :class:`ActionPermission`.
        """
        withheld_names = {item.action.value for item in withheld}
        names = set(permitted) | withheld_names | set(proposed)
        resolution: dict[str, ActionPermission] = {}
        for name in sorted(names):
            if name in withheld_names:
                resolution[name] = ActionPermission.WITHHELD
            elif level is AuthorityLevel.HUMAN_REQUIRED or level is AuthorityLevel.RESTRICTED:
                resolution[name] = ActionPermission.NEEDS_APPROVAL
            else:
                resolution[name] = ActionPermission.PERMITTED
        return resolution

    def _record_entry(self, learner_id: str, entry: MemoryEntry) -> None:
        """File an entry in the learner's memory and note it in the ledger.

        Args:
            learner_id: The learner.
            entry: The entry to file.
        """
        store = self._memories.get(learner_id)
        if store is None:
            store = InterventionOutcomeMemory(learner_id=learner_id)
        self._memories[learner_id] = store.record(entry)
        self._ledger.append(
            event_type="decay_applied",
            learner_id=learner_id,
            context_type=entry.context_type,
            decision_ref=None,
            detail={
                "action": entry.action,
                "outcome": entry.outcome,
                "weight": entry.weight,
                "recorded_at": entry.recorded_at.isoformat(),
            },
            recorded_at=self._clock.now(),
        )

    def _expiry_for(self, now: datetime) -> datetime:
        """Return when an envelope minted at ``now`` stops being valid.

        Args:
            now: The minting instant.

        Returns:
            ``now`` plus :attr:`AuthoritySettings.expiry_hours`.
        """
        return now + timedelta(hours=self._settings.expiry_hours)
