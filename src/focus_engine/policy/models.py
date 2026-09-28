"""The vocabulary of a policy decision, and the rules that keep it self-consistent.

A policy decision is the answer to one question: *should something be done, and if so
what?* It is deliberately a separate record from the prediction it consumed, for the
reason the architecture states plainly - when an outcome is bad, the only way to tell
whether the prediction, the policy, or the delivery was at fault is to be able to read
those three apart afterwards.

Two properties are enforced here rather than described, because a decision that
contradicts itself is worse than no decision at all:

- ``INTERVENE`` must name a candidate, and ``NO_INTERVENTION`` must not. A decision that
  says "no" while carrying a selected candidate invites a caller to act on it anyway.
- A decision that declined to act must say which restraint bound it. A refusal with no
  stated cause cannot be distinguished from a bug, so it is refused at construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from focus_engine.schemas.primitives import (
    BehavioralEngagementState,
    ConfidenceLevel,
    DataOrigin,
)
from focus_engine.schemas.versioning import PolicyVersion
from focus_engine.uncertainty.outcomes import Verdict

__all__ = [
    "ACTIONABLE_STATES",
    "CONFIDENCE_RANK",
    "POLICY_V1",
    "InterventionCandidate",
    "NonActionReason",
    "PolicyDecision",
    "PolicyDecisionType",
    "Restraint",
    "RestraintKind",
    "meets_confidence",
]

#: The first and only policy definition. Bumping this is required before changing any
#: restraint, candidate, or state rule, so that a recorded decision can always be read
#: back under the rules that produced it.
POLICY_V1: Final[PolicyVersion] = "POLICY_V1"

#: Relative strength of the confidence bands, used to answer "at least this band?".
#:
#: This ordering is the *policy's* requirement, not a second derivation of the
#: uncertainty engine's bands. The engine decides which band a value falls in; the policy
#: separately decides which bands are strong enough to justify interrupting a learner.
#: Keeping the two apart is what stops a change to band boundaries from silently
#: loosening the policy's restraint.
CONFIDENCE_RANK: Final[Mapping[ConfidenceLevel, int]] = MappingProxyType(
    {
        ConfidenceLevel.UNKNOWN: 0,
        ConfidenceLevel.INSUFFICIENT_DATA: 0,
        ConfidenceLevel.LOW: 1,
        ConfidenceLevel.MEDIUM: 2,
        ConfidenceLevel.HIGH: 3,
    }
)

#: The behavioural states in which an intervention may be justified.
#:
#: ``STABLE`` is absent because a learner whose behaviour already matches their own recent
#: history has nothing to be helped with, and ``RECOVERING`` is absent because that
#: learner is already returning towards baseline without help - interrupting them buys
#: nothing and costs their attention. Excluding both is the single most important
#: expression of "optimised for useful intervention, not maximal intervention": the
#: states a maximal policy would act on are precisely the states where acting is waste.
ACTIONABLE_STATES: Final[frozenset[BehavioralEngagementState]] = frozenset(
    {
        BehavioralEngagementState.DECLINING,
        BehavioralEngagementState.INCREASING_DEVIATION,
        BehavioralEngagementState.HIGH_DEVIATION,
    }
)


def meets_confidence(observed: ConfidenceLevel, required: ConfidenceLevel) -> bool:
    """Whether an observed band is at least as strong as a required band.

    Args:
        observed: The band the uncertainty engine assigned.
        required: The band the policy insists on.

    Returns:
        ``True`` when ``observed`` ranks at or above ``required``. The two refusal bands
        rank equal to each other and below every numeric band, so requiring ``LOW`` can
        never be satisfied by ``UNKNOWN``.
    """
    return CONFIDENCE_RANK[observed] >= CONFIDENCE_RANK[required]


class PolicyDecisionType(StrEnum):
    """Whether the policy chose to act.

    ``NO_INTERVENTION`` is a member of the decision vocabulary, not an exception and not a
    ``None``. A system that can only return a decision when it decides to intervene has no
    way to record restraint, and restraint that leaves no trace is restraint nobody can
    audit. Most decisions are expected to be ``NO_INTERVENTION``.
    """

    INTERVENE = "intervene"
    """A candidate was selected and may be delivered."""

    NO_INTERVENTION = "no_intervention"
    """Nothing will be done, and the decision says which restraint bound it."""


class NonActionReason(StrEnum):
    """The one restraint that prevented an intervention.

    Each member corresponds to exactly one :class:`RestraintKind`, so a consumer
    translating a refusal into a metric never has to pattern-match free text.
    """

    NOT_RESOLVED = "not_resolved"
    """The upstream outcome was a refusal, so there was no probability to weigh."""

    PROBABILITY_BELOW_FLOOR = "probability_below_floor"
    """The probability did not clear the floor, or the model did not cross its threshold."""

    CONFIDENCE_TOO_LOW = "confidence_too_low"
    """The confidence band was below the minimum the policy requires."""

    STATE_INSUFFICIENT = "state_insufficient"
    """The behavioural state could not be established."""

    STATE_NOT_INDICATED = "state_not_indicated"
    """The state was established and did not warrant an intervention."""

    MINIMUM_EVIDENCE_NOT_MET = "minimum_evidence_not_met"
    """Too few observations stood behind the decision to justify acting."""

    IN_COOLDOWN = "in_cooldown"
    """An intervention was delivered to this learner too recently."""

    SESSION_CAP_REACHED = "session_cap_reached"
    """This session has already used its intervention budget."""

    WINDOW_CAP_REACHED = "window_cap_reached"
    """The sliding window has already used its intervention budget."""

    REPEATED_TYPE_LIMIT = "repeated_type_limit"
    """Every remaining candidate was delivered too many times in a row."""

    TYPE_RETIRED = "type_retired"
    """Every remaining candidate was retired after repeated observed non-responses."""

    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"
    """No candidate applies to this behavioural state."""


class RestraintKind(StrEnum):
    """Which restraint was evaluated."""

    PROBABILITY_FLOOR = "probability_floor"
    CONFIDENCE_FLOOR = "confidence_floor"
    STATE_RULE = "state_rule"
    MINIMUM_EVIDENCE = "minimum_evidence"
    COOLDOWN = "cooldown"
    SESSION_CAP = "session_cap"
    WINDOW_CAP = "window_cap"
    REPEATED_TYPE_LIMIT = "repeated_type_limit"
    TYPE_RETIRED = "type_retired"
    CANDIDATE_APPLICABILITY = "candidate_applicability"


@dataclass(frozen=True, slots=True)
class Restraint:
    """One restraint, with the numbers it was judged against.

    Recording the limit and the observed value - not merely the fact that a restraint
    existed - is what makes a restraint reviewable. "Cooldown blocked it" and "cooldown
    blocked it with 40 seconds left" call for different responses from whoever is tuning
    the policy, and only the second one can be acted on.

    Attributes:
        kind: Which restraint this is.
        limit: The configured bound. Seconds for a cooldown, a count for a cap, a
            probability for a floor.
        observed: The measured value at decision time.
        detail: A sentence explaining the comparison in domain terms.
        binding: Whether this restraint was the one that prevented the intervention. A
            decision carries at least one binding restraint when it declines to act, and
            exactly one, so that a refusal can never be attributed to two causes at once.
    """

    kind: RestraintKind
    limit: float
    observed: float
    detail: str
    binding: bool = False

    @property
    def margin(self) -> float:
        """The signed difference between the limit and the observation.

        Returns:
            ``limit - observed``, unclamped and direction-agnostic. This is the number
            that answers "how far from the limit was it" for *any* restraint: ``0.25`` for
            a probability that fell short of its floor, ``-2`` for a cap that was exceeded
            by two. A clamped reading cannot serve both, because for a cap "past the
            limit" means zero headroom while for a floor it means the observation is the
            thing that is too small.
        """
        return self.limit - self.observed

    @property
    def headroom(self) -> float:
        """How much capacity is left before this restraint binds.

        Returns:
            ``margin`` clamped at zero. Meaningful only for restraints whose limit is an
            upper bound - caps and cooldowns. For a floor restraint, which binds when the
            observation is *below* the limit, the informative number is :attr:`margin`.
        """
        return max(0.0, self.margin)

    def describe(self) -> str:
        """Render the restraint as a short human-readable line.

        Returns:
            A one-line description naming the restraint, the limit, and the observation.
        """
        marker = "binding" if self.binding else "clear"
        return f"[{marker}] {self.kind.value}: {self.detail} (limit {self.limit:g}, observed {self.observed:g})"


@dataclass(frozen=True, slots=True)
class InterventionCandidate:
    """A thing the policy is allowed to choose, described rather than delivered.

    The policy names a candidate; it does not construct the artefact, schedule it, or
    tell a learner anything. That separation is what lets a decision be replayed and
    compared without the delivery side having to be reconstructed.

    Attributes:
        intervention_type: The controlled-vocabulary label the intervention engine owns.
        priority: Lower sorts first. Ties are impossible within a catalogue, and the
            catalogue is rejected at construction if two entries share a priority, so
            selection cannot depend on dictionary order.
        min_confidence: The weakest band this particular candidate will accept. A
            disruptive intervention should be able to demand a higher bar than a gentle
            one, which is why the floor is per-candidate and not only global.
        applies_to_states: The states this candidate is suitable for. Empty means every
            actionable state.
    """

    intervention_type: str
    priority: int = 0
    min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    applies_to_states: tuple[BehavioralEngagementState, ...] = ()

    def __post_init__(self) -> None:
        """Validate the candidate.

        Raises:
            ValueError: If the type label is blank, or the candidate claims to apply to
                a state that never warrants an intervention - which would let a
                configuration reintroduce the "act on a stable learner" behaviour the
                state rule exists to prevent.
        """
        if not self.intervention_type.strip():
            raise ValueError("an intervention candidate must have a non-empty type label")
        if not set(self.applies_to_states).issubset(ACTIONABLE_STATES):
            raise ValueError(
                "a candidate may only apply to states that warrant an intervention; got "
                f"{sorted(state.value for state in self.applies_to_states)}"
            )

    def applies_to(self, state: BehavioralEngagementState) -> bool:
        """Whether this candidate is suitable for a state.

        Args:
            state: The observed behavioural state.

        Returns:
            ``True`` when the candidate has no state restriction, or lists this state.
        """
        return not self.applies_to_states or state in self.applies_to_states

    def describe(self) -> str:
        """Render the candidate as a short human-readable line.

        Returns:
            A one-line description naming the type, its priority, and its confidence bar.
        """
        states = (
            "any actionable state"
            if not self.applies_to_states
            else "|".join(sorted(state.value for state in self.applies_to_states))
        )
        return (
            f"{self.intervention_type} (priority {self.priority}, "
            f"min {self.min_confidence.value}, {states})"
        )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A complete answer from the policy layer, including when it declines.

    Attributes:
        decision: Whether the policy chose to act.
        selected: The chosen candidate, present only when acting.
        reason: The restraint that bound the decision, present only when not acting.
        detail: A sentence a human can read, present in every case.
        restraints: Every restraint that was evaluated, binding or not, so a decision can
            be read as "this held, and this nearly held" rather than as a bare yes or no.
        excluded: The candidate types removed before selection, in the order removed.
        learner_id: The learner the decision is about.
        session_id: The session the decision was made in.
        policy_version: The policy definition in force.
        settings_fingerprint: A digest of the settings and catalogue, so a decision made
            under one configuration is not silently compared against another.
        evaluated_at: When the decision was made.
        upstream_verdict: The verdict the policy received, carried through so a decision
            can be traced back to the inference that prompted it.
        probability: The probability that prompted the decision, or ``None`` when the
            upstream outcome was a refusal.
        confidence: The confidence band that prompted the decision, likewise.
        data_origin: Whether the inputs behind this decision were real or synthetic.
    """

    decision: PolicyDecisionType
    selected: InterventionCandidate | None = None
    reason: NonActionReason | None = None
    detail: str = ""
    restraints: tuple[Restraint, ...] = ()
    excluded: tuple[str, ...] = ()
    learner_id: str = ""
    session_id: str = ""
    policy_version: PolicyVersion = POLICY_V1
    settings_fingerprint: str = ""
    evaluated_at: datetime | None = None
    upstream_verdict: Verdict | None = None
    probability: float | None = None
    confidence: ConfidenceLevel | None = None
    data_origin: DataOrigin = DataOrigin.SYNTHETIC

    def __post_init__(self) -> None:
        """Validate that the decision agrees with itself.

        Raises:
            ValueError: If the decision acts without naming a candidate, declines without
                naming a reason, claims exactly one binding restraint while declining, or
                carries a probability from an outcome that never issued one.
        """
        if self.decision is PolicyDecisionType.INTERVENE:
            if self.selected is None:
                raise ValueError(
                    "an INTERVENE decision must name the candidate it selected; a decision "
                    "to act that does not say what to do is not actionable"
                )
            if self.reason is not None:
                raise ValueError(
                    f"an INTERVENE decision must not carry a non-action reason; got "
                    f"{self.reason.value}"
                )
            return

        if self.selected is not None:
            raise ValueError(
                f"a NO_INTERVENTION decision must not carry a selected candidate; got "
                f"{self.selected.intervention_type}. A caller that reads the candidate "
                "from a decision to decline would still deliver it."
            )
        if self.reason is None:
            raise ValueError(
                "a NO_INTERVENTION decision must state which restraint bound it; an "
                "unexplained refusal cannot be distinguished from a bug"
            )
        if self.reason is not NonActionReason.NOT_RESOLVED and not any(
            restraint.binding for restraint in self.restraints
        ):
            raise ValueError(
                f"a decision to decline for {self.reason.value} must record the restraint "
                "that bound it; the reason and the restraint are the same fact stated "
                "twice, and disagreement between them would be undetectable"
            )
        binding = [restraint for restraint in self.restraints if restraint.binding]
        if len(binding) > 1:
            raise ValueError(
                f"a decision to decline must name exactly one binding restraint; got "
                f"{[restraint.kind.value for restraint in binding]}. Two binding "
                "restraints make the cause of the refusal ambiguous."
            )

    @property
    def is_intervention(self) -> bool:
        """Whether the policy chose to act."""
        return self.decision is PolicyDecisionType.INTERVENE

    @property
    def binding_restraint(self) -> Restraint | None:
        """The restraint that bound a refusal.

        Returns:
            The single binding restraint, or ``None`` when the policy acted or when the
            upstream outcome was a refusal and so no restraint was ever reached.
        """
        for restraint in self.restraints:
            if restraint.binding:
                return restraint
        return None

    def describe(self) -> str:
        """Render the decision as a short human-readable block.

        Returns:
            A multi-line summary of the decision, the reason, and every restraint.
        """
        lines = [f"{self.decision.value}: {self.detail}"]
        if self.selected is not None:
            lines.append(f"  selected {self.selected.describe()}")
        for restraint in self.restraints:
            lines.append(f"  {restraint.describe()}")
        if self.excluded:
            lines.append(f"  excluded candidates: {', '.join(self.excluded)}")
        return "\n".join(lines)

    def to_summary_dict(self) -> dict[str, object]:
        """Reduce the decision to flat, countable fields.

        Returns:
            A mapping with no nested containers, suitable for a counter or a metrics
            frame. Candidate details are counted rather than embedded so a summary of
            thousands of decisions stays small.
        """
        return {
            "decision": self.decision.value,
            "reason": self.reason.value if self.reason is not None else None,
            "selected_type": self.selected.intervention_type if self.selected else None,
            "selected_priority": self.selected.priority if self.selected else None,
            "learner_id": self.learner_id,
            "session_id": self.session_id,
            "policy_version": self.policy_version,
            "settings_fingerprint": self.settings_fingerprint,
            "evaluated_at": self.evaluated_at,
            "upstream_verdict": self.upstream_verdict.value if self.upstream_verdict else None,
            "probability": self.probability,
            "confidence": self.confidence.value if self.confidence is not None else None,
            "restraints_checked": len(self.restraints),
            "binding_restraint": (
                self.binding_restraint.kind.value if self.binding_restraint else None
            ),
            "excluded_count": len(self.excluded),
            "data_origin": self.data_origin.value,
        }
