"""The deterministic policy: whether to act, and under what restraints.

Everything here is a threshold comparison against a frozen configuration. There is no
learned component, no reward, and no state accumulated inside the policy object, so the
same inputs always produce the same decision. That is a deliberate limitation rather than
a shortfall: a policy that adapts to live responses is a policy whose decisions cannot be
explained after the fact, and the roadmap defers reinforcement learning until a
deterministic policy has been shown to be insufficient.

The evaluation order is the substance of the module, so it is stated plainly. The
restraints fall into three groups, checked in this order:

1. **Is there a case for acting at all?** The upstream outcome must be a resolved
   probability; it must clear the probability floor; the confidence must meet the
   catalogue's weakest bar; the behavioural state must be one that warrants help; and
   enough evidence must stand behind the decision to justify interrupting someone.
2. **Is acting permitted right now?** The cooldown, the per-session cap, and the
   sliding-window cap.
3. **Which candidate, if any, survives?** The repeat limit, retirement after repeated
   observed non-responses, state applicability, and each candidate's own confidence bar.

The order decides which reason gets reported, and the reported reason is what anyone
tuning this policy will be reading. Cheap checks come first, and the "nothing to act on"
group precedes the "not allowed to act now" group, because "the learner is doing fine"
is the more informative answer when both are true.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final

from focus_engine.configuration.thresholds import InterventionSettings
from focus_engine.policy.history import InterventionHistory
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
from focus_engine.schemas.primitives import BehavioralEngagementState, ConfidenceLevel
from focus_engine.schemas.versioning import PolicyVersion
from focus_engine.uncertainty.outcomes import PredictionOutcome, Verdict
from focus_engine.utils.clock import Clock, SystemClock
from focus_engine.utils.determinism import content_digest

__all__ = [
    "DEFAULT_CATALOGUE",
    "WINDOW",
    "InterventionPolicy",
]

#: The width of the sliding window the per-hour cap is applied over.
WINDOW: Final[timedelta] = timedelta(hours=1)

#: The default catalogue, ordered from gentlest to most disruptive.
#:
#: Selection walks this list in priority order and takes the first survivor, so the
#: ordering *is* the escalation policy: a learner is offered the least intrusive thing
#: that applies, and the more disruptive options are reached only when the earlier ones
#: are suppressed. The two disruptive entries also demand a higher confidence than the
#: gentle ones, which is the restraint the per-candidate bar exists to express.
DEFAULT_CATALOGUE: Final[tuple[InterventionCandidate, ...]] = (
    InterventionCandidate("recap", priority=0, min_confidence=ConfidenceLevel.MEDIUM),
    InterventionCandidate("micro_question", priority=1, min_confidence=ConfidenceLevel.MEDIUM),
    InterventionCandidate("checkpoint_prompt", priority=2, min_confidence=ConfidenceLevel.HIGH),
    InterventionCandidate("focus_quiz", priority=3, min_confidence=ConfidenceLevel.HIGH),
)

#: The reason each candidate-level filter produces when it empties the catalogue.
_CANDIDATE_REASONS: Final[Mapping[RestraintKind, NonActionReason]] = MappingProxyType(
    {
        RestraintKind.REPEATED_TYPE_LIMIT: NonActionReason.REPEATED_TYPE_LIMIT,
        RestraintKind.TYPE_RETIRED: NonActionReason.TYPE_RETIRED,
        RestraintKind.CANDIDATE_APPLICABILITY: NonActionReason.NO_ELIGIBLE_CANDIDATE,
        RestraintKind.CONFIDENCE_FLOOR: NonActionReason.CONFIDENCE_TOO_LOW,
    }
)

#: Which filter is blamed when several emptied the catalogue together. The most specific
#: cause wins, so a decision is never reported as "no candidate applies" when the real
#: story is that every candidate had already been repeatedly declined. Retirement outranks
#: the repeat limit because it explains *why*: a run of declines is a verdict on the type,
#: whereas the repeat limit only counts deliveries and would report a type the learner
#: never even saw as merely "delivered twice in a row".
_BLAME_PRECEDENCE: Final[tuple[RestraintKind, ...]] = (
    RestraintKind.TYPE_RETIRED,
    RestraintKind.REPEATED_TYPE_LIMIT,
    RestraintKind.CONFIDENCE_FLOOR,
    RestraintKind.CANDIDATE_APPLICABILITY,
)


@dataclass(frozen=True, slots=True)
class InterventionPolicy:
    """The deterministic policy, as a frozen value.

    Holding no per-learner state is the point. Two policies configured identically are
    interchangeable, a decision can be reproduced from its recorded inputs, and there is
    no hidden accumulator whose contents would have to be snapshotted to explain why the
    policy did what it did.

    Attributes:
        settings: The restraint configuration.
        catalogue: The candidates the policy may choose from, in priority order.
        clock: The source of the current instant.
        policy_version: The policy definition in force.
    """

    settings: InterventionSettings = field(default_factory=InterventionSettings)
    catalogue: tuple[InterventionCandidate, ...] = DEFAULT_CATALOGUE
    clock: Clock = field(default_factory=SystemClock)
    policy_version: PolicyVersion = POLICY_V1

    def __post_init__(self) -> None:
        """Validate the catalogue.

        Raises:
            ValueError: If the catalogue is empty, repeats a type label, or reuses a
                priority. Duplicate priorities would make selection depend on iteration
                order, which is exactly the hidden non-determinism this layer exists to
                exclude.
        """
        if not self.catalogue:
            raise ValueError(
                "a policy with an empty catalogue can never intervene; the catalogue is "
                "part of the policy definition, so an empty one is a configuration error "
                "rather than a policy that always declines"
            )
        types = [candidate.intervention_type for candidate in self.catalogue]
        duplicates = sorted({name for name in types if types.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"the catalogue repeats intervention type(s) {duplicates}; a type must "
                "name one thing or the repeat limit cannot be reasoned about"
            )
        priorities = [candidate.priority for candidate in self.catalogue]
        if len(set(priorities)) != len(priorities):
            raise ValueError(
                f"the catalogue reuses priorities {sorted(priorities)}; selection must be "
                "a total order, so every candidate needs a distinct priority"
            )

    @property
    def cooldown_seconds(self) -> float:
        """The cooldown expressed in seconds, the unit the history is measured in."""
        return self.settings.cooldown_minutes * 60.0

    @property
    def ordered_catalogue(self) -> tuple[InterventionCandidate, ...]:
        """The catalogue in priority order, gentlest first."""
        return tuple(sorted(self.catalogue, key=lambda item: item.priority))

    def settings_fingerprint(self) -> str:
        """Digest the policy definition.

        Returns:
            A digest over the version, the settings, and the catalogue. Recorded on every
            decision so a decision made under one configuration is never silently
            compared against another.
        """
        return content_digest(
            {
                "policy_version": self.policy_version,
                "settings": repr(self.settings),
                "catalogue": [
                    [
                        candidate.intervention_type,
                        candidate.priority,
                        candidate.min_confidence.value,
                        sorted(state.value for state in candidate.applies_to_states),
                    ]
                    for candidate in self.ordered_catalogue
                ],
            }
        )

    def catalogue_confidence_floor(self) -> ConfidenceLevel:
        """The weakest confidence any candidate in the catalogue will accept.

        Returns:
            The lowest per-candidate bar. Used as the policy-wide floor, so that a
            confidence refusal is reported only when literally nothing could have acted.
        """
        return min(
            (candidate.min_confidence for candidate in self.catalogue),
            key=lambda level: CONFIDENCE_RANK[level],
        )

    def eligible_candidates(
        self,
        outcome: PredictionOutcome,
        history: InterventionHistory,
        state: BehavioralEngagementState,
    ) -> tuple[tuple[InterventionCandidate, ...], tuple[RestraintKind, ...]]:
        """Filter the catalogue down to the candidates that may be selected.

        Args:
            outcome: The upstream outcome, already known to be resolved.
            history: The learner's delivery history, already truncated at the decision
                instant.
            state: The observed behavioural state.

        Returns:
            The surviving candidates in priority order, and the distinct restraint kinds
            that removed at least one candidate. The second element lets a caller explain
            an empty result without re-running the filter. A single candidate can fail more
            than one filter - the repeat limit and retirement both default to a run of two -
            and every applicable kind is reported rather than just the first one hit, so a
            suppressed type is not explained away by whichever filter happened to be
            evaluated first.
        """
        settings = self.settings
        retired = history.retired_types(settings.abandon_after_no_response_count)
        observed_band = outcome.confidence or ConfidenceLevel.UNKNOWN
        eligible: list[InterventionCandidate] = []
        removed: set[RestraintKind] = set()

        for candidate in self.ordered_catalogue:
            reasons: list[RestraintKind] = []
            if history.trailing_run_length(candidate.intervention_type) >= (
                settings.max_repeat_same_type
            ):
                reasons.append(RestraintKind.REPEATED_TYPE_LIMIT)
            if candidate.intervention_type in retired:
                reasons.append(RestraintKind.TYPE_RETIRED)
            if not candidate.applies_to(state):
                reasons.append(RestraintKind.CANDIDATE_APPLICABILITY)
            if not meets_confidence(observed_band, candidate.min_confidence):
                reasons.append(RestraintKind.CONFIDENCE_FLOOR)
            if reasons:
                removed.update(reasons)
            else:
                eligible.append(candidate)

        return tuple(eligible), tuple(item for item in _BLAME_PRECEDENCE if item in removed)

    def decide(
        self,
        outcome: PredictionOutcome,
        history: InterventionHistory,
        *,
        learner_id: str,
        session_id: str,
        state: BehavioralEngagementState,
        now: datetime | None = None,
    ) -> PolicyDecision:
        """Decide whether to intervene, and under which restraints.

        Args:
            outcome: The prediction outcome from the uncertainty engine. A refusal is
                never acted on, because there is no number to weigh.
            history: The learner's delivery history. Truncated at the decision instant, so
                a history containing later deliveries cannot influence this decision.
            learner_id: The learner the decision is about. Must match the history.
            session_id: The session the decision is being made in.
            state: The observed behavioural state.
            now: The decision instant. Defaults to the policy's clock.

        Returns:
            A :class:`PolicyDecision` that either selects a candidate or names the single
            restraint that prevented one.

        Raises:
            ValueError: If the learner does not match the history, or a resolved outcome
                arrives without a probability or a confidence band. Both are checked here
                rather than trusted, because acting on a decision built from mismatched
                inputs is unrecoverable and would otherwise surface as an intervention
                nobody can explain.
        """
        moment = now if now is not None else self.clock.now()
        settings = self.settings

        if learner_id != history.learner_id:
            raise ValueError(
                f"the decision is for learner {learner_id} but the history belongs to "
                f"{history.learner_id}; a decision computed from another learner's history "
                "would be wrong in a way that looks entirely plausible"
            )

        visible = history.through(moment)

        if outcome.verdict is not Verdict.RESOLVED:
            return PolicyDecision(
                decision=PolicyDecisionType.NO_INTERVENTION,
                reason=NonActionReason.NOT_RESOLVED,
                detail=(
                    f"the uncertainty engine returned {outcome.verdict.value} rather than a "
                    "probability, so there is nothing to weigh; its remedy is "
                    f"{outcome.remedy.value}"
                ),
                learner_id=learner_id,
                session_id=session_id,
                policy_version=self.policy_version,
                settings_fingerprint=self.settings_fingerprint(),
                evaluated_at=moment,
                upstream_verdict=outcome.verdict,
                data_origin=outcome.data_origin,
            )

        probability = outcome.probability
        band = outcome.confidence
        if probability is None or band is None:
            raise ValueError(
                "a resolved outcome must carry both a probability and a confidence band "
                f"before it reaches the policy; got probability={probability}, "
                f"confidence={band}"
            )

        # Two floors apply and the stricter one binds, so a policy cannot be configured
        # into acting below the model's own decision threshold.
        floor = max(settings.min_probability_to_act, outcome.threshold)
        catalogue_floor = self.catalogue_confidence_floor()
        restraints: list[Restraint] = [
            Restraint(
                kind=RestraintKind.PROBABILITY_FLOOR,
                limit=floor,
                observed=probability,
                detail=(
                    f"the calibrated probability was {probability:.4f} against a floor of "
                    f"{floor:.4f} (the stricter of the policy floor "
                    f"{settings.min_probability_to_act:.2f} and the model's own threshold "
                    f"{outcome.threshold:.2f})"
                ),
            ),
            Restraint(
                kind=RestraintKind.CONFIDENCE_FLOOR,
                limit=float(CONFIDENCE_RANK[catalogue_floor]),
                observed=float(CONFIDENCE_RANK[band]),
                detail=(
                    f"confidence was {band.value}; the policy requires at least "
                    f"{catalogue_floor.value} before any candidate may act"
                ),
            ),
            Restraint(
                kind=RestraintKind.STATE_RULE,
                limit=1.0,
                observed=1.0 if state in ACTIONABLE_STATES else 0.0,
                detail=f"the state was {state.value}",
            ),
            Restraint(
                kind=RestraintKind.MINIMUM_EVIDENCE,
                limit=float(settings.min_evidence_to_act),
                observed=float(outcome.evidence_units),
                detail=(
                    f"{outcome.evidence_units} observation(s) stood behind the decision "
                    f"against a policy floor of {settings.min_evidence_to_act}"
                ),
            ),
        ]

        def decline(reason: NonActionReason, binding: int, detail: str) -> PolicyDecision:
            return self._decline(
                reason=reason,
                binding=binding,
                detail=detail,
                restraints=restraints,
                outcome=outcome,
                learner_id=learner_id,
                session_id=session_id,
                evaluated_at=moment,
            )

        if probability < floor:
            return decline(
                NonActionReason.PROBABILITY_BELOW_FLOOR,
                0,
                f"the probability {probability:.4f} did not clear the floor {floor:.4f}",
            )

        if not meets_confidence(band, catalogue_floor):
            return decline(
                NonActionReason.CONFIDENCE_TOO_LOW,
                1,
                f"confidence was {band.value}, below the policy minimum of {catalogue_floor.value}",
            )

        if state is BehavioralEngagementState.INSUFFICIENT_DATA:
            return decline(
                NonActionReason.STATE_INSUFFICIENT,
                2,
                "the behavioural state could not be established, so the policy has no "
                "grounds to judge whether help is warranted",
            )

        if state not in ACTIONABLE_STATES:
            return decline(
                NonActionReason.STATE_NOT_INDICATED,
                2,
                f"the state was {state.value}, which does not warrant an intervention. "
                "Acting on a learner whose behaviour already matches their own history, or "
                "who is already recovering, costs their attention and buys nothing.",
            )

        if outcome.evidence_units < settings.min_evidence_to_act:
            return decline(
                NonActionReason.MINIMUM_EVIDENCE_NOT_MET,
                3,
                f"only {outcome.evidence_units} observation(s) stood behind this decision, "
                f"against a policy floor of {settings.min_evidence_to_act}",
            )

        elapsed = visible.seconds_since_last(moment)
        restraints.append(
            Restraint(
                kind=RestraintKind.COOLDOWN,
                limit=self.cooldown_seconds,
                observed=0.0 if elapsed is None else elapsed,
                detail=(
                    "this learner has never been intervened with"
                    if elapsed is None
                    else (
                        f"an intervention was delivered {elapsed:.0f}s ago and the "
                        f"cooldown is {self.cooldown_seconds:.0f}s"
                    )
                ),
            )
        )
        if elapsed is not None and elapsed < self.cooldown_seconds:
            return decline(
                NonActionReason.IN_COOLDOWN,
                len(restraints) - 1,
                f"the cooldown has {self.cooldown_seconds - elapsed:.0f}s left to run",
            )

        in_session = visible.count_in_session(session_id)
        restraints.append(
            Restraint(
                kind=RestraintKind.SESSION_CAP,
                limit=float(settings.max_interventions_per_session),
                observed=float(in_session),
                detail=(
                    f"{in_session} of {settings.max_interventions_per_session} permitted "
                    "interventions have been used in this session"
                ),
            )
        )
        if in_session >= settings.max_interventions_per_session:
            return decline(
                NonActionReason.SESSION_CAP_REACHED,
                len(restraints) - 1,
                f"this session has used all {settings.max_interventions_per_session} of its "
                "permitted interventions",
            )

        recent = visible.in_window(moment, WINDOW)
        restraints.append(
            Restraint(
                kind=RestraintKind.WINDOW_CAP,
                limit=float(settings.max_interventions_per_hour),
                observed=float(len(recent)),
                detail=(
                    f"{len(recent)} intervention(s) fell in the last hour against a cap of "
                    f"{settings.max_interventions_per_hour}"
                ),
            )
        )
        if len(recent) >= settings.max_interventions_per_hour:
            return decline(
                NonActionReason.WINDOW_CAP_REACHED,
                len(restraints) - 1,
                f"the hourly cap of {settings.max_interventions_per_hour} has been reached",
            )

        eligible, removed_kinds = self.eligible_candidates(outcome, visible, state)
        restraints.extend(self._candidate_restraints(visible, eligible, removed_kinds))
        if not eligible:
            blamed = next(
                (kind for kind in removed_kinds if kind in _CANDIDATE_REASONS),
                RestraintKind.CANDIDATE_APPLICABILITY,
            )
            index = next(
                position
                for position, item in enumerate(restraints)
                if item.kind is blamed and item.binding is False
            )
            names = ", ".join(sorted(_CANDIDATE_REASONS[kind].value for kind in removed_kinds))
            return decline(
                _CANDIDATE_REASONS[blamed],
                index,
                f"every candidate was filtered out ({names}), and no option remained",
            )

        selected = eligible[0]
        return PolicyDecision(
            decision=PolicyDecisionType.INTERVENE,
            selected=selected,
            detail=(
                f"acting: probability {probability:.4f}, confidence {band.value}, state "
                f"{state.value}, and every restraint was clear. {selected.intervention_type} "
                "is the least intrusive option that applies."
            ),
            restraints=tuple(restraints),
            excluded=tuple(
                candidate.intervention_type
                for candidate in self.ordered_catalogue
                if candidate is not selected
            ),
            learner_id=learner_id,
            session_id=session_id,
            policy_version=self.policy_version,
            settings_fingerprint=self.settings_fingerprint(),
            evaluated_at=moment,
            upstream_verdict=outcome.verdict,
            probability=probability,
            confidence=band,
            data_origin=outcome.data_origin,
        )

    def _decline(
        self,
        *,
        reason: NonActionReason,
        binding: int,
        detail: str,
        restraints: list[Restraint],
        outcome: PredictionOutcome,
        learner_id: str,
        session_id: str,
        evaluated_at: datetime,
    ) -> PolicyDecision:
        """Build a refusal in which exactly one restraint is marked binding.

        The binding restraint is identified by position rather than by kind, because a
        kind can legitimately appear twice - the confidence floor is checked both as a
        policy-wide gate and again per candidate - and marking by kind would then flag
        two restraints as binding, making the cause of the refusal ambiguous.

        Args:
            reason: The refusal reason to report.
            binding: The index of the restraint that bound the decision.
            detail: A sentence explaining the refusal.
            restraints: Every restraint evaluated, in the order they were checked.
            outcome: The upstream outcome, for the probability and band to carry through.
            learner_id: The learner the decision is about.
            session_id: The session the decision was made in.
            evaluated_at: The decision instant.

        Returns:
            A decision that declines to act, attributed to exactly one restraint.

        Raises:
            ValueError: If the index does not name a restraint. A refusal attributed to
                nothing is the specific defect this method exists to prevent.
        """
        if not 0 <= binding < len(restraints):
            raise ValueError(
                f"binding index {binding} names no restraint out of {len(restraints)}; a "
                "refusal must be attributable to a restraint that was actually evaluated"
            )
        marked = tuple(
            Restraint(
                kind=item.kind,
                limit=item.limit,
                observed=item.observed,
                detail=item.detail,
                binding=position == binding,
            )
            for position, item in enumerate(restraints)
        )
        return PolicyDecision(
            decision=PolicyDecisionType.NO_INTERVENTION,
            reason=reason,
            detail=detail,
            restraints=marked,
            excluded=(),
            learner_id=learner_id,
            session_id=session_id,
            policy_version=self.policy_version,
            settings_fingerprint=self.settings_fingerprint(),
            evaluated_at=evaluated_at,
            upstream_verdict=outcome.verdict,
            probability=outcome.probability,
            confidence=outcome.confidence,
            data_origin=outcome.data_origin,
        )

    def _candidate_restraints(
        self,
        history: InterventionHistory,
        eligible: Sequence[InterventionCandidate],
        removed_kinds: Sequence[RestraintKind],
    ) -> list[Restraint]:
        """Describe the candidate-level filters that ran, whether or not they bound.

        These are recorded even when they did not bind. "The disruptive option was
        suppressed" is part of why the gentler one was chosen, and leaving it out makes
        the choice look arbitrary to anyone reading the decision back.

        Args:
            history: The learner's truncated delivery history.
            eligible: The candidates that survived, for counting what was removed.
            removed_kinds: The filter kinds that removed at least one candidate.

        Returns:
            One restraint per filter that removed something, none marked binding - a
            candidate survived, so nothing here bound the decision.
        """
        settings = self.settings
        total = len(self.catalogue)
        kept = len(eligible)
        retired = history.retired_types(settings.abandon_after_no_response_count)
        restraints: list[Restraint] = []

        for kind in removed_kinds:
            worst: float
            if kind is RestraintKind.REPEATED_TYPE_LIMIT:
                worst = float(
                    max(
                        (
                            history.trailing_run_length(item.intervention_type)
                            for item in self.catalogue
                        ),
                        default=0,
                    )
                )
                limit = float(settings.max_repeat_same_type)
                detail = (
                    f"at least one candidate had already been delivered {worst} time(s) in a "
                    f"row, reaching the consecutive limit of {settings.max_repeat_same_type}"
                )
            elif kind is RestraintKind.TYPE_RETIRED:
                worst = float(
                    max(
                        (history.consecutive_non_responses(name) for name in retired),
                        default=0,
                    )
                )
                limit = float(settings.abandon_after_no_response_count)
                detail = (
                    f"{len(retired)} candidate type(s) were retired after {worst} consecutive "
                    f"observed non-response(s), reaching the threshold of "
                    f"{settings.abandon_after_no_response_count}"
                )
            elif kind is RestraintKind.CONFIDENCE_FLOOR:
                worst = float(
                    CONFIDENCE_RANK[self.catalogue_confidence_floor()]
                    - CONFIDENCE_RANK[ConfidenceLevel.LOW]
                )
                limit = float(CONFIDENCE_RANK[ConfidenceLevel.HIGH])
                detail = (
                    f"at least one candidate requires a higher confidence band than "
                    f"{ConfidenceLevel.LOW.value} to proceed"
                )
            else:
                worst = float(total - kept)
                limit = float(total)
                detail = f"{total - kept} of {total} candidates do not apply to this state"
            restraints.append(
                Restraint(kind=kind, limit=limit, observed=worst, detail=detail, binding=False)
            )
        return restraints
