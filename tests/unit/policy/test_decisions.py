"""Tests for policy decisions (Phase 10).

The exit criteria for this phase are mostly negative: the policy must decline in a dozen
specific situations, and each refusal must be attributable to one named restraint. These
tests therefore assert both halves - that the policy did not act, and that it can say why
- because a policy that silently declines passes every "did not act" assertion while
remaining impossible to tune.
"""

from __future__ import annotations

import pytest

from focus_engine.policy.models import (
    ACTIONABLE_STATES,
    InterventionCandidate,
    NonActionReason,
    PolicyDecisionType,
    RestraintKind,
    meets_confidence,
)
from focus_engine.schemas.primitives import BehavioralEngagementState, ConfidenceLevel
from focus_engine.uncertainty.outcomes import Verdict
from tests.unit.policy import scenarios as s

pytestmark = pytest.mark.unit


class TestActing:
    """When the policy acts, it acts on the least intrusive option available."""

    def test_a_clear_case_produces_an_intervention(self) -> None:
        decision = s.decide(s.policy())

        assert decision.decision is PolicyDecisionType.INTERVENE
        assert decision.is_intervention
        assert decision.selected is not None
        assert decision.selected.intervention_type == "recap"
        assert decision.reason is None

    def test_an_intervention_carries_no_non_action_reason(self) -> None:
        decision = s.decide(s.policy())

        assert decision.reason is None, (
            "a decision to act that also explains why it did not is reporting two "
            "contradictory things"
        )

    def test_the_gentlest_option_is_preferred_over_a_disruptive_one(self) -> None:
        decision = s.decide(s.policy())

        assert decision.selected is not None
        assert decision.selected.intervention_type == "recap"
        assert "focus_quiz" in decision.excluded

    def test_every_restraint_that_ran_is_recorded_even_when_clear(self) -> None:
        decision = s.decide(s.policy())

        kinds = {restraint.kind for restraint in decision.restraints}
        assert RestraintKind.PROBABILITY_FLOOR in kinds
        assert RestraintKind.CONFIDENCE_FLOOR in kinds
        assert RestraintKind.STATE_RULE in kinds
        assert RestraintKind.MINIMUM_EVIDENCE in kinds
        assert RestraintKind.COOLDOWN in kinds
        assert RestraintKind.SESSION_CAP in kinds
        assert RestraintKind.WINDOW_CAP in kinds
        assert not any(restraint.binding for restraint in decision.restraints), (
            "nothing bound an intervention, so no restraint may claim to have"
        )

    def test_the_decision_records_how_close_each_restraint_came(self) -> None:
        decision = s.decide(s.policy())

        session_cap = next(
            item for item in decision.restraints if item.kind is RestraintKind.SESSION_CAP
        )
        assert session_cap.limit == 3
        assert session_cap.observed == 0
        assert session_cap.headroom == 3

    def test_the_provenance_of_the_upstream_inference_travels_with_the_decision(self) -> None:
        decision = s.decide(s.policy())

        assert decision.upstream_verdict is Verdict.RESOLVED
        assert decision.probability == pytest.approx(0.9)
        assert decision.confidence is ConfidenceLevel.HIGH
        assert decision.data_origin is not None
        assert decision.policy_version == "POLICY_V1"
        assert decision.settings_fingerprint
        assert decision.evaluated_at == s.START


class TestNoInterventionIsFirstClass:
    """``NO_INTERVENTION`` is a value the policy returns, not a failure it raises."""

    def test_a_refused_inference_is_never_acted_on(self) -> None:
        decision = s.decide(s.policy(), outcome=s.insufficient_data())

        assert decision.decision is PolicyDecisionType.NO_INTERVENTION
        assert decision.reason is NonActionReason.NOT_RESOLVED
        assert decision.selected is None

    def test_an_untrustworthy_model_is_never_acted_on(self) -> None:
        decision = s.decide(s.policy(), outcome=s.unknown())

        assert decision.decision is PolicyDecisionType.NO_INTERVENTION
        assert decision.reason is NonActionReason.NOT_RESOLVED
        assert decision.selected is None

    def test_a_refusal_names_the_remedy_the_upstream_engine_offered(self) -> None:
        decision = s.decide(s.policy(), outcome=s.unknown())

        assert "replace_or_retrain_model" in decision.detail

    def test_a_refusal_on_a_refused_inference_carries_no_probability(self) -> None:
        decision = s.decide(s.policy(), outcome=s.insufficient_data())

        assert decision.probability is None
        assert decision.confidence is None
        assert "probability" not in decision.detail or "nothing to weigh" in decision.detail


class TestProbabilityFloor:
    def test_a_probability_below_the_floor_is_refused(self) -> None:
        decision = s.decide(s.policy(), outcome=s.resolved(probability=0.4))

        assert decision.reason is NonActionReason.PROBABILITY_BELOW_FLOOR
        assert decision.binding_restraint is not None
        assert decision.binding_restraint.kind is RestraintKind.PROBABILITY_FLOOR

    def test_the_floor_records_the_margin_that_was_missed(self) -> None:
        decision = s.decide(s.policy(), outcome=s.resolved(probability=0.4))

        restraint = decision.binding_restraint
        assert restraint is not None
        assert restraint.limit == pytest.approx(0.65)
        assert restraint.observed == pytest.approx(0.4)
        assert restraint.margin == pytest.approx(0.25)
        assert restraint.headroom == pytest.approx(0.25), (
            "headroom is clamped at zero and is only meaningful for an upper-bound "
            "restraint; for a floor the informative number is the signed margin"
        )

    def test_the_stricter_of_the_two_floors_binds(self) -> None:
        """A policy configured below the model's threshold still may not act on it."""
        the_policy = s.policy()

        decision = s.decide(the_policy, outcome=s.resolved(probability=0.7, threshold=0.8))

        assert decision.reason is NonActionReason.PROBABILITY_BELOW_FLOOR
        restraint = decision.binding_restraint
        assert restraint is not None
        assert restraint.limit == pytest.approx(0.8), (
            "the model's own threshold is stricter than the policy floor and must bind"
        )

    def test_a_probability_above_the_floor_is_allowed(self) -> None:
        decision = s.decide(s.policy(), outcome=s.resolved(probability=0.7))

        assert decision.is_intervention


class TestConfidenceFloor:
    def test_a_low_band_is_refused(self) -> None:
        decision = s.decide(s.policy(), outcome=s.resolved(band=ConfidenceLevel.LOW))

        assert decision.reason is NonActionReason.CONFIDENCE_TOO_LOW
        assert decision.binding_restraint is not None
        assert decision.binding_restraint.kind is RestraintKind.CONFIDENCE_FLOOR

    def test_a_medium_band_satisfies_a_catalogue_whose_weakest_bar_is_medium(self) -> None:
        decision = s.decide(s.policy(), outcome=s.resolved(band=ConfidenceLevel.MEDIUM))

        assert decision.is_intervention

    def test_a_medium_band_still_fails_a_catalogue_that_demands_high(self) -> None:
        demanding = s.policy(
            catalogue=(
                InterventionCandidate(
                    "focus_quiz", priority=0, min_confidence=ConfidenceLevel.HIGH
                ),
            )
        )

        decision = s.decide(demanding, outcome=s.resolved(band=ConfidenceLevel.MEDIUM))

        assert decision.reason is NonActionReason.CONFIDENCE_TOO_LOW

    def test_the_floor_is_never_satisfied_by_a_refusal_band(self) -> None:
        assert not meets_confidence(ConfidenceLevel.UNKNOWN, ConfidenceLevel.LOW)
        assert not meets_confidence(ConfidenceLevel.INSUFFICIENT_DATA, ConfidenceLevel.LOW)
        assert meets_confidence(ConfidenceLevel.LOW, ConfidenceLevel.LOW)
        assert meets_confidence(ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)

    def test_a_disruptive_candidate_is_skipped_when_the_confidence_is_only_medium(self) -> None:
        decision = s.decide(s.policy(), outcome=s.resolved(band=ConfidenceLevel.MEDIUM))

        assert decision.selected is not None
        assert decision.selected.intervention_type in {"recap", "micro_question"}
        assert "focus_quiz" in decision.excluded


class TestStateRule:
    @pytest.mark.parametrize("state", sorted(ACTIONABLE_STATES))
    def test_an_actionable_state_may_proceed(self, state: BehavioralEngagementState) -> None:
        decision = s.decide(s.policy(), state=state)

        assert decision.is_intervention

    @pytest.mark.parametrize(
        "state",
        [
            BehavioralEngagementState.STABLE,
            BehavioralEngagementState.RECOVERING,
        ],
    )
    def test_a_learner_who_does_not_need_help_is_not_interrupted(
        self, state: BehavioralEngagementState
    ) -> None:
        decision = s.decide(s.policy(), state=state)

        assert decision.decision is PolicyDecisionType.NO_INTERVENTION
        assert decision.reason is NonActionReason.STATE_NOT_INDICATED

    def test_a_learner_who_is_already_recovering_is_left_alone(self) -> None:
        """The most important restraint: a learner returning to baseline needs no prompt.

        A maximal policy intervenes here, because the probability may well be high and the
        caps are not exhausted. The intervention would cost attention to confirm something
        that is already happening.
        """
        decision = s.decide(s.policy(), state=BehavioralEngagementState.RECOVERING)

        assert decision.selected is None
        assert decision.reason is NonActionReason.STATE_NOT_INDICATED

    def test_an_unknown_state_is_refused_for_a_different_reason_than_a_known_good_one(
        self,
    ) -> None:
        unknown = s.decide(s.policy(), state=BehavioralEngagementState.INSUFFICIENT_DATA)
        known_good = s.decide(s.policy(), state=BehavioralEngagementState.RECOVERING)

        assert unknown.reason is NonActionReason.STATE_INSUFFICIENT
        assert unknown.reason is not known_good.reason, (
            "not knowing the state is a different situation from knowing the learner is "
            "fine, and the two must not be reported identically"
        )
        assert known_good.reason is NonActionReason.STATE_NOT_INDICATED, (
            "the comparison above is only meaningful while the known-good state is still "
            "refused for its own reason"
        )

    def test_the_state_restraint_reports_whether_the_rule_was_satisfied(self) -> None:
        declined = s.decide(s.policy(), state=BehavioralEngagementState.STABLE)
        acted = s.decide(s.policy(), state=BehavioralEngagementState.DECLINING)

        declined_rule = next(
            item for item in declined.restraints if item.kind is RestraintKind.STATE_RULE
        )
        acted_rule = next(
            item for item in acted.restraints if item.kind is RestraintKind.STATE_RULE
        )
        assert declined_rule.observed == 0.0
        assert acted_rule.observed == 1.0

    def test_a_candidate_may_be_restricted_to_particular_states(self) -> None:
        """A state-specific candidate is offered only in its state."""
        narrow = s.policy(
            catalogue=(
                InterventionCandidate(
                    "focus_quiz",
                    priority=0,
                    min_confidence=ConfidenceLevel.MEDIUM,
                    applies_to_states=(BehavioralEngagementState.HIGH_DEVIATION,),
                ),
            )
        )

        deep = s.decide(narrow, state=BehavioralEngagementState.HIGH_DEVIATION)
        mild = s.decide(narrow, state=BehavioralEngagementState.DECLINING)

        assert deep.is_intervention
        assert deep.selected is not None
        assert deep.selected.intervention_type == "focus_quiz"
        assert mild.reason is NonActionReason.NO_ELIGIBLE_CANDIDATE

    def test_a_candidate_may_not_claim_a_state_that_never_warrants_help(self) -> None:
        with pytest.raises(ValueError, match="only apply to states that warrant"):
            InterventionCandidate(
                "focus_quiz",
                applies_to_states=(BehavioralEngagementState.STABLE,),
            )


class TestMinimumEvidence:
    def test_too_few_observations_are_refused(self) -> None:
        decision = s.decide(s.policy(), outcome=s.resolved(evidence_units=2))

        assert decision.reason is NonActionReason.MINIMUM_EVIDENCE_NOT_MET
        assert decision.binding_restraint is not None
        assert decision.binding_restraint.kind is RestraintKind.MINIMUM_EVIDENCE

    def test_the_evidence_floor_records_the_shortfall(self) -> None:
        decision = s.decide(s.policy(), outcome=s.resolved(evidence_units=2))

        restraint = decision.binding_restraint
        assert restraint is not None
        assert restraint.limit == 5
        assert restraint.observed == 2

    def test_the_policy_floor_is_separate_from_the_engines_own_evidence_floor(self) -> None:
        """A probability can be worth reporting and still be too thin to act on.

        The uncertainty engine's floor decides whether an answer may be *issued*. This one
        decides whether that answer may justify *interrupting a person*. Collapsing them
        would make the stricter of the two silently disappear.
        """
        from focus_engine.configuration.thresholds import (
            InterventionSettings,
            UncertaintySettings,
        )

        assert InterventionSettings().min_evidence_to_act != (
            UncertaintySettings().min_evidence_for_prediction
        )

    def test_exactly_at_the_floor_is_allowed(self) -> None:
        decision = s.decide(s.policy(), outcome=s.resolved(evidence_units=5))

        assert decision.is_intervention


class TestDeterminism:
    def test_the_same_inputs_give_the_same_decision(self) -> None:
        the_policy = s.policy()

        first = s.decide(the_policy)
        second = s.decide(the_policy)

        assert first == second

    def test_two_independently_configured_policies_agree(self) -> None:
        assert s.decide(s.policy()) == s.decide(s.policy())

    def test_the_fingerprint_changes_when_a_restraint_changes(self) -> None:
        from focus_engine.configuration.thresholds import InterventionSettings

        loose = s.policy()
        tight = s.policy(settings=InterventionSettings(cooldown_minutes=30.0))

        assert loose.settings_fingerprint() != tight.settings_fingerprint()

    def test_the_fingerprint_is_stable_for_one_configuration(self) -> None:
        assert s.policy().settings_fingerprint() == s.policy().settings_fingerprint()

    def test_the_fingerprint_changes_when_the_catalogue_changes(self) -> None:
        narrowed = s.policy(catalogue=(InterventionCandidate("recap", priority=0),))

        assert s.policy().settings_fingerprint() != narrowed.settings_fingerprint()


class TestInputIntegrity:
    def test_a_history_belonging_to_another_learner_is_refused(self) -> None:
        other = s.history(s.delivery(), learner_id="learner-0002")

        with pytest.raises(ValueError, match="history belongs to"):
            s.decide(s.policy(), the_history=other)

    def test_a_delivery_from_the_future_cannot_influence_the_decision(self) -> None:
        """A decision must not depend on interventions that had not happened yet.

        Without the truncation, a decision made now and replayed later from the same
        persisted history would give different answers - and the live behaviour would look
        correct right up until someone tried to reproduce it.
        """
        the_policy = s.policy()
        future = s.history(
            s.delivery("recap", minutes_ago=-60.0, suffix="third"),
            s.delivery("recap", minutes_ago=-90.0, suffix="second"),
            s.delivery("recap", minutes_ago=-120.0, suffix="first"),
        )

        decision = s.decide(the_policy, the_history=future)

        assert decision.is_intervention, (
            "three deliveries inside the cooldown and the session cap had not happened yet"
        )
