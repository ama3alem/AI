"""Tests for the authority layer.

The exit criteria here are almost entirely negative, and that is the point of the layer.
Each test below asserts that something the system *could* plausibly do - act on a stale
grant, act alone in a proctored exam, let a track record manufacture permission, treat an
unreachable human as consent, quietly train on a correction - does not happen.

Where a positive behaviour is asserted it is asserted together with the boundary it must
respect, because a test that only checks "it allowed the recap" passes just as happily
when the same code would also have allowed a grade change.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from focus_engine.authority import (
    ActionPermission,
    AuthorityEnvelope,
    AuthorityLevel,
    ContextDescriptor,
    ContextMemory,
    ContextRisk,
    DecayEngine,
    FeedbackOutcome,
    FeedbackType,
    FinalActionOutcome,
    HumanAvailability,
    HumanFeedback,
    MemoryEntry,
    RestrictedAction,
    confidence_ceiling,
)
from focus_engine.schemas.primitives import ConfidenceLevel
from focus_engine.utils.clock import FixedClock
from tests.unit.authority import scenarios as s

pytestmark = pytest.mark.unit


class TestAuthorityIsNotConfidence:
    """The same prediction earns different authority in different rooms."""

    def test_identical_bands_diverge_by_context(self) -> None:
        """The required demonstration: one probability, two contexts, two outcomes."""
        engine = s.engine()
        permissive = engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.MEDIUM)
        guarded = engine.authorize(s.LEARNER, s.GRADED, ConfidenceLevel.MEDIUM)

        assert permissive.envelope.confidence_weight == guarded.envelope.confidence_weight, (
            "the band is identical, so the weight must be too; if it is not, the contexts "
            "are leaking into the calculation"
        )
        assert permissive.envelope.authority_level is AuthorityLevel.ALLOWED
        assert guarded.envelope.authority_level is AuthorityLevel.HUMAN_REQUIRED

    def test_a_permission_does_not_travel_between_contexts(self) -> None:
        engine = s.engine()
        permissive = engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope
        guarded = engine.authorize(s.LEARNER, s.GRADED, ConfidenceLevel.HIGH).envelope

        assert permissive.permits("recap", at=s.START) is True
        assert guarded.permits("recap", at=s.START) is False, (
            "the same widget in a recorded assessment is not the same act, and a grant "
            "from free practice must not carry into a mark"
        )

    def test_an_unknown_band_buys_no_authority(self) -> None:
        """Absent confidence is absent evidence, not a neutral default."""
        assert confidence_ceiling(None) == 0.0
        assert confidence_ceiling(ConfidenceLevel.UNKNOWN) == 0.0
        assert confidence_ceiling(ConfidenceLevel.INSUFFICIENT_DATA) == 0.0

        assessment = s.engine().authorize(s.LEARNER, s.FREE, None)

        assert assessment.envelope.confidence_weight == 0.0
        assert assessment.envelope.authority_level is AuthorityLevel.RESTRICTED


class TestHumanAbsence:
    """An unreachable person must never be read as consent."""

    def test_absence_does_not_downgrade_an_allow(self) -> None:
        """Restraint is a property of the grant, not of who happens to be awake."""
        engine = s.engine()
        envelope = engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope

        for availability in HumanAvailability:
            final = engine.finalise(s.intervene(), envelope, availability)
            assert final.outcome is FinalActionOutcome.ALLOWED, (
                f"availability={availability.value!r} changed an allow; a system that "
                "acts more freely at 3am is the failure this layer exists to prevent"
            )

    def test_absence_does_not_upgrade_a_refusal(self) -> None:
        engine = s.engine()
        envelope = engine.authorize(s.LEARNER, s.EXAM, ConfidenceLevel.HIGH).envelope

        for availability in HumanAvailability:
            final = engine.finalise(s.intervene(), envelope, availability)
            assert final.outcome is not FinalActionOutcome.ALLOWED
            assert final.requires_human_approval is True, (
                "an action awaiting a person must keep saying so; reporting it as refused "
                "makes a live escalation look like a closed question"
            )

    def test_a_human_required_envelope_is_queued_not_refused(self) -> None:
        """Deferred and refused are different operational states and must stay distinct."""
        engine = s.engine()
        envelope = engine.authorize(s.LEARNER, s.EXAM, ConfidenceLevel.HIGH).envelope

        final = engine.finalise(s.intervene(), envelope, HumanAvailability.AVAILABLE)

        assert final.outcome is FinalActionOutcome.DEFERRED
        assert envelope.permitted_actions == (), (
            "a human-required envelope must permit nothing, or the level is decorative"
        )


class TestSafetyNeverOverrides:
    """The safety gate adds restrictions; it never removes them."""

    def test_the_safety_gate_does_not_invent_an_action(self) -> None:
        engine = s.engine()
        envelope = engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope

        final = engine.finalise(s.decline(), envelope, HumanAvailability.AVAILABLE)

        assert final.outcome is FinalActionOutcome.REFUSED
        assert final.intervention_type is None

    def test_an_expired_envelope_is_refused(self) -> None:
        """Expiry is checked at the decision instant, not at the envelope's own mint time."""
        minted = s.engine().authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope
        later = s.engine(at=s.START + timedelta(days=15))

        final = later.finalise(s.intervene(), minted, HumanAvailability.AVAILABLE)

        assert final.outcome is FinalActionOutcome.REFUSED
        assert final.envelope_expired is True

    def test_expiry_boundary_is_exact(self) -> None:
        """Valid strictly before the expiry instant, expired at it. The 336h default is
        what the boundary is being measured against."""
        envelope = s.engine().authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope

        last_valid = s.START + timedelta(hours=335)
        at_expiry = s.START + timedelta(hours=336)

        assert envelope.expires_at == at_expiry
        assert envelope.permits("recap", at=last_valid) is True
        assert envelope.permits("recap", at=at_expiry) is False

    def test_a_high_impact_action_is_refused_below_its_weight_floor(self) -> None:
        """A permissive context still cannot make a grade change automatic."""
        engine = s.engine()
        envelope = engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope

        final = engine.finalise(
            s.intervene(RestrictedAction.GRADE_MODIFICATION.value),
            envelope,
            HumanAvailability.AVAILABLE,
        )

        assert final.outcome is FinalActionOutcome.REFUSED
        assert envelope.confidence_weight < 0.75, (
            "the test is only meaningful while a fresh context sits below the high-impact "
            "floor; if this fails the default changed and the scenario needs revisiting"
        )


class TestDecay:
    """History may take authority away. It may never grant it."""

    def test_good_outcomes_cannot_lift_a_context_past_its_base(self) -> None:
        """The bug this test was written for: credit used to compound to full authority."""
        memory = ContextMemory(
            learner_id=s.LEARNER,
            context_type=s.FREE.context_type,
            entries=tuple(
                MemoryEntry(
                    action="recap",
                    context_type=s.FREE.context_type,
                    recorded_at=s.START,
                    outcome="improved",
                )
                for _ in range(20)
            ),
        )

        report = DecayEngine(clock=FixedClock(s.START)).evaluate(
            memory, s.FREE, ConfidenceLevel.HIGH
        )

        assert report.credit_total > 0.0, "the scenario must actually earn credit"
        assert report.weight == report.base, (
            "twenty successful interventions is evidence the last action was not harmful, "
            "not that the system may now act without limit"
        )
        assert report.binding == "base"

    def test_a_low_band_binds_even_with_a_perfect_history(self) -> None:
        memory = ContextMemory(
            learner_id=s.LEARNER,
            context_type=s.FREE.context_type,
            entries=(MemoryEntry("recap", s.FREE.context_type, s.START, "improved"),),
        )

        report = DecayEngine(clock=FixedClock(s.START)).evaluate(
            memory, s.FREE, ConfidenceLevel.LOW
        )

        assert report.weight == report.ceiling == 0.35
        assert report.binding == "ceiling"

    def test_disputes_decay_authority_toward_the_floor(self) -> None:
        engine = s.engine()
        for index in range(6):
            engine.record_feedback(
                s.dispute(feedback_id=f"fb-{index}"), learner_id=s.LEARNER, action="recap"
            )

        assessment = engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH)

        assert assessment.envelope.confidence_weight == assessment.decay.weight
        assert assessment.decay.binding == "floor"
        assert assessment.envelope.authority_level is not AuthorityLevel.ALLOWED

    def test_a_stale_dispute_stops_counting(self) -> None:
        """Decay must actually decay; a counter would report the same weight for ever."""
        fresh = ContextMemory(
            learner_id=s.LEARNER,
            context_type=s.FREE.context_type,
            entries=(MemoryEntry("recap", s.FREE.context_type, s.START, "disputed"),),
        )
        old = ContextMemory(
            learner_id=s.LEARNER,
            context_type=s.FREE.context_type,
            entries=(
                MemoryEntry(
                    "recap",
                    s.FREE.context_type,
                    s.START - timedelta(days=365),
                    "disputed",
                ),
            ),
        )
        engine = DecayEngine(clock=FixedClock(s.START))

        assert (
            engine.evaluate(fresh, s.FREE, ConfidenceLevel.HIGH).weight
            < engine.evaluate(old, s.FREE, ConfidenceLevel.HIGH).weight
        )

    def test_an_unrecognised_outcome_is_penalised_not_ignored(self) -> None:
        memory = ContextMemory(
            learner_id=s.LEARNER,
            context_type=s.FREE.context_type,
            entries=(MemoryEntry("recap", s.FREE.context_type, s.START, "wat"),),
        )

        report = DecayEngine(clock=FixedClock(s.START)).evaluate(
            memory, s.FREE, ConfidenceLevel.HIGH
        )

        assert report.penalty_total > 0.0, (
            "a system that trusts itself most when its logging has broken is exactly the "
            "failure this rule prevents"
        )


class TestContextIsolation:
    """A permission earned in one room does not lend weight in another."""

    def test_history_is_keyed_by_context(self) -> None:
        engine = s.engine()
        for index in range(3):
            engine.record_feedback(
                s.dispute(feedback_id=f"fb-{index}"), learner_id=s.LEARNER, action="recap"
            )
        elsewhere = ContextDescriptor("graded_practice", ContextRisk.LOW)

        disputed = engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH)
        untouched = engine.authorize(s.LEARNER, elsewhere, ConfidenceLevel.HIGH)

        assert disputed.envelope.confidence_weight < untouched.envelope.confidence_weight

    def test_the_decay_engine_rejects_a_mismatched_memory(self) -> None:
        memory = ContextMemory(learner_id=s.LEARNER, context_type="graded_practice")

        with pytest.raises(ValueError, match="same context"):
            DecayEngine(clock=FixedClock(s.START)).evaluate(memory, s.FREE, ConfidenceLevel.HIGH)


class TestRestrictedActions:
    """High-impact actions are refused structurally, with a named route out."""

    def test_every_restricted_action_names_a_role_and_a_route(self) -> None:
        assessment = s.engine().authorize(s.LEARNER, s.EXAM, ConfidenceLevel.HIGH)

        for withheld in assessment.envelope.restricted_actions:
            assert withheld.required_approval, f"{withheld.action.value} names no role"
            assert withheld.escalation_path, f"{withheld.action.value} names no route"
            assert withheld.reason

    def test_escalation_never_says_human_in_the_loop(self) -> None:
        """A role is answerable; a generic human is nobody in particular."""
        assessment = s.engine().authorize(s.LEARNER, s.EXAM, ConfidenceLevel.HIGH)

        assert assessment.envelope.required_approvals
        for role in assessment.envelope.required_approvals:
            assert "human in the loop" not in role.lower()

    def test_restricted_actions_are_withheld_in_every_context(self) -> None:
        engine = s.engine()
        for context in (s.FREE, s.GRADED, s.EXAM):
            assessment = engine.authorize(s.LEARNER, context, ConfidenceLevel.HIGH)
            names = {item.action.value for item in assessment.envelope.restricted_actions}

            assert names == {action.value for action in RestrictedAction}
            for action in RestrictedAction:
                assert assessment.permission[action.value] is ActionPermission.WITHHELD


class TestFeedback:
    """A correction is evidence about the authority layer, not a training signal."""

    def test_a_dispute_reduces_authority_and_never_trains(self) -> None:
        engine = s.engine()

        outcome, detail = engine.record_feedback(s.dispute(), learner_id=s.LEARNER, action="recap")

        assert outcome is FeedbackOutcome.ACCEPTED
        assert "no model update" in detail.lower()
        applied = [
            entry for entry in engine.ledger.entries() if entry.event_type == "feedback_applied"
        ]
        assert applied and applied[0].detail["retrained"] is False

    def test_a_confirmation_changes_nothing(self) -> None:
        engine = s.engine()
        before = engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope

        engine.record_feedback(
            s.dispute(feedback_type=FeedbackType.CONFIRMED),
            learner_id=s.LEARNER,
            action="recap",
        )
        after = engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope

        assert before.confidence_weight == after.confidence_weight, (
            "a system that gained authority for being agreed with would be incentivised "
            "to seek agreement rather than to be right"
        )

    def test_a_correction_must_carry_a_reading(self) -> None:
        with pytest.raises(ValueError, match="corrected_reading"):
            HumanFeedback(
                feedback_id="fb-1",
                decision_ref="decision-1",
                feedback_type=FeedbackType.CORRECTED,
                actor_role="instructor",
                submitted_at=s.START,
                context_type="free_practice",
            )

    def test_a_non_correction_must_not_carry_a_reading(self) -> None:
        with pytest.raises(ValueError, match="only valid for CORRECTED"):
            HumanFeedback(
                feedback_id="fb-1",
                decision_ref="decision-1",
                feedback_type=FeedbackType.DISPUTED,
                actor_role="instructor",
                submitted_at=s.START,
                context_type="free_practice",
                corrected_reading="something else entirely",
            )


class TestLedger:
    """The record must be verifiable by the reader, not trusted by the writer."""

    def test_a_fresh_chain_is_intact(self) -> None:
        engine = s.engine()
        engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH)
        engine.finalise(
            s.intervene(),
            engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope,
            HumanAvailability.AVAILABLE,
        )

        report = engine.integrity()

        assert report.is_valid is True
        assert report.total_entries > 0

    def test_tampering_with_an_entry_is_detected(self) -> None:
        engine = s.engine()
        engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH)
        entries = list(engine.ledger.entries())
        assert entries, "the scenario must produce at least one entry to corrupt"

        engine.ledger._entries[0] = dataclasses.replace(  # noqa: SLF001
            entries[0], detail={"confidence_weight": 0.999, "invented": True}
        )
        report = engine.integrity()

        assert report.is_valid is False
        assert report.broken_at == entries[0].entry_id
        assert "altered" in report.reason

    def test_breaking_a_link_is_detected(self) -> None:
        engine = s.engine()
        for _ in range(3):
            engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH)
        entries = list(engine.ledger.entries())

        engine.ledger._entries[2] = dataclasses.replace(  # noqa: SLF001
            entries[2], previous_hash="0" * 64
        )

        assert engine.integrity().is_valid is False

    def test_every_calculation_is_recorded(self) -> None:
        engine = s.engine()
        engine.authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH)
        engine.authorize(s.LEARNER, s.EXAM, ConfidenceLevel.HIGH)

        kinds = [entry.event_type for entry in engine.ledger.entries()]

        assert kinds.count("authority_calculation") == 2, (
            "the refusals matter at least as much as the permissions, and a ledger that "
            "only records what fired cannot be audited"
        )


class TestEnvelopeInvariants:
    """An envelope that contradicts itself must not be constructible."""

    def _envelope(self, **overrides: object) -> AuthorityEnvelope:
        base = {
            "authority_level": AuthorityLevel.ALLOWED,
            "context": s.FREE,
            "permitted_actions": ("recap",),
            "restricted_actions": (),
            "confidence_weight": 0.6,
            "restrictions": (),
            "required_approvals": (),
            "escalation_path": (),
            "calculated_at": s.START,
            "expires_at": s.START + timedelta(hours=336),
        }
        return AuthorityEnvelope(**{**base, **overrides})  # type: ignore[arg-type]

    def test_a_valid_envelope_is_constructible(self) -> None:
        assert self._envelope().permits("recap", at=s.START) is True

    def test_an_envelope_born_expired_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="born expired"):
            self._envelope(expires_at=s.START - timedelta(hours=1))

    def test_an_allowed_envelope_with_no_actions_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one action"):
            self._envelope(permitted_actions=())

    def test_permitting_and_withholding_the_same_action_is_rejected(self) -> None:
        from focus_engine.authority import WithheldAction

        with pytest.raises(ValueError, match="both permitted and withheld"):
            self._envelope(
                restricted_actions=(
                    WithheldAction(
                        action=RestrictedAction.GRADE_MODIFICATION,
                        reason="because",
                        required_approval="course_coordinator",
                    ),
                ),
                permitted_actions=("recap", RestrictedAction.GRADE_MODIFICATION.value),
            )

    def test_a_weight_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            self._envelope(confidence_weight=1.5)


class TestDeterminism:
    """A replay must reproduce the decision, or the audit is fiction."""

    def test_the_same_clock_gives_an_identical_envelope(self) -> None:
        first = s.engine().authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH)
        second = s.engine().authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH)

        assert first.envelope.to_summary_dict() == second.envelope.to_summary_dict()

    def test_authority_is_not_cached_across_a_decision(self) -> None:
        """A later decision must see the later clock, not the first one's answer."""
        first = s.engine().authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH).envelope
        second = (
            s.engine(at=s.START + timedelta(days=1))
            .authorize(s.LEARNER, s.FREE, ConfidenceLevel.HIGH)
            .envelope
        )

        assert first.calculated_at != second.calculated_at
        assert first.expires_at != second.expires_at
