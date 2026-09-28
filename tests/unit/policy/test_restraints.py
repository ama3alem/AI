"""Tests for the policy's restraints (Phase 10).

Two kinds of test live here. The first are the exit-criteria cases - cooldown expiry, each
cap, the repeat limit, repeated-failure protection. The second is a property test over a
grid of scenarios, asserting that *tightening any restraint can only ever remove
interventions*.

That property is the load-bearing one. "Optimised for useful intervention, not maximal
intervention" is otherwise a slogan, and every individual threshold test can pass while
the policy as a whole quietly acts more when pushed harder. The property test is what
turns the slogan into something a change to the code can break.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from focus_engine.configuration.thresholds import InterventionSettings
from focus_engine.policy.engine import InterventionPolicy
from focus_engine.policy.history import DeliveredIntervention, InterventionHistory
from focus_engine.policy.models import (
    InterventionCandidate,
    NonActionReason,
    PolicyDecisionType,
    RestraintKind,
)
from focus_engine.schemas.primitives import BehavioralEngagementState, ConfidenceLevel
from focus_engine.uncertainty.outcomes import PredictionOutcome
from tests.unit.policy import scenarios as s

pytestmark = pytest.mark.unit

#: Types cycled through when a test needs several deliveries without tripping the repeat
#: limit, which would otherwise confound a test about a different restraint.
CYCLE = ("recap", "micro_question", "checkpoint_prompt", "focus_quiz")


def _spaced(
    count: int,
    *,
    minutes: float,
    step: float,
    start_index: int = 0,
    session_id: str = s.SESSION,
) -> tuple[DeliveredIntervention, ...]:
    """Build ``count`` deliveries of alternating types.

    Args:
        count: How many deliveries to build.
        minutes: How long before the decision the first was delivered.
        step: The gap between consecutive deliveries.
        start_index: An offset into :data:`CYCLE`, so two sessions can continue the
            rotation instead of repeating the same types.
        session_id: The session the deliveries belong to. Overridden by the cross-session
            tests, which need the current session's count to be zero.

    Returns:
        Deliveries, earliest first, with no two consecutive ones sharing a type.
    """
    return tuple(
        s.delivery(
            CYCLE[(start_index + index) % len(CYCLE)],
            minutes_ago=minutes - (step * index),
            session_id=session_id,
            suffix=f"seq{session_id}{start_index + index}",
        )
        for index in range(count)
    )


def _scattered(
    count: int,
    *,
    minutes: float,
    step: float,
    start_index: int = 0,
    sessions: int = 2,
) -> tuple[DeliveredIntervention, ...]:
    """Build deliveries round-robin across several earlier sessions.

    The window cap is a rate limit rather than a session limit, so a test about it needs
    enough deliveries to breach the window without any one session breaching its own cap
    first - otherwise the per-session cap binds and the test passes for the wrong reason.

    Args:
        count: How many deliveries to build.
        minutes: How long before the decision the first was delivered.
        step: The gap between consecutive deliveries.
        start_index: An offset into :data:`CYCLE`.
        sessions: How many sessions to spread them across.

    Returns:
        Deliveries, earliest first, no session holding more than ``ceil(count / sessions)``.
    """
    return tuple(
        s.delivery(
            CYCLE[(start_index + index) % len(CYCLE)],
            minutes_ago=minutes - (step * index),
            session_id=f"old-sess-{index % sessions}",
            suffix=f"sc{index}",
        )
        for index in range(count)
    )


class TestCooldown:
    def test_a_recent_delivery_blocks_a_second_one(self) -> None:
        decision = s.decide(s.policy(), the_history=s.history(s.delivery(minutes_ago=1.0)))

        assert decision.reason is NonActionReason.IN_COOLDOWN
        assert decision.binding_restraint is not None
        assert decision.binding_restraint.kind is RestraintKind.COOLDOWN

    def test_the_cooldown_reports_how_long_is_left_to_run(self) -> None:
        decision = s.decide(s.policy(), the_history=s.history(s.delivery(minutes_ago=4.0)))

        restraint = decision.binding_restraint
        assert restraint is not None
        assert restraint.limit == 600.0
        assert restraint.observed == pytest.approx(240.0)
        assert restraint.headroom == pytest.approx(360.0), (
            "the remaining cooldown is the number an operator acts on, which is why the "
            "restraint carries a headroom rather than leaving it to be recomputed from the "
            "limit and the elapsed time"
        )

    def test_the_cooldown_expires_exactly_on_its_boundary(self) -> None:
        """A cooldown of ten minutes must permit an intervention at ten minutes."""
        settings = InterventionSettings(cooldown_minutes=10.0)
        the_policy = s.policy(settings=settings)
        moment = s.START

        inside = the_policy.decide(
            s.resolved(),
            s.history(s.delivery(minutes_ago=9.999)),
            learner_id=s.LEARNER,
            session_id=s.SESSION,
            state=BehavioralEngagementState.DECLINING,
            now=moment,
        )
        on_boundary = the_policy.decide(
            s.resolved(),
            s.history(s.delivery(minutes_ago=10.0)),
            learner_id=s.LEARNER,
            session_id=s.SESSION,
            state=BehavioralEngagementState.DECLINING,
            now=moment,
        )

        assert inside.reason is NonActionReason.IN_COOLDOWN
        assert on_boundary.is_intervention, (
            "at exactly the cooldown the learner may be acted on again; an off-by-one "
            "here silently doubles or halves the intervention rate"
        )

    def test_a_learner_never_intervened_with_is_not_in_cooldown(self) -> None:
        decision = s.decide(s.policy(), the_history=s.empty_history())

        assert decision.is_intervention
        restraint = next(
            item for item in decision.restraints if item.kind is RestraintKind.COOLDOWN
        )
        assert "never been intervened with" in restraint.detail

    def test_never_having_been_intervened_is_not_the_same_as_just_having_been(self) -> None:
        """The cooldown distinguishes absence of history from recent delivery.

        Reading "never" as "0 seconds ago" would put every first-time learner into a
        permanent cooldown and quietly stop the system working for exactly the learners it
        knows least about.
        """
        never = s.decide(s.policy(), the_history=s.empty_history())
        just_now = s.decide(s.policy(), the_history=s.history(s.delivery(minutes_ago=0.0)))

        assert never.is_intervention
        assert just_now.reason is NonActionReason.IN_COOLDOWN


class TestSessionCap:
    def test_the_cap_blocks_the_fourth_intervention_in_a_session(self) -> None:
        the_history = s.history(*_spaced(3, minutes=40.0, step=15.0))

        decision = s.decide(s.policy(), the_history=the_history)

        assert decision.reason is NonActionReason.SESSION_CAP_REACHED
        assert decision.binding_restraint is not None
        assert decision.binding_restraint.kind is RestraintKind.SESSION_CAP
        assert decision.binding_restraint.observed == 3
        assert decision.binding_restraint.headroom == 0.0

    def test_the_third_intervention_in_a_session_is_still_permitted(self) -> None:
        the_history = s.history(*_spaced(2, minutes=40.0, step=15.0))

        decision = s.decide(s.policy(), the_history=the_history)

        assert decision.is_intervention

    def test_the_cap_counts_only_the_current_session(self) -> None:
        """A learner who used their budget in another session is not capped."""
        yesterday = _spaced(5, minutes=60.0 * 24.0, step=20.0, session_id="old-session")
        the_history = s.history(*yesterday)

        decision = s.decide(s.policy(), the_history=the_history)

        assert decision.is_intervention


class TestSlidingWindowCap:
    def _five_across_two_sessions(self) -> InterventionHistory:
        """Five deliveries inside the hour, split so no single session hits its cap."""
        return s.history(
            *_spaced(3, minutes=55.0, step=10.0, start_index=0, session_id="old-sess"),
            *_spaced(2, minutes=25.0, step=10.0, start_index=1, session_id="new-sess"),
        )

    def test_the_hourly_cap_blocks_the_sixth_intervention(self) -> None:
        decision = s.decide(s.policy(), the_history=self._five_across_two_sessions())

        assert decision.reason is NonActionReason.WINDOW_CAP_REACHED
        assert decision.binding_restraint is not None
        assert decision.binding_restraint.kind is RestraintKind.WINDOW_CAP
        assert decision.binding_restraint.observed == 5

    def test_the_window_cap_binds_across_a_session_boundary(self) -> None:
        """This is the reason the window cap exists.

        The per-session cap alone permits three interventions in every session, so a learner
        opening four sessions in an hour would be interrupted twelve times. The window cap
        is what bounds the rate rather than the session.
        """
        the_history = self._five_across_two_sessions()

        decision = s.decide(s.policy(), the_history=the_history)

        assert decision.reason is NonActionReason.WINDOW_CAP_REACHED
        assert decision.binding_restraint is not None
        assert decision.binding_restraint.limit == 5.0

    def test_a_delivery_exactly_an_hour_old_has_aged_out(self) -> None:
        """The cap slides rather than tumbling on the hour.

        The window is half-open. A delivery an hour old is already out of it, so this
        history holds only four interventions inside the window and the learner is
        permitted a fifth.
        """
        the_history = s.history(
            s.delivery("recap", minutes_ago=60.0, suffix="aged", session_id="old-sess"),
            *_scattered(4, minutes=50.0, step=10.0, start_index=1),
        )

        decision = s.decide(s.policy(), the_history=the_history)

        assert decision.is_intervention
        assert decision.reason is not NonActionReason.WINDOW_CAP_REACHED

    def test_a_delivery_just_inside_the_hour_still_counts(self) -> None:
        """The same history a tenth of a second inside the window does breach it.

        This is the boundary the half-open window is really asserting. A sixth delivery
        ten seconds short of the hour has not aged out, so the learner is capped - and an
        inclusive boundary here would quietly raise the effective hourly rate above the
        configured cap.
        """
        the_history = s.history(
            s.delivery("recap", minutes_ago=59.9, suffix="inside", session_id="old-sess"),
            *_scattered(4, minutes=50.0, step=10.0, start_index=1),
        )

        decision = s.decide(s.policy(), the_history=the_history)

        assert decision.reason is NonActionReason.WINDOW_CAP_REACHED
        assert decision.binding_restraint is not None
        assert decision.binding_restraint.observed == 5


class TestRepeatedTypeLimit:
    def test_a_type_delivered_twice_in_a_row_is_withdrawn(self) -> None:
        """After the limit, the policy must try something else rather than repeat."""
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, suffix="a"),
            s.delivery("recap", minutes_ago=20.0, suffix="b"),
        )

        decision = s.decide(s.policy(), the_history=the_history)

        assert decision.is_intervention
        assert decision.selected is not None
        assert decision.selected.intervention_type != "recap", (
            "the third consecutive recap in a row is exactly the loop the limit exists to prevent"
        )
        assert decision.selected.intervention_type == "micro_question"

    def test_the_limit_is_recorded_even_though_another_option_survived(self) -> None:
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, suffix="a"),
            s.delivery("recap", minutes_ago=20.0, suffix="b"),
        )

        decision = s.decide(s.policy(), the_history=the_history)

        restraint = next(
            item for item in decision.restraints if item.kind is RestraintKind.REPEATED_TYPE_LIMIT
        )
        assert restraint.limit == 2
        assert restraint.observed == 2
        assert not restraint.binding, "a candidate survived, so this did not bind the decision"

    def test_the_limit_counts_only_consecutive_deliveries_of_that_type(self) -> None:
        """A different type in between resets the run.

        Two recaps, then a quiz, then a recap is not three recaps in a row. The run is
        counted over that type's own trailing deliveries, so the interruption genuinely
        resets it. The deliveries sit in an earlier session, because this test is about the
        repeat limit and the per-session cap would otherwise bind first.
        """
        the_history = s.history(
            s.delivery("recap", minutes_ago=40.0, suffix="a", session_id="old-sess"),
            s.delivery("recap", minutes_ago=30.0, suffix="b", session_id="old-sess"),
            s.delivery("focus_quiz", minutes_ago=20.0, suffix="c", session_id="old-sess"),
        )

        decision = s.decide(s.policy(), the_history=the_history)

        assert decision.is_intervention
        assert decision.selected is not None
        assert decision.selected.intervention_type == "recap"

    def test_every_type_being_withdrawn_leaves_nothing_to_do(self) -> None:
        """A catalogue of one type, repeated to its limit, must decline rather than loop."""
        only_one = s.policy(catalogue=(InterventionCandidate("recap", priority=0),))
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, suffix="a"),
            s.delivery("recap", minutes_ago=20.0, suffix="b"),
        )

        decision = s.decide(only_one, the_history=the_history)

        assert decision.decision is PolicyDecisionType.NO_INTERVENTION
        assert decision.reason is NonActionReason.REPEATED_TYPE_LIMIT
        assert decision.selected is None


class TestRepeatedFailureProtection:
    """Retirement after repeated observed non-responses.

    Every test here raises ``max_repeat_same_type`` well above the number of deliveries it
    makes. Retirement and the repeat limit both suppress a type, and with the default of
    two they fire together - so a test that meant to isolate retirement would pass or fail
    on the repeat limit's behaviour instead. The one place they are deliberately combined
    is :meth:`test_retirement_outranks_the_repeat_limit`.
    """

    #: A policy whose only candidate-level filter in play is retirement.
    RELAXED_REPEAT = InterventionSettings(max_repeat_same_type=99)

    @staticmethod
    def _past(*deliveries: DeliveredIntervention) -> InterventionHistory:
        """Attribute deliveries to an earlier session.

        Retirement is about the learner's history, not the current session, so most of
        these tests would otherwise be measuring the per-session cap whenever they needed
        three or more deliveries to describe a run.
        """
        return s.history(*(replace(item, session_id="old-sess") for item in deliveries))

    @staticmethod
    def _policy(settings: InterventionSettings | None = None) -> InterventionPolicy:
        return s.policy(settings=settings or TestRepeatedFailureProtection.RELAXED_REPEAT)

    def test_a_type_declined_twice_is_retired_for_the_learner(self) -> None:
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, outcome="ignored", suffix="a"),
            s.delivery("recap", minutes_ago=20.0, outcome="dismissed", suffix="b"),
        )

        decision = s.decide(self._policy(), the_history=the_history)

        assert decision.is_intervention
        assert decision.selected is not None
        assert decision.selected.intervention_type != "recap", (
            "offering a third recap after two declines is the behaviour the retirement "
            "rule exists to stop"
        )

    def test_the_retirement_is_recorded_with_its_evidence(self) -> None:
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, outcome="ignored", suffix="a"),
            s.delivery("recap", minutes_ago=20.0, outcome="dismissed", suffix="b"),
        )

        decision = s.decide(self._policy(), the_history=the_history)

        restraint = next(
            item for item in decision.restraints if item.kind is RestraintKind.TYPE_RETIRED
        )
        assert restraint.limit == 2
        assert restraint.observed == 2
        assert "non-response" in restraint.detail

    def test_a_single_decline_does_not_retire_a_type(self) -> None:
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, outcome="ignored", suffix="a"),
        )

        decision = s.decide(self._policy(), the_history=the_history)

        assert decision.is_intervention
        assert decision.selected is not None
        assert decision.selected.intervention_type == "recap"

    def test_a_response_clears_the_run_of_declines(self) -> None:
        """Retirement requires declines *in a row*, not declines in general."""
        the_history = self._past(
            s.delivery("recap", minutes_ago=50.0, outcome="ignored", suffix="a"),
            s.delivery("recap", minutes_ago=40.0, outcome="accepted", suffix="b"),
            s.delivery("recap", minutes_ago=30.0, outcome="ignored", suffix="c"),
        )

        decision = s.decide(self._policy(), the_history=the_history)

        assert decision.selected is not None
        assert decision.selected.intervention_type == "recap", (
            "one decline after an acceptance is a run of one, which is below the threshold"
        )

    def test_an_unrecognised_outcome_label_does_not_retire_a_type(self) -> None:
        """A label this layer does not know must not be read as a decline.

        Treating an unknown label as a non-response would let a naming change elsewhere
        quietly manufacture a run of failures, and the learner would silently stop being
        offered something.
        """
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, outcome="shrugged", suffix="a"),
            s.delivery("recap", minutes_ago=20.0, outcome="shrugged", suffix="b"),
        )

        decision = s.decide(self._policy(), the_history=the_history)

        assert decision.selected is not None
        assert decision.selected.intervention_type == "recap"

    def test_an_unrecognised_label_terminates_a_run_rather_than_extending_it(self) -> None:
        the_history = self._past(
            s.delivery("recap", minutes_ago=40.0, outcome="ignored", suffix="a"),
            s.delivery("recap", minutes_ago=30.0, outcome="shrugged", suffix="b"),
            s.delivery("recap", minutes_ago=20.0, outcome="ignored", suffix="c"),
        )

        decision = s.decide(self._policy(), the_history=the_history)

        assert decision.selected is not None
        assert decision.selected.intervention_type == "recap", (
            "the unrecognised label breaks consecutiveness, so the run is one, not two"
        )

    def test_a_missing_outcome_terminates_a_run_rather_than_extending_it(self) -> None:
        the_history = self._past(
            s.delivery("recap", minutes_ago=40.0, outcome="ignored", suffix="a"),
            s.delivery("recap", minutes_ago=30.0, outcome=None, suffix="b"),
            s.delivery("recap", minutes_ago=20.0, outcome="ignored", suffix="c"),
        )

        decision = s.decide(self._policy(), the_history=the_history)

        assert decision.selected is not None
        assert decision.selected.intervention_type == "recap"

    def test_retirement_removes_an_option_without_promoting_another(self) -> None:
        """Retirement is a suppression, not a ranking.

        The learner who declined recaps is offered the next option in catalogue order, not
        the option with the best observed response - that would be a feedback rule, and
        feedback belongs to a later, separately versioned layer.
        """
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, outcome="ignored", suffix="a"),
            s.delivery("recap", minutes_ago=20.0, outcome="dismissed", suffix="b"),
        )

        decision = s.decide(self._policy(), the_history=the_history)

        assert decision.selected is not None
        assert decision.selected.intervention_type == "micro_question", (
            "micro_question is simply next in priority order; nothing observed about it "
            "was used to choose it over the others"
        )

    def test_a_retired_type_returns_once_its_run_is_broken(self) -> None:
        """Retirement is not a permanent blacklist.

        A learner who later engages with a recap should be offered one again, because the
        rule is about a *consecutive* run of declines rather than a verdict on the type.
        """
        the_history = self._past(
            s.delivery("recap", minutes_ago=50.0, outcome="ignored", suffix="a"),
            s.delivery("recap", minutes_ago=40.0, outcome="dismissed", suffix="b"),
            s.delivery("recap", minutes_ago=30.0, outcome="accepted", suffix="c"),
        )

        decision = s.decide(self._policy(), the_history=the_history)

        assert decision.selected is not None
        assert decision.selected.intervention_type == "recap"

    def test_every_type_retired_leaves_nothing_to_do(self) -> None:
        every_type = tuple(
            s.delivery(name, minutes_ago=400.0 - index * 20.0, outcome=outcome, suffix=f"r{index}")
            for index, (name, outcome) in enumerate(
                (name, outcome) for name in CYCLE for outcome in ("ignored", "dismissed")
            )
        )
        the_history = self._past(*every_type)

        decision = s.decide(self._policy(), the_history=the_history)

        assert decision.decision is PolicyDecisionType.NO_INTERVENTION
        assert decision.reason is NonActionReason.TYPE_RETIRED
        assert decision.selected is None

    def test_retirement_outranks_the_repeat_limit(self) -> None:
        """When both suppress the same type, the reported reason is the explanatory one.

        With the default thresholds, two declined recaps trip the repeat limit *and*
        retirement simultaneously. Blaming the repeat limit would tell an operator the type
        had been "delivered twice in a row", which is not what happened - the learner was
        shown it twice and declined both times, and that is the fact worth surfacing.
        """
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, outcome="ignored", suffix="a"),
            s.delivery("recap", minutes_ago=20.0, outcome="dismissed", suffix="b"),
        )

        decision = s.decide(s.policy(), the_history=the_history)

        kinds = {item.kind for item in decision.restraints}
        assert RestraintKind.REPEATED_TYPE_LIMIT in kinds, (
            "the repeat limit did fire here, so the two really did overlap"
        )
        assert RestraintKind.TYPE_RETIRED in kinds, (
            "retirement fired but was masked by the short-circuit in eligible_candidates"
        )


class TestResponseGateIsNotUsedHere:
    def test_the_response_observation_gate_belongs_to_a_later_layer(self) -> None:
        """This layer must not rank candidates by observed response.

        ``min_response_observations`` is validated configuration, but the policy never
        consults it. If it did, a candidate would be chosen because of what happened when
        it was last used - which is the feedback architecture, and doing it here would
        introduce an unversioned selection rule into a layer whose whole claim is that it
        is deterministic and transparent.
        """
        strict = s.policy(settings=InterventionSettings(min_response_observations=1_000))
        the_history = s.history(
            s.delivery("recap", minutes_ago=30.0, outcome="accepted", suffix="a"),
        )

        baseline = s.decide(s.policy(), the_history=the_history)
        gated = s.decide(strict, the_history=the_history)

        assert baseline.selected is not None
        assert gated.selected is not None
        assert gated.selected.intervention_type == baseline.selected.intervention_type, (
            "raising the response-observation gate must not change this layer's choice"
        )


class TestTighteningARestraintOnlyRemovesInterventions:
    """The property that makes the restraints restraints.

    Every individual threshold can pass its own test while the policy as a whole behaves
    perversely - for instance if raising a cap somehow made the policy *more* willing to
    act. Asserting the direction of the whole configuration over a grid of scenarios is
    what rules that out.
    """

    #: Each pair is (looser, tighter) versions of the same policy.
    PAIRS = (
        pytest.param(
            InterventionSettings(),
            InterventionSettings(min_probability_to_act=0.95),
            id="probability_floor",
        ),
        pytest.param(
            InterventionSettings(),
            InterventionSettings(cooldown_minutes=600.0),
            id="cooldown",
        ),
        pytest.param(
            InterventionSettings(),
            InterventionSettings(max_interventions_per_session=1),
            id="session_cap",
        ),
        pytest.param(
            InterventionSettings(),
            InterventionSettings(max_interventions_per_hour=1),
            id="window_cap",
        ),
        pytest.param(
            InterventionSettings(),
            InterventionSettings(max_repeat_same_type=1),
            id="repeat_limit",
        ),
        pytest.param(
            InterventionSettings(),
            InterventionSettings(abandon_after_no_response_count=1),
            id="retirement",
        ),
        pytest.param(
            InterventionSettings(),
            InterventionSettings(min_evidence_to_act=10_000),
            id="evidence_floor",
        ),
        pytest.param(
            InterventionSettings(min_probability_to_act=0.7),
            InterventionSettings(min_probability_to_act=0.95),
            id="probability_floor_from_non_default",
        ),
    )

    @staticmethod
    def _histories() -> dict[str, InterventionHistory]:
        return {
            "empty": s.empty_history(),
            "old_delivery": s.history(s.delivery(minutes_ago=120.0)),
            "recent_delivery": s.history(s.delivery(minutes_ago=2.0)),
            "one_declined": s.history(
                s.delivery("recap", minutes_ago=30.0, outcome="ignored", suffix="a")
            ),
            "one_responded": s.history(
                s.delivery("recap", minutes_ago=30.0, outcome="accepted", suffix="a")
            ),
            "session_full": s.history(*_spaced(3, minutes=40.0, step=15.0)),
            "hour_full": s.history(
                *_spaced(3, minutes=55.0, step=10.0, start_index=0),
                *_spaced(2, minutes=25.0, step=10.0, start_index=1),
            ),
            "recap_repeated": s.history(
                s.delivery("recap", minutes_ago=30.0, suffix="a"),
                s.delivery("recap", minutes_ago=20.0, suffix="b"),
            ),
        }

    @staticmethod
    def _outcomes() -> dict[str, PredictionOutcome]:
        return {
            "high_prob_high_band": s.resolved(probability=0.9),
            "just_above_floor": s.resolved(probability=0.66),
            "below_floor": s.resolved(probability=0.5),
            # Above the tightest probability floor in ``PAIRS``, so the grid is not
            # degenerate for that pair - every scenario would decline and the property
            # would hold vacuously.
            "very_high_prob": s.resolved(probability=0.99),
            "medium_band": s.resolved(band=ConfidenceLevel.MEDIUM),
            "low_band": s.resolved(band=ConfidenceLevel.LOW),
            "thin_evidence": s.resolved(evidence_units=1),
            # Above the tightest evidence floor in ``PAIRS``, for the same reason.
            "abundant_evidence": s.resolved(evidence_units=20_000),
            "unknown": s.unknown(),
            "insufficient": s.insufficient_data(),
        }

    @staticmethod
    def _states() -> dict[str, BehavioralEngagementState]:
        return {
            "declining": BehavioralEngagementState.DECLINING,
            "high": BehavioralEngagementState.HIGH_DEVIATION,
            "stable": BehavioralEngagementState.STABLE,
            "recovering": BehavioralEngagementState.RECOVERING,
            "unknown_state": BehavioralEngagementState.INSUFFICIENT_DATA,
        }

    @pytest.mark.parametrize(("loose", "tight"), PAIRS)
    def test_a_tighter_policy_never_acts_where_a_looser_one_declines(
        self, loose: InterventionSettings, tight: InterventionSettings
    ) -> None:
        loose_policy = s.policy(settings=loose)
        tight_policy = s.policy(settings=tight)
        acted = 0
        total = 0

        for history in self._histories().values():
            for outcome in self._outcomes().values():
                for state in self._states().values():
                    total += 1
                    loose_decision = s.decide(
                        loose_policy, outcome=outcome, the_history=history, state=state
                    )
                    tight_decision = s.decide(
                        tight_policy, outcome=outcome, the_history=history, state=state
                    )
                    if tight_decision.decision is PolicyDecisionType.INTERVENE:
                        acted += 1
                        assert loose_decision.decision is PolicyDecisionType.INTERVENE, (
                            f"the tighter policy acted where the looser one declined: "
                            f"outcome={outcome.verdict.value}/{outcome.probability}, "
                            f"state={state.value}. A restraint that tightens the policy and "
                            f"also loosens it is not a restraint."
                        )
                        assert (
                            tight_decision.selected is not None
                            and loose_decision.selected is not None
                        )

        assert total == 8 * 10 * 5, (
            "the grid shrank, so the property is being checked over fewer scenarios than it looks"
        )
        assert acted > 0, (
            "no scenario in the grid produced an intervention under the tighter policy, "
            "so this test would pass vacuously"
        )
