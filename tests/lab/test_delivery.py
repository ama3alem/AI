"""Lab delivery tests: Phase B consumes the frozen decision and never recomputes it.

The whole Phase A/Phase B seam exists to prevent one specific failure: a system that
re-runs the analysis after acting, then reports the second run's numbers as if they were
the ones that justified the action. This file makes that seam testable rather than
asserted:

* the delivery stage is driven from a frozen :class:`AnalysisPhase` whose decision and
  final action are constructed directly, so no coincidence of a scenario can mask a scope
  slip;
* Phase A's decision-making entry points are frozen (monkeypatched to raise), so any
  recomputation inside ``execute_decision`` fails immediately and loudly instead of
  quietly producing a different number;
* every delivery status is exercised: ``NOT_SELECTED``, ``SELECTED_NOT_DELIVERED`` under
  each gate, and ``DELIVERED``;
* the lifecycle events are asserted to be the engine's own native synthetic envelopes,
  byte-identical across replays;
* the outcome and evaluation stages run only when a delivery actually happened, and
  report ``INSUFFICIENT_DATA`` honestly when they cannot measure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from api.lab.adapter import (
    _BASELINE_SETTINGS,
    AnalysisPhase,
    _dimension_value,
    execute_decision,
    prepare_analysis,
)
from api.lab.contract import DeliveryStatus
from api.lab.execution import (
    DELIVERY_OUTCOME_ACCEPTED,
    post_window_reading,
)

from focus_engine.authority import (
    AuthorityLevel,
    FinalAction,
    FinalActionOutcome,
    HumanAvailability,
)
from focus_engine.baseline import (
    BASELINE_DIMENSIONS,
    BaselineEngine,
)
from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    InterventionCompletedPayload,
)
from focus_engine.features import FEATURE_SET_V1
from focus_engine.policy import (
    DEFAULT_CATALOGUE,
    NonActionReason,
    PolicyDecision,
    PolicyDecisionType,
)
from focus_engine.schemas import DataOrigin, Provenance
from focus_engine.simulator import LearnerArchetype, SimulationConfig, generate_session
from focus_engine.temporal import TemporalEngine
from focus_engine.utils.clock import FixedClock

#: The engine's own recap candidate, reused verbatim so a crafted decision still carries
#: the real catalogue, not a test-local stand-in.
RECAP = DEFAULT_CATALOGUE[0]


def _decline_decision(phase: AnalysisPhase, reason: NonActionReason) -> PolicyDecision:
    """A NO_INTERVENTION decision the engine's own validator will accept.

    A decline must name the restraint that bound it — the validator refuses a reason
    without a matching binding ``Restraint`` — so each reason maps to its restraint kind.
    """
    from focus_engine.policy.models import Restraint, RestraintKind

    kind_for_reason = {
        NonActionReason.PROBABILITY_BELOW_FLOOR: RestraintKind.PROBABILITY_FLOOR,
        NonActionReason.STATE_NOT_INDICATED: RestraintKind.STATE_RULE,
        NonActionReason.MINIMUM_EVIDENCE_NOT_MET: RestraintKind.MINIMUM_EVIDENCE,
    }
    kind = kind_for_reason[reason]
    restraint = Restraint(kind=kind, limit=1.0, observed=0.0, detail="test binding", binding=True)
    assert phase.uncertainty is not None
    return PolicyDecision(
        decision=PolicyDecisionType.NO_INTERVENTION,
        selected=None,
        reason=reason,
        detail="test: the restraint bound the decision",
        restraints=(restraint,),
        learner_id=phase.learner_id,
        session_id=phase.session_id,
        evaluated_at=phase.cut_at,
        data_origin=phase.uncertainty.data_origin,
    )


def _declining_probe() -> tuple[tuple[EventEnvelope, ...], datetime]:
    """A real declining learner whose decision surface actually warrants a recap.

    ``base_seed`` is fixed per session (``900110``), so the stream is byte-identical on
    every call. 40 sessions of 40 questions leaves a populated history and a populated
    after-cut window, so the outcome and evaluation stages have something honest to read.
    """
    anchor = datetime(2026, 6, 1, 8, 0, 0, tzinfo=UTC)
    events: list[EventEnvelope] = []
    for index in range(40):
        result = generate_session(
            SimulationConfig(
                archetype=LearnerArchetype.GRADUAL_DECLINE,
                learner_id="probe-s-0001",
                session_id=f"probe-s-0001-s{index + 1:02d}",
                base_seed=900_110,
                question_count=40,
                start_time=anchor + timedelta(days=index),
                inter_event_gap_min=timedelta(seconds=45),
                inter_event_gap_max=timedelta(seconds=90),
            )
        )
        events.extend(result.events)
    ordered = tuple(sorted(events, key=lambda event: event.timestamp))
    return ordered, ordered[-1].timestamp


def _deliverable_phase() -> AnalysisPhase:
    """Phase A over the probe, with the policy/gate inputs replaced by deterministic ones.

    Phase A here is real: the state, vector, uncertainty, profile, and events all come from
    the real engines over the real probe stream. Only the *decision* and *final action* are
    replaced, because those are the two inputs ``execute_decision`` legendarily must consume
    as-is. Replacing them with engine-native objects lets every gate be exercised
    deterministically rather than at the mercy of a particular seed.
    """
    events, last = _declining_probe()
    phase = prepare_analysis(
        events,
        learner_id="probe-s-0001",
        session_id=events[-1].session_id,
        clock=FixedClock(last),
    )
    assert phase.decision is not None
    assert phase.final_action is not None
    assert phase.uncertainty is not None
    assert phase.state is not None
    decision = PolicyDecision(
        decision=PolicyDecisionType.INTERVENE,
        selected=RECAP,
        detail="test: recap is the least intrusive option that applies",
        learner_id=phase.learner_id,
        session_id=phase.session_id,
        evaluated_at=phase.cut_at,
        upstream_verdict=phase.uncertainty.verdict,
        probability=phase.uncertainty.probability,
        confidence=phase.uncertainty.confidence,
        data_origin=phase.uncertainty.data_origin,
    )
    allowed = FinalAction(
        outcome=FinalActionOutcome.ALLOWED,
        action=RECAP.intervention_type,
        intervention_type=RECAP.intervention_type,
        reason="test: the envelope permits autonomous action",
        authority_level=AuthorityLevel.ALLOWED,
        confidence_weight=0.6,
        envelope_expired=False,
        requires_human_approval=False,
        escalation_path=(),
        available_human=HumanAvailability.UNKNOWN,
    )
    return replace(phase, decision=decision, final_action=allowed)


# ----------------------------------------------------------------------------------
# 1. Phase B consumes the frozen decision; it never recomputes Phase A.
# ----------------------------------------------------------------------------------


def test_phase_b_never_recomputes_phase_a(
    phase: AnalysisPhase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``execute_decision`` must not touch any Phase A decision-making entry point.

    If Phase B even *consulted* the model, the policy, the authority engine, or the
    uncertainty engine, the patched entry points below raise and the delivery fails with a
    stack trace pointing at the scope slip.
    """
    import api.lab.adapter as adapter

    monkeypatch.setattr(adapter, "get_cached_artifact", _boom("get_cached_artifact"))
    monkeypatch.setattr(adapter, "predict_with_artifact", _boom("predict_with_artifact"))
    monkeypatch.setattr(adapter, "InterventionPolicy", _boom("InterventionPolicy(decide)"))
    monkeypatch.setattr(adapter, "AuthorityEngine", _boom("AuthorityEngine(authorize)"))

    output = execute_decision(phase, clock=FixedClock(phase.events[-1].timestamp))
    result = output.response.results[0]
    assert result.intervention.delivery_status is DeliveryStatus.DELIVERED


def _boom(name: str) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            f"execute_decision recomputed Phase A by calling {name}; the seam is broken"
        )

    return _raise


# ----------------------------------------------------------------------------------
# 2. Every delivery status is reachable, and each is distinct.
# ----------------------------------------------------------------------------------


def test_not_selected_emits_nothing(phase: AnalysisPhase) -> None:
    output = execute_decision(
        replace(phase, decision=_decline_decision(phase, NonActionReason.PROBABILITY_BELOW_FLOOR)),
        clock=FixedClock(phase.events[-1].timestamp),
    )
    stage = _intervention(output)
    assert stage["status"] == DeliveryStatus.NOT_SELECTED.value
    assert stage["gate"] == "policy"
    assert stage["delivered"] is False
    assert stage["intervention_id"] is None
    assert stage["lifecycle_event_count"] == 0


def test_safety_refusal_is_selected_but_not_delivered(phase: AnalysisPhase) -> None:
    assert phase.final_action is not None
    refused = replace(
        phase.final_action,
        outcome=FinalActionOutcome.REFUSED,
        requires_human_approval=False,
        reason="test: structural rule prevents this context",
    )
    output = execute_decision(
        replace(phase, final_action=refused), clock=FixedClock(phase.events[-1].timestamp)
    )
    stage = _intervention(output)
    assert stage["status"] == DeliveryStatus.SELECTED_NOT_DELIVERED.value
    assert stage["gate"] == "safety"
    assert stage["delivered"] is False
    assert stage["lifecycle_event_count"] == 0, (
        "a start event for an intervention that was never sent is a lie in the stream"
    )


def test_human_approval_is_selected_but_not_delivered(phase: AnalysisPhase) -> None:
    assert phase.final_action is not None
    deferred = replace(
        phase.final_action,
        outcome=FinalActionOutcome.DEFERRED,
        requires_human_approval=True,
        reason="test: restricted action awaits a named human",
    )
    output = execute_decision(
        replace(phase, final_action=deferred), clock=FixedClock(phase.events[-1].timestamp)
    )
    stage = _intervention(output)
    assert stage["status"] == DeliveryStatus.SELECTED_NOT_DELIVERED.value
    assert stage["gate"] == "human_approval"
    assert stage["requires_human_approval"] is True
    assert stage["lifecycle_event_count"] == 0


def test_allowed_delivers_and_emits_lifecycle_events(phase: AnalysisPhase) -> None:
    output = execute_decision(phase, clock=FixedClock(phase.events[-1].timestamp))
    stage = _intervention(output)
    assert stage["status"] == DeliveryStatus.DELIVERED.value
    assert stage["gate"] == "delivered"
    assert stage["delivered"] is True
    assert stage["lifecycle_event_count"] == 2
    types = [event["event_type"] for event in stage["lifecycle_events"]]
    assert types == [
        EventType.INTERVENTION_STARTED.value,
        EventType.INTERVENTION_COMPLETED.value,
    ]


def test_three_way_status_split_is_mutually_exclusive(phase: AnalysisPhase) -> None:
    """NOT_SELECTED, SELECTED_NOT_DELIVERED, and DELIVERED are each one fact."""
    statuses: set[str] = set()
    variants: list[AnalysisPhase] = []

    no_choice = _decline_decision(phase, NonActionReason.STATE_NOT_INDICATED)
    variants.append(replace(phase, decision=no_choice))

    assert phase.final_action is not None
    refused = replace(
        phase.final_action,
        outcome=FinalActionOutcome.REFUSED,
        requires_human_approval=False,
    )
    variants.append(replace(phase, final_action=refused))

    variants.append(phase)

    for variant in variants:
        output = execute_decision(variant, clock=FixedClock(variant.events[-1].timestamp))
        statuses.add(_intervention(output)["status"])
    assert statuses == {
        DeliveryStatus.NOT_SELECTED.value,
        DeliveryStatus.SELECTED_NOT_DELIVERED.value,
        DeliveryStatus.DELIVERED.value,
    }


# ----------------------------------------------------------------------------------
# 3. The lifecycle events are native engine events that say what they are.
# ----------------------------------------------------------------------------------


def test_delivery_events_are_native_synthetic_envelopes(phase: AnalysisPhase) -> None:
    output = execute_decision(phase, clock=FixedClock(phase.events[-1].timestamp))
    result = output.response.results[0]
    assert result.intervention.intervention_id is not None
    for event in result.intervention.lifecycle_events:
        assert type(event) is EventEnvelope
        assert event.origin is DataOrigin.SYNTHETIC
        assert event.provenance is Provenance.SYNTHETIC_LABEL
        assert event.synthetic_stamp is not None, "a synthetic event needs its stamp"
        assert event.event_id.startswith(f"event-{result.intervention.intervention_id}")


def test_delivery_events_carry_the_engine_trigger_values(phase: AnalysisPhase) -> None:
    output = execute_decision(phase, clock=FixedClock(phase.events[-1].timestamp))
    result = output.response.results[0]
    started = result.intervention.lifecycle_events[0]
    assert started.event_type is EventType.INTERVENTION_STARTED
    assert phase.state is not None
    assert phase.uncertainty is not None
    assert started.payload.trigger_state == phase.state.state.value
    assert started.payload.trigger_probability == pytest.approx(
        phase.uncertainty.probability or 0.0
    )
    completed = result.intervention.lifecycle_events[1]
    assert isinstance(completed.payload, InterventionCompletedPayload)
    assert completed.payload.outcome == DELIVERY_OUTCOME_ACCEPTED


def test_delivery_anchors_at_the_decision_instant(phase: AnalysisPhase) -> None:
    """The lifecycle starts at ``decided_at``, not at the injected wall clock."""
    wall_clock = FixedClock(phase.events[-1].timestamp + timedelta(days=1))
    output = execute_decision(phase, clock=wall_clock)
    stage = _intervention(output)
    assert stage["started_at"] == phase.cut_at.isoformat()
    assert phase.decision is not None
    assert phase.cut_at == phase.decision.evaluated_at


# ----------------------------------------------------------------------------------
# 4. Replays are byte-identical including the delivery.
# ----------------------------------------------------------------------------------


def test_delivery_replay_is_byte_identical(phase: AnalysisPhase) -> None:
    clock = FixedClock(phase.events[-1].timestamp)
    first = execute_decision(phase, clock=clock)
    second = execute_decision(phase, clock=clock)
    raw_first = first.response.model_dump_json()
    raw_second = second.response.model_dump_json()
    assert raw_first == raw_second

    first_result = first.response.results[0]
    second_result = second.response.results[0]
    assert first_result.intervention.intervention_id == second_result.intervention.intervention_id
    assert first_result.intervention.lifecycle_events == second_result.intervention.lifecycle_events
    assert first_result.outcome_record == second_result.outcome_record
    assert first_result.evaluation_record == second_result.evaluation_record


# ----------------------------------------------------------------------------------
# 5. Outcome and evaluation run only after a real delivery, and are honest otherwise.
# ----------------------------------------------------------------------------------


def test_outcome_and_evaluation_run_after_delivery(phase: AnalysisPhase) -> None:
    output = execute_decision(phase, clock=FixedClock(phase.events[-1].timestamp))
    result = output.response.results[0]
    assert result.outcome_record is not None, "a delivered recap must be measured"
    assert result.evaluation_record is not None, "a delivered recap must be evaluated"


def test_outcome_and_evaluation_degrade_honestly_without_delivery(phase: AnalysisPhase) -> None:
    output = execute_decision(
        replace(phase, decision=_decline_decision(phase, NonActionReason.MINIMUM_EVIDENCE_NOT_MET)),
        clock=FixedClock(phase.events[-1].timestamp),
    )
    assert output.response.results[0].outcome_record is None
    assert output.response.results[0].evaluation_record is None
    stages = {stage["stage"]: stage["detail"] for stage in output.response.stage_log}
    assert stages["outcome"]["status"] == "INSUFFICIENT_DATA"
    assert stages["evaluation"]["status"] == "INSUFFICIENT_DATA"
    assert stages["outcome"]["reason"]
    assert stages["evaluation"]["reason"]


# ----------------------------------------------------------------------------------
# 6. The post-window reading is independent of the decision-time state.
# ----------------------------------------------------------------------------------


def test_post_window_reading_rebuilds_in_window_state_not_decision_state(
    phase: AnalysisPhase,
) -> None:
    horizon = phase.events[-1].timestamp + timedelta(minutes=15)
    in_window = tuple(
        event
        for event in phase.events
        if event.learner_id == phase.learner_id
        and event.session_id == phase.session_id
        and phase.cut_at <= event.timestamp < horizon
    )
    engine = TemporalEngine(clock=FixedClock(phase.cut_at))
    reading = post_window_reading(
        learner_id=phase.learner_id,
        in_window=in_window,
        window_start=phase.cut_at,
        profile=phase.profile,
        baseline=BaselineEngine(_BASELINE_SETTINGS),
        dimensions=BASELINE_DIMENSIONS,
        dimension_value=_dimension_value,
        engine=engine,
        feature_set_version=FEATURE_SET_V1,
    )
    assert reading is not None, "the probe must leave enough in-window evidence"
    assert reading is not phase.state
    assert reading.reference_time > phase.cut_at, (
        "the reading must be anchored inside the window, not at the decision instant"
    )
    assert reading.reference_time <= phase.events[-1].timestamp


def _intervention(output: Any) -> dict[str, Any]:
    stages: dict[str, Any] = {
        stage["stage"]: stage["detail"] for stage in output.response.stage_log
    }
    return dict(stages["intervention"])


@pytest.fixture(scope="module")
def phase() -> AnalysisPhase:
    return _deliverable_phase()
