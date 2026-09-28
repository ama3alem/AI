"""Shared builders for the policy tests (Phase 10).

The policy is a pure function of a small number of inputs, so the tests state each input
directly rather than arranging a whole session to produce it. A test that has to replay a
simulated session in order to check a cooldown is testing two things at once, and when it
fails it is not obvious which.

The default outcome here clears every restraint by default. That is deliberate: a test
about one restraint should not have to defeat the other five to reach it, and a test that
has to is a test that will silently start testing the wrong thing when a default changes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from focus_engine.baseline.models import BaselineMaturity
from focus_engine.configuration.thresholds import InterventionSettings
from focus_engine.policy.engine import InterventionPolicy
from focus_engine.policy.history import (
    DeliveredIntervention,
    InterventionHistory,
)
from focus_engine.policy.models import InterventionCandidate, PolicyDecision
from focus_engine.schemas.primitives import (
    BehavioralEngagementState,
    ConfidenceLevel,
    DataOrigin,
    InferenceBasis,
)
from focus_engine.uncertainty.outcomes import (
    ConfidenceAssessment,
    Constraint,
    PredictionOutcome,
    Verdict,
)
from focus_engine.utils.clock import FixedClock

#: A fixed instant every test starts from. Written out rather than taken from the clock so
#: that a failure message reads as a date instead of as "now".
START = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)

#: Long enough to be clearly outside any default window.
DISTANT_PAST = START - timedelta(days=30)

LEARNER = "learner-0001"
SESSION = "session-0001"


def assessment(
    *,
    value: float = 0.8,
    band: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> ConfidenceAssessment:
    """Build a self-consistent confidence assessment.

    Args:
        value: The confidence value. It must be the lowest of the ceilings or the
            assessment will refuse to construct, which is the point.
        band: The band to record. The caller states it, because the policy's per-candidate
            bars are what these tests are exercising and the band cannot be inferred from
            the value without re-implementing the engine's thresholds.

    Returns:
        An assessment whose value, ceilings, and band agree.
    """
    return ConfidenceAssessment(
        value=value,
        level=band,
        model_assertion=value,
        calibration_ceiling=1.0,
        evidence_ceiling=1.0,
        maturity_ceiling=1.0,
        binding=Constraint.MODEL_ASSERTION,
    )


def resolved(
    *,
    probability: float = 0.9,
    band: ConfidenceLevel = ConfidenceLevel.HIGH,
    evidence_units: int = 200,
    threshold: float = 0.5,
) -> PredictionOutcome:
    """Build a resolved outcome that clears every restraint by default.

    Args:
        probability: The calibrated decline probability.
        band: The confidence band.
        evidence_units: Observations behind the decision. Well above the default policy
            floor, so a test about the evidence restraint has to lower it deliberately.
        threshold: The model's own decision threshold.

    Returns:
        A resolved outcome carrying a probability, an assessment, and a band.
    """
    return PredictionOutcome(
        verdict=Verdict.RESOLVED,
        confidence=band,
        assessment=assessment(value=0.8, band=band),
        probability=probability,
        is_positive=probability >= threshold,
        threshold=threshold,
        basis=InferenceBasis.PERSONAL,
        maturity=BaselineMaturity.ESTABLISHED,
        evidence_units=evidence_units,
        model_id="model-0001",
        model_version="MODEL_V1",
        feature_set_version="FEATURE_SET_V1",
        target_definition_version="TARGET_V1",
        data_origin=DataOrigin.SYNTHETIC,
        computed_at=START,
    )


def insufficient_data(reason: str = "not enough history about this learner") -> PredictionOutcome:
    """Build an ``INSUFFICIENT_DATA`` outcome.

    Args:
        reason: The refusal reason to record.

    Returns:
        A refusal that carries neither a probability nor a confidence assessment.
    """
    return PredictionOutcome(
        verdict=Verdict.INSUFFICIENT_DATA,
        confidence=ConfidenceLevel.INSUFFICIENT_DATA,
        evidence_units=3,
        refusal_reason=reason,
        data_origin=DataOrigin.SYNTHETIC,
        computed_at=START,
    )


def unknown(reason: str = "the model's calibration has never been measured") -> PredictionOutcome:
    """Build an ``UNKNOWN`` outcome.

    Args:
        reason: The refusal reason to record.

    Returns:
        A refusal that carries neither a probability nor a confidence assessment.
    """
    return PredictionOutcome(
        verdict=Verdict.UNKNOWN,
        confidence=ConfidenceLevel.UNKNOWN,
        evidence_units=200,
        refusal_reason=reason,
        data_origin=DataOrigin.SYNTHETIC,
        computed_at=START,
    )


def delivery(
    intervention_type: str = "recap",
    *,
    minutes_ago: float = 30.0,
    outcome: str | None = None,
    session_id: str = SESSION,
    suffix: str = "0",
) -> DeliveredIntervention:
    """Build one recorded delivery.

    Args:
        intervention_type: The type that was delivered.
        minutes_ago: How long before ``START`` it was delivered.
        outcome: The reported outcome, or ``None`` for no completion.
        session_id: The session it was delivered in.
        suffix: A discriminator so several deliveries get distinct identifiers.

    Returns:
        A delivery positioned relative to ``START``.
    """
    return DeliveredIntervention(
        intervention_id=f"intervention-{minutes_ago:.0f}-{suffix}",
        session_id=session_id,
        intervention_type=intervention_type,
        delivered_at=START - timedelta(minutes=minutes_ago),
        outcome=outcome,
        responded_at=None if outcome is None else START - timedelta(minutes=minutes_ago / 2),
    )


def history(
    *deliveries: DeliveredIntervention,
    learner_id: str = LEARNER,
) -> InterventionHistory:
    """Build a history from deliveries, in order.

    Args:
        deliveries: The deliveries, earliest first.
        learner_id: The learner the history belongs to.

    Returns:
        An ordered history. Passing deliveries out of order is refused rather than
        silently sorted, so a test that builds a nonsense timeline fails loudly.
    """
    return InterventionHistory(learner_id=learner_id, interventions=tuple(deliveries))


def empty_history(learner_id: str = LEARNER) -> InterventionHistory:
    """Build a history with no deliveries.

    Args:
        learner_id: The learner the history belongs to.

    Returns:
        An empty history.
    """
    return InterventionHistory(learner_id=learner_id)


def policy(
    *,
    settings: InterventionSettings | None = None,
    catalogue: tuple[InterventionCandidate, ...] | None = None,
) -> InterventionPolicy:
    """Build a policy with a fixed clock.

    Args:
        settings: The restraint configuration. Defaults to the shipped configuration.
        catalogue: The candidate catalogue. Defaults to the shipped catalogue.

    Returns:
        A policy whose clock reads ``START``, so a test that does not care about time gets
        the same instant every run.
    """
    return InterventionPolicy(
        settings=settings or InterventionSettings(),
        catalogue=catalogue if catalogue is not None else InterventionPolicy().catalogue,
        clock=FixedClock(START),
    )


def decide(
    the_policy: InterventionPolicy,
    *,
    outcome: PredictionOutcome | None = None,
    the_history: InterventionHistory | None = None,
    state: BehavioralEngagementState = BehavioralEngagementState.DECLINING,
    now: datetime = START,
    learner_id: str = LEARNER,
    session_id: str = SESSION,
) -> PolicyDecision:
    """Run one decision with the common arguments filled in.

    Args:
        the_policy: The policy to ask.
        outcome: The upstream outcome. Defaults to one that clears every restraint.
        the_history: The learner's history. Defaults to an empty one.
        state: The observed behavioural state.
        now: The decision instant.
        learner_id: The learner the decision is about.
        session_id: The session the decision is made in.

    Returns:
        The resulting policy decision.
    """
    return the_policy.decide(
        outcome if outcome is not None else resolved(),
        the_history if the_history is not None else empty_history(learner_id),
        learner_id=learner_id,
        session_id=session_id,
        state=state,
        now=now,
    )
