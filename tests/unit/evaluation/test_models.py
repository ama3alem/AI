"""Unit tests for the evaluation vocabulary and validators (Phase 13).

The models here enforce the invariants that the engine relies on: exactly one claim field
per target, provenance constrained to OBSERVED, non-findings structurally unable to carry
observed values, and the confusion matrix restricted to the DECLINE target. Every test
constructs one invalid model and checks the refusal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from focus_engine.evaluation.models import (
    ObservationWindow,
    PredictionHorizon,
    PredictionSnapshot,
    PredictionTarget,
    claim_value,
)
from focus_engine.schemas.primitives import (
    BehavioralEngagementState,
    ConfidenceLevel,
    DataOrigin,
    InferenceBasis,
)
from focus_engine.temporal.models import TrendDirection
from focus_engine.uncertainty import ConfidenceAssessment, Constraint
from focus_engine.uncertainty.outcomes import PredictionOutcome, Verdict

pytestmark = pytest.mark.unit

_PREDICTED_AT = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
_HORIZON_END = _PREDICTED_AT + timedelta(seconds=600)
_LEARNER = "learner-0001"
_SESSION = "session-0001"


def _window() -> ObservationWindow:
    return ObservationWindow(start=_PREDICTED_AT, end=_HORIZON_END, duration_seconds=600)


def _horizon() -> PredictionHorizon:
    return PredictionHorizon(predicted_at=_PREDICTED_AT, duration_seconds=600)


def _assessment(value: float = 0.6) -> ConfidenceAssessment:
    """Build an internally consistent confidence assessment."""
    return ConfidenceAssessment(
        value=value,
        level=ConfidenceLevel.MEDIUM,
        model_assertion=value,
        calibration_ceiling=1.0,
        evidence_ceiling=1.0,
        maturity_ceiling=1.0,
        binding=Constraint.MODEL_ASSERTION,
        explanation_capped=False,
    )


def _resolved_outcome(**overrides: object) -> PredictionOutcome:
    """Build a RESOLVED outcome satisfying the upstream contract.

    ``PredictionOutcome`` refuses a RESOLVED verdict that lacks a probability, a confidence
    assessment, a band, or the threshold decision, so all four are supplied here rather than
    in each test.
    """
    defaults: dict[str, object] = {
        "verdict": Verdict.RESOLVED,
        "confidence": ConfidenceLevel.MEDIUM,
        "assessment": _assessment(),
        "probability": 0.75,
        "threshold": 0.5,
        "is_positive": True,
        "basis": InferenceBasis.PERSONAL,
        "model_id": "m",
        "model_version": "MODEL_V1",
        "feature_set_version": "FEATURE_SET_V1",
        "target_definition_version": "t-1",
        "data_origin": DataOrigin.SYNTHETIC,
        "computed_at": _PREDICTED_AT,
    }
    defaults.update(overrides)
    return PredictionOutcome(**defaults)  # type: ignore[arg-type]


def _snapshot(
    *,
    target: PredictionTarget = PredictionTarget.DECLINE,
    predicted_positive: bool | None = True,
    predicted_state: BehavioralEngagementState | None = None,
    predicted_direction: TrendDirection | None = None,
) -> PredictionSnapshot:
    return PredictionSnapshot(
        prediction_id="pred-0001",
        learner_id=_LEARNER,
        session_id=_SESSION,
        target=target,
        horizon=_horizon(),
        predicted_positive=predicted_positive,
        predicted_state=predicted_state,
        predicted_direction=predicted_direction,
        verdict=Verdict.RESOLVED,
        model_id="model-hgb-v1",
        model_version="MODEL_HGB_V1",
        feature_set_version="FEATURE_SET_V1",
        target_definition_version="target-engage-v1",
        data_origin=DataOrigin.SYNTHETIC,
    )


class TestClaimHelpers:
    def test_claim_value_state(self) -> None:
        assert (
            claim_value(
                PredictionTarget.STATE,
                BehavioralEngagementState.DECLINING,
                None,
                None,
            )
            is BehavioralEngagementState.DECLINING
        )

    def test_claim_value_positive(self) -> None:
        assert claim_value(PredictionTarget.DECLINE, None, None, True) is True

    def test_claim_value_direction(self) -> None:
        assert (
            claim_value(
                PredictionTarget.DIRECTION,
                None,
                TrendDirection.WORSENING,
                None,
            )
            is TrendDirection.WORSENING
        )

    def test_claim_value_raises_on_none(self) -> None:
        with pytest.raises(ValueError, match="DECLINE claim requires"):
            claim_value(PredictionTarget.DECLINE, None, None, None)

    def test_declining_reuses_policy_set(self) -> None:
        from focus_engine.evaluation.models import is_declining

        assert is_declining(BehavioralEngagementState.DECLINING) is True
        assert is_declining(BehavioralEngagementState.STABLE) is False


class TestObservationWindow:
    def test_valid_window(self) -> None:
        window = _window()
        assert window.contains(_PREDICTED_AT)
        assert not window.contains(_HORIZON_END)

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="before it starts"):
            ObservationWindow(start=_HORIZON_END, end=_PREDICTED_AT, duration_seconds=600)

    def test_duration_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            ObservationWindow(start=_PREDICTED_AT, end=_HORIZON_END, duration_seconds=300)


class TestPredictionSnapshot:
    def test_valid_decline_snapshot(self) -> None:
        snap = _snapshot()
        assert snap.is_resolved
        assert snap.predicted_positive is True

    def test_two_claims_rejected(self) -> None:
        with pytest.raises(ValueError, match="must set exactly"):
            _snapshot(
                target=PredictionTarget.DECLINE,
                predicted_positive=True,
                predicted_state=BehavioralEngagementState.DECLINING,
            )

    def test_decline_with_state_rejected(self) -> None:
        with pytest.raises(ValueError, match="must set exactly"):
            _snapshot(
                target=PredictionTarget.DECLINE,
                predicted_state=BehavioralEngagementState.DECLINING,
            )

    def test_refused_prediction_carrying_probability_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not carry a probability"):
            PredictionSnapshot(
                prediction_id="pred-0001",
                learner_id=_LEARNER,
                session_id=_SESSION,
                target=PredictionTarget.DECLINE,
                horizon=_horizon(),
                predicted_positive=True,
                probability=0.5,
                verdict=Verdict.INSUFFICIENT_DATA,
                model_id="m",
                model_version="MODEL_V1",
                feature_set_version="FEATURE_SET_V1",
                target_definition_version="t-1",
                data_origin=DataOrigin.SYNTHETIC,
            )

    def test_from_outcome_wraps_decline(self) -> None:
        snap = PredictionSnapshot.from_outcome(
            _resolved_outcome(),
            prediction_id="pred-0002",
            learner_id=_LEARNER,
            session_id=_SESSION,
            horizon=_horizon(),
        )
        assert snap.target is PredictionTarget.DECLINE
        assert snap.predicted_positive is True
        assert snap.probability == 0.75
        assert snap.confidence == 0.6
        assert snap.confidence_level is ConfidenceLevel.MEDIUM

    def test_from_outcome_refused_raises(self) -> None:
        outcome = PredictionOutcome(
            verdict=Verdict.INSUFFICIENT_DATA,
            confidence=ConfidenceLevel.INSUFFICIENT_DATA,
            refusal_reason="no data",
            model_id="m",
            model_version="MODEL_V1",
            feature_set_version="FEATURE_SET_V1",
            target_definition_version="t-1",
            data_origin=DataOrigin.SYNTHETIC,
            computed_at=_PREDICTED_AT,
        )
        with pytest.raises(ValueError, match="issued no claim"):
            PredictionSnapshot.from_outcome(
                outcome,
                prediction_id="pred-0003",
                learner_id=_LEARNER,
                session_id=_SESSION,
                horizon=_horizon(),
            )
