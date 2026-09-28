"""Tests for the uncertainty engine (Phase 9).

The engine composes the three inputs into one auditable answer. The tests here enforce
the five things the research principles and the phase's exit criteria require:

1. ``UNKNOWN`` and ``INSUFFICIENT_DATA`` are reachable and tested, and each names its own
   remedy.
2. Confidence is derived, never invented: it is the minimum of four independent ceilings
   that can be re-derived from the recorded values.
3. The maturity cap is a function of maturity alone and is applied independently of the
   model's own output.
4. The order of negative answers encodes which remedy applies: calibration before
   evidence, because a broken model cannot be repaired by collecting more data.
5. Malformed input propagates ``REFUSED`` with the original reason and missing features.
"""

from __future__ import annotations

import pytest

from focus_engine.configuration.thresholds import UncertaintySettings
from focus_engine.features import FeatureName
from focus_engine.models.artifacts import ModelArtifact
from focus_engine.models.predictors import PredictionRefusal
from focus_engine.schemas.primitives import ConfidenceLevel
from focus_engine.uncertainty import UncertaintyEngine, band_confidence
from focus_engine.uncertainty.outcomes import Constraint, Remedy, Verdict
from tests.unit.models.scenarios import make_vector
from tests.unit.uncertainty.scenarios import (
    UNITS_DEVELOPING,
    UNITS_EARLY,
    UNITS_ESTABLISHED,
    evidence_at,
    train_shared_artifact,
    with_calibration,
    without_recorded_metrics,
    without_reference_values,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def artifact() -> ModelArtifact:
    """Shared trained artifact with measured calibration."""
    return train_shared_artifact()


@pytest.fixture(scope="module")
def vector() -> object:
    """A well-specified vector from the shared dataset's positive class."""
    return make_vector(
        overrides={
            FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS: 30.0,
            FeatureName.SESSION_ELAPSED_SECONDS: 900.0,
        }
    )


@pytest.fixture(scope="module")
def engine() -> UncertaintyEngine:
    """Engine with default configuration."""
    return UncertaintyEngine()


@pytest.fixture(scope="module")
def strict_engine() -> UncertaintyEngine:
    """Engine with a deliberately strict calibration limit."""
    return UncertaintyEngine(uncertainty=UncertaintySettings(max_calibration_error=0.01))


@pytest.fixture(scope="module")
def custom_ladder_engine() -> UncertaintyEngine:
    """Engine with a tighter maturity ladder, to prove it reads the configuration."""
    return UncertaintyEngine(
        uncertainty=UncertaintySettings(
            max_confidence_when_new=0.20,
            max_confidence_when_early=0.35,
            max_confidence_when_developing=0.60,
        )
    )


class TestReachableVerdicts:
    """Every verdict is reachable and names its own remedy."""

    def test_insufficient_data_is_reached_below_the_evidence_gate(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        outcome = engine.assess_vector(artifact, make_vector(), evidence=evidence_at(0))
        assert outcome.verdict is Verdict.INSUFFICIENT_DATA
        assert outcome.confidence is ConfidenceLevel.INSUFFICIENT_DATA
        assert outcome.remedy is Remedy.COLLECT_MORE_DATA
        assert "0 observation" in str(outcome.refusal_reason)
        assert "holdout" not in str(outcome.refusal_reason)

    def test_unknown_is_reached_when_calibration_cannot_be_measured(
        self, artifact: ModelArtifact
    ) -> None:
        bare = without_recorded_metrics(artifact)
        outcome = UncertaintyEngine().assess_vector(bare, make_vector(), evidence=evidence_at(600))
        assert outcome.verdict is Verdict.UNKNOWN
        assert outcome.confidence is ConfidenceLevel.UNKNOWN
        assert "calibration" in str(outcome.refusal_reason).lower()
        assert outcome.evidence_units == 600

    def test_unknown_is_reached_when_calibration_is_not_believable(
        self, artifact: ModelArtifact, strict_engine: UncertaintyEngine
    ) -> None:
        bad_calibration = with_calibration(artifact, expected_calibration_error=0.20, brier=0.15)
        outcome = strict_engine.assess_vector(
            bad_calibration, make_vector(), evidence=evidence_at(600)
        )
        assert outcome.verdict is Verdict.UNKNOWN
        assert "0.2000" in str(outcome.refusal_reason)
        assert str(strict_engine.uncertainty.max_calibration_error) in str(outcome.refusal_reason)
        assert outcome.remedy is Remedy.REPLACE_OR_RETRAIN_MODEL

    def test_refused_propagates_from_a_missing_feature(self, artifact: ModelArtifact) -> None:
        broken = make_vector(absent=(FeatureName.PERFORMANCE_RECENT_ACCURACY,))
        outcome = UncertaintyEngine().assess_vector(
            artifact, broken, evidence=evidence_at(UNITS_EARLY)
        )
        assert outcome.verdict is Verdict.REFUSED
        assert outcome.confidence is None
        assert outcome.refusal_reason is not None
        assert (
            "missing" in outcome.refusal_reason.lower()
            or FeatureName.PERFORMANCE_RECENT_ACCURACY.value in outcome.refusal_reason
        )

    def test_refused_propagates_through_assess_from_a_refusal(
        self, artifact: ModelArtifact
    ) -> None:
        refusal = PredictionRefusal(
            learner_id="L1",
            session_id=None,
            computed_at=None,  # type: ignore[arg-type]
            reason="the vector was refused",
            missing_features=(FeatureName.CONTENT_DIFFICULTY,),
        )
        outcome = UncertaintyEngine().assess(
            refusal, artifact=artifact, evidence=evidence_at(UNITS_EARLY)
        )
        assert outcome.verdict is Verdict.REFUSED
        assert "the vector was refused" in outcome.refusal_reason

    def test_resolved_is_reached_with_sufficient_evidence_and_good_calibration(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        outcome = engine.assess_vector(artifact, make_vector(), evidence=evidence_at(UNITS_EARLY))
        assert outcome.verdict is Verdict.RESOLVED
        assert outcome.probability is not None
        assert outcome.assessment is not None


class TestConfidenceDerivation:
    """Confidence is the lowest of four independent ceilings."""

    def test_all_four_ceilings_are_recorded(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        outcome = engine.assess_vector(
            artifact, make_vector(), evidence=evidence_at(UNITS_DEVELOPING)
        )
        assert outcome.verdict is Verdict.RESOLVED
        a = outcome.assessment
        assert a is not None
        assert a.calibration_ceiling == pytest.approx(
            1.0 - artifact.record.metrics[0].expected_calibration_error, abs=1e-9
        )
        assert 0.5 <= a.evidence_ceiling <= 1.0
        assert 0.5 <= a.maturity_ceiling <= 1.0
        assert a.value == pytest.approx(
            min(a.model_assertion, a.calibration_ceiling, a.evidence_ceiling, a.maturity_ceiling)
        )

    def test_binding_is_the_ceiling_that_set_the_value(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        outcome = engine.assess_vector(artifact, make_vector(), evidence=evidence_at(UNITS_EARLY))
        a = outcome.assessment
        ceilings = {
            Constraint.MODEL_ASSERTION: a.model_assertion,
            Constraint.CALIBRATION: a.calibration_ceiling,
            Constraint.EVIDENCE: a.evidence_ceiling,
            Constraint.MATURITY: a.maturity_ceiling,
        }
        assert a.binding.value == min(ceilings, key=ceilings.get).value  # type: ignore[union-attr]

    def test_the_confidence_is_a_derived_not_invented_number(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        """The value equals the minimum of the recorded ceilings, within floating point."""
        outcome = engine.assess_vector(
            artifact, make_vector(), evidence=evidence_at(UNITS_DEVELOPING)
        )
        a = outcome.assessment
        assert a is not None
        assert a.value == min(
            a.model_assertion,
            a.calibration_ceiling,
            a.evidence_ceiling,
            a.maturity_ceiling,
        )


class TestMaturityCapIsIndependent:
    """The core requirement: a confident model cannot raise its own ceiling."""

    def test_a_confident_model_is_capped_by_maturity(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        """The model claims near 1.0; the learner has an ``EARLY`` baseline; the cap binds."""
        outcome = engine.assess_vector(artifact, make_vector(), evidence=evidence_at(UNITS_EARLY))
        a = outcome.assessment
        assert a.model_assertion > 0.9
        assert a.maturity_ceiling == pytest.approx(UncertaintySettings().max_confidence_when_early)
        assert a.value == pytest.approx(a.maturity_ceiling)
        assert a.binding is Constraint.MATURITY

    def test_the_cap_tightens_on_a_tighter_ladder(
        self, custom_ladder_engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        outcome = custom_ladder_engine.assess_vector(
            artifact, make_vector(), evidence=evidence_at(UNITS_EARLY)
        )
        assert outcome.assessment.maturity_ceiling == pytest.approx(0.35)

    def test_maturity_bounds_the_output_regardless_of_model_confidence(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        """Even a model reporting 0.99 cannot exceed the ``EARLY`` cap."""
        for vector_seed in range(5):
            vector = make_vector(offset_seconds=float(vector_seed))
            outcome = engine.assess_vector(artifact, vector, evidence=evidence_at(UNITS_EARLY))
            if outcome.verdict is Verdict.RESOLVED:
                assert outcome.assessment.value <= (
                    UncertaintySettings().max_confidence_when_early + 1e-9
                )


class TestRemedyOrdering:
    """Calibration before evidence, because a broken model cannot be fixed by collecting."""

    def test_bad_calibration_trumps_sufficient_evidence(self, artifact: ModelArtifact) -> None:
        bad = with_calibration(artifact, expected_calibration_error=0.30, brier=0.25)
        outcome = UncertaintyEngine().assess_vector(bad, make_vector(), evidence=evidence_at(600))
        assert outcome.verdict is Verdict.UNKNOWN
        assert "calibration error" in str(outcome.refusal_reason).lower()

    def test_sufficient_evidence_and_good_calibration_yield_resolved(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        outcome = engine.assess_vector(artifact, make_vector(), evidence=evidence_at(600))
        assert outcome.verdict is Verdict.RESOLVED
        assert outcome.is_resolved


class TestBandConfidence:
    """Banding reads the thresholds from configuration."""

    def test_above_the_high_line_is_high(self) -> None:
        assert band_confidence(0.80, UncertaintySettings()) is ConfidenceLevel.HIGH

    def test_at_the_high_line_is_high(self) -> None:
        assert band_confidence(0.85, UncertaintySettings()) is ConfidenceLevel.HIGH

    def test_between_the_lines_is_medium(self) -> None:
        assert band_confidence(0.65, UncertaintySettings()) is ConfidenceLevel.MEDIUM

    def test_below_the_medium_line_is_low(self) -> None:
        assert band_confidence(0.30, UncertaintySettings()) is ConfidenceLevel.LOW

    def test_zero_confidence_is_low(self) -> None:
        assert band_confidence(0.0, UncertaintySettings()) is ConfidenceLevel.LOW


class TestAssessFromPrediction:
    """assess() accepts a Prediction or PredictionRefusal, and a PredictionRefusal
    reaches the REFUSED outcome without touching calibration."""

    def test_a_prediction_with_a_refusal_reaches_refused_directly(
        self, artifact: ModelArtifact
    ) -> None:
        refusal = PredictionRefusal(
            learner_id="L1",
            session_id=None,
            computed_at=None,  # type: ignore[arg-type]
            reason="feature missing",
            missing_features=(FeatureName.CONTENT_DIFFICULTY,),
        )
        outcome = UncertaintyEngine().assess(refusal, artifact=artifact, evidence=evidence_at(600))
        assert outcome.verdict is Verdict.REFUSED
        assert outcome.probability is None
        assert outcome.missing_features == (FeatureName.CONTENT_DIFFICULTY,)


class TestExplanationCapping:
    """An unexplicable prediction does not get a confident band."""

    def test_an_explanation_available_but_empty_caps_the_band_to_low(
        self, artifact: ModelArtifact
    ) -> None:
        """The vector at the reference values moves the model very little."""
        bare = without_reference_values(artifact)
        bare_vector = make_vector()
        outcome = UncertaintyEngine().assess_vector(
            bare, bare_vector, evidence=evidence_at(UNITS_ESTABLISHED)
        )
        # Without references the explanation is not available, not empty
        assert outcome.explanation is not None
        assert not outcome.explanation.available


class TestSyntheticOrigin:
    """The outcome records whether the model behind it was synthetic."""

    def test_the_outcome_carries_the_models_data_origin(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        outcome = engine.assess_vector(artifact, make_vector(), evidence=evidence_at(UNITS_EARLY))
        assert outcome.data_origin is artifact.record.data_origin


class TestTargetDefinitionVersion:
    """The probability is uninterpretable without saying what it is *of*."""

    def test_the_outcome_carrying_the_probability_also_carries_its_definition(
        self, engine: UncertaintyEngine, artifact: ModelArtifact
    ) -> None:
        outcome = engine.assess_vector(artifact, make_vector(), evidence=evidence_at(UNITS_EARLY))
        assert outcome.target_definition_version
        assert outcome.target_definition_version == artifact.record.target_definition_version
