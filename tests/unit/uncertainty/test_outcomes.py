"""Tests for the outcome vocabulary (Phase 9).

The value of these tests is almost entirely in what they refuse to construct. An outcome
that carries a probability it never issued, or a confidence band that contradicts its
verdict, is the shape of defect that reaches production and gets read as a finding.
"""

from __future__ import annotations

import pytest

from focus_engine.baseline.models import BaselineMaturity
from focus_engine.schemas.primitives import ConfidenceLevel, InferenceBasis
from focus_engine.uncertainty.explanation import (
    PredictionExplanation,
)
from focus_engine.uncertainty.outcomes import (
    REMEDIES,
    ConfidenceAssessment,
    Constraint,
    PredictionOutcome,
    Remedy,
    Verdict,
)

pytestmark = pytest.mark.unit


def _assessment(
    *,
    value: float = 0.6,
    binding: Constraint = Constraint.MODEL_ASSERTION,
    model_assertion: float | None = None,
    calibration_ceiling: float = 1.0,
    evidence_ceiling: float = 1.0,
    maturity_ceiling: float = 1.0,
    explanation_capped: bool = False,
) -> ConfidenceAssessment:
    """Build a consistent assessment, letting a test break one thing."""
    assertion = model_assertion if model_assertion is not None else value
    return ConfidenceAssessment(
        value=value,
        level=ConfidenceLevel.MEDIUM,
        model_assertion=assertion,
        calibration_ceiling=calibration_ceiling,
        evidence_ceiling=evidence_ceiling,
        maturity_ceiling=maturity_ceiling,
        binding=binding,
        explanation_capped=explanation_capped,
    )


def _resolved(**overrides: object) -> PredictionOutcome:
    """Build a resolved outcome, letting a test break one thing."""
    defaults: dict[str, object] = {
        "verdict": Verdict.RESOLVED,
        "confidence": ConfidenceLevel.MEDIUM,
        "assessment": _assessment(),
        "probability": 0.72,
        "is_positive": True,
        "basis": InferenceBasis.PERSONAL,
        "maturity": BaselineMaturity.DEVELOPING,
        "evidence_units": 150,
    }
    defaults.update(overrides)
    return PredictionOutcome(**defaults)  # type: ignore[arg-type]


class TestRemedies:
    """Each negative verdict names the thing that would change it."""

    def test_every_verdict_has_exactly_one_remedy(self) -> None:
        assert set(REMEDIES) == set(Verdict)
        assert len(REMEDIES) == len(Verdict)

    def test_the_two_negative_verdicts_have_different_remedies(self) -> None:
        """The distinction the research principles require, held in configuration."""
        assert REMEDIES[Verdict.INSUFFICIENT_DATA] is Remedy.COLLECT_MORE_DATA
        assert REMEDIES[Verdict.UNKNOWN] is Remedy.REPLACE_OR_RETRAIN_MODEL
        assert REMEDIES[Verdict.UNKNOWN] is not REMEDIES[Verdict.INSUFFICIENT_DATA]

    def test_a_refusal_is_fixed_at_the_input_not_by_waiting(self) -> None:
        assert REMEDIES[Verdict.REFUSED] is Remedy.FIX_THE_INPUT

    def test_outcomes_resolve_their_own_remedy(self) -> None:
        assert _resolved().remedy is Remedy.NONE
        assert (
            PredictionOutcome(
                verdict=Verdict.UNKNOWN,
                confidence=ConfidenceLevel.UNKNOWN,
                refusal_reason="because",
            ).remedy
            is Remedy.REPLACE_OR_RETRAIN_MODEL
        )


class TestConfidenceAssessment:
    """A confidence must be re-derivable from the ceilings recorded beside it."""

    def test_a_consistent_assessment_is_accepted(self) -> None:
        assessment = _assessment(
            value=0.5,
            binding=Constraint.MATURITY,
            model_assertion=0.99,
            maturity_ceiling=0.5,
        )
        assert assessment.value == 0.5

    def test_a_value_that_is_not_the_lowest_ceiling_is_refused(self) -> None:
        """Storing this would leave a number nobody can trace back to its inputs."""
        with pytest.raises(ValueError, match="not the lowest of its ceilings"):
            _assessment(value=0.9, model_assertion=0.99, maturity_ceiling=0.5)

    @pytest.mark.parametrize("field", ["value", "calibration_ceiling", "maturity_ceiling"])
    def test_out_of_range_values_are_refused(self, field: str) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[0.0, 1.0\]"):
            _assessment(**{field: 1.5})

    def test_a_model_assertion_below_a_half_is_permitted(self) -> None:
        """A prediction near its own threshold can be an undecided one.

        At a threshold of 0.65 a probability of 0.60 emits a negative decision while the
        model still favours the positive side, so its confidence in what it emitted is
        0.40. That is a real low-confidence prediction, not a contradiction.
        """
        assert _assessment(value=0.40, model_assertion=0.40).value == 0.40

    def test_describe_names_the_binding_constraint_and_every_ceiling(self) -> None:
        text = _assessment(
            value=0.5, binding=Constraint.MATURITY, model_assertion=0.99, maturity_ceiling=0.5
        ).describe()
        assert "maturity" in text
        assert "0.990" in text
        assert "0.500" in text

    def test_an_explanation_capped_assessment_says_so(self) -> None:
        assert "nothing was attributable" in _assessment(explanation_capped=True).describe()


class TestResolvedOutcome:
    """A resolved outcome carries a number and the derivation of it."""

    def test_a_well_formed_outcome_is_accepted(self) -> None:
        outcome = _resolved()
        assert outcome.is_resolved
        assert outcome.verdict is Verdict.RESOLVED

    def test_a_resolved_outcome_without_a_probability_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must carry both a probability"):
            _resolved(probability=None)

    def test_a_resolved_outcome_without_an_assessment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must carry both a probability"):
            _resolved(assessment=None)

    def test_a_resolved_outcome_must_say_whether_the_threshold_was_met(self) -> None:
        with pytest.raises(ValueError, match="whether the threshold was met"):
            _resolved(is_positive=None)

    def test_a_resolved_outcome_must_be_banded_in_the_numeric_range(self) -> None:
        with pytest.raises(ValueError, match="banded high, medium, or low"):
            _resolved(confidence=ConfidenceLevel.UNKNOWN)

    def test_an_out_of_range_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="threshold must lie"):
            _resolved(threshold=1.5)


class TestRefusingOutcomes:
    """A verdict that declined to issue a number may not carry one."""

    @pytest.mark.parametrize("verdict", [Verdict.INSUFFICIENT_DATA, Verdict.UNKNOWN])
    def test_a_probability_may_not_accompany_a_refusal(self, verdict: Verdict) -> None:
        with pytest.raises(ValueError, match="must not carry probability"):
            PredictionOutcome(
                verdict=verdict,
                confidence=ConfidenceLevel.LOW,
                probability=0.9,
                refusal_reason="because",
            )

    def test_an_assessment_may_not_accompany_a_refusal(self) -> None:
        with pytest.raises(ValueError, match="must not carry assessment"):
            PredictionOutcome(
                verdict=Verdict.UNKNOWN,
                confidence=ConfidenceLevel.UNKNOWN,
                assessment=_assessment(),
                refusal_reason="because",
            )

    def test_insufficient_data_must_be_banded_as_such(self) -> None:
        with pytest.raises(ValueError, match="banded insufficient_data"):
            PredictionOutcome(
                verdict=Verdict.INSUFFICIENT_DATA,
                confidence=ConfidenceLevel.LOW,
                refusal_reason="because",
            )

    def test_unknown_must_be_banded_as_such(self) -> None:
        with pytest.raises(ValueError, match="banded unknown"):
            PredictionOutcome(
                verdict=Verdict.UNKNOWN, confidence=ConfidenceLevel.LOW, refusal_reason="because"
            )

    def test_a_refusal_carries_no_confidence_at_all(self) -> None:
        """Nothing was scored, so no confidence was assessed.

        Reporting a band here would assert that the system had an opinion about the input,
        and the only bands available describe opinions about predictions.
        """
        with pytest.raises(ValueError, match="no confidence band"):
            PredictionOutcome(
                verdict=Verdict.REFUSED,
                confidence=ConfidenceLevel.LOW,
                refusal_reason="bad payload",
            )

    def test_a_refusal_may_name_the_missing_features(self) -> None:
        outcome = PredictionOutcome(
            verdict=Verdict.REFUSED,
            refusal_reason="schema mismatch",
            missing_features=("performance_recent_accuracy",),
        )
        assert outcome.missing_features == ("performance_recent_accuracy",)
        assert outcome.confidence is None

    @pytest.mark.parametrize("verdict", [Verdict.INSUFFICIENT_DATA, Verdict.UNKNOWN])
    def test_a_negative_verdict_must_explain_itself(self, verdict: Verdict) -> None:
        """An unexplained negative answer cannot be acted on."""
        band = (
            ConfidenceLevel.INSUFFICIENT_DATA
            if verdict is Verdict.INSUFFICIENT_DATA
            else ConfidenceLevel.UNKNOWN
        )
        with pytest.raises(ValueError, match="must state why it declined"):
            PredictionOutcome(verdict=verdict, confidence=band)


class TestOutcomeDescription:
    """What an outcome says when someone reads it."""

    def test_a_resolved_outcome_names_the_number_and_its_confidence(self) -> None:
        text = _resolved().describe()
        assert "0.7200" in text
        assert "positive" in text
        assert "medium" in text

    def test_a_refusal_leads_with_its_reason(self) -> None:
        outcome = PredictionOutcome(
            verdict=Verdict.REFUSED, refusal_reason="the schema did not match"
        )
        text = outcome.describe()
        assert text.startswith("refused")
        assert "the schema did not match" in text
        assert "fix_the_input" in text

    def test_an_empty_explanation_is_reported_rather_than_hidden(self) -> None:
        explanation = PredictionExplanation(base_probability=0.5, min_contribution=10.0)
        assert explanation.is_empty
        assert "no feature moved" in explanation.describe()
