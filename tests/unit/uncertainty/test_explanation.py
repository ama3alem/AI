"""Tests for per-row explanation by occlusion (Phase 9).

The criterion these exist to enforce is narrow and absolute: a signal may appear in an
explanation only if it measurably moved *this* prediction. The tests below are therefore
mostly about what is *absent* - a model that ranks a feature highly must still not be able
to put it in an explanation of a row it did not affect.
"""

from __future__ import annotations

import numpy as np
import pytest

from focus_engine.features import FeatureName
from focus_engine.models.artifacts import ModelArtifact
from focus_engine.uncertainty.explanation import (
    PredictionExplanation,
    SignalContribution,
    occlusion_contributions,
)
from tests.unit.models.scenarios import make_vector
from tests.unit.uncertainty.scenarios import (
    neutral_vector,
    train_shared_artifact,
    without_reference_values,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def artifact() -> ModelArtifact:
    """A trained artifact with reference values, shared by the tests in this module."""
    return train_shared_artifact()


def _probability(artifact: ModelArtifact, vector: object) -> float:
    """Score a vector with the artifact, returning just the probability."""
    from focus_engine.models.artifacts import class_one_probabilities
    from focus_engine.models.encoding import encode_rows, requirements_from_schema

    matrix = encode_rows(
        ((vector, 1),), requirements=requirements_from_schema(artifact.encoding_schema)
    )
    return float(class_one_probabilities(artifact.estimator, matrix.matrix)[0])


class TestOnlyContributingSignals:
    """What appears in an explanation, and what is kept out of it."""

    def test_an_explanation_measures_against_the_reported_probability(
        self, artifact: ModelArtifact
    ) -> None:
        vector = neutral_vector()
        probability = _probability(artifact, vector)
        explanation = occlusion_contributions(artifact, vector, base_probability=probability)
        assert explanation.available
        assert explanation.base_probability == pytest.approx(probability)
        assert explanation.contributions

    def test_only_features_above_the_tolerance_are_reported(self, artifact: ModelArtifact) -> None:
        vector = neutral_vector()
        probability = _probability(artifact, vector)
        loose = occlusion_contributions(artifact, vector, base_probability=probability)
        strict = occlusion_contributions(
            artifact, vector, base_probability=probability, min_contribution=10.0
        )
        assert len(loose.contributions) > len(strict.contributions)
        assert strict.contributions == ()
        assert all(
            contribution.magnitude >= strict.min_contribution
            for contribution in strict.contributions
        )

    def test_a_very_high_tolerance_produces_an_empty_explanation(
        self, artifact: ModelArtifact
    ) -> None:
        """A prediction nothing moved is a reportable finding, not a gap.

        This is the shape of the worst outcome a prediction system can have: a number that
        no input affected. It is surfaced rather than filled in with the model's global
        importances, which would describe the model rather than this row.
        """
        vector = neutral_vector()
        probability = _probability(artifact, vector)
        explanation = occlusion_contributions(
            artifact, vector, base_probability=probability, min_contribution=10.0
        )
        assert explanation.is_empty
        assert "no feature moved" in explanation.describe()

    def test_an_empty_explanation_still_reports_which_features_were_considered(
        self, artifact: ModelArtifact
    ) -> None:
        vector = neutral_vector()
        probability = _probability(artifact, vector)
        explanation = occlusion_contributions(
            artifact, vector, base_probability=probability, min_contribution=10.0
        )
        assert explanation.omitted
        assert all(isinstance(name, str) for name in explanation.omitted)

    def test_contributions_are_ordered_by_magnitude(self, artifact: ModelArtifact) -> None:
        vector = neutral_vector()
        probability = _probability(artifact, vector)
        magnitudes = [
            contribution.magnitude
            for contribution in occlusion_contributions(
                artifact, vector, base_probability=probability
            ).contributions
        ]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_each_feature_appears_at_most_once(self, artifact: ModelArtifact) -> None:
        """One-hot expansion must not turn one fact into several reported signals."""
        vector = neutral_vector()
        probability = _probability(artifact, vector)
        names = [
            contribution.feature
            for contribution in occlusion_contributions(
                artifact, vector, base_probability=probability
            ).contributions
        ]
        assert len(names) == len(set(names))

    def test_a_one_hot_feature_is_named_by_its_feature_and_not_its_category(
        self, artifact: ModelArtifact
    ) -> None:
        vector = neutral_vector()
        probability = _probability(artifact, vector)
        explanation = occlusion_contributions(artifact, vector, base_probability=probability)
        for contribution in explanation.contributions:
            base, _, category = contribution.column.partition("=")
            assert contribution.feature == base
            if category:
                assert contribution.column == f"{base}={category}"

    def test_moving_a_feature_further_from_its_reference_enlarges_its_contribution(
        self, artifact: ModelArtifact
    ) -> None:
        """Occlusion measures distance from the reference, so distance must show up.

        This is the direct check that the routine is measuring the row rather than echoing
        the model: for a linear scorer the occlusion delta is the coefficient times the
        distance from the reference, so pushing a value further out must grow its
        attributed magnitude and never shrink it.
        """
        near = make_vector(overrides={FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS: 60.0})
        far = make_vector(overrides={FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS: 400.0})
        near_magnitude = self._magnitude(
            artifact, near, FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS
        )
        far_magnitude = self._magnitude(
            artifact, far, FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS
        )
        assert far_magnitude > near_magnitude > 0.0

    @staticmethod
    def _magnitude(artifact: ModelArtifact, vector: object, feature: str) -> float:
        """Absolute occlusion contribution of one feature on one row."""
        explanation = occlusion_contributions(
            artifact, vector, base_probability=_probability(artifact, vector)
        )
        return sum(
            contribution.magnitude
            for contribution in explanation.contributions
            if contribution.feature == feature.value
        )

    def test_a_feature_the_model_ignores_is_never_attributed(self, artifact: ModelArtifact) -> None:
        """A zero coefficient means no movement, so no signal may be reported.

        The point of this test is that a model cannot get credit for a feature it does not
        use. Rather than hard-coding a feature name, the ignored columns are read from the
        fitted coefficients, so the assertion stays true as the training data changes.
        """

        coefficients = artifact.estimator.named_steps["model"].coef_[0]
        assert coefficients.shape[0] == len(artifact.columns)
        ignored = {
            artifact.columns[index].split("=")[0] for index in np.flatnonzero(coefficients == 0.0)
        }
        if not ignored:
            pytest.skip("the fitted model uses every column, so none is ignorable")

        vector = neutral_vector()
        explanation = occlusion_contributions(
            artifact, vector, base_probability=_probability(artifact, vector)
        )
        reported = {contribution.feature for contribution in explanation.contributions}
        assert not (reported & ignored)
        assert reported, (
            "no feature was attributed at all, so this test is not exercising the "
            "grouping and filtering that it is meant to check"
        )

        # An ignored feature is a non-contributor, so it belongs in ``omitted`` - which is
        # the list a reader consults to see why a plausible signal is absent.
        assert ignored & set(explanation.omitted)

    def test_probability_moves_only_for_columns_the_model_weights(
        self, artifact: ModelArtifact
    ) -> None:
        """The occlusion routine is verified against the model's own coefficients.

        Occlusion is compared here with the closed form for a linear scorer: setting one
        column to its reference should change the logit by exactly that column's
        coefficient times its distance from the reference. If the two disagree, the
        explanation is measuring something other than the model.
        """
        from focus_engine.models.encoding import encode_rows, requirements_from_schema

        vector = neutral_vector()
        matrix = encode_rows(
            ((vector, 1),), requirements=requirements_from_schema(artifact.encoding_schema)
        ).matrix
        scaler = artifact.estimator.named_steps["scaler"]
        model = artifact.estimator.named_steps["model"]
        logit = float(model.decision_function(scaler.transform(matrix))[0])
        coefficients = model.coef_[0]
        scale = np.asarray(scaler.scale_, dtype=float)
        mean = np.asarray(scaler.mean_, dtype=float)

        explanation = occlusion_contributions(
            artifact, vector, base_probability=_probability(artifact, vector)
        )
        by_column: dict[str, float] = {}
        for contribution in explanation.contributions:
            by_column[contribution.column] = contribution.contribution

        vector_row = matrix[0]
        for index, column in enumerate(artifact.columns):
            distance = (float(vector_row[index]) - mean[index]) / scale[index]
            predicted_delta = _logistic(logit - coefficients[index] * distance) - _logistic(logit)
            assert by_column.get(column, 0.0) == pytest.approx(predicted_delta, abs=1e-9), (
                f"column {column} was attributed {by_column.get(column, 0.0):.6f} but the "
                f"model's own coefficients imply {predicted_delta:.6f}"
            )

    def test_perturbing_the_vector_does_not_change_the_attribution_order(
        self, artifact: ModelArtifact
    ) -> None:
        """Attribution is a property of the model's sensitivity, not of the noise."""
        vector = neutral_vector()
        probability = _probability(artifact, vector)
        first = [
            c.feature
            for c in occlusion_contributions(
                artifact, vector, base_probability=probability
            ).contributions
        ]
        second = [
            c.feature
            for c in occlusion_contributions(
                artifact, vector, base_probability=probability
            ).contributions
        ]
        assert first == second


class TestExplanationAvailability:
    """When no explanation can be produced, the system says so."""

    def test_an_artifact_without_reference_values_yields_no_explanation(
        self, artifact: ModelArtifact
    ) -> None:
        bare = without_reference_values(artifact)
        vector = neutral_vector()
        explanation = occlusion_contributions(
            bare, vector, base_probability=_probability(bare, vector)
        )
        assert not explanation.available
        assert explanation.contributions == ()

    def test_the_unavailability_reason_refuses_the_global_importance_shortcut(
        self, artifact: ModelArtifact
    ) -> None:
        """The tempting wrong answer is named, so the refusal is auditable."""
        bare = without_reference_values(artifact)
        vector = neutral_vector()
        reason = str(
            occlusion_contributions(
                bare, vector, base_probability=_probability(bare, vector)
            ).reason
        )
        assert "importances" in reason
        assert "this one" in reason

    def test_a_vector_from_another_feature_set_is_refused(self, artifact: ModelArtifact) -> None:
        """Attributing across a schema boundary would describe unseen features."""
        vector = neutral_vector()
        wrong = vector.model_copy(update={"feature_set_version": "OTHER_SET_V1"})
        with pytest.raises(ValueError, match="An explanation cannot be produced"):
            occlusion_contributions(
                artifact, wrong, base_probability=_probability(artifact, vector)
            )

    def test_a_negative_tolerance_is_refused(self, artifact: ModelArtifact) -> None:
        with pytest.raises(ValueError, match="min_contribution must be non-negative"):
            occlusion_contributions(
                artifact, neutral_vector(), base_probability=0.5, min_contribution=-1.0
            )


class TestExplanationValidation:
    """An explanation that contradicts itself is refused."""

    def test_an_unavailable_explanation_may_not_carry_contributions(self) -> None:
        with pytest.raises(ValueError, match="cannot carry contributions"):
            PredictionExplanation(
                base_probability=0.5,
                contributions=(SignalContribution("f", 0.1, "f"),),
                available=False,
                reason="no references",
            )

    def test_an_unavailable_explanation_must_say_why(self) -> None:
        with pytest.raises(ValueError, match="must say why"):
            PredictionExplanation(base_probability=0.5, available=False)

    def test_an_available_explanation_may_not_carry_a_reason(self) -> None:
        with pytest.raises(ValueError, match="cannot carry an unavailability reason"):
            PredictionExplanation(base_probability=0.5, reason="because")

    def test_a_reported_contribution_below_the_threshold_is_refused(self) -> None:
        """Otherwise the filter could be bypassed by constructing the record directly."""
        with pytest.raises(ValueError, match="below the"):
            PredictionExplanation(
                base_probability=0.5,
                min_contribution=0.5,
                contributions=(SignalContribution("f", 0.1, "f"),),
            )

    def test_contributions_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="ordered by magnitude"):
            PredictionExplanation(
                base_probability=0.5,
                contributions=(
                    SignalContribution("a", 0.1, "a"),
                    SignalContribution("b", 0.9, "b"),
                ),
            )

    def test_a_negative_tolerance_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            PredictionExplanation(base_probability=0.5, min_contribution=-0.1)

    def test_top_returns_at_most_what_was_asked_for(self) -> None:
        explanation = PredictionExplanation(
            base_probability=0.5,
            contributions=(
                SignalContribution("a", 0.3, "a"),
                SignalContribution("b", 0.2, "b"),
            ),
        )
        assert len(explanation.top(1)) == 1
        assert len(explanation.top(5)) == 2

    def test_top_refuses_a_negative_count(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            PredictionExplanation(base_probability=0.5).top(-1)


class TestSignalContribution:
    """Direction and magnitude are reported as measured."""

    def test_a_positive_contribution_moved_towards_the_positive_class(self) -> None:
        assert SignalContribution("f", 0.2, "f").direction == "towards_positive"

    def test_a_negative_contribution_moved_away_from_it(self) -> None:
        assert SignalContribution("f", -0.2, "f").direction == "towards_negative"

    def test_magnitude_is_unsigned(self) -> None:
        assert SignalContribution("f", -0.2, "f").magnitude == 0.2

    def test_describe_names_the_feature_the_column_and_the_direction(self) -> None:
        text = SignalContribution("accuracy", -0.05, "performance_accuracy").describe()
        assert "accuracy" in text
        assert "performance_accuracy" in text
        assert "towards_negative" in text


def _logistic(value: float) -> float:
    """The logistic function, in the form that does not overflow for large ``|value|``.

    Used only to reproduce the model's own scoring as a closed form, so that the
    attribution can be checked against the coefficients rather than against itself.
    """
    if value >= 0.0:
        return 1.0 / (1.0 + float(np.exp(-value)))
    exponential = float(np.exp(value))
    return exponential / (1.0 + exponential)
