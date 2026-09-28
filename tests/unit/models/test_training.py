"""Tests for model training and its reproducibility contract (Phase 8).

A training run is a claim that a model learned something, and these tests check the parts
of that claim which are cheap to get wrong: that the three declared algorithms actually
fit, that a recorded seed reproduces the same model, that the metrics describe a real
evaluation rather than a plausible-looking one, and that the training run refuses inputs
it cannot honestly be run on.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from focus_engine.models.artifacts import _metadata_path, load_artifact, save_artifact
from focus_engine.models.encoding import encode_rows
from focus_engine.models.trainers import (
    Algorithm,
    TrainingConfig,
    _default_hyperparameters,
    train_model,
)
from focus_engine.schemas.primitives import DataOrigin, Provenance
from focus_engine.schemas.versioning import ModelStatus
from tests.unit.models.scenarios import make_labelled_rows

pytestmark = pytest.mark.unit

SEED = 4242


def make_config(
    algorithm: Algorithm = Algorithm.LOGISTIC_REGRESSION,
    *,
    seed: int = SEED,
    model_id: str = "risk-logreg",
    model_version: str = "RISK_MODEL_V1",
    data_origin: DataOrigin = DataOrigin.REAL,
    label_provenance: Provenance = Provenance.OBSERVED,
    hyperparameters: dict[str, str | int | float | bool | None] | None = None,
) -> TrainingConfig:
    """Build a valid training configuration for a test."""
    return TrainingConfig(
        algorithm=algorithm,
        model_id=model_id,
        model_version=model_version,
        feature_set_version="FEATURE_SET_V1",
        dataset_version="DATASET_REAL_V1",
        target_definition_version="td-1.0.0",
        label_provenance=label_provenance,
        data_origin=data_origin,
        seed=seed,
        hyperparameters=hyperparameters or {},
    )


def make_split(train_n: int = 60, holdout_n: int = 20, overlap: bool = False):
    """Build an independent train and holdout split from the scenario generator.

    Each split gets its own disjoint block of learner identifiers, so a row can never
    appear in both. That is the minimum a split is: sharing a learner's rows across the
    boundary would let the model memorise a learner rather than learn a pattern, and no
    amount of metric care afterwards would reveal it.
    """
    train = encode_rows(
        make_labelled_rows(train_n, positive=train_n // 3, start_index=0, overlap=overlap)
    )
    holdout = encode_rows(
        make_labelled_rows(holdout_n, positive=holdout_n // 4, start_index=1000, overlap=overlap)
    )
    return train, holdout


class TestAlgorithms:
    """Every declared algorithm fits and produces a usable artifact."""

    @pytest.mark.parametrize("algorithm", list(Algorithm))
    def test_each_algorithm_trains(self, algorithm: Algorithm) -> None:
        """Each supported algorithm returns a fitted estimator and a full metric suite."""
        train, holdout = make_split()
        config = make_config(algorithm)

        outcome = train_model(config, train, holdout)

        assert hasattr(outcome.artifact.estimator, "predict")
        assert len(outcome.metrics) == 1
        assert outcome.metrics[0].n_samples == holdout.n_rows

    @pytest.mark.parametrize("algorithm", list(Algorithm))
    def test_each_algorithm_learns_a_learnable_signal(self, algorithm: Algorithm) -> None:
        """A model fitted on separable data should score well above chance.

        This is the test that would catch a silent encoding failure: if the columns were
        misaligned or the labels shuffled, the model would still train and would still
        report metrics, and only the quality of those metrics would show the problem.
        """
        train, holdout = make_split()
        config = make_config(algorithm)

        outcome = train_model(config, train, holdout)

        roc_auc = outcome.metrics[0].roc_auc
        assert roc_auc is not None
        assert roc_auc > 0.85, f"{algorithm.value} scored a roc_auc of only {roc_auc}"

    def test_logistic_regression_has_a_coefficient_per_column(self) -> None:
        """The pipeline's scaler and model expose one coefficient per encoded column."""
        train, holdout = make_split()

        outcome = train_model(make_config(Algorithm.LOGISTIC_REGRESSION), train, holdout)

        estimator = outcome.artifact.estimator
        coefficients = estimator.named_steps["model"].coef_
        assert coefficients.shape == (1, train.n_columns)

    def test_the_record_names_the_concrete_estimator(self) -> None:
        """The record identifies the class path, not the enum name.

        The registry has to be readable by someone who did not write this module, and a
        record saying "logistic_regression" leaves them to guess the implementation.
        """
        train, holdout = make_split()

        outcome = train_model(make_config(Algorithm.RANDOM_FOREST), train, holdout)

        assert outcome.record.algorithm == "sklearn.ensemble.RandomForestClassifier"


class TestReproducibility:
    """The recorded seed reproduces the model."""

    def test_the_same_seed_produces_identical_predictions(self) -> None:
        """Two runs with the same seed and data agree on every prediction.

        The data overlaps, so agreement is a real constraint: on separable data any two
        forests would agree whether or not the seed was threaded through.
        """
        train, holdout = make_split(overlap=True)
        config = make_config(Algorithm.RANDOM_FOREST)

        first = train_model(config, train, holdout)
        second = train_model(config, train, holdout)

        first_probabilities = first.artifact.estimator.predict_proba(holdout.matrix)
        second_probabilities = second.artifact.estimator.predict_proba(holdout.matrix)
        np.testing.assert_array_equal(first_probabilities, second_probabilities)

    def test_a_different_seed_produces_a_different_model(self) -> None:
        """The seed actually reaches the estimator.

        The data overlaps deliberately. On separable data every forest predicts perfectly
        and two fits are indistinguishable no matter what seed they were given, so this
        test would pass even if the seed were dropped on the floor.
        """
        train, holdout = make_split(overlap=True)
        first = train_model(make_config(Algorithm.RANDOM_FOREST, seed=1), train, holdout)
        second = train_model(make_config(Algorithm.RANDOM_FOREST, seed=2), train, holdout)

        first_probabilities = first.artifact.estimator.predict_proba(holdout.matrix)
        second_probabilities = second.artifact.estimator.predict_proba(holdout.matrix)
        assert not np.allclose(first_probabilities, second_probabilities)

    def test_the_derived_child_seed_is_deterministic_and_documented(self) -> None:
        """The child seed is derived from the run's coordinates, not from a global RNG."""
        train, holdout = make_split()
        config = make_config(seed=99)

        first = train_model(config, train, holdout)
        second = train_model(config, train, holdout)

        assert first.seed == second.seed
        assert first.seed != config.seed
        assert 0 <= first.seed < 2**32

    def test_different_model_identities_do_not_share_a_seed(self) -> None:
        """Two models trained on the same data do not draw the same random stream.

        Otherwise an experiment with one model would silently change another's behaviour.
        """
        train, holdout = make_split()
        first = train_model(make_config(model_id="model-a"), train, holdout)
        second = train_model(make_config(model_id="model-b"), train, holdout)

        assert first.seed != second.seed

    def test_the_record_keeps_the_run_seed_not_the_derived_one(self) -> None:
        """The record stores what a reader needs to re-derive the child seed.

        Storing the derived value instead would break the derivation, which takes the run
        seed as its input.
        """
        train, holdout = make_split()
        config = make_config(seed=1234)

        outcome = train_model(config, train, holdout)

        assert outcome.record.seed == 1234
        assert outcome.seed != 1234


class TestHyperparameters:
    """The recorded hyperparameters are the ones the model was actually fitted with."""

    def test_defaults_are_recorded(self) -> None:
        """The record shows the resolved configuration, not an empty dict."""
        train, holdout = make_split()
        config = make_config(Algorithm.RANDOM_FOREST)

        outcome = train_model(config, train, holdout)

        assert outcome.record.hyperparameters == _default_hyperparameters(Algorithm.RANDOM_FOREST)

    def test_caller_overrides_are_recorded(self) -> None:
        """An experiment is recorded as the configuration that was actually run."""
        train, holdout = make_split()
        config = make_config(Algorithm.RANDOM_FOREST, hyperparameters={"n_estimators": 7})

        outcome = train_model(config, train, holdout)

        assert outcome.record.hyperparameters["n_estimators"] == 7
        assert outcome.artifact.estimator.n_estimators == 7

    def test_the_default_forest_uses_a_single_job(self) -> None:
        """Reproducibility is claimed, so thread scheduling must not be part of the model.

        Work distribution across threads is documented as an acceptable difference for
        some sklearn estimators, which means it cannot be relied on to produce the same
        fit. ``lbfgs`` is single-threaded already, so logistic regression sets no such
        option - and passing one would be deprecated.
        """
        assert _default_hyperparameters(Algorithm.RANDOM_FOREST)["n_jobs"] == 1
        assert "n_jobs" not in _default_hyperparameters(Algorithm.LOGISTIC_REGRESSION)

    def test_a_forest_leaf_cannot_be_a_single_observation(self) -> None:
        """``min_samples_leaf`` is set so a leaf cannot memorise one session."""
        assert _default_hyperparameters(Algorithm.RANDOM_FOREST)["min_samples_leaf"] > 1
        assert _default_hyperparameters(Algorithm.GRADIENT_BOOSTING)["min_samples_leaf"] > 1


class TestRefusals:
    """A training run that cannot be honest refuses to start."""

    def test_an_empty_training_set_is_refused(self) -> None:
        """There is no model to fit."""
        train, holdout = make_split()
        empty = encode_rows([])

        with pytest.raises(ValueError, match="training set is empty"):
            train_model(make_config(), empty, holdout)

    def test_a_single_class_training_set_is_refused(self) -> None:
        """A model fitted on one class learns the class prior, not a pattern."""
        train = encode_rows(make_labelled_rows(20, positive=0))
        _, holdout = make_split()

        with pytest.raises(ValueError, match="single class"):
            train_model(make_config(), train, holdout)

    def test_an_empty_holdout_is_refused(self) -> None:
        """Metrics without observations are not results."""
        train, _ = make_split()

        with pytest.raises(ValueError, match="holdout set is empty"):
            train_model(make_config(), train, encode_rows([]))

    def test_mismatched_schemas_are_refused(self) -> None:
        """Training on one encoding and evaluating on another is confusion, not evaluation."""
        train, holdout = make_split()
        narrow_holdout = encode_rows(
            make_labelled_rows(20, positive=5, start_index=1000),
            requirements=train.requirements[:3],
        )

        with pytest.raises(ValueError, match="schemas disagree"):
            train_model(make_config(), train, narrow_holdout)

    def test_a_negative_seed_is_refused(self) -> None:
        """A negative seed cannot be re-derived, so it breaks the contract."""
        with pytest.raises(ValueError, match="non-negative"):
            make_config(seed=-1)

    def test_a_blank_model_id_is_refused(self) -> None:
        """A model with no identity is untrackable."""
        with pytest.raises(ValueError, match="model_id must not be blank"):
            make_config(model_id="   ")

    def test_a_blank_target_definition_is_refused(self) -> None:
        """A model whose label definition is unknown cannot be compared to anything."""
        config = make_config()
        with pytest.raises(ValueError, match="target_definition_version"):
            TrainingConfig(
                algorithm=config.algorithm,
                model_id=config.model_id,
                model_version=config.model_version,
                feature_set_version=config.feature_set_version,
                dataset_version=config.dataset_version,
                target_definition_version="  ",
                label_provenance=config.label_provenance,
                data_origin=config.data_origin,
                seed=config.seed,
            )


class TestMetrics:
    """Metrics report what was measured, and ``None`` where nothing was."""

    def test_the_holdout_split_is_named(self) -> None:
        """A metric without a split name cannot be attributed."""
        train, holdout = make_split()

        outcome = train_model(make_config(), train, holdout)

        assert outcome.metrics[0].split_name == "holdout"

    def test_the_record_carries_the_same_metrics(self) -> None:
        """The registry entry and the outcome agree, so an audit sees one set of numbers."""
        train, holdout = make_split()

        outcome = train_model(make_config(), train, holdout)

        assert outcome.record.metrics == outcome.metrics

    def test_a_single_class_holdout_reports_no_roc_auc(self) -> None:
        """ROC-AUC is undefined on a single-class split and is reported as absent.

        Reporting 0.5 would be a number that was never measured, and a later reader could
        not distinguish it from a real 0.5.
        """
        train, _ = make_split()
        all_negative_holdout = encode_rows(make_labelled_rows(20, positive=0, start_index=2000))

        outcome = train_model(make_config(), train, all_negative_holdout)

        metrics = outcome.metrics[0]
        assert metrics.roc_auc is None
        assert metrics.pr_auc is None
        assert metrics.expected_calibration_error is None
        # Metrics that remain defined are still reported.
        assert metrics.accuracy is not None
        assert metrics.brier is not None

    def test_probability_metrics_stay_in_range(self) -> None:
        """Brier score and ECE are only meaningful inside their natural ranges."""
        train, holdout = make_split()

        outcome = train_model(make_config(), train, holdout)

        metrics = outcome.metrics[0]
        assert metrics.brier is not None and 0.0 <= metrics.brier <= 1.0
        assert (
            metrics.expected_calibration_error is not None
            and 0.0 <= metrics.expected_calibration_error <= 1.0
        )

    def test_a_well_calibrated_model_scores_a_low_brier(self) -> None:
        """A model that learned the signal should not be wildly overconfident."""
        train, holdout = make_split()

        outcome = train_model(make_config(Algorithm.LOGISTIC_REGRESSION), train, holdout)

        assert outcome.metrics[0].brier is not None
        assert outcome.metrics[0].brier < 0.25


class TestRecordProvenance:
    """The record carries the coordinates that make the run traceable."""

    def test_a_new_model_is_recorded_as_experimental(self) -> None:
        """Training does not promote. The lifecycle starts at the bottom."""
        train, holdout = make_split()

        outcome = train_model(make_config(), train, holdout)

        assert outcome.record.status is ModelStatus.EXPERIMENTAL

    def test_the_artifact_and_the_record_are_the_same_run(self) -> None:
        """The artifact's record is the record returned alongside it."""
        train, holdout = make_split()

        outcome = train_model(make_config(), train, holdout)

        assert outcome.artifact.record == outcome.record

    def test_the_artifact_carries_the_training_schema(self) -> None:
        """A model without its schema cannot verify the data it later receives."""
        train, holdout = make_split()

        outcome = train_model(make_config(), train, holdout)

        assert outcome.artifact.columns == train.columns
        assert outcome.artifact.encoding_digest == train.schema_digest()
        assert outcome.encoding_digest == train.schema_digest()

    def test_synthetic_training_is_recorded_as_synthetic(self) -> None:
        """The origin travels with the model, so promotion can be refused later."""
        train, holdout = make_split()
        config = make_config(
            data_origin=DataOrigin.SYNTHETIC,
            label_provenance=Provenance.SYNTHETIC_LABEL,
        )

        outcome = train_model(config, train, holdout)

        assert outcome.record.data_origin is DataOrigin.SYNTHETIC
        assert outcome.record.label_provenance is Provenance.SYNTHETIC_LABEL
        # And it therefore still starts experimental, never validated.
        assert outcome.record.status is ModelStatus.EXPERIMENTAL

    def test_a_trained_at_timestamp_is_recorded(self) -> None:
        """A model with no training time cannot be placed in a history."""
        train, holdout = make_split()

        outcome = train_model(make_config(), train, holdout)

        assert outcome.record.trained_at.tzinfo is not None
        assert outcome.artifact.trained_at == outcome.record.trained_at


class TestCalibrationMetric:
    """The calibration error is a measure of calibration, not of prediction accuracy.

    A previous implementation compared hard-prediction accuracy against stated confidence,
    which produces near-0.5 for a perfect model. This regression test makes sure the
    corrected metric agrees with the Brier score on clean data.
    """

    def test_a_perfect_model_on_separated_data_reports_near_zero_ece(self) -> None:
        train, holdout = make_split()
        outcome = train_model(make_config(), train, holdout)
        metrics = outcome.metrics[0]
        assert metrics.expected_calibration_error is not None
        assert metrics.expected_calibration_error < 0.05, (
            f"a perfect-holdout model (brier={metrics.brier:.6f}) reported "
            f"ECE={metrics.expected_calibration_error:.4f}, which means the metric "
            "is not measuring calibration"
        )

    def test_ece_and_brier_are_consistent_order(self) -> None:
        """A model with near-zero Brier cannot have a large calibration error."""
        train, holdout = make_split()
        outcome = train_model(make_config(), train, holdout)
        metrics = outcome.metrics[0]
        if metrics.brier is not None and metrics.expected_calibration_error is not None:
            # Brier is an upper bound on ECE for binary calibration; a large
            # divergence indicates a metric bug.
            assert metrics.expected_calibration_error <= metrics.brier + 0.1

    def test_ece_uses_observed_frequency_not_accuracy(self) -> None:
        """Known-answer check: 100% accuracy does not confuse the metric.

        In a separable model where predictions exactly match labels, ECE is 0 not 0.5.
        """
        import numpy as np

        from focus_engine.models.trainers import _expected_calibration_error as ece

        probs = np.concatenate([np.zeros(100), np.ones(100)])
        labels = np.concatenate([np.zeros(100), np.ones(100)])
        error = ece(probs, labels)
        assert error is not None
        assert error < 0.01, (
            f"a perfectly calibrated model reported ECE={error:.4f}; "
            "the metric is comparing accuracy rather than observed frequency"
        )


class TestFeatureReference:
    """Per-column reference values for explanation."""

    def test_the_reference_covers_all_columns(self) -> None:
        train, holdout = make_split()
        outcome = train_model(make_config(), train, holdout)
        assert set(outcome.artifact.feature_reference.keys()) == set(outcome.artifact.columns)

    def test_the_reference_values_are_medians(self) -> None:
        train, holdout = make_split()
        outcome = train_model(make_config(), train, holdout)
        for column, reference in outcome.artifact.feature_reference.items():
            column_index = outcome.artifact.columns.index(column)
            expected = float(np.median(train.matrix[:, column_index]))
            assert reference == pytest.approx(expected)

    def test_the_reference_is_persisted_and_reloaded(self, tmp_path: Path) -> None:
        """A reloaded artifact must carry the same reference mapping."""
        train, holdout = make_split()
        outcome = train_model(make_config(), train, holdout)
        path = tmp_path / "model.joblib"
        save_artifact(outcome.artifact, path)
        loaded = load_artifact(path)
        assert loaded.feature_reference == outcome.artifact.feature_reference

    def test_a_pre_existing_artifact_with_no_reference_is_loaded(self, tmp_path: Path) -> None:
        """An artifact saved before Phase 9 must still load, with no reference mapping.

        Backward compatibility here is a correctness requirement, not a convenience: an
        existing sidecar without the field must not turn into a load failure, and the empty
        mapping is what tells the explanation layer to report an explanation as
        unavailable rather than invent reference values.
        """
        train, holdout = make_split()
        outcome = train_model(make_config(), train, holdout)
        path = tmp_path / "model.joblib"
        save_artifact(outcome.artifact, path)
        metadata = joblib.load(_metadata_path(path))
        del metadata["feature_reference"]
        joblib.dump(metadata, _metadata_path(path))
        loaded = load_artifact(path)
        assert loaded.feature_reference == {}
