"""Training entry point for scikit-learn classifiers.

This module owns the path from a labelled design matrix to a
:class:`~focus_engine.models.artifacts.ModelArtifact`. The path is deliberately narrow:
the caller supplies the train and holdout splits, so that split policy is not a decision
this module makes. A module that decides its own splits is a module that can be
reconfigured into leaking, and one that can be reconfigured into leaking is one that will
be, because the next person who touches it will not know the history of the split that was
chosen and why.

**The seed is the reproducibility contract.** Every estimator is configured with a
``random_state`` derived deterministically from the training seed, the model identity, and
the label ``"training"``. Two runs with the same seed, the same data, and the same schema
therefore fit the same model. Reproducibility here means *identical fitted parameters*, not
byte-identical pickles: a pickle embeds library versions, so byte identity would make the
contract unmeetable across a dependency upgrade even when the model's behaviour is
unchanged. Predicting from both artefacts and comparing the outputs is the test that
matches the claim being made.

**Imputation is not performed.** The encoding layer refuses rows with absent features, so
the data reaching this module has no gaps. A model fitted only on measured data is the
precondition for its coefficients to mean what they say, and a pipeline that silently
filled gaps would move that precondition from the caller into this module, where it would
be invisible.

**A model is a claim, not a measurement.** The return value carries both the fitted
estimator and a complete audit record: algorithm, resolved hyperparameters, metrics, seed,
``data_origin``, and ``label_provenance``. The record is created at ``EXPERIMENTAL``. This
module has no standing to promote it, because the decision to serve a model to learners is
not the fact that one was fitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from focus_engine.models.artifacts import ModelArtifact, class_one_probabilities
from focus_engine.models.encoding import DesignMatrix, column_references
from focus_engine.schemas.primitives import DataOrigin, Provenance, utc_now
from focus_engine.schemas.versioning import (
    DatasetVersion,
    FeatureSetVersion,
    ModelMetrics,
    ModelRecord,
    ModelStatus,
    ModelVersion,
)
from focus_engine.utils.determinism import derive_seed, library_versions

__all__ = [
    "Algorithm",
    "TrainingConfig",
    "TrainingOutcome",
    "train_model",
]

#: Number of equal-width probability bins used for expected calibration error.
_CALIBRATION_BINS: Final[int] = 10


class Algorithm(StrEnum):
    """The classifier algorithms currently supported.

    Every supported algorithm is a classical, auditable estimator. No deep learning
    algorithm is included, because the question at this stage is whether the feature set
    carries a signal at all, not whether a more flexible architecture can extract one.
    Deep learning becomes a legitimate question only after the simpler estimators have
    been shown to be capacity-limited.
    """

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


#: The class path recorded in ``ModelRecord.algorithm`` for each algorithm.
_ALGORITHM_NAMES: Final[dict[Algorithm, str]] = {
    Algorithm.LOGISTIC_REGRESSION: "sklearn.linear_model.LogisticRegression",
    Algorithm.RANDOM_FOREST: "sklearn.ensemble.RandomForestClassifier",
    Algorithm.GRADIENT_BOOSTING: "sklearn.ensemble.HistGradientBoostingClassifier",
}


def _default_hyperparameters(algorithm: Algorithm) -> dict[str, Any]:
    """Return the default hyperparameters for an algorithm.

    ``n_jobs=1`` is a deliberate determinism choice for the two tree ensembles. Parallel
    job schedulers across different library versions may assign work to threads in a
    different order, which is documented as an acceptable difference for some sklearn
    estimators; a claim of exact reproducibility for a fit with ``n_jobs>1`` would therefore
    be a claim this module cannot verify. ``lbfgs`` is already single-threaded, so logistic
    regression needs no such setting. The cost is slower training; the benefit is that the
    same seed and data always produce the same model, which is the property the roadmap
    asks for.

    ``min_samples_leaf=5`` is set for both tree ensembles so that a leaf cannot be a single
    learner's session. A leaf that memorises one observation produces a confident
    prediction for that observation and an unjustified one everywhere else, which is
    overfitting dressed as accuracy.

    Returns:
        Default hyperparameters suitable for passing to the estimator constructor.

    Raises:
        ValueError: If the algorithm is not one of the supported values.
    """
    if algorithm is Algorithm.LOGISTIC_REGRESSION:
        return {"max_iter": 1000, "solver": "lbfgs"}
    if algorithm is Algorithm.RANDOM_FOREST:
        return {"n_estimators": 100, "min_samples_leaf": 5, "n_jobs": 1}
    if algorithm is Algorithm.GRADIENT_BOOSTING:
        return {"max_iter": 100, "learning_rate": 0.1, "min_samples_leaf": 5, "max_leaf_nodes": 31}
    raise ValueError(f"unknown algorithm: {algorithm!r}")


def _build_estimator(algorithm: Algorithm, hyperparameters: dict[str, Any], seed: int) -> Any:
    """Construct the estimator for an algorithm, with its randomness pinned.

    Logistic regression is wrapped in a pipeline with a ``StandardScaler``, because its
    features live on incompatible scales - accuracy near 0.8 against response time near
    20.0 - and an unscaled optimiser converges slowly and stops early for reasons that
    look like a low signal. Tree models are left unscaled: a split criterion is invariant to
    monotonic rescaling of a feature, so a scaler would add latency and obscure the
    importances without changing a single split.

    Args:
        algorithm: The algorithm to instantiate.
        hyperparameters: The resolved hyperparameters, including ``random_state``.
        seed: The derived child seed for this training run.

    Returns:
        An unfitted scikit-learn estimator.

    Raises:
        ValueError: If the algorithm is not one of the supported values.
    """
    params = {**hyperparameters, "random_state": seed}
    if algorithm is Algorithm.LOGISTIC_REGRESSION:
        return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(**params))])
    if algorithm is Algorithm.RANDOM_FOREST:
        return RandomForestClassifier(**params)
    if algorithm is Algorithm.GRADIENT_BOOSTING:
        return HistGradientBoostingClassifier(**params)
    raise ValueError(f"unknown algorithm: {algorithm!r}")


def _safe_metric(metric_fn: Any, *args: Any, **kwargs: Any) -> float | None:
    """Evaluate a metric, returning ``None`` when it is undefined.

    Several metrics are undefined for a single-class split. Returning ``None`` records that
    absence; returning a default such as 0.5 would report a number that was never measured,
    and a later reader could not tell the difference between a real 0.5 and an invented one.

    Args:
        metric_fn: The metric to evaluate.
        *args: Positional arguments for the metric.
        **kwargs: Keyword arguments for the metric.

    Returns:
        The metric value, or ``None`` if undefined.
    """
    try:
        value = float(metric_fn(*args, **kwargs))
    except (ValueError, IndexError):
        return None
    return None if math.isnan(value) else value


def _expected_calibration_error(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float | None:
    """Compute equal-width-bin expected calibration error.

    Calibration asks whether a stated probability matches the frequency of the event it
    predicts, so the comparison is between the mean predicted probability in a bin and the
    *observed positive frequency* in that bin.

    It is worth being explicit about what that is not, because an earlier version of this
    function compared against the fraction of hard predictions that were correct, and that
    quantity answers a different question. Consider a model that is perfectly calibrated
    and perfectly accurate: it assigns 0.3% to a bin in which nothing positive ever
    occurs. Its stated confidence there is 0.003 and the observed frequency is 0.000, so
    the calibration error is 0.003. Measured against hard-prediction accuracy the same bin
    scores 1.0 against a confidence of 0.003, an error of 0.997 - so a flawless model
    reports an error of roughly 0.5, and a model whose predictions are systematically
    inverted reports a small one. The metric was not measuring calibration at all; it was
    measuring agreement between two predictions, and it disagreed with the Brier score
    recorded beside it by a factor of thousands on the same data.

    ECE needs both classes present and at least one observation; when either is absent it
    is undefined, and is reported as ``None`` rather than as zero, which would read as
    perfect calibration.

    Args:
        probabilities: Predicted class-1 probabilities.
        labels: Ground-truth labels.

    Returns:
        The ECE, or ``None`` if undefined.
    """
    n_samples = int(labels.shape[0])
    n_positive = int((labels == 1).sum())
    if n_samples == 0 or n_positive == 0 or n_positive == n_samples:
        return None

    edges = np.linspace(0.0, 1.0, min(_CALIBRATION_BINS, n_samples) + 1)
    total = 0.0
    n_bins = len(edges) - 1
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        # The final bin is closed on the right, so a probability of exactly 1.0 is counted
        # rather than falling outside every bin. Dropping it would leave the top of the
        # confidence range unmeasured while the reported error still claimed to cover it.
        if index == n_bins - 1:
            in_bin = (probabilities >= low) & (probabilities <= high)
        else:
            in_bin = (probabilities >= low) & (probabilities < high)
        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            continue
        observed_frequency = float(labels[in_bin].mean())
        confidence_in_bin = float(probabilities[in_bin].mean())
        total += abs(observed_frequency - confidence_in_bin) * (n_in_bin / n_samples)
    return total


def _compute_metrics(
    probabilities: NDArray[np.float64],
    predictions: NDArray[np.int64],
    labels: NDArray[np.int64],
    *,
    split_name: str,
) -> ModelMetrics:
    """Compute the metric suite for one split.

    Rank-based metrics are gated on both classes being present rather than on whether the
    underlying scikit-learn function happens to raise. It does not raise: it emits a
    warning and returns ``nan`` or, for average precision, ``0.0``. A ``0.0`` returned
    without being asked for is a number this system did not measure, and it is exactly the
    kind of value that later reads as a result.

    Args:
        probabilities: Predicted class-1 probabilities.
        predictions: Hard class predictions.
        labels: Ground-truth labels.
        split_name: Identifier of the split being evaluated.

    Returns:
        The metric suite, with ``None`` for any metric the split cannot support.
    """
    n_samples = int(labels.shape[0])
    n_positive = int((labels == 1).sum())
    both_classes = 0 < n_positive < n_samples

    return ModelMetrics(
        split_name=split_name,
        n_samples=n_samples,
        n_positive=n_positive,
        roc_auc=_safe_metric(roc_auc_score, labels, probabilities) if both_classes else None,
        pr_auc=_safe_metric(average_precision_score, labels, probabilities)
        if both_classes
        else None,
        f1=_safe_metric(f1_score, labels, predictions, zero_division=0.0),
        precision=_safe_metric(precision_score, labels, predictions, zero_division=0.0),
        recall=_safe_metric(recall_score, labels, predictions, zero_division=0.0),
        brier=_safe_metric(brier_score_loss, labels, probabilities),
        expected_calibration_error=_expected_calibration_error(probabilities, labels),
        accuracy=_safe_metric(accuracy_score, labels, predictions),
    )


def _class_one_probabilities(estimator: Any, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract class-1 probabilities from a fitted estimator.

    Delegates to :func:`focus_engine.models.artifacts.class_one_probabilities`, so that
    training and scoring cannot disagree about which column is the positive class.

    Args:
        estimator: A fitted scikit-learn estimator.
        matrix: The feature matrix to score.

    Returns:
        A one-dimensional array of class-1 probabilities.
    """
    return class_one_probabilities(estimator, matrix)


def _resolved_hyperparameters(config: TrainingConfig) -> dict[str, Any]:
    """Merge the algorithm defaults with the caller's overrides.

    The caller's values win, so that a deliberate experiment is recorded as the model's
    actual configuration rather than being folded into a hidden default.

    Args:
        config: The training configuration.

    Returns:
        The resolved hyperparameters.
    """
    return {**_default_hyperparameters(config.algorithm), **config.hyperparameters}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Everything a training run needs that is not the data itself.

    The fields here are the coordinates that make a result reproducible and traceable:
    algorithm, hyperparameters, seed, dataset version, feature-set version, target
    definition version, label provenance, and data origin. ``model_id`` and
    ``model_version`` are supplied by the caller because the caller is the one who owns
    the identity of the model being trained.

    ``target_definition_version`` is required rather than defaulted. A model is
    uninterpretable without knowing what its label meant; a model trained against one
    definition and evaluated against another is not evaluated, it is an anecdote.

    Attributes:
        algorithm: Which classifier to fit.
        model_id: Stable identity of the modelling approach.
        model_version: Identity of this specific artefact.
        feature_set_version: The feature definitions the model is trained against.
        dataset_version: The training dataset's identity.
        target_definition_version: Version of the label definition.
        label_provenance: What kind of label the model is trained against.
        data_origin: Whether the training data is real or synthetic.
        seed: The run's master seed.
        hyperparameters: Caller overrides for the algorithm's defaults.
    """

    algorithm: Algorithm
    model_id: str
    model_version: ModelVersion
    feature_set_version: FeatureSetVersion
    dataset_version: DatasetVersion
    target_definition_version: str
    label_provenance: Provenance
    data_origin: DataOrigin
    seed: int = 0
    hyperparameters: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the seed and the model identity.

        Raises:
            ValueError: If the seed is negative or the model identity is blank. A negative
                seed cannot be re-derived, and a blank identity makes the model
                untrackable.
        """
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative so it can be re-derived; got {self.seed}")
        if not self.model_id.strip():
            raise ValueError("model_id must not be blank; a model with no identity is untrackable")
        if not self.target_definition_version.strip():
            raise ValueError(
                "target_definition_version must not be blank; a model whose label definition "
                "is unknown cannot be evaluated or compared against anything"
            )


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """The result of a single training run.

    This is a data object, not a behaviour. The caller decides whether to register it,
    evaluate it further, or discard it; nothing here has the side effect of advancing the
    model's lifecycle.

    Attributes:
        artifact: The fitted model with its schema and provenance.
        metrics: The metric suite computed on the holdout split.
        record: The registry entry describing the run, at ``EXPERIMENTAL``.
        encoding_digest: The digest of the schema the model was trained against.
        seed: The derived child seed actually used to fit the estimator.
    """

    artifact: ModelArtifact
    record: ModelRecord
    metrics: tuple[ModelMetrics, ...] = field(default_factory=tuple)
    encoding_digest: str = ""
    seed: int = 0


def train_model(
    config: TrainingConfig,
    train: DesignMatrix,
    holdout: DesignMatrix,
) -> TrainingOutcome:
    """Fit a classifier and return the result with a complete audit record.

    The caller supplies both splits, so split policy stays outside this function. The
    holdout is read once, to produce metrics, and never used to fit anything.

    Args:
        config: The training configuration.
        train: The training set, fully encoded and labelled.
        holdout: The held-out set, fully encoded and labelled.

    Returns:
        The training outcome, carrying the artifact, metrics, and registry record.

    Raises:
        ValueError: If either split is empty, if the training set is single-class, or if
            the two splits do not share an encoding schema.
    """
    if train.n_rows == 0:
        raise ValueError("the training set is empty; a model cannot be fitted on no observations")
    if train.n_rows == 1:
        raise ValueError(
            "the training set has one observation; a model fitted on one observation will "
            "predict that observation and nothing else"
        )
    if holdout.n_rows == 0:
        raise ValueError(
            "the holdout set is empty; evaluation without observations is not a result"
        )
    if train.n_positive == 0 or train.n_positive == train.n_rows:
        raise ValueError(
            f"the training set has {train.n_positive} positive and "
            f"{train.n_rows - train.n_positive} negative examples. A model fitted on a "
            "single class learns the class prior rather than a pattern, and ROC-AUC is "
            "undefined on the result; use a split that contains both classes."
        )
    if train.schema_digest() != holdout.schema_digest():
        raise ValueError(
            "the train and holdout schemas disagree. A model fitted against one encoding "
            "and evaluated against another is not evaluated, it is confused."
        )

    estimator_seed = derive_seed(
        config.seed, config.model_id, str(config.model_version), "training"
    )
    hyperparameters = _resolved_hyperparameters(config)
    estimator = _build_estimator(config.algorithm, hyperparameters, estimator_seed)
    estimator.fit(train.matrix, train.labels)

    probabilities = _class_one_probabilities(estimator, holdout.matrix)
    predictions = (probabilities >= 0.5).astype(np.int64)
    holdout_metrics = _compute_metrics(
        probabilities, predictions, holdout.labels, split_name="holdout"
    )

    trained_at = utc_now()
    record = ModelRecord(
        model_id=config.model_id,
        model_version=config.model_version,
        status=ModelStatus.EXPERIMENTAL,
        algorithm=_ALGORITHM_NAMES[config.algorithm],
        hyperparameters=hyperparameters,
        feature_set_version=config.feature_set_version,
        dataset_version=config.dataset_version,
        trained_at=trained_at,
        metrics=(holdout_metrics,),
        target_definition_version=config.target_definition_version,
        label_provenance=config.label_provenance,
        data_origin=config.data_origin,
        seed=config.seed,
    )

    artifact = ModelArtifact(
        estimator=estimator,
        record=record,
        encoding_schema=train.schema(),
        encoding_digest=train.schema_digest(),
        columns=train.columns,
        algorithm_name=_ALGORITHM_NAMES[config.algorithm],
        seed=estimator_seed,
        trained_at=trained_at,
        library_versions=library_versions(),
        # Taken from the training split only. The holdout is not consulted for anything
        # that ends up inside the model, because a reference value derived from held-out
        # data would be information about the evaluation set leaking into the artifact.
        feature_reference=column_references(train),
    )

    return TrainingOutcome(
        artifact=artifact,
        metrics=(holdout_metrics,),
        record=record,
        encoding_digest=train.schema_digest(),
        seed=estimator_seed,
    )
