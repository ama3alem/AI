"""Shared builders for the uncertainty-engine tests (Phase 9).

The uncertainty engine's behaviour is almost entirely a function of three inputs - a
model's recorded calibration, a count of evidence, and a baseline maturity - so the tests
need to vary those three precisely and hold the rest still. Training a fresh model per
test would be both slow and imprecise: it is the *recorded* calibration on the record that
the engine reads, so a test about a miscalibrated model has to be able to state a
calibration error, not hope a fit produces one.

These builders therefore do the expensive work once and let a test adjust the artifact's
record to the situation it is about. Adjusting the record is not a shortcut around the
engine: the engine reads calibration from the record precisely so that there is one
number in the system and it cannot disagree with the one the model was approved on.
"""

from __future__ import annotations

import numpy as np

from focus_engine.configuration.thresholds import BaselineSettings, UncertaintySettings
from focus_engine.features import FEATURE_SET_V1
from focus_engine.models.artifacts import ModelArtifact
from focus_engine.models.encoding import encode_rows
from focus_engine.models.trainers import (
    Algorithm,
    TrainingConfig,
    train_model,
)
from focus_engine.schemas.primitives import DataOrigin
from focus_engine.schemas.versioning import ModelMetrics, ModelRecord, ModelStatus
from focus_engine.uncertainty.evidence import EvidenceVolume
from tests.unit.models.scenarios import make_labelled_rows, make_vector

#: Sizes used by the shared artifacts. Large enough for a logistic fit to converge on
#: separated data, small enough that the whole Phase 9 suite stays fast.
TRAIN_ROWS = 400
HOLDOUT_ROWS = 200

#: Observation counts chosen to land on each rung of the default maturity ladder. These are
#: the counts the ``BaselineSettings`` defaults define, restated here so a test reads as
#: "a learner with a developing baseline" rather than as a magic number.
UNITS_NEW = 5
UNITS_EARLY = 30
UNITS_DEVELOPING = 150
UNITS_ESTABLISHED = 600


def interleaved_dataset(
    *,
    total: int = TRAIN_ROWS + HOLDOUT_ROWS,
) -> list[tuple[object, int]]:
    """Build a labelled dataset whose every contiguous half holds both classes.

    The underlying generator emits all positives before all negatives, which is realistic
    for a file sorted by label and is a trap for a test: a holdout taken as the tail of such
    a list is single-class, no calibration error can be measured on it, and the engine
    correctly answers ``UNKNOWN``. That is the right behaviour and it is tested
    deliberately - but it makes an ordinary test unable to reach the ``RESOLVED`` path for
    reasons that have nothing to do with what it is testing. Interleaving removes the
    accident without weakening the deliberate case, which is built explicitly instead.

    Args:
        total: Number of rows to generate.

    Returns:
        Labelled rows alternating positive and negative.
    """
    raw = make_labelled_rows(total, positive=total // 2, overlap=False)
    positives = [row for row in raw if row[1] == 1]
    negatives = [row for row in raw if row[1] == 0]
    return [row for pair in zip(positives, negatives, strict=True) for row in pair]


def train_shared_artifact(
    *,
    algorithm: Algorithm = Algorithm.LOGISTIC_REGRESSION,
) -> ModelArtifact:
    """Train a model on separated data and return its artifact.

    The data is separable by construction, which is the one property a calibration test
    needs: a model that predicts correctly and reports near-0 or near-1 probabilities is
    well calibrated, and the engine's calibration ceiling for it should sit at the top of
    its range. Nothing else about this model is claimed to be realistic.

    Args:
        algorithm: Which algorithm to fit.

    Returns:
        A fitted artifact whose record carries a genuine holdout measurement.
    """
    rows = interleaved_dataset()
    train = encode_rows(rows[:TRAIN_ROWS])
    holdout = encode_rows(rows[TRAIN_ROWS:])
    config = TrainingConfig(
        algorithm=algorithm,
        model_id="uncertainty-test-model",
        model_version="MODEL_V1",
        feature_set_version=FEATURE_SET_V1,
        dataset_version="SYNTHETIC_V1",
        target_definition_version="TARGET_V1",
        label_provenance="synthetic_label",
        data_origin=DataOrigin.SYNTHETIC,
    )
    return train_model(config, train, holdout).artifact


def with_calibration(
    artifact: ModelArtifact,
    *,
    expected_calibration_error: float | None,
    brier: float | None = None,
    n_samples: int = HOLDOUT_ROWS,
    n_positive: int = HOLDOUT_ROWS // 2,
    split_name: str = "holdout",
) -> ModelArtifact:
    """Return a copy of the artifact whose holdout reports a chosen calibration error.

    The engine reads calibration from the record rather than recomputing it, so this is how
    a test states "what if the model's calibration had been this". It is a legitimate way
    to set up a test because it changes the same field a real training run would.

    Args:
        artifact: The artifact to copy.
        expected_calibration_error: The calibration error to record, or ``None`` to record
            that calibration could not be measured.
        brier: The Brier score to record. Defaults to a value consistent with the error.
        n_samples: Observations the measurement is claimed to rest on.
        n_positive: Positives among them.
        split_name: Which split the metrics belong to.

    Returns:
        A new artifact with the same estimator and a record carrying the given metrics.

    Raises:
        ValueError: If only one of the two metrics is supplied as ``None``.
    """
    if (expected_calibration_error is None) != (brier is None):
        raise ValueError(
            "brier and expected_calibration_error must be set together; a report with one "
            "and not the other is not a measurement"
        )
    metrics = ModelMetrics(
        split_name=split_name,
        n_samples=n_samples,
        n_positive=n_positive,
        brier=brier,
        expected_calibration_error=expected_calibration_error,
    )
    record = artifact.record.model_copy(update={"metrics": (metrics,)})
    return ModelArtifact(
        estimator=artifact.estimator,
        record=record,
        encoding_schema=artifact.encoding_schema,
        encoding_digest=artifact.encoding_digest,
        columns=artifact.columns,
        algorithm_name=artifact.algorithm_name,
        seed=artifact.seed,
        trained_at=artifact.trained_at,
        library_versions=artifact.library_versions,
        feature_reference=artifact.feature_reference,
    )


def without_recorded_metrics(artifact: ModelArtifact) -> ModelArtifact:
    """Return a copy of the artifact whose record holds no holdout measurement at all.

    Args:
        artifact: The artifact to copy.

    Returns:
        A new artifact whose record carries an empty metrics tuple.
    """
    record = artifact.record.model_copy(update={"metrics": ()})
    return ModelArtifact(
        estimator=artifact.estimator,
        record=record,
        encoding_schema=artifact.encoding_schema,
        encoding_digest=artifact.encoding_digest,
        columns=artifact.columns,
        algorithm_name=artifact.algorithm_name,
        seed=artifact.seed,
        trained_at=artifact.trained_at,
        library_versions=artifact.library_versions,
        feature_reference=artifact.feature_reference,
    )


def without_reference_values(artifact: ModelArtifact) -> ModelArtifact:
    """Return a copy of the artifact carrying no per-column reference values.

    This reproduces a model saved before reference values were recorded, and is how the
    "explanation unavailable" path is reached honestly rather than by patching.

    Args:
        artifact: The artifact to copy.

    Returns:
        A new artifact with an empty reference mapping.
    """
    return ModelArtifact(
        estimator=artifact.estimator,
        record=artifact.record,
        encoding_schema=artifact.encoding_schema,
        encoding_digest=artifact.encoding_digest,
        columns=artifact.columns,
        algorithm_name=artifact.algorithm_name,
        seed=artifact.seed,
        trained_at=artifact.trained_at,
        library_versions=artifact.library_versions,
        feature_reference={},
    )


def as_production(record: ModelRecord) -> ModelRecord:
    """Return a record moved to ``PRODUCTION``, for tests that check lifecycle guards.

    Args:
        record: The record to move.

    Returns:
        A record with the production status.
    """
    return record.model_copy(update={"status": ModelStatus.PRODUCTION})


def evidence_at(units: int, *, baseline: BaselineSettings | None = None) -> EvidenceVolume:
    """Build an evidence volume from a raw observation count.

    Args:
        units: The observation count.
        baseline: The maturity ladder, defaulting to the engine's own.

    Returns:
        The derived evidence volume.
    """
    return EvidenceVolume.from_observation_count(units, baseline=baseline or BaselineSettings())


def default_engine_settings() -> UncertaintySettings:
    """Return the default uncertainty configuration.

    Returns:
        A fresh settings instance.
    """
    return UncertaintySettings()


def confident_vector():
    """Build a vector from the positive class of the shared dataset.

    Returns:
        A feature vector the shared artifact scores confidently.
    """
    return interleaved_dataset()[0][0]


def neutral_vector():
    """Build a vector whose features sit near the training medians.

    A vector at the reference values is the one least likely to move a prediction, which
    makes it the right subject for the "nothing was attributable" case.

    Returns:
        A feature vector built from the default scenario values.
    """
    return make_vector()


def probabilities_around(values: list[float]) -> np.ndarray:
    """Wrap a list of probabilities as a one-dimensional float array.

    Args:
        values: The probabilities.

    Returns:
        A float64 array.
    """
    return np.asarray(values, dtype=np.float64)


def labels_from(values: list[int]) -> np.ndarray:
    """Wrap a list of labels as a one-dimensional integer array.

    Args:
        values: The binary labels.

    Returns:
        An int64 array.
    """
    return np.asarray(values, dtype=np.int64)
