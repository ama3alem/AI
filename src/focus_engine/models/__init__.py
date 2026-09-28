"""Baseline ML models (Phase 8).

This layer provides a minimal, auditable modelling surface that fits the project's
separations: prediction is distinct from decision; population vs personal is explicit; and
an implementation being complete is distinct from it being scientifically validated.

Constraints:
- Algorithms start with scikit-learn classifiers (logistic regression, random forest,
  gradient boosting). No deep learning.
- Every trained artifact carries the ``ModelRecord`` that documents it, and recording it in
  the append-only registry is an explicit call the caller makes. Training does not register
  anything by itself: a side effect that writes to shared state at fit time would make a
  training call unsafe to repeat, and a registry entry nobody chose to make is not
  evidence that a model was reviewed.
- Status transitions are explicit, with no automatic promotion. A synthetic-only model
  may never be marked ``VALIDATED`` or ``PRODUCTION`` (the guard is enforced by
  :class:`~focus_engine.schemas.versioning.ModelRecord`).
- Training, prediction, serialisation and reloading are reproducible from a recorded seed
  under a fixed library environment. The versions that can move a numeric result are
  recorded with the artifact, and :meth:`~focus_engine.models.artifacts.ModelArtifact.version_drift`
  reports them on reload; a claim of reproducibility that omits the environment is not a
  claim anyone can check.
- A prediction is never returned without its provenance: model identity, schema, feature-set
  version, and the threshold applied all travel with the value. Uncertainty is deliberately
  **not** claimed here - a calibrated confidence derived from calibration, evidence volume,
  and baseline maturity is Phase 9's work, and this layer returns the probability and the
  threshold rather than a confidence it has no standing to produce.

This package imports only from layers above it or supporting layers, and never from later
layers in the pipeline.
"""

from __future__ import annotations

from focus_engine.models.artifacts import (
    ModelArtifact,
    load_artifact,
    save_artifact,
)
from focus_engine.models.predictors import (
    Prediction,
    PredictionRefusal,
    predict_with_artifact,
    score_batch,
)
from focus_engine.models.registry import (
    ModelRegistry,
    ModelRegistryError,
    ModelRegistryNotFoundError,
)
from focus_engine.models.trainers import (
    Algorithm,
    TrainingConfig,
    TrainingOutcome,
    train_model,
)

__all__ = [
    "Algorithm",
    "ModelArtifact",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelRegistryNotFoundError",
    "Prediction",
    "PredictionRefusal",
    "TrainingConfig",
    "TrainingOutcome",
    "load_artifact",
    "predict_with_artifact",
    "save_artifact",
    "score_batch",
    "train_model",
]
