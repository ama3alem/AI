"""Versioning and record schemas.

Reproducibility requires that any result can be traced to the exact inputs that
produced it. These types make that trace part of the data model rather than a
convention: a result that carries a :class:`ReproductionStamp` cannot be separated from
its dataset version, feature-set version, model version, code version, and random seed.

The version identifiers are opaque strings validated against a strict grammar. They are
deliberately *not* auto-incrementing integers, because an integer invites silent
reuse. ``FEATURE_SET_V1`` and ``FEATURE_SET_V2`` are distinct historical facts: changing
a feature calculation must mint a new version rather than edit the old one, so that
results computed under the old definition remain interpretable.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Final

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from focus_engine.schemas.primitives import (
    DataOrigin,
    Probability,
    Provenance,
    Timestamp,
    non_empty_text,
    utc_now,
)

__all__ = [
    "BaselineVersion",
    "CodeVersion",
    "DatasetVersion",
    "ExperimentConclusion",
    "ExperimentRecord",
    "EvaluationVersion",
    "FeatureSetVersion",
    "ModelMetrics",
    "ModelRecord",
    "ModelStatus",
    "ModelVersion",
    "OutcomeVersion",
    "PolicyVersion",
    "ReproductionStamp",
    "TemporalVersion",
    "VersionStamp",
    "checked_version",
]

_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_V[0-9]+$")


def _validate_version(value: str) -> str:
    """Validate a version identifier against the version grammar.

    Args:
        value: Candidate version string, e.g. ``FEATURE_SET_V2``.

    Returns:
        The validated version string.

    Raises:
        ValueError: If the string does not match the version grammar.
    """
    if not _VERSION_PATTERN.match(value):
        raise ValueError(
            f"invalid version identifier {value!r}; expected uppercase snake case ending in "
            "_V<number>, for example FEATURE_SET_V1 or DATASET_SYNTHETIC_V3"
        )
    return value


#: Identifies a specific, frozen set of feature definitions and their calculations.
FeatureSetVersion = Annotated[str, AfterValidator(_validate_version)]
#: Identifies a specific, frozen set of baseline dimensions and their statistics.
#:
#: Minted for the same reason as :data:`FeatureSetVersion`: changing how a centre or a
#: spread is computed would otherwise silently reinterpret every baseline already stored
#: against the old definition, with no record that the meaning had moved.
BaselineVersion = Annotated[str, AfterValidator(_validate_version)]
#: Identifies a specific, frozen set of temporal state definitions.
#:
#: Minted for the same reason again. The state ladder, the persistence rule, and the
#: stability measure together determine what a recorded state *means*; changing any of
#: them without a new version would leave previously recorded states explainable only
#: under definitions that no longer exist.
TemporalVersion = Annotated[str, AfterValidator(_validate_version)]
#: Identifies a specific, frozen set of intervention policy definitions.
#:
#: The restraint ladder, the candidate catalogue, and the rule that decides which states
#: warrant an intervention together determine what a recorded policy decision *means*.
#: Changing any of them without minting a new version would leave past decisions
#: re-interpretable only under rules that no longer exist, which is the same failure the
#: other version aliases exist to prevent.
PolicyVersion = Annotated[str, AfterValidator(_validate_version)]
#: Identifies a specific, frozen set of outcome measurement definitions.
#:
#: Minted for the same reason again, and this one has a second reason on top of the first.
#: The before/after window widths determine *which events are even inside the
#: measurement*; a narrower after window measures a shorter stretch of the learner's
#: behaviour and can report a deterioration where a wider one reports no change. So the
#: window widths are not a presentational detail of a recorded outcome, they determine
#: what the recorded outcome means, and a stored record read under different widths would
#: be a record of a measurement that was never made. The measure set, the direction
#: vocabulary, and the window geometry are therefore frozen together under one identifier.
OutcomeVersion = Annotated[str, AfterValidator(_validate_version)]
#: Identifies a specific, frozen set of evaluation definitions.
#:
#: Minted for the same reason again, and this one carries a second reason on top of the
#: first. A :class:`~focus_engine.evaluation.models.PredictionHorizon` is the *only* thing
#: that decides which events a ground truth is permitted to read, so the way horizons are
#: turned into windows, and the way an insufficient reading is separated from an
#: unassessable one, together determine what a recorded evaluation *means*. Changing the
#: boundary rule from half-open to inclusive would not merely reformat stored results: it
#: would move evidence across the prediction edge, and an event exactly at the horizon
#: boundary would be counted in a store of ground truths that had excluded it. The target
#: vocabulary, the horizon rule, and the evidence gate are therefore frozen together under
#: one identifier.
EvaluationVersion = Annotated[str, AfterValidator(_validate_version)]
#: Identifies a specific, frozen dataset artifact.
DatasetVersion = Annotated[str, AfterValidator(_validate_version)]
#: Identifies a specific trained model artifact.
ModelVersion = Annotated[str, AfterValidator(_validate_version)]
#: Identifies the engine source revision that produced a result.
CodeVersion = Annotated[str, AfterValidator(_validate_version)]


def checked_version(value: str) -> str:
    """Validate a version identifier without constructing a model.

    The version aliases above enforce the grammar whenever a value passes through a
    pydantic model. A registry that accepts a version string as a plain argument has no
    such model boundary, so it needs the same check directly; otherwise an unvalidated
    version could be registered and later recorded in a result.

    Args:
        value: Candidate version identifier, e.g. ``FEATURE_SET_V2``.

    Returns:
        The validated version string, unchanged.

    Raises:
        ValueError: If the string does not match the version grammar.
    """
    return _validate_version(value)


class ModelStatus(StrEnum):
    """Lifecycle position of a model artifact.

    A model may only become ``PRODUCTION`` by explicit promotion. There is no
    automatic promotion path, because automatic promotion from live data would let an
    unevaluated model silently take over decisions.
    """

    EXPERIMENTAL = "experimental"
    """Trained and scored once. No claim attached."""

    VALIDATED = "validated"
    """Passed a pre-registered evaluation on a held-out split that it was not selected on."""

    CANDIDATE = "candidate"
    """Passed evaluation and is proposed for production use, pending approval."""

    PRODUCTION = "production"
    """Approved for use by a serving path."""

    RETIRED = "retired"
    """Withdrawn. Retained for reproducibility of historical results."""


class VersionStamp(BaseModel):
    """The version coordinates of a single computed artifact.

    Attached to predictions, feature vectors, model artifacts, and experiment results so
    that no result exists without its provenance coordinates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: DatasetVersion
    """The dataset the artifact was computed from."""

    feature_set_version: FeatureSetVersion
    """The feature definitions in force when it was computed."""

    code_version: CodeVersion
    """The engine source revision."""

    def describe(self) -> str:
        """Render the stamp as a compact, human-readable coordinate.

        Returns:
            A single-line description such as ``dataset=DATASET_X_V1 features=FEATURE_SET_V1``.
        """
        return (
            f"dataset={self.dataset_version} features={self.feature_set_version} "
            f"code={self.code_version}"
        )


class ReproductionStamp(VersionStamp):
    """A :class:`VersionStamp` extended with the determinism coordinates.

    Two runs carrying equal reproduction stamps and equal inputs are expected to agree
    within the tolerance documented for the algorithm in use.
    """

    model_version: ModelVersion
    """The model artifact, when the stamped result involved a model."""

    seed: int = Field(ge=0)
    """The master random seed governing the run."""

    library_versions: dict[str, str] = Field(default_factory=dict)
    """Resolved versions of the numeric/ML libraries, because library upgrades can move
    results and a version stamp that omits them is not a reproducible stamp."""


class ModelMetrics(BaseModel):
    """Evaluation metrics for one model on one evaluation split.

    Accuracy is deliberately not the only field. For an imbalanced behavioural-forecasting
    target, accuracy is dominated by the majority class and is close to uninformative.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split_name: str
    """Identifier of the split these metrics were computed on, e.g. ``temporal_holdout``."""

    n_samples: int = Field(ge=0)
    """Number of scored samples."""

    n_positive: int | None = Field(default=None, ge=0)
    """Positive-class count, so class balance is visible alongside any metric."""

    roc_auc: float | None = None
    """Ranking quality. ``None`` when undefined, e.g. a single-class split."""

    pr_auc: float | None = None
    """Precision-recall area. More informative than ROC-AUC under imbalance."""

    f1: float | None = None
    """Harmonic mean of precision and recall at the operating threshold."""

    precision: float | None = None
    """Precision at the operating threshold."""

    recall: float | None = None
    """Recall at the operating threshold."""

    brier: float | None = None
    """Mean squared error of predicted probabilities. Lower is better calibrated."""

    expected_calibration_error: float | None = None
    """Mean absolute gap between confidence and empirical accuracy across bins."""

    accuracy: float | None = None
    """Included for completeness and explicitly not treated as a decision criterion."""

    @field_validator("roc_auc", "pr_auc", "f1", "precision", "recall", "expected_calibration_error")
    @classmethod
    def _validate_unit_range(cls, value: float | None) -> float | None:
        """Validate that rate metrics lie within ``[0.0, 1.0]``.

        Args:
            value: Candidate metric value.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value falls outside the unit interval.
        """
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"metric must lie within [0.0, 1.0]; got {value}")
        return value

    @field_validator("brier")
    @classmethod
    def _validate_non_negative(cls, value: float | None) -> float | None:
        """Validate that the Brier score is non-negative.

        Args:
            value: Candidate Brier score.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value is negative.
        """
        if value is None:
            return None
        if value != value:  # NaN
            raise ValueError("brier must not be NaN")
        if value < 0.0:
            raise ValueError(f"brier score must be non-negative; got {value}")
        return value


class ModelRecord(BaseModel):
    """Registry entry describing one model artifact.

    The registry is append-only. Status transitions are recorded, never overwritten, so
    that the path a model took to production remains inspectable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: non_empty_text
    """Stable identity of the modelling approach, shared across its versions."""

    model_version: ModelVersion
    """Identity of this specific artifact."""

    status: ModelStatus
    """Current lifecycle position."""

    algorithm: non_empty_text
    """Concrete algorithm, e.g. ``sklearn.ensemble.HistGradientBoostingClassifier``."""

    hyperparameters: dict[str, str | int | float | bool | None]
    """Full resolved hyperparameters. Recorded as primitives so the record stays portable."""

    feature_set_version: FeatureSetVersion
    """Feature definitions this model was trained against."""

    dataset_version: DatasetVersion
    """Training dataset identity."""

    trained_at: Timestamp
    """When training completed."""

    metrics: tuple[ModelMetrics, ...] = ()
    """Evaluation results, one entry per split evaluated."""

    target_definition_version: str
    """Version of the label definition. A model is meaningless without it."""

    label_provenance: Provenance
    """What kind of label the model was trained against.

    ``SYNTHETIC_LABEL`` here means the model has only ever seen simulator output and
    therefore has no established validity on real learners."""

    data_origin: DataOrigin
    """Whether the training data was real or synthetic."""

    seed: int = Field(ge=0)
    """Random seed used for training."""

    notes: str = ""
    """Free-text caveats. Not a substitute for ``EXPERIMENTS.md``."""

    @model_validator(mode="after")
    def _reject_synthetic_as_validated(self) -> ModelRecord:
        """Forbid marking a synthetic-only model as validated.

        A model trained exclusively on synthetic labels can demonstrate that the pipeline
        learns the simulator's structure, but it cannot be validated for real learners.
        Allowing that status would be the single most damaging false claim this system
        could make.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If status is ``VALIDATED`` or ``PRODUCTION`` while every recorded
                metric came from synthetic-origin data.
        """
        if (
            self.status in (ModelStatus.VALIDATED, ModelStatus.PRODUCTION)
            and self.data_origin is DataOrigin.SYNTHETIC
        ):
            raise ValueError(
                f"model {self.model_id}:{self.model_version} cannot be "
                f"{self.status.value!r}: it was trained on {self.data_origin.value} data. "
                "Synthetic-only results demonstrate pipeline behaviour, not validity on "
                "real learners. Use status 'experimental'."
            )
        return self


class ExperimentConclusion(StrEnum):
    """Standardised outcome vocabulary for a recorded experiment.

    Forcing experiments to conclude with one of these, rather than free-text enthusiasm,
    makes "we could not establish this" a reportable result.
    """

    HYPOTHESIS_SUPPORTED = "hypothesis_supported"
    """The pre-registered hypothesis held on the evaluation split."""

    HYPOTHESIS_REJECTED = "hypothesis_rejected"
    """The pre-registered hypothesis did not hold."""

    INCONCLUSIVE = "inconclusive"
    """Evidence was insufficient in either direction. A legitimate and common result."""

    BLOCKED = "blocked"
    """The experiment could not be run, e.g. missing data or a failed prerequisite."""


class ExperimentRecord(BaseModel):
    """One reproducible experiment.

    ``EXPERIMENTS.md`` is rendered from these records. A result with no experiment record
    is not a result; it is an anecdote.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: non_empty_text
    """Stable identifier, e.g. ``EXP-0007``."""

    hypothesis: non_empty_text
    """What was expected, stated so that it could have been falsified."""

    conclusion: ExperimentConclusion
    """What actually happened."""

    dataset_version: DatasetVersion
    """Dataset used."""

    feature_set_version: FeatureSetVersion
    """Feature definitions used."""

    model_version: ModelVersion | None = None
    """Model used, if any."""

    model_id: str | None = None
    """Model identity, if any."""

    seed: int = Field(ge=0)
    """Random seed governing the run."""

    metrics: tuple[ModelMetrics, ...] = ()
    """Measured results."""

    data_origin: DataOrigin
    """Whether the underlying data was real or synthetic."""

    label_provenance: Provenance
    """What kind of label was scored against."""

    evaluation_protocol: non_empty_text
    """How the split was constructed. Temporal splits and learner-disjoint splits are
    materially different and must never be conflated."""

    leakage_checks: tuple[str, ...] = ()
    """Leakage checks executed for this run, by name."""

    limitations: tuple[str, ...] = ()
    """Stated limitations. A record with none is treated as incomplete."""

    recorded_at: Timestamp = Field(default_factory=utc_now)
    """When the record was written."""

    conclusion_detail: Probability | None = None
    """Optional strength-of-conclusion value in ``[0.0, 1.0]``. Not a p-value; no
    significance testing is claimed anywhere in this project."""
