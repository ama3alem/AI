"""Unit tests for :mod:`focus_engine.schemas.versioning`.

The load-bearing test in this file is
:func:`test_synthetic_only_model_cannot_be_marked_validated`. It encodes the rule that
the engine may never report a simulator-trained model as validated, which is the single
most damaging false claim this system could make.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from focus_engine.schemas.primitives import DataOrigin, Provenance, utc_now
from focus_engine.schemas.versioning import (
    ExperimentConclusion,
    ExperimentRecord,
    ModelMetrics,
    ModelRecord,
    ModelStatus,
    ReproductionStamp,
    VersionStamp,
)

pytestmark = pytest.mark.unit

_BASE_RECORD: dict[str, object] = {
    "model_id": "declining_engagement_classifier",
    "model_version": "MODEL_DECL_V1",
    "algorithm": "sklearn.linear_model.LogisticRegression",
    "hyperparameters": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
    "feature_set_version": "FEATURE_SET_V1",
    "dataset_version": "DATASET_SYNTHETIC_V1",
    "trained_at": "2026-09-26T10:00:00+00:00",
    "target_definition_version": "TARGET_DECLINING_V1",
    "label_provenance": Provenance.SYNTHETIC_LABEL,
    "data_origin": DataOrigin.SYNTHETIC,
    "seed": 20260926,
}


def _record(**overrides: object) -> ModelRecord:
    """Build a model record from the shared base, with overrides applied.

    Args:
        **overrides: Fields to override.

    Returns:
        The constructed record.
    """
    return ModelRecord(**{**_BASE_RECORD, **overrides})


# --------------------------------------------------------------------------------------
# Version grammar
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version",
    ["FEATURE_SET_V1", "DATASET_SYNTHETIC_V3", "MODEL_DECL_V12", "CODE_V2"],
)
def test_well_formed_versions_are_accepted(version: str) -> None:
    stamp = VersionStamp(
        dataset_version="DATASET_SYNTHETIC_V1",
        feature_set_version=version if version.startswith("FEATURE") else "FEATURE_SET_V1",
        code_version="CODE_V1",
    )
    assert stamp.feature_set_version == (
        version if version.startswith("FEATURE") else "FEATURE_SET_V1"
    )


@pytest.mark.parametrize(
    "version",
    [
        "feature_set_v1",  # lowercase
        "FEATURE_SET",  # no version number
        "FEATURE_SET_V",  # empty version number
        "FEATURE_SET_V1_",  # trailing separator
        "FEATURE SET V1",  # spaces
        "V1",  # no component
        "",
    ],
)
def test_malformed_versions_are_rejected(version: str) -> None:
    with pytest.raises(ValidationError):
        VersionStamp(
            dataset_version="DATASET_SYNTHETIC_V1",
            feature_set_version=version,
            code_version="CODE_V1",
        )


def test_version_stamp_describe_is_stable_and_complete() -> None:
    stamp = VersionStamp(
        dataset_version="DATASET_SYNTHETIC_V1",
        feature_set_version="FEATURE_SET_V2",
        code_version="CODE_V1",
    )
    assert stamp.describe() == "dataset=DATASET_SYNTHETIC_V1 features=FEATURE_SET_V2 code=CODE_V1"


def test_reproduction_stamp_adds_model_and_seed() -> None:
    stamp = ReproductionStamp(
        dataset_version="DATASET_SYNTHETIC_V1",
        feature_set_version="FEATURE_SET_V1",
        code_version="CODE_V1",
        model_version="MODEL_DECL_V1",
        seed=7,
        library_versions={"numpy": "2.5.3", "scikit-learn": "1.9.1"},
    )
    assert stamp.seed == 7
    assert stamp.library_versions["numpy"] == "2.5.3"
    assert "MODEL_DECL_V1" not in stamp.describe()


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReproductionStamp(
            dataset_version="DATASET_SYNTHETIC_V1",
            feature_set_version="FEATURE_SET_V1",
            code_version="CODE_V1",
            model_version="MODEL_DECL_V1",
            seed=-1,
        )


# --------------------------------------------------------------------------------------
# The synthetic-cannot-be-validated invariant
# --------------------------------------------------------------------------------------


def test_synthetic_only_model_cannot_be_marked_validated() -> None:
    with pytest.raises(ValidationError, match="cannot be 'validated'"):
        _record(status=ModelStatus.VALIDATED)


def test_synthetic_only_model_cannot_be_marked_production() -> None:
    with pytest.raises(ValidationError, match="cannot be 'production'"):
        _record(status=ModelStatus.PRODUCTION)


@pytest.mark.parametrize("status", [ModelStatus.EXPERIMENTAL, ModelStatus.CANDIDATE])
def test_synthetic_only_model_may_hold_non_validating_statuses(status: ModelStatus) -> None:
    assert _record(status=status).status is status


def test_synthetic_model_may_be_retired() -> None:
    """Retirement is always permitted: recording that something was withdrawn is honest."""
    assert _record(status=ModelStatus.RETIRED).status is ModelStatus.RETIRED


def test_real_origin_model_may_be_marked_validated() -> None:
    """The guard blocks the specific false claim, not the legitimate status transition."""
    record = _record(
        status=ModelStatus.VALIDATED,
        data_origin=DataOrigin.REAL,
        label_provenance=Provenance.GROUND_TRUTH,
        dataset_version="DATASET_COHORT_V1",
    )
    assert record.status is ModelStatus.VALIDATED


@pytest.mark.parametrize("status", [ModelStatus.VALIDATED, ModelStatus.PRODUCTION])
def test_rejection_message_names_the_requested_status_and_suggests_a_permitted_one(
    status: ModelStatus,
) -> None:
    """The message must name both the refused status and an allowed alternative.

    An operator who hit this needs to know which status was refused and what to use
    instead, without reading the source.
    """
    with pytest.raises(ValidationError) as excinfo:
        _record(status=status, model_id="other_model")
    message = str(excinfo.value)
    assert f"cannot be '{status.value}'" in message
    assert "Use status 'experimental'" in message


# --------------------------------------------------------------------------------------
# Model metrics
# --------------------------------------------------------------------------------------


def test_metrics_reject_out_of_range_rates() -> None:
    with pytest.raises(ValidationError, match=r"\[0.0, 1.0\]"):
        ModelMetrics(split_name="temporal_holdout", n_samples=10, roc_auc=1.5)


def test_metrics_reject_negative_brier() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        ModelMetrics(split_name="temporal_holdout", n_samples=10, brier=-0.01)


def test_metrics_reject_nan_brier() -> None:
    with pytest.raises(ValidationError, match="NaN"):
        ModelMetrics(split_name="temporal_holdout", n_samples=10, brier=float("nan"))


def test_metrics_allow_undefined_ranking_scores() -> None:
    """A single-class split has no defined ROC-AUC; that must be representable as None."""
    metrics = ModelMetrics(split_name="degenerate_split", n_samples=4, n_positive=0)
    assert metrics.roc_auc is None
    assert metrics.pr_auc is None


def test_metrics_record_positive_count_for_balance_visibility() -> None:
    metrics = ModelMetrics(split_name="temporal_holdout", n_samples=100, n_positive=7)
    assert metrics.n_positive == 7


def test_metrics_reject_negative_sample_count() -> None:
    with pytest.raises(ValidationError):
        ModelMetrics(split_name="temporal_holdout", n_samples=-1)


# --------------------------------------------------------------------------------------
# Experiment records
# --------------------------------------------------------------------------------------


def test_experiment_record_requires_stated_limitations_to_be_explicit() -> None:
    """A record must carry its evaluation protocol; a bare metric is not reproducible."""
    record = ExperimentRecord(
        experiment_id="EXP-0001",
        hypothesis="Personal baseline features improve decline detection.",
        conclusion=ExperimentConclusion.INCONCLUSIVE,
        dataset_version="DATASET_SYNTHETIC_V1",
        feature_set_version="FEATURE_SET_V1",
        seed=1,
        data_origin=DataOrigin.SYNTHETIC,
        label_provenance=Provenance.SYNTHETIC_LABEL,
        evaluation_protocol="temporal holdout per learner, last 20% of each timeline",
        limitations=("synthetic labels only",),
    )
    assert record.conclusion is ExperimentConclusion.INCONCLUSIVE
    assert record.limitations == ("synthetic labels only",)


def test_inconclusive_is_a_first_class_conclusion() -> None:
    """Not establishing a result must be reportable, not a failure to record."""
    assert ExperimentConclusion.INCONCLUSIVE.value == "inconclusive"
    assert ExperimentConclusion.HYPOTHESIS_REJECTED.value == "hypothesis_rejected"
    assert ExperimentConclusion.BLOCKED.value == "blocked"


def test_experiment_record_defaults_recorded_at_to_aware_utc() -> None:
    record = ExperimentRecord(
        experiment_id="EXP-0002",
        hypothesis="h",
        conclusion=ExperimentConclusion.HYPOTHESIS_SUPPORTED,
        dataset_version="DATASET_SYNTHETIC_V1",
        feature_set_version="FEATURE_SET_V1",
        seed=1,
        data_origin=DataOrigin.SYNTHETIC,
        label_provenance=Provenance.SYNTHETIC_LABEL,
        evaluation_protocol="temporal holdout",
    )
    assert record.recorded_at <= utc_now()


def test_experiment_conclusion_detail_is_optional_and_bounded() -> None:
    base: dict[str, object] = {
        "experiment_id": "EXP-0003",
        "hypothesis": "h",
        "conclusion": ExperimentConclusion.HYPOTHESIS_SUPPORTED,
        "dataset_version": "DATASET_SYNTHETIC_V1",
        "feature_set_version": "FEATURE_SET_V1",
        "seed": 1,
        "data_origin": DataOrigin.SYNTHETIC,
        "label_provenance": Provenance.SYNTHETIC_LABEL,
        "evaluation_protocol": "temporal holdout",
    }
    assert ExperimentRecord(**base).conclusion_detail is None
    assert ExperimentRecord(**base, conclusion_detail=0.9).conclusion_detail == 0.9
    with pytest.raises(ValidationError):
        ExperimentRecord(**base, conclusion_detail=1.4)
