"""Scoring a feature vector with a trained artifact.

Prediction is the point at which a model's assumptions meet a real vector, and it is the
point at which most of a modelling system's avoidable damage happens. Three decisions in
this module exist to prevent specific failures, and each is worth stating plainly.

**The artifact's schema governs, not the incoming vector.** A caller may hand over a vector
with the right features in a different order, or with a newer feature set, or with a
feature the model never saw. Scoring that vector would produce a number, and the number
would be wrong in a way nothing downstream could detect. This module rebuilds the row from
the artifact's recorded requirements and refuses a vector that does not satisfy them, so
that "the model accepted this input" means the input was the one it was trained for.

**An absent feature is not a zero.** A missing measurement carries no information about
the learner's future performance, and imputing one produces a confident prediction from
nothing. Vectors with unavailable required features are refused and the caller is told
which features were missing. There is no partial-prediction fallback, because a prediction
computed from a subset of a model's inputs is not that model's output and should not be
presented as one.

**A probability is not a decision.** This module returns the probability and the threshold
used to apply it, and does not select an intervention. ``threshold`` is a parameter with a
default, not a hidden one, so that a caller who disagrees with it says so. A model that
chose its own operating point would be making a policy decision on the grounds that it had
a probability available, which is not a reason.

**Nothing here is a claim about the learner.** A prediction describes what a model produced
from a vector, and carries the model identity and feature-set version so that it can be
traced later. Whether that output should alter a learner's experience is a decision for a
different layer, and this module has no standing to make it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from focus_engine.features.models import FeatureValue
from focus_engine.models.artifacts import ModelArtifact, class_one_probabilities
from focus_engine.models.encoding import DesignMatrix, encode_rows, requirements_from_schema
from focus_engine.schemas.versioning import FeatureSetVersion, ModelVersion

__all__ = [
    "Prediction",
    "PredictionRefusal",
    "predict_with_artifact",
    "score_batch",
]

#: The probability at or above which a prediction is reported as positive. It is a
#: parameter rather than a constant so that a caller who has a reason to disagree with it
#: has to state one.
DEFAULT_THRESHOLD: float = 0.5


@dataclass(frozen=True, slots=True)
class Prediction:
    """One model output, with the identity needed to trace it later.

    A bare probability is not a usable output: six months later, nobody can tell which
    model produced it, against which feature-set version, or under which threshold. Every
    field here exists to make the number answerable.

    Attributes:
        probability: The predicted probability of the positive class, in ``[0, 1]``.
        is_positive: Whether the probability met the threshold.
        threshold: The threshold that produced ``is_positive``.
        model_id: Identity of the scoring model.
        model_version: Identity of the specific artefact.
        feature_set_version: The feature definitions the model was trained against.
        computed_at: When the score was produced.
        encoding_digest: The schema digest the input was verified against.
    """

    probability: float
    is_positive: bool
    threshold: float
    model_id: str
    model_version: ModelVersion
    feature_set_version: FeatureSetVersion
    computed_at: datetime
    encoding_digest: str


@dataclass(frozen=True, slots=True)
class PredictionRefusal:
    """A vector the model would not score, and the reason it would not.

    Refusal is a first-class outcome rather than an exception, because a production
    pipeline encountering an unusable vector needs to record and count it, not crash. An
    exception that halts a batch is a fine way to hide a systematic encoding problem.

    Attributes:
        learner_id: The vector's learner identifier, when it has one.
        session_id: The vector's session identifier, when it has one.
        computed_at: The vector's reference time.
        reason: Why the vector was refused.
        missing_features: The required features that were absent, when the refusal is due
            to missing features.
    """

    learner_id: str | None
    session_id: str | None
    computed_at: datetime
    reason: str
    missing_features: tuple[str, ...] = ()


def _score_matrix(artifact: ModelArtifact, matrix: DesignMatrix) -> NDArray[np.float64]:
    """Score an already-encoded matrix with the artifact's estimator.

    Args:
        artifact: The artifact holding the fitted estimator.
        matrix: The encoded rows, in the artifact's column order.

    Returns:
        One-dimensional array of class-1 probabilities.

    Raises:
        EncodingError: If the matrix's columns are not the artifact's columns. A model
            handed columns in another order produces a number, and it is wrong.
        ValueError: If the estimator does not produce two-class probabilities.
    """
    from focus_engine.models.encoding import EncodingError

    if matrix.columns != artifact.columns:
        raise EncodingError(
            "the encoded columns do not match the columns the model was trained on "
            f"({len(matrix.columns)} supplied, {len(artifact.columns)} expected). A model "
            "handed the right values in the wrong order produces a confident, wrong answer."
        )
    return class_one_probabilities(artifact.estimator, matrix.matrix)


def predict_with_artifact(
    artifact: ModelArtifact,
    vector: FeatureValue,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Prediction | PredictionRefusal:
    """Score one feature vector, or explain why it cannot be scored.

    Args:
        artifact: The fitted model to score with.
        vector: The feature vector to score.
        threshold: The probability at or above which the prediction is positive.

    Returns:
        A :class:`Prediction`, or a :class:`PredictionRefusal` naming what was missing.

    Raises:
        ValueError: If the threshold is outside ``[0, 1]``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1]; got {threshold}")

    if not _schema_matches(artifact, vector):
        return PredictionRefusal(
            learner_id=vector.learner_id,
            session_id=vector.session_id,
            computed_at=vector.computed_at,
            reason=(
                f"the vector was computed with feature set {vector.feature_set_version} but "
                f"the model was trained on {artifact.record.feature_set_version}. The column "
                "names may match while the quantities behind them do not, so the vector is "
                "refused rather than scored."
            ),
        )

    requirements = requirements_from_schema(artifact.encoding_schema)
    matrix = encode_rows(((vector, 1),), requirements=requirements)

    if matrix.rejected:
        rejection = matrix.rejected[0]
        return PredictionRefusal(
            learner_id=rejection.learner_id,
            session_id=rejection.session_id,
            computed_at=rejection.computed_at,
            reason=rejection.reason,
            missing_features=tuple(name.value for name in rejection.missing_features),
        )

    probability = float(_score_matrix(artifact, matrix)[0])
    return Prediction(
        probability=probability,
        is_positive=probability >= threshold,
        threshold=threshold,
        model_id=artifact.record.model_id,
        model_version=artifact.record.model_version,
        feature_set_version=artifact.record.feature_set_version,
        # The vector's own reference time, which is what the two refusal branches above
        # already stamp from. Stamping a successful prediction from the wall clock
        # instead would make the same function date its refusals from the evidence and
        # its answers from whenever the scorer happened to run, so two runs over
        # identical events would disagree about when the prediction was made.
        computed_at=vector.computed_at,
        encoding_digest=artifact.encoding_digest,
    )


def score_batch(
    artifact: ModelArtifact,
    vectors: tuple[FeatureValue, ...],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[Prediction], list[PredictionRefusal]]:
    """Score a batch of vectors, separating predictions from refusals.

    A batch is scored in one call so that the encoding cost is paid once and so that
    refusals are visible as a count. A caller that wants an all-or-nothing result should
    assert that ``refusals`` is empty; a caller that wants to proceed anyway can, but only
    after having been shown what it is proceeding without.

    Args:
        artifact: The fitted model to score with.
        vectors: The feature vectors to score.
        threshold: The probability at or above which a prediction is positive.

    Returns:
        A pair of the predictions and the refusals, each in input order.

    Raises:
        ValueError: If the threshold is outside ``[0, 1]``, or if the vectors disagree on
            their feature-set version. A mixed batch has no single meaning, and scoring it
            anyway would hide the disagreement in a per-row refusal.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1]; got {threshold}")
    if not vectors:
        return [], []

    versions = {vector.feature_set_version for vector in vectors}
    if len(versions) > 1:
        raise ValueError(
            f"the batch mixes {len(versions)} feature-set versions ({sorted(versions)}). A "
            "batch spanning versions has no single schema, and scoring it would either "
            "reject most of it or, worse, accept some rows under a schema they were not "
            "produced under."
        )

    predictions: list[Prediction] = []
    refusals: list[PredictionRefusal] = []
    for vector in vectors:
        outcome = predict_with_artifact(artifact, vector, threshold=threshold)
        if isinstance(outcome, PredictionRefusal):
            refusals.append(outcome)
        else:
            predictions.append(outcome)
    return predictions, refusals


def _schema_matches(artifact: ModelArtifact, vector: FeatureValue) -> bool:
    """Report whether a vector's feature-set version matches the model's.

    Args:
        artifact: The artifact that would score the vector.
        vector: The candidate vector.

    Returns:
        ``True`` if the versions agree.
    """
    return vector.feature_set_version == artifact.record.feature_set_version
