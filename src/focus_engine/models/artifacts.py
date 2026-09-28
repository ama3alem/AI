"""Model artifacts: serialisation, reload, and schema enforcement.

An artifact is more than a fitted estimator. It is the estimator *plus* the encoding
schema it was trained against, the record that documents it, and the seed that produced it.
A model reloaded without its schema is a model that will silently reinterpret features on
a column order it was never trained to understand, and the result would be a valid
prediction with a completely wrong meaning. This module makes the schema load-bearing: a
model whose schema does not match the data offered to it is refused, not silently
reinterpreted.

**Serialisation uses joblib.** Scikit-learn estimators are complex nested objects with
closures, Cython extensions, and references to ``numpy`` arrays. ``joblib`` is the
serialisation layer scikit-learn's own documentation recommends, and it handles ``numpy``
arrays efficiently by default. The trade-off is a larger artefact and a slower save; the
benefit is that the payload is a scikit-learn object graph rather than a bare ``pickle``
stream.

**The schema is not optional.** :func:`save_artifact` writes a sidecar metadata file
alongside the estimator. :func:`load_artifact` reads it back, recomputes the schema digest,
and refuses to continue if the two disagree. A model whose schema has drifted from the
data it will receive is not ready for use - it is a bug that happens to still be silent -
and failing loudly at load time is cheaper than failing quietly in production.

**This module does not decide whether a model is safe to load.** It verifies integrity of
the bytes it was given. Whether an artefact corresponds to an approved model version is a
governance question that belongs to the registry and to a human, and this module has no
standing to answer it.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from focus_engine.schemas.versioning import ModelRecord

__all__ = [
    "ModelArtifact",
    "class_one_probabilities",
    "load_artifact",
    "save_artifact",
]

#: Suffix used for the metadata sidecar written beside each serialised estimator.
_METADATA_SUFFIX: str = ".meta.joblib"


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A fitted model together with everything needed to interpret it.

    ``estimator`` is the fitted scikit-learn model. ``record`` is the registry entry that
    documents the training run. ``encoding_schema`` and ``encoding_digest`` are the column
    names and types the model was trained to expect, so that a prediction call can verify
    the data it receives matches the data it learned from. ``columns`` is the explicit
    column order the estimator sees, which is not derivable from the schema alone once
    one-hot expansion has reordered or repeated a feature.

    The artifact deliberately does not carry training rows, because those contain training
    labels and learner identities that would travel with the artefact if held as fields.
    The schema is sufficient for a prediction call to verify incoming data and refuse a
    mismatch.

    Attributes:
        estimator: The fitted scikit-learn estimator.
        record: The registry entry documenting the training run.
        encoding_schema: A JSON-serialisable description of the training columns.
        encoding_digest: The content digest of ``encoding_schema``.
        columns: The explicit column order the estimator expects.
        algorithm_name: The algorithm's recorded name, as stored in ``record.algorithm``.
        seed: The seed the model was trained with.
        trained_at: When training completed.
        library_versions: Versions of the packages that can move a numeric result, captured
            at training time. A reload under different versions is not guaranteed to
            reproduce the original predictions, and without these the difference would be
            invisible.
        feature_reference: One neutral value per column, summarising the training
            distribution. This is what makes an honest per-row explanation possible
            downstream without the artifact carrying any training row: the uncertainty
            engine substitutes these values to measure how much each signal moved a
            particular prediction. Empty for an artifact built before this field existed,
            in which case the uncertainty engine reports that no explanation is available
            rather than attributing the prediction to nothing in particular.
    """

    estimator: Any
    record: ModelRecord
    encoding_schema: dict[str, Any]
    encoding_digest: str
    columns: tuple[str, ...] = field(default_factory=tuple)
    algorithm_name: str = ""
    seed: int = 0
    trained_at: datetime.datetime | None = None
    library_versions: dict[str, str] = field(default_factory=dict)
    feature_reference: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the estimator can predict and the schema is present.

        Raises:
            ValueError: If the estimator has no ``predict`` method. A ``ModelArtifact``
                carrying an unfitted model is a claim recorded before it was tested.
            EncodingError: If the encoding schema is missing. A model without a schema
                cannot verify the data it receives.
        """
        from focus_engine.models.encoding import EncodingError

        if not hasattr(self.estimator, "predict"):
            raise ValueError(
                "the estimator must have a predict() method. A ModelArtifact carrying an "
                "unfitted model is a claim that has been recorded before it was tested."
            )
        if not self.encoding_schema:
            raise EncodingError(
                "the encoding schema is empty; a model without a schema cannot verify the "
                "data it receives and will silently reinterpret features on a column order "
                "it was never trained to understand"
            )

    def version_drift(self) -> dict[str, tuple[str, str]]:
        """Report packages whose version now differs from the one recorded at save time.

        This layer's reproducibility claim is predictive equivalence under a fixed
        environment, not byte identity of a pickle. Recording the environment is what
        makes the difference between those two situations visible: a reloaded model whose
        scikit-learn version has moved is not guaranteed to reproduce its original
        predictions, and the drift is otherwise indistinguishable from a genuine
        behavioural change.

        The result is informational. Refusing to load would be wrong, because a patch-level
        upgrade frequently does not move a fit at all, and a loader that refused on any
        version difference would be unusable in practice. A caller that needs a guarantee
        can act on an empty result; a caller that does not can still see that one exists.

        Returns:
            A mapping of package name to ``(recorded, current)`` for each package that
            differs. Empty when nothing drifted or when the artifact recorded no versions.
        """
        from focus_engine.utils.determinism import library_versions

        if not self.library_versions:
            return {}
        current = library_versions()
        return {
            name: (recorded, current.get(name, "absent"))
            for name, recorded in self.library_versions.items()
            if current.get(name) != recorded
        }


def save_artifact(artifact: ModelArtifact, path: Path) -> Path:
    """Serialise a model artifact to disk.

    The estimator is written to ``path``. The record, schema, digest, column order, seed,
    and training timestamp are written to a sidecar whose name is derived from ``path``.
    Keeping the two files separate means the provenance can be inspected, and its digest
    checked, without deserialising the model at all.

    Args:
        artifact: The artifact to serialise.
        path: Destination path for the estimator.

    Returns:
        The path to the written estimator file.

    Raises:
        OSError: If the write fails.
    """
    from focus_engine.schemas.primitives import utc_now

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = _metadata_path(path)

    from focus_engine.utils.determinism import library_versions

    joblib.dump(artifact.estimator, path)
    joblib.dump(
        {
            "record": artifact.record.model_dump(mode="json"),
            "encoding_schema": artifact.encoding_schema,
            "encoding_digest": artifact.encoding_digest,
            "columns": list(artifact.columns),
            "algorithm_name": artifact.algorithm_name,
            "seed": artifact.seed,
            "trained_at": (
                artifact.trained_at.isoformat() if artifact.trained_at else utc_now().isoformat()
            ),
            "metadata_written_at": utc_now().isoformat(),
            # Captured at save time rather than at fit time, and only when the artifact
            # does not already carry them: a model that was trained in one environment and
            # re-saved in another must report the environment it was *fitted* in, since
            # that is the one its parameters are a consequence of.
            "library_versions": artifact.library_versions or library_versions(),
            # Likewise captured once at fit time and preserved across re-saves: the
            # counterfactual an explanation uses must be the distribution the model
            # learned, not the distribution of whatever run happened to save it last.
            "feature_reference": dict(artifact.feature_reference),
        },
        metadata_path,
    )
    return path


def load_artifact(path: Path) -> ModelArtifact:
    """Load a model artifact and verify its schema.

    The stored schema's digest is recomputed and compared against the digest recorded
    beside it. A disagreement means the sidecar and the estimator no longer describe the
    same training run, and the artefact is refused rather than used, because a model
    whose provenance cannot be verified is a model whose behaviour cannot be explained.

    Args:
        path: Path to the serialised estimator.

    Returns:
        The loaded artifact, with its schema verified.

    Raises:
        FileNotFoundError: If the estimator or its sidecar is missing.
        EncodingError: If the sidecar is unreadable or its digest disagrees with the
            stored schema.
        ValueError: If the stored object is not a usable estimator.
    """
    from focus_engine.models.encoding import EncodingError
    from focus_engine.utils.determinism import content_digest

    metadata_path = _metadata_path(path)

    if not path.exists():
        raise FileNotFoundError(f"model file does not exist: {path}")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"model metadata file does not exist: {metadata_path}. A model loaded without "
            "its schema can silently reinterpret features on a column order it was never "
            "trained to understand."
        )

    estimator = joblib.load(path)
    metadata = joblib.load(metadata_path)

    if not hasattr(estimator, "predict"):
        raise ValueError("the loaded object is not a valid estimator: it has no predict() method")
    if not isinstance(metadata, dict) or "encoding_schema" not in metadata:
        raise EncodingError(
            f"the metadata sidecar at {metadata_path} is not a recognised artefact "
            "metadata file. Refusing to load a model whose provenance is unreadable."
        )

    stored_digest = str(metadata["encoding_digest"])
    recomputed_digest = content_digest(metadata["encoding_schema"])
    if stored_digest != recomputed_digest:
        raise EncodingError(
            f"the stored encoding schema digest ({stored_digest[:16]}) does not match the "
            f"schema itself ({recomputed_digest[:16]}). The model was trained on one "
            "encoding and reloaded with another, and would silently reinterpret features "
            "on a column order it never learned."
        )

    trained_at: datetime.datetime | None = None
    if metadata.get("trained_at") is not None:
        trained_at = datetime.datetime.fromisoformat(str(metadata["trained_at"]))

    return ModelArtifact(
        estimator=estimator,
        record=ModelRecord.model_validate(metadata["record"]),
        encoding_schema=metadata["encoding_schema"],
        encoding_digest=stored_digest,
        columns=tuple(str(column) for column in metadata.get("columns", ())),
        algorithm_name=str(metadata.get("algorithm_name", "")),
        seed=int(metadata.get("seed", 0)),
        trained_at=trained_at,
        library_versions={
            str(name): str(value)
            for name, value in dict(metadata.get("library_versions") or {}).items()
        },
        feature_reference={
            str(name): float(value)
            for name, value in dict(metadata.get("feature_reference") or {}).items()
        },
    )


def class_one_probabilities(estimator: Any, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract class-1 probabilities from a fitted estimator.

    A ``Pipeline`` wrapping ``LogisticRegression`` exposes ``predict_proba`` at the
    pipeline level, and so do a bare forest and a bare booster, so the positive-class
    column is read in both cases rather than assuming a layout. The check on the shape is
    not defensive padding: a single-column output cannot be interpreted as a class-1
    score, and choosing which column is positive without evidence would invert half the
    predictions in a way that every downstream metric would happily report.

    This lives here, next to the artifact, because it is a property of the estimator
    rather than of either the training or the scoring path, and both need the same answer.

    Args:
        estimator: A fitted scikit-learn estimator.
        matrix: The feature matrix to score.

    Returns:
        A one-dimensional array of class-1 probabilities.

    Raises:
        ValueError: If the estimator does not produce two-class probabilities.
    """
    raw = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(
            f"expected two-class probabilities, got an array of shape {raw.shape}. A "
            "single-class model cannot produce a class-1 score; the label definition or the "
            "training data is wrong, and guessing which class is positive would be worse "
            "than refusing."
        )
    return raw[:, 1]


def _metadata_path(path: Path) -> Path:
    """Derive the sidecar metadata path for an estimator path.

    Args:
        path: The estimator's path.

    Returns:
        The path of the metadata sidecar.
    """
    return path.with_name(f"{path.stem}{_METADATA_SUFFIX}")
