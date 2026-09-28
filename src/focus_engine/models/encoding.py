"""Feature vectors to numeric design matrices, without inventing a value for an absent one.

A scikit-learn classifier consumes a dense numeric matrix. The feature engine produces
something quite different: fourteen heterogeneous entries, each of which may be a float, an
int, or a string, and each of which may be *absent* - reported as
:attr:`~focus_engine.features.models.FeatureAvailability.INSUFFICIENT_DATA` with a reason.
The gap between those two shapes is where a great deal of dishonesty enters behavioural
modelling, and this module exists to make the conversion auditable rather than convenient.

**An absent feature is not zero.** The Phase 5 principle is that a missing measurement must
never be replaced by a plausible constant, because the replacement is indistinguishable
downstream from a measurement. Substituting ``0.0`` for an absent accuracy would tell the
model that the learner answered everything wrongly, and substituting the column mean would
tell it the learner is exactly average. Both are fabrications wearing a number's clothes.
This module therefore does not impute. It **refuses**.

**A refused row is an honest outcome, not an error.** When a required feature is absent,
:func:`encode_rows` reports the row as unusable *with a reason*, and the caller decides what
to do. Training on a partially-measured row would teach the model a relationship between
missingness and the target that will not hold at serving time, where the pattern of absence
is different. Predicting from one would be worse. So a row with an absent required feature
is dropped from training and refused at prediction, and both outcomes are stated rather
than silently handled.

**String features are one-hot encoded against a declared, closed vocabulary.** Four of the
fourteen features are categorical. One-hot encoding is only safe when the category set is
known in advance and fixed, because a category that appears for the first time at serving
time has no column and would otherwise be silently dropped or folded into an "other" bucket
the model has never seen. The vocabularies here are therefore declared in code, and an
unrecognised category is refused with a reason. The alternative - an ``other`` bucket - is
exactly the substitution this project refuses elsewhere: it makes an unknown category
indistinguishable from a real one.

**The encoding is part of the artifact.** A model whose columns mean something is
uninterpretable without knowing what the columns were, and a reloaded model that re-derived
its columns from the *current* feature registry would silently change meaning if the
registry changed. :class:`DesignMatrix` therefore carries its own column names, dtypes, and
vocabularies, and an artifact is refused at load time if its encoding does not match the
data offered to it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from focus_engine.features.models import (
    FeatureAvailability,
    FeatureName,
    FeatureValue,
    FeatureValueType,
    TypedFeatureValue,
)
from focus_engine.schemas.primitives import Timestamp
from focus_engine.utils.determinism import content_digest

__all__ = [
    "CATEGORICAL_VOCABULARIES",
    "UNATTRIBUTED",
    "DesignMatrix",
    "EncodingError",
    "FeatureRequirement",
    "RejectedRow",
    "column_references",
    "default_features",
    "encode_rows",
    "requirements_from_schema",
]


class EncodingError(Exception):
    """Raised when a feature vector cannot be encoded as specified."""


#: Stands in for a missing identifier in a rejection record, so that a rejected row is
#: still reportable. It is deliberately not a valid pseudonymous identifier: it cannot be
#: mistaken for a real learner, and any attempt to use it as one is refused by the
#: identifier grammar.
UNATTRIBUTED: Final[str] = "unattributed"


#: Closed category sets for every categorical feature, declared in code rather than
#: learned from the data.
#:
#: A vocabulary learned from a training set is a vocabulary that can grow at serving time,
#: and a new category would then either have no column or be folded into a catch-all the
#: model was never trained against. Declaring the sets here means an unrecognised value is
#: a *refusal* - a visible defect - rather than a silent reinterpretation. The sets are
#: derived from the closed enums the feature engine already encodes into, so they cannot
#: drift from the definitions they mirror.
CATEGORICAL_VOCABULARIES: Final[dict[FeatureName, tuple[str, ...]]] = {
    FeatureName.CONTENT_TYPE_ENCODED: ("multiple_choice", "free_response", "video", "other"),
    FeatureName.TRAJECTORY_ENCODED: ("improving", "stable", "declining", "other"),
    FeatureName.BASELINE_MATURITY_ENCODED: ("new", "early", "developing", "established"),
    FeatureName.INTERVENTION_COOLDOWN_ACTIVE: ("true", "false"),
}

#: Features whose absence makes a row unusable. Declared explicitly so that a new feature
#: added to the registry is not silently treated as required or optional.
_REQUIRED_FEATURES: Final[frozenset[FeatureName]] = frozenset(
    name for name in FeatureName if name not in CATEGORICAL_VOCABULARIES
)


def default_features() -> tuple[FeatureName, ...]:
    """Return the features a design matrix is built from when none are specified.

    This is every feature the engine knows, not just the numeric ones, and the name says
    so. A function called ``required_numeric_features`` that returned the categorical
    features as well would be read as a contract it did not honour, and a caller relying on
    that reading would build a model expecting no one-hot columns.

    Returns:
        All feature names, in :class:`~focus_engine.features.models.FeatureName` declaration
        order, so that a matrix's column order is a property of the feature set rather than
        of dictionary iteration order.
    """
    return tuple(FeatureName)


@dataclass(frozen=True, slots=True)
class FeatureRequirement:
    """What a single model column needs from a feature vector.

    Args:
        name: The feature this column comes from.
        value_type: The declared type of the value.
        vocabulary: Closed category set, for categorical features only.
    """

    name: FeatureName
    value_type: FeatureValueType
    vocabulary: tuple[str, ...] | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        """The design-matrix column names this requirement produces.

        A numeric feature occupies one column named for itself. A categorical feature
        occupies one column per declared category, so that a model's coefficient for
        "free_response" is attributable to a real category rather than to a contrast
        against an implicit baseline the matrix does not contain.

        Returns:
            The column names, in a stable order.
        """
        if self.vocabulary is None:
            return (self.name.value,)
        return tuple(f"{self.name.value}={category}" for category in self.vocabulary)


@dataclass(frozen=True, slots=True)
class RejectedRow:
    """One feature vector that could not be encoded, and why.

    A rejected row is a data-quality fact, not a discarded input. Recording the reason
    matters: a dataset where four percent of rows are unusable for one missing feature is a
    different dataset from one where four percent are unusable for three, and a training
    run that reported only the accepted count would hide which.
    """

    learner_id: str
    session_id: str
    computed_at: Timestamp
    reason: str
    missing_features: tuple[FeatureName, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        """Render the rejection as a single line.

        Returns:
            The learner, the reason, and the absent features.
        """
        absent = ",".join(name.value for name in self.missing_features) or "none"
        return f"{self.learner_id} at {self.computed_at}: {self.reason} [{absent}]"


@dataclass(frozen=True, slots=True)
class DesignMatrix:
    """An encoded numeric matrix together with the schema that gives it meaning.

    A bare ``numpy`` array is not an interpretable artifact. This carries the column names
    in order, the requirement that produced each column group, and a digest of the whole
    schema, so that a reloaded model can refuse data it was not trained to interpret.
    """

    matrix: NDArray[np.float64]
    labels: NDArray[np.int64]
    columns: tuple[str, ...]
    requirements: tuple[FeatureRequirement, ...]
    learner_ids: tuple[str, ...] = field(default_factory=tuple)
    rejected: tuple[RejectedRow, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate that the matrix agrees with its declared schema.

        Raises:
            EncodingError: If the column count, row count, or label length disagrees with
                the declared columns. A matrix whose shape contradicts its schema would
                silently misalign every column with its name.
        """
        if self.matrix.ndim != 2:
            raise EncodingError(f"design matrix must be two-dimensional; got {self.matrix.ndim}")
        if self.matrix.shape[1] != len(self.columns):
            raise EncodingError(
                f"design matrix has {self.matrix.shape[1]} columns but the schema declares "
                f"{len(self.columns)}; every column must be named"
            )
        if self.matrix.shape[0] != self.labels.shape[0]:
            raise EncodingError(
                f"design matrix has {self.matrix.shape[0]} rows but carries "
                f"{self.labels.shape[0]} labels; every row must have exactly one label"
            )
        if self.learner_ids and len(self.learner_ids) != self.matrix.shape[0]:
            raise EncodingError(
                f"design matrix has {self.matrix.shape[0]} rows but carries "
                f"{len(self.learner_ids)} learner identifiers"
            )

    @property
    def n_rows(self) -> int:
        """Number of encoded rows."""
        return int(self.matrix.shape[0])

    @property
    def n_columns(self) -> int:
        """Number of encoded columns."""
        return int(self.matrix.shape[1])

    @property
    def n_positive(self) -> int:
        """Number of positive-class rows, so class balance is visible on the artifact."""
        return int((self.labels == 1).sum())

    def schema(self) -> dict[str, object]:
        """Return the schema as a JSON-serialisable payload.

        Returns:
            Column names, feature names, value types, and vocabularies.
        """
        return {
            "columns": list(self.columns),
            "requirements": [
                {
                    "name": requirement.name.value,
                    "value_type": requirement.value_type.value,
                    "vocabulary": (
                        list(requirement.vocabulary) if requirement.vocabulary is not None else None
                    ),
                }
                for requirement in self.requirements
            ],
        }

    def schema_digest(self) -> str:
        """Return a stable digest over the schema.

        Two matrices with the same digest interpret their columns identically, which is
        the precondition for a reloaded model to accept one of them.

        Returns:
            A hex digest.
        """
        return content_digest(self.schema())


def column_references(matrix: DesignMatrix) -> dict[str, float]:
    """Reduce a design matrix to one neutral value per column.

    An honest per-row explanation needs a counterfactual: to say that a signal moved this
    prediction, the system must know what the prediction would have been without it. The
    counterfactual value has to come from the training distribution, because "what an
    average row looks like" is a property of the data the model learned, not of the model.

    The median is used rather than the mean. An explanation's counterfactual is asked to
    stand in for a *typical* row, and a mean is pulled toward whichever extreme happened to
    be common in the sample, which for a heavily skewed feature is not typical at all. The
    median is also defined for the one-hot columns, where the mean of an indicator
    degenerates to its own base rate and would replace "this is a quiz" with "quizzes
    sometimes happen".

    What this deliberately does not do is carry the rows. A single ``float`` per column
    summarises the column; it cannot reconstruct a learner, a label, or a session, so the
    privacy reason for keeping training data out of the artifact survives intact. The
    output is a distribution summary, not data about any learner.

    Args:
        matrix: The encoded training matrix.

    Returns:
        A mapping of column name to that column's median value. Every declared column is
        present.
    """
    if matrix.n_columns == 0:
        return {}
    return {
        column: float(np.median(matrix.matrix[:, index]))
        for index, column in enumerate(matrix.columns)
    }


def _is_measured(entry: TypedFeatureValue) -> bool:
    """Report whether a feature entry carries a usable value.

    Both ``AVAILABLE`` and ``SYNTHETIC_ONLY`` carry a real measurement. ``SYNTHETIC_ONLY``
    means the value was measured but its provenance is synthetic, which is a labelling
    property recorded elsewhere; the number itself is as usable as any other. Only
    ``INSUFFICIENT_DATA`` means there is no value, and only that is treated as absent here.

    Args:
        entry: A typed feature value.

    Returns:
        ``True`` when a value is present.
    """
    return entry.availability is not FeatureAvailability.INSUFFICIENT_DATA


def _numeric_value(entry: TypedFeatureValue, requirement: FeatureRequirement) -> float:
    """Coerce a measured numeric feature to a float.

    Args:
        entry: The feature entry.
        requirement: The requirement being satisfied, used for the error message.

    Returns:
        The value as a float.

    Raises:
        EncodingError: If the value is not numeric, or is not finite. An infinite or NaN
            feature would propagate into the model's coefficients as a NaN, which is a
            model that trains successfully and predicts nonsense.
        EncodingError: If the feature is declared as an integer and carries a fraction.
            Rounding it would change a count the learner actually produced, and the change
            would be invisible in every downstream metric.
    """
    value = entry.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EncodingError(
            f"feature {requirement.name.value!r} is declared "
            f"{requirement.value_type.value!r} but carries {value!r}"
        )
    number = float(value)
    if not np.isfinite(number):
        raise EncodingError(
            f"feature {requirement.name.value!r} carries a non-finite value {value!r}; a "
            "non-finite feature produces a model that trains and predicts nonsense"
        )
    if requirement.value_type is FeatureValueType.INT and not number.is_integer():
        raise EncodingError(
            f"feature {requirement.name.value!r} is declared an integer but carries {value!r}. "
            "A count that is not a whole number was not rounded, because rounding would "
            "silently change a quantity the learner actually produced."
        )
    return number


def _categorical_columns(entry: TypedFeatureValue, requirement: FeatureRequirement) -> list[float]:
    """One-hot encode a categorical feature against its declared vocabulary.

    Args:
        entry: The feature entry.
        requirement: The requirement being satisfied.

    Returns:
        One indicator per declared category.

    Raises:
        EncodingError: If the value is not in the declared vocabulary. A category outside
            the declared set has no column, and folding it into a catch-all would make it
            indistinguishable from a category the model was actually trained on.
    """
    value = entry.value
    if not isinstance(value, str):
        raise EncodingError(
            f"categorical feature {requirement.name.value!r} carries {value!r}, which is "
            "not a string"
        )
    if value not in (requirement.vocabulary or ()):
        raise EncodingError(
            f"categorical feature {requirement.name.value!r} carries {value!r}, which is "
            f"outside its declared vocabulary {list(requirement.vocabulary or ())}. An "
            "unrecognised category is refused rather than folded into a catch-all, because "
            "the catch-all would be indistinguishable from a real category."
        )
    return [1.0 if category == value else 0.0 for category in requirement.vocabulary or ()]


def _requirements() -> tuple[FeatureRequirement, ...]:
    """Build the requirement list for the current feature set.

    Returns:
        One requirement per feature, in :class:`~focus_engine.features.models.FeatureName`
        declaration order.
    """
    return tuple(
        FeatureRequirement(
            name=name,
            value_type=(
                FeatureValueType.STR
                if name in CATEGORICAL_VOCABULARIES
                else (
                    FeatureValueType.INT
                    if name
                    in (
                        FeatureName.SESSION_EVENT_COUNT,
                        FeatureName.INTERVENTION_COUNT,
                        FeatureName.INTERVENTION_SECONDS_SINCE_LAST,
                    )
                    else FeatureValueType.FLOAT
                )
            ),
            vocabulary=CATEGORICAL_VOCABULARIES.get(name),
        )
        for name in FeatureName
    )


def requirements_from_schema(schema: dict[str, Any]) -> tuple[FeatureRequirement, ...]:
    """Rebuild the requirement list from a stored schema payload.

    An artifact stores its schema as plain JSON-serialisable data, because that is what
    survives a save and a reload. This function is the inverse of
    :meth:`DesignMatrix.schema`, and it is what lets a reloaded model encode new data
    against the requirements it was actually trained on rather than against whatever the
    feature set looks like now.

    The round trip is exact or it is refused. A schema naming an unknown feature, or
    carrying a vocabulary that no longer matches the declared one, means the artifact and
    the code have diverged; scoring against either would be a guess.

    Args:
        schema: A payload produced by :meth:`DesignMatrix.schema`.

    Returns:
        The requirements, in schema order.

    Raises:
        EncodingError: If the payload is malformed, names an unknown feature, or declares a
            vocabulary that differs from the one the code holds.
    """
    raw_requirements = schema.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise EncodingError(
            "the stored schema has no requirements. A schema without requirements cannot "
            "describe how a column was produced, and a model that cannot describe its own "
            "inputs should not be asked to score them."
        )

    requirements: list[FeatureRequirement] = []
    for entry in raw_requirements:
        if not isinstance(entry, dict) or "name" not in entry:
            raise EncodingError(f"malformed requirement in the stored schema: {entry!r}")
        try:
            name = FeatureName(entry["name"])
        except ValueError as error:
            raise EncodingError(
                f"the stored schema names {entry['name']!r}, which is not a known feature. "
                "The artifact and the code have diverged, and encoding against either would "
                "be a guess."
            ) from error

        declared = CATEGORICAL_VOCABULARIES.get(name)
        stored_vocabulary = entry.get("vocabulary")
        if stored_vocabulary is not None:
            stored = tuple(str(category) for category in stored_vocabulary)
            if declared is not None and stored != declared:
                raise EncodingError(
                    f"the stored vocabulary for {name.value!r} is {list(stored)} but the code "
                    f"declares {list(declared)}. One-hot columns are positional, so a changed "
                    "vocabulary silently reinterprets every category column; refusing is the "
                    "only safe response."
                )
            vocabulary: tuple[str, ...] | None = stored
        else:
            vocabulary = declared

        requirements.append(
            FeatureRequirement(
                name=name,
                value_type=FeatureValueType(entry.get("value_type", "float")),
                vocabulary=vocabulary,
            )
        )
    return tuple(requirements)


def encode_rows(
    rows: Iterable[tuple[FeatureValue, int]],
    *,
    requirements: Sequence[FeatureRequirement] | None = None,
) -> DesignMatrix:
    """Encode labelled feature vectors into a numeric design matrix.

    Rows whose required features are absent are rejected and reported, never imputed. A
    caller that needs a rejection to be fatal can assert on
    :attr:`DesignMatrix.rejected`; a caller that can tolerate a smaller dataset can record
    why it is smaller. Neither is forced, but neither is invisible.

    Args:
        rows: Labelled feature vectors. The label is ``1`` for the positive class and ``0``
            otherwise, and is supplied by the caller because labelling is a claim about the
            future that this layer does not make.
        requirements: The feature requirements to encode against. Defaults to the full
            feature set.

    Returns:
        The encoded matrix, with any rejected rows described.

    Raises:
        EncodingError: If a row is structurally unusable in a way that is a producer defect
            rather than an absent measurement - a missing feature entry, a non-finite
            value, or an unrecognised category.
    """
    resolved = tuple(requirements) if requirements is not None else _requirements()
    columns: list[str] = []
    for requirement in resolved:
        columns.extend(requirement.columns)

    accepted_rows: list[list[float]] = []
    accepted_labels: list[int] = []
    accepted_learners: list[str] = []
    rejected: list[RejectedRow] = []

    for vector, label in rows:
        entries = {entry.name: entry for entry in vector.values}
        absent = tuple(
            requirement.name
            for requirement in resolved
            if requirement.name not in entries or not _is_measured(entries[requirement.name])
        )
        if absent:
            details = "; ".join(
                f"{name.value} ({entries[name].reason})"
                if name in entries and entries[name].reason
                else f"{name.value} (absent from the vector)"
                for name in absent
            )
            rejected.append(
                RejectedRow(
                    learner_id=vector.learner_id or UNATTRIBUTED,
                    session_id=vector.session_id or UNATTRIBUTED,
                    computed_at=vector.computed_at,
                    reason=f"required feature(s) unavailable: {details}",
                    missing_features=absent,
                )
            )
            continue
        if vector.learner_id is None:
            # An unattributed vector is a genuine measurement but not one that can be
            # sliced per learner, held out per learner, or traced back when an evaluation
            # result is questioned. Accepting it would make the model's training set
            # partly unaccountable, so it is rejected and reported rather than absorbed.
            rejected.append(
                RejectedRow(
                    learner_id=UNATTRIBUTED,
                    session_id=vector.session_id or UNATTRIBUTED,
                    computed_at=vector.computed_at,
                    reason="the vector carries no learner_id, so it cannot be attributed "
                    "for per-learner evaluation or leakage checks",
                )
            )
            continue

        row: list[float] = []
        for requirement in resolved:
            entry = entries[requirement.name]
            if requirement.vocabulary is None:
                row.append(_numeric_value(entry, requirement))
            else:
                row.extend(_categorical_columns(entry, requirement))
        accepted_rows.append(row)
        accepted_labels.append(int(label))
        accepted_learners.append(vector.learner_id)

    return DesignMatrix(
        matrix=np.asarray(accepted_rows, dtype=np.float64).reshape(
            len(accepted_rows), len(columns)
        ),
        labels=np.asarray(accepted_labels, dtype=np.int64),
        columns=tuple(columns),
        requirements=resolved,
        learner_ids=tuple(accepted_learners),
        rejected=tuple(rejected),
    )


def describe_encoding(matrix: DesignMatrix) -> Mapping[str, str]:
    """Return a human-readable map of column name to how it was derived.

    Args:
        matrix: The encoded matrix.

    Returns:
        A mapping of column name to a short description.
    """
    described: dict[str, str] = {}
    for requirement in matrix.requirements:
        if requirement.vocabulary is None:
            described[requirement.name.value] = f"numeric ({requirement.value_type.value})"
        else:
            for category in requirement.vocabulary:
                described[f"{requirement.name.value}={category}"] = f"one-hot ({category})"
    return described
