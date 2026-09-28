"""Per-row explanations that report only what actually moved the prediction.

An explanation that lists a model's globally important features is not an explanation of
any particular prediction. It is a description of the model, dressed as a description of
this learner, and it is the single most common way a prediction system produces confident
narratives about people. This module refuses that substitution: a signal appears in a
:func:`PredictionExplanation` only if replacing *this* vector's value for it measurably
moved *this* prediction.

**The method is occlusion against the training distribution.** For each column, the
vector's own value is replaced with a neutral reference value taken from the training data
and the model is rescored. The change in the emitted probability is that column's
contribution. A column that does not change the answer did not contribute, and is omitted
rather than listed with a small number next to it.

Two properties of this choice are worth stating, because alternatives are tempting.

*Why not global feature importance.* ``feature_importances_`` and coefficients describe a
model averaged over all its rows. They are not a function of any individual input, they are
identical for every learner the model ever scores, and using them would produce the
unacceptable outcome of explaining a prediction by citing the model's overall behaviour.
*Why not a Shapley value.* Shapley attribution is better founded and would be a
reasonable upgrade, but its cost grows factorially in the number of features, and this
project has a portability and runtime budget that does not accommodate that. Occlusion is
O(columns) per row, is deterministic, needs no background sample, and is a recognised
method with a known weakness: it attributes interactions to whichever member is occluded
first, so a row whose signal is genuinely shared between two correlated features will
show one of them. That limitation is the reason :class:`SignalContribution` describes a
*local attribution* and never a cause.

**Grouping is at the feature level, not the column level.** A categorical feature is
one-hot expanded into one column per category, and those columns are not independent
signals - "the content was a quiz" is one fact, not one fact per quiz column. Grouping by
feature and reporting the most influential member keeps the explanation in terms a reader
can act on, and the group is signed by its strongest member so the direction is not
averaged into meaninglessness.

**An empty explanation is a real result, not a failure.** It happens when the model
outputs the same probability whatever the input says, and it is the most important thing
this module can report: a number that no feature moved is not a prediction about anybody.
It is surfaced rather than papered over with the model's top three global importances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from focus_engine.features.models import FeatureValue
from focus_engine.models.artifacts import ModelArtifact, class_one_probabilities
from focus_engine.models.encoding import encode_rows, requirements_from_schema

__all__ = [
    "PredictionExplanation",
    "SignalContribution",
    "occlusion_contributions",
]

#: Default smallest probability change that counts as a contribution. A tenth of a
#: percentage point is small enough that a real signal is never discarded for being
#: merely modest, and large enough that the residual wobble of a tree ensemble on one
#: row is not presented to a reader as a finding.
DEFAULT_MIN_CONTRIBUTION: float = 1e-3


def _base_feature_of(column: str) -> str:
    """Return the feature a column came from.

    Categorical features expand to ``feature=category`` columns and numeric ones occupy a
    single column named for themselves, so the first ``=`` separates the two cases. The
    feature names in the registry contain no ``=``, which makes the split unambiguous.

    Args:
        column: The encoded column name.

    Returns:
        The base feature name the column belongs to.
    """
    head, separator, _ = column.partition("=")
    return head if separator else column


@dataclass(frozen=True, slots=True)
class SignalContribution:
    """How much one feature moved one prediction.

    Attributes:
        feature: The feature whose value moved the prediction.
        contribution: Signed change in the emitted probability when the feature was
            replaced by its neutral reference value. Positive means the feature's presence
            pushed the probability towards the positive class, negative means it pulled
            away from it.
        column: The encoded column that carried the contribution. Recorded because a
            one-hot feature's direction is meaningless without knowing which of its
            columns moved.
    """

    feature: str
    contribution: float
    column: str

    @property
    def magnitude(self) -> float:
        """The unsigned size of the contribution, for ranking and thresholding."""
        return abs(self.contribution)

    @property
    def direction(self) -> str:
        """Which way the signal pushed the prediction.

        Returns:
            ``"towards_positive"`` or ``"towards_negative"``.
        """
        return "towards_positive" if self.contribution > 0 else "towards_negative"

    def describe(self) -> str:
        """Render the contribution as a short human-readable line.

        Returns:
            A one-line description naming the feature, its column, and its effect.
        """
        return (
            f"{self.feature} ({self.column}) moved the probability "
            f"{self.contribution:+.4f} {self.direction}"
        )


@dataclass(frozen=True, slots=True)
class PredictionExplanation:
    """The signals that moved one prediction, and the ones that did not.

    Both halves are recorded. The omitted features are as informative as the included
    ones: they are the reasons a reader should not expect some seemingly relevant signal
    to have been considered, and a system that reported only its contributors would leave
    the impression that it had looked at everything.

    Attributes:
        base_probability: The probability the contributions are measured against.
        min_contribution: The threshold below which a movement was not reported.
        contributions: Reported signals, largest first.
        omitted: Features whose movement was below the threshold.
        available: Whether an explanation could be produced at all. ``False`` when the
            artifact carries no reference values to occlude against, which is the honest
            result for a model saved before reference values were recorded - and is
            deliberately not papered over with global importances.
        reason: Why no explanation is available, when ``available`` is ``False``.
    """

    base_probability: float
    min_contribution: float = DEFAULT_MIN_CONTRIBUTION
    contributions: tuple[SignalContribution, ...] = field(default_factory=tuple)
    omitted: tuple[str, ...] = field(default_factory=tuple)
    available: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the explanation's internal consistency.

        Raises:
            ValueError: If the tolerance is negative, if a reported contribution is below
                it, if the contributions are not ordered by magnitude, or if an
                unavailable explanation carries contributions.
        """
        if self.min_contribution < 0.0:
            raise ValueError(f"min_contribution must be non-negative; got {self.min_contribution}")
        if not self.available and self.contributions:
            raise ValueError(
                "an explanation marked unavailable cannot carry contributions; reporting "
                f"{len(self.contributions)} of them would contradict the availability flag"
            )
        if not self.available and self.reason is None:
            raise ValueError("an unavailable explanation must say why it is unavailable")
        if self.available and self.reason is not None:
            raise ValueError(
                "an available explanation cannot carry an unavailability reason; got "
                f"reason={self.reason!r}"
            )
        magnitudes = [contribution.magnitude for contribution in self.contributions]
        if any(magnitude < self.min_contribution for magnitude in magnitudes):
            raise ValueError(
                f"a reported contribution is below the {self.min_contribution} threshold it "
                "claims to have been filtered against; the filter did not run"
            )
        if magnitudes != sorted(magnitudes, reverse=True):
            raise ValueError(
                f"contributions must be ordered by magnitude, largest first; got {magnitudes}"
            )

    @property
    def is_empty(self) -> bool:
        """Whether the prediction was moved by no feature above the threshold.

        This is a reportable finding, not a gap. It means the model returned the same
        probability regardless of what the vector said.
        """
        return self.available and not self.contributions

    def top(self, count: int = 3) -> tuple[SignalContribution, ...]:
        """Return the largest contributions.

        Args:
            count: Maximum number to return.

        Returns:
            Up to ``count`` contributions, largest first.

        Raises:
            ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative; got {count}")
        return self.contributions[:count]

    def describe(self) -> str:
        """Render the explanation as a short human-readable block.

        Returns:
            A multi-line summary naming the base probability, the reported signals, and
            the omitted ones.
        """
        if not self.available:
            return f"no explanation available: {self.reason}"
        if not self.contributions:
            return (
                f"no feature moved the probability {self.base_probability:.4f} by at least "
                f"{self.min_contribution}; the model returned the same answer regardless of "
                "this vector's values"
            )
        lines = [
            f"probability {self.base_probability:.4f}, attributed to "
            f"{len(self.contributions)} signal(s):"
        ]
        lines.extend(f"  {contribution.describe()}" for contribution in self.contributions)
        if self.omitted:
            lines.append(f"  no measurable effect: {', '.join(self.omitted)}")
        return "\n".join(lines)


def occlusion_contributions(
    artifact: ModelArtifact,
    vector: FeatureValue,
    *,
    base_probability: float,
    min_contribution: float = DEFAULT_MIN_CONTRIBUTION,
) -> PredictionExplanation:
    """Measure which signals moved one prediction, by occluding each against training.

    Args:
        artifact: The fitted artifact, which must carry reference values.
        vector: The feature vector that was scored.
        base_probability: The probability the prediction reported. Passed in rather than
            recomputed so that there is exactly one base score in the system, and an
            explanation can never be measured against a different number from the one
            that was reported.
        min_contribution: Smallest probability movement to report as a contribution.

    Returns:
        A :class:`PredictionExplanation`. When the artifact carries no reference values the
        explanation is marked unavailable, with a reason, rather than being approximated
        from global importances.

    Raises:
        ValueError: If the vector's feature-set version does not match the artifact's, or
            if the artifact's reference values do not cover its columns. Either means the
            artifact cannot describe its own inputs, and an explanation built on it would
            be unmeasured.
    """
    if min_contribution < 0.0:
        raise ValueError(f"min_contribution must be non-negative; got {min_contribution}")

    if not artifact.feature_reference:
        return PredictionExplanation(
            base_probability=base_probability,
            min_contribution=min_contribution,
            available=False,
            reason=(
                f"artifact {artifact.record.model_id} version "
                f"{artifact.record.model_version} carries no per-column reference values, so "
                "there is no neutral value to occlude against and no way to measure which "
                "signal moved this prediction. The model's global feature importances are "
                "not a substitute: they describe the model across all its rows, not this "
                "one, and reporting them here would present an average as an explanation "
                "of an individual."
            ),
        )

    if vector.feature_set_version != artifact.record.feature_set_version:
        raise ValueError(
            f"the vector was computed with feature set {vector.feature_set_version} but the "
            f"artifact was trained on {artifact.record.feature_set_version}. An "
            "explanation cannot be produced across a schema boundary, because the "
            "attributions would be about features the model never saw."
        )

    missing = [column for column in artifact.columns if column not in artifact.feature_reference]
    if missing:
        raise ValueError(
            f"the artifact's reference values cover none of {len(missing)} of its columns, "
            f"beginning with {missing[0]!r}. An artifact must be able to describe its own "
            "input columns; a partial mapping would silently produce explanations that "
            "omitted exactly the features it could not measure."
        )

    requirements = requirements_from_schema(artifact.encoding_schema)
    matrix = encode_rows(((vector, 1),), requirements=requirements)
    if matrix.rejected:
        rejection = matrix.rejected[0]
        raise ValueError(
            "the vector cannot be explained because it could not be encoded: "
            f"{rejection.reason} A prediction is never produced from a partially encoded "
            "row, so neither should an explanation be."
        )

    row: NDArray[np.float64] = matrix.matrix[0]
    per_column: dict[str, float] = {}
    for index, column in enumerate(artifact.columns):
        occluded = row.copy()
        occluded[index] = artifact.feature_reference[column]
        moved = float(class_one_probabilities(artifact.estimator, occluded.reshape(1, -1))[0])
        per_column[column] = moved - base_probability

    # One feature may own several columns, and the contribution attributed to the feature
    # is that of its most influential column. Averaging them would be wrong for a one-hot
    # group: the columns are mutually exclusive indicators, so their movements partly
    # cancel by construction and the average describes no state of the world.
    strongest: dict[str, SignalContribution] = {}
    for column, contribution in per_column.items():
        base = _base_feature_of(column)
        candidate = SignalContribution(feature=base, contribution=contribution, column=column)
        incumbent = strongest.get(base)
        if incumbent is None or candidate.magnitude > incumbent.magnitude:
            strongest[base] = candidate

    ordered = sorted(strongest.values(), key=lambda item: item.magnitude, reverse=True)
    reported = tuple(item for item in ordered if item.magnitude >= min_contribution)
    omitted = tuple(item.feature for item in ordered if item.magnitude < min_contribution)

    return PredictionExplanation(
        base_probability=base_probability,
        min_contribution=min_contribution,
        contributions=reported,
        omitted=omitted,
        available=True,
    )
