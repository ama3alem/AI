"""Turning a corpus of evaluations into counts, rates, and groups.

This module aggregates. It does not interpret, does not select, and does not rank. Every
number it produces is a description of a corpus that already exists, and none of them is a
recommendation about what to do next.

**An undefined rate is ``None``, never ``0.0``.** A model that predicted decline for nobody
in a corpus with no declining learners has a false-positive rate that is zero out of zero.
That is not a rate of zero -- zero would mean it made no false positives in a corpus where
false positives were possible. The distinction matters because a summary that silently
substitutes zero for "there was nothing to measure" will show a model improving when nothing
was tested, and that is the shape a metric fakes. Every denominator of zero yields ``None``,
and every ``None`` is accompanied by the counts that explain it, so a reader can always see
which case they are in.

**A prediction that could not be scored is counted, not discarded.** Unassessable records
appear in :attr:`EvaluationSummary.not_assessable` and in the confusion matrix's fifth
cell. Dropping them would inflate every rate, because the records most likely to be
unassessable are those made when the evidence was thin, and a corpus of those is exactly the
corpus where the model looks worst and is being measured least. The rate denominators are
therefore the *assessed* records, and the assessed count is reported beside them.

**Groups are descriptions, not rankings.** :func:`group_by_learner`,
:func:`group_by_model_version`, and :func:`group_by_target` return a mapping from identifier
to summary, keyed and ordered deterministically. They deliberately expose no ordering by
accuracy and no comparison helper. A per-learner table is the natural place for a reader to
conclude that the model is better for some learners than others, and a learner with a
thousand records will outrank one with ten regardless of the underlying quality. The
per-group record counts are reported so that a comparison is at least not made blind to them.

The module holds no state and reads no clock, so aggregation is a pure function of its input
and two calls over the same records return equal summaries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.evaluation.models import (
    BinaryVerdict,
    EvaluationRecord,
    EvaluationVerdict,
    GroundTruthStatus,
    InterventionResponse,
    PredictionTarget,
)

__all__ = [
    "EVALUATION_VERDICTS",
    "PREDICTION_TARGETS",
    "RESPONSE_MEMBERS",
    "ConfusionCounts",
    "EvaluationSummary",
    "group_by",
    "group_by_intervention_response",
    "group_by_learner",
    "group_by_model_version",
    "group_by_target",
    "summarise",
]


def _rate(numerator: int, denominator: int) -> float | None:
    """Divide, refusing to invent an answer when there is nothing to divide.

    Args:
        numerator: The count in the numerator.
        denominator: The count in the denominator.

    Returns:
        The ratio, or ``None`` when ``denominator`` is zero. A zero denominator is not a
        numerator of zero over a positive number; it is the absence of the population the
        rate is defined over.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def _harmonic(first: float | None, second: float | None) -> float | None:
    """Return the harmonic mean of two rates, or ``None`` when either is undefined.

    Args:
        first: One rate.
        second: The other rate.

    Returns:
        The harmonic mean, or ``None`` if either input is ``None`` or both are zero. A
        harmonic mean of two zeros is undefined rather than zero, since the formula divides
        by their sum.
    """
    if first is None or second is None:
        return None
    total = first + second
    if total == 0.0:
        return None
    return 2.0 * first * second / total


class ConfusionCounts(BaseModel):
    """The four cells of a binary confusion matrix, plus the records outside it.

    Unassessable records have a cell of their own rather than being dropped. They are the
    records that could not be scored at all, and a matrix that silently excluded them would
    report the same counts whether the model was evaluated against a full corpus or against
    the small part of it that happened to be measurable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    true_positive: int = Field(default=0, ge=0)
    false_positive: int = Field(default=0, ge=0)
    true_negative: int = Field(default=0, ge=0)
    false_negative: int = Field(default=0, ge=0)
    not_assessable: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        """Every record counted, including the unassessable ones."""
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
            + self.not_assessable
        )

    @property
    def assessed(self) -> int:
        """The records that were actually scored against a confusion cell.

        Note:
            This is the *matrix* view, and it deliberately does not subtract only the
            records whose verdict was not assessable. A record for a ``STATE`` or
            ``DIRECTION`` target is scored -- it was either confirmed or refuted -- yet it
            sits in :attr:`not_assessable` because a five-way claim has no positive class to
            put in a cell. Use :attr:`EvaluationSummary.assessed` for how much of the corpus
            was really measured; this counts only what the matrix can hold.
        """
        return self.total - self.not_assessable

    @property
    def predicted_positive(self) -> int:
        """How many records claimed a decline."""
        return self.true_positive + self.false_positive

    @property
    def observed_positive(self) -> int:
        """How many records found a decline."""
        return self.true_positive + self.false_negative

    @classmethod
    def from_records(cls, records: Iterable[EvaluationRecord]) -> ConfusionCounts:
        """Count a corpus into the confusion matrix.

        Args:
            records: The evaluated records to count.

        Returns:
            The counts. Records whose target is not ``DECLINE`` land in
            :attr:`not_assessable`, because a five-way or four-way claim has no positive
            class and cannot be placed in a binary cell.
        """
        counts = Counter(record.binary_verdict for record in records)
        return cls(
            true_positive=counts[BinaryVerdict.TRUE_POSITIVE],
            false_positive=counts[BinaryVerdict.FALSE_POSITIVE],
            true_negative=counts[BinaryVerdict.TRUE_NEGATIVE],
            false_negative=counts[BinaryVerdict.FALSE_NEGATIVE],
            not_assessable=counts[BinaryVerdict.NOT_ASSESSABLE],
        )

    def describe(self) -> str:
        """Render the counts as a single human-readable line.

        Returns:
            A description naming each cell and the assessed total.
        """
        return (
            f"TP {self.true_positive}, FP {self.false_positive}, TN {self.true_negative}, "
            f"FN {self.false_negative}, unassessable {self.not_assessable} "
            f"({self.assessed} assessed of {self.total})"
        )


#: Every intervention response member, in canonical order. Every summary reports all five,
#: zero-filled, so an absent key can never be misread as a category that was never defined.
RESPONSE_MEMBERS: Final[tuple[InterventionResponse, ...]] = tuple(InterventionResponse)


class EvaluationSummary(BaseModel):
    """Counts and rates for one corpus of evaluated records.

    Every rate is ``float | None``, and every ``None`` is a statement that the rate's
    denominator was zero. The counts are reported beside the rates so a reader never has to
    guess which of the two situations they are looking at, and so a rate can be recomputed
    from the record.

    The rates are the standard five for an imbalanced binary target -- precision, recall, F1,
    false-positive rate, and false-negative rate -- plus accuracy for completeness. None of
    them is a decision criterion, and no field here combines them into a single score: no such
    reduction is defensible without knowing which errors matter for which learner, and that
    weighting is a policy decision this layer does not make.

    Two different counts of "could not be used" appear here, and they are not the same count.
    The four status fields partition the corpus by what the evaluation concluded, so
    :attr:`assessed` is ``confirmed + refuted`` and everything else is a non-finding.
    :attr:`EvaluationSummary.confusion` is partitioned by what a *binary* matrix can hold, so
    its :attr:`~ConfusionCounts.not_assessable` additionally swallows every record whose target
    is not ``DECLINE``. On a corpus of only ``DECLINE`` predictions the two partition it the
    same way; on a mixed corpus they do not, and neither is wrong. Reading the matrix cell as
    the number of unscored records overstates how much of the corpus was measured, because a
    confirmed five-way claim is outside the matrix and inside the measured set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_records: int = Field(ge=0)
    confusion: ConfusionCounts
    confirmed: int = Field(default=0, ge=0)
    refuted: int = Field(default=0, ge=0)
    insufficient_data: int = Field(default=0, ge=0)
    not_assessable: int = Field(default=0, ge=0)
    intervention_responses: dict[InterventionResponse, int] = Field(default_factory=dict)
    learners: int = Field(default=0, ge=0)
    """Distinct learners in this corpus, so a per-group summary states its own sample size."""

    @model_validator(mode="after")
    def _validate_totals(self) -> Self:
        """Check the status counts add up to the record total.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the four status counts do not sum to ``total_records``. A summary
                whose parts do not account for the whole cannot be trusted for any rate
                derived from it, because a miscount would silently redistribute into a
                numerator.
        """
        parts = self.confirmed + self.refuted + self.insufficient_data + self.not_assessable
        if parts != self.total_records:
            raise ValueError(
                f"summary counts do not account for the corpus: "
                f"{self.confirmed} + {self.refuted} + {self.insufficient_data} + "
                f"{self.not_assessable} = {parts}, but total_records is {self.total_records}"
            )
        return self

    @property
    def assessed(self) -> int:
        """The records that were scored, i.e. the ones the model either got right or wrong.

        This is ``confirmed + refuted`` rather than ``total - not_assessable`` because the two
        non-findings are not interchangeable: a window that held too little activity
        (:attr:`insufficient_data`) produced a reading nobody could act on, exactly as a
        missing reading did (:attr:`not_assessable`), and subtracting only one of them would
        report the unmeasurable windows as though the model had been tested on them.
        """
        return self.confirmed + self.refuted

    @property
    def unscored(self) -> int:
        """The records that produced no finding: too little evidence, or no usable reading."""
        return self.insufficient_data + self.not_assessable

    @property
    def coverage(self) -> float | None:
        """The share of records that could be scored at all.

        Returns:
            The assessed share, or ``None`` for an empty corpus. This is the number to read
            before any other: a corpus that is 5% covered and a corpus that is 100% covered
            can produce the same accuracy, and they are not the same result.
        """
        return _rate(self.assessed, self.total_records)

    @property
    def accuracy(self) -> float | None:
        """The share of assessed records whose prediction was borne out.

        Returns:
            The rate, or ``None`` when nothing was assessed. Dominated by the majority class
            under imbalance, and reported for completeness rather than as a criterion.
        """
        return _rate(self.confirmed, self.assessed)

    @property
    def precision(self) -> float | None:
        """The share of predicted declines that were borne out.

        Returns:
            The rate, or ``None`` when no record claimed a decline.
        """
        return _rate(self.confusion.true_positive, self.confusion.predicted_positive)

    @property
    def recall(self) -> float | None:
        """The share of actual declines that were predicted.

        Returns:
            The rate, or ``None`` when no record found a decline.
        """
        return _rate(self.confusion.true_positive, self.confusion.observed_positive)

    @property
    def f1(self) -> float | None:
        """The harmonic mean of precision and recall.

        Returns:
            The rate, or ``None`` when either input is undefined.
        """
        return _harmonic(self.precision, self.recall)

    @property
    def false_positive_rate(self) -> float | None:
        """The share of non-declining records that were predicted as declining.

        Returns:
            The rate, or ``None`` when the corpus held no non-declining records. This is the
            cost of a prompt delivered to a learner who did not need it.
        """
        return _rate(
            self.confusion.false_positive,
            self.confusion.false_positive + self.confusion.true_negative,
        )

    @property
    def false_negative_rate(self) -> float | None:
        """The share of declining records that were not predicted.

        Returns:
            The rate, or ``None`` when the corpus held no declining records.
        """
        return _rate(
            self.confusion.false_negative,
            self.confusion.false_negative + self.confusion.true_positive,
        )

    def describe(self) -> str:
        """Render the summary as a short human-readable block.

        Returns:
            A multi-line summary of the counts and of every defined rate. An undefined rate
            is printed as ``undefined`` rather than as a number, so the gap is visible in the
            rendering instead of only in the type.
        """

        def show(value: float | None) -> str:
            return "undefined" if value is None else f"{value:.4f}"

        return "\n".join(
            [
                f"{self.total_records} record(s) over {self.learners} learner(s); "
                f"{self.assessed} assessed, {self.insufficient_data} insufficient data, "
                f"{self.not_assessable} not assessable",
                f"  {self.confusion.describe()}",
                f"  accuracy {show(self.accuracy)}, precision {show(self.precision)}, "
                f"recall {show(self.recall)}, f1 {show(self.f1)}",
                f"  fpr {show(self.false_positive_rate)}, fnr {show(self.false_negative_rate)}",
            ]
        )


def summarise(records: Iterable[EvaluationRecord]) -> EvaluationSummary:
    """Reduce a corpus of evaluated records to one summary.

    Args:
        records: The records to summarise. Consumed once; order does not affect the result.

    Returns:
        A summary whose rates are ``None`` wherever their denominator was zero. An empty
        corpus yields a summary of zeros with every rate ``None``, which is the honest
        description: nothing was measured.
    """
    materialised = tuple(records)
    statuses = Counter(record.ground_truth_status for record in materialised)
    responses = Counter(
        record.intervention_response
        for record in materialised
        if record.intervention_response is not None
    )
    return EvaluationSummary(
        total_records=len(materialised),
        confusion=ConfusionCounts.from_records(materialised),
        confirmed=statuses[GroundTruthStatus.CONFIRMED],
        refuted=statuses[GroundTruthStatus.REFUTED],
        insufficient_data=statuses[GroundTruthStatus.INSUFFICIENT_DATA],
        not_assessable=statuses[GroundTruthStatus.NOT_ASSESSABLE],
        intervention_responses={member: responses[member] for member in RESPONSE_MEMBERS},
        learners=len({record.learner_id for record in materialised}),
    )


def group_by(
    records: Iterable[EvaluationRecord],
    key: Callable[[EvaluationRecord], str],
) -> dict[str, EvaluationSummary]:
    """Summarise a corpus once per distinct value of an identifier.

    Args:
        records: The records to group.
        key: Extracts the group identifier from a record.

    Returns:
        A mapping from identifier to summary, ordered by identifier so two runs over the same
        corpus produce the same iteration order and a rendered table does not shuffle.

    Note:
        The keys are sorted rather than returned in first-seen order, and no group is
        annotated with its relative performance. Ordering a table by a metric invites the
        reader to read the order as a ranking, and for per-learner groups that ranking would
        be dominated by how many records each learner happens to have. The per-group
        :attr:`EvaluationSummary.learners` and :attr:`EvaluationSummary.total_records` are
        reported so a comparison is not blind to that.
    """
    buckets: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        buckets.setdefault(key(record), []).append(record)
    return {name: summarise(buckets[name]) for name in sorted(buckets)}


def group_by_learner(records: Iterable[EvaluationRecord]) -> dict[str, EvaluationSummary]:
    """Summarise separately for each learner.

    Args:
        records: The records to group.

    Returns:
        A mapping from learner identifier to that learner's summary.
    """
    return group_by(records, lambda record: record.learner_id)


def group_by_model_version(records: Iterable[EvaluationRecord]) -> dict[str, EvaluationSummary]:
    """Summarise separately for each model version.

    Grouping on the version rather than on :attr:`EvaluationRecord.model_id` keeps two
    artifacts of the same approach separate, which is the comparison that matters: the
    question is whether a *version* did better, and grouping by approach would pool a
    retired artifact with its successor and answer a different one.

    Args:
        records: The records to group.

    Returns:
        A mapping from model version to that version's summary.
    """
    return group_by(records, lambda record: record.model_version)


def group_by_target(records: Iterable[EvaluationRecord]) -> dict[str, EvaluationSummary]:
    """Summarise separately for each prediction target.

    Args:
        records: The records to group.

    Returns:
        A mapping from target to that target's summary. Only the ``DECLINE`` target carries
        confusion counts, so the ``STATE`` and ``DIRECTION`` groups report every confusion rate
        as ``None``; their :attr:`EvaluationSummary.confirmed` and :attr:`EvaluationSummary.refuted`
        counts are still populated, and their accuracy and coverage are still defined. A
        multi-state claim is measured, it simply has no confusion matrix.
    """
    return group_by(records, lambda record: record.target.value)


def group_by_intervention_response(
    records: Iterable[EvaluationRecord],
) -> dict[InterventionResponse, EvaluationSummary]:
    """Summarise the predictions associated with each observed intervention response.

    Args:
        records: The records to group. Those carrying no response are excluded, since a
            prediction that is not tied to a delivery has no response to group under.

    Returns:
        A mapping from response member to the summary for the predictions carrying it, in
        canonical member order. Predictions are grouped by what followed a delivery, not by
        the delivery's effect on the prediction: the response describes a window either side
        of a prompt, and a prediction made inside that window is not a consequence of it.
    """
    buckets: dict[InterventionResponse, list[EvaluationRecord]] = {}
    for record in records:
        if record.intervention_response is None:
            continue
        buckets.setdefault(record.intervention_response, []).append(record)
    return {member: summarise(buckets[member]) for member in RESPONSE_MEMBERS if member in buckets}


#: The prediction targets, in canonical order, for consumers switching exhaustively.
PREDICTION_TARGETS: Final[tuple[PredictionTarget, ...]] = tuple(PredictionTarget)

#: The evaluation verdicts, in canonical order, for consumers switching exhaustively.
EVALUATION_VERDICTS: Final[tuple[EvaluationVerdict, ...]] = tuple(EvaluationVerdict)
