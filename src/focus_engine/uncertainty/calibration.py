"""Calibration as a measured property of a model, not an assumed one.

Confidence in this system is a claim about how much a number should be trusted, and the
only honest way to make that claim is to derive it from something. This module supplies
one of the three inputs: whether the model, on data it did not train on, reports its
confidence accurately.

**Calibration is separate from discrimination.** A model can rank learners perfectly and
still be wildly overconfident, and a ranking-only metric will never show it. The practical
consequence is that "the model was right" and "the model knew it was right" are different
questions, and only the second one licenses a high confidence value. This is why the
report here is built from a held-out evaluation rather than from training accuracy, which
is close to meaningless as a calibration signal.

**An unmeasured calibration is not a good calibration.** The distinction this module is
built around is between three states that a naive implementation collapses into one:

* measured, and close enough to believe;
* measured, and too far off to believe;
* never measured, because the evaluation split could not support the measurement.

The third is the one that gets lost. A model evaluated on a single-class split has no
calibration error, not a perfect one, and code that treats an absent ECE as ``0.0`` will
report flawless calibration for a model nobody has ever checked. :class:`CalibrationReport`
therefore refuses to have a calibration ceiling at all when it has not been measured, and
the uncertainty engine turns that refusal into ``UNKNOWN`` rather than into a number.

**The ceiling is deliberately conservative.** Where a calibration error *e* has been
measured, the model's stated confidence of ``c`` is taken to be worth at most ``1 - e``.
That is not a remapping and does not claim to be one; a true isotonic recalibration needs
more held-out data than exists here. It is the reading that cannot be wrong in the
dangerous direction: a model whose stated confidence is off by twenty-five points is not
treated as able to speak with ninety-five-point confidence about anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from focus_engine.schemas.versioning import ModelMetrics, ModelRecord

__all__ = [
    "CalibrationReport",
    "ReliabilityBin",
    "reliability_table",
]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One equal-width probability bin and what was observed inside it.

    Attributes:
        lower: Inclusive lower edge of the bin.
        upper: Exclusive upper edge of the bin, except for the final bin which is closed.
        n: Number of observations that fell in the bin.
        mean_predicted: Mean predicted probability inside the bin.
        observed_frequency: Fraction of the bin's observations that were actually positive.
    """

    lower: float
    upper: float
    n: int
    mean_predicted: float
    observed_frequency: float

    @property
    def is_populated(self) -> bool:
        """Whether the bin holds any observation, and so can be read at all."""
        return self.n > 0

    def gap(self) -> float:
        """Signed distance between what the model claimed and what was observed.

        A positive gap means the model was underconfident in this bin; negative means it
        overstated. The sign is reported rather than discarded because an
        underconfident model and an overconfident one fail in different directions, and a
        single unsigned error number cannot tell them apart.
        """
        return self.observed_frequency - self.mean_predicted

    def describe(self) -> str:
        """Render the bin as a short human-readable line.

        Returns:
            A one-line description of the bin's contents.
        """
        return (
            f"[{self.lower:.2f}, {self.upper:.2f}) n={self.n} "
            f"predicted={self.mean_predicted:.3f} observed={self.observed_frequency:.3f}"
        )


def reliability_table(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    bins: int,
) -> tuple[ReliabilityBin, ...]:
    """Bin predictions by stated confidence and record what actually happened.

    Bins are equal-width, which makes the resulting error a weighted mean of per-bin gaps
    that can be read directly. Equal-frequency bins would give a better-resolved curve at
    the low end, but they make the per-bin weights uneven in a way that has to be carried
    through every downstream number, and the point here is to answer "is this model's
    stated confidence usable", not to publish a calibration curve.

    Args:
        probabilities: Predicted class-1 probabilities.
        labels: Ground-truth binary labels.
        bins: Requested number of equal-width bins.

    Returns:
        One :class:`ReliabilityBin` per bin, including empty ones. Empty bins are kept
        rather than dropped so that the shape of the confidence range is visible; an
        empty bin has nothing to say, and :attr:`ReliabilityBin.is_populated` says so.
    """
    if bins < 2:
        raise ValueError(f"at least two bins are required for a reliability table; got {bins}")
    if probabilities.ndim != 1 or labels.ndim != 1:
        raise ValueError(
            f"probabilities and labels must both be one-dimensional; got "
            f"{probabilities.ndim} and {labels.ndim}"
        )
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probabilities and labels disagree in length: {probabilities.shape[0]} against "
            f"{labels.shape[0]}"
        )

    edges = np.linspace(0.0, 1.0, bins + 1)
    table: list[ReliabilityBin] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        # The final bin is closed on the right so that a probability of exactly 1.0 is
        # counted rather than falling outside every bin and silently vanishing from the
        # table, which would leave the top of the confidence range unmeasured.
        if index == bins - 1:
            in_bin = (probabilities >= low) & (probabilities <= high)
        else:
            in_bin = (probabilities >= low) & (probabilities < high)
        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            table.append(
                ReliabilityBin(
                    lower=float(low),
                    upper=float(high),
                    n=0,
                    mean_predicted=0.0,
                    observed_frequency=0.0,
                )
            )
            continue
        table.append(
            ReliabilityBin(
                lower=float(low),
                upper=float(high),
                n=n_in_bin,
                mean_predicted=float(probabilities[in_bin].mean()),
                observed_frequency=float(labels[in_bin].mean()),
            )
        )
    return tuple(table)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """How far a model's stated confidence departs from observed frequency.

    A report can exist without having a measurement, and that is a first-class state
    rather than a missing field. :attr:`is_measured` distinguishes the two, and
    :meth:`ceiling` returns ``None`` for an unmeasured report so that no caller can
    accidentally read an absent measurement as a perfect one.

    Attributes:
        brier: Mean squared error of the stated probabilities, or ``None`` when
            unmeasured. A probability is penalised more for a confident error than for an
            unconfident one, which is why this is reported alongside rather than instead of
            the calibration error.
        expected_calibration_error: Weighted mean absolute gap between stated confidence
            and observed frequency, or ``None`` when unmeasured.
        n_samples: Observations the calibration was measured on.
        n_positive: Positives among those observations.
        bins: Reliability bins, when the caller had the raw predictions to build them.
        reason: Why the report is unmeasured. ``None`` when it was measured.
    """

    brier: float | None = None
    expected_calibration_error: float | None = None
    n_samples: int = 0
    n_positive: int = 0
    bins: tuple[ReliabilityBin, ...] = field(default_factory=tuple)
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate that a report is internally consistent.

        Raises:
            ValueError: If one calibration metric is present without the other, or if a
                stated count contradicts the other. Two metrics always enter a report
                together, because they are computed by the same evaluation; one being
                present and the other absent means the report was assembled incorrectly
                and the result would be read as a partial measurement.
        """
        if (self.brier is None) != (self.expected_calibration_error is None):
            raise ValueError(
                "brier and expected_calibration_error must be present or absent together; "
                f"got brier={self.brier} and ece={self.expected_calibration_error}"
            )
        if self.n_samples < 0 or self.n_positive < 0:
            raise ValueError("observation counts must be non-negative")
        if self.n_positive > self.n_samples:
            raise ValueError(f"n_positive ({self.n_positive}) exceeds n_samples ({self.n_samples})")
        if self.is_measured and self.n_samples == 0:
            raise ValueError(
                "a measured calibration report must have observed at least one sample; a "
                "report with metrics and no observations cannot be a measurement"
            )
        if self.is_measured and self.reason is not None:
            raise ValueError(
                "a measured calibration report cannot also carry an unmeasured reason; the "
                f"two contradict each other (reason={self.reason!r})"
            )

    @property
    def is_measured(self) -> bool:
        """Whether calibration was actually measured on held-out data.

        This is the property the uncertainty engine keys on, and it is deliberately not a
        truthiness test on the error value: an error of ``0.0`` is a *measured* result and
        a genuinely good one, while ``None`` means nothing was measured.
        """
        return self.expected_calibration_error is not None

    def ceiling(self) -> float | None:
        """The highest confidence this model's own output can justify.

        Returns:
            ``1 - expected_calibration_error`` when measured, clamped into ``[0, 1]``, or
            ``None`` when unmeasured.
        """
        if self.expected_calibration_error is None:
            return None
        return max(0.0, min(1.0, 1.0 - float(self.expected_calibration_error)))

    def is_believable(self, max_calibration_error: float) -> bool:
        """Whether a measured calibration is close enough to build confidence on.

        Args:
            max_calibration_error: The error above which the model's own stated
                confidence is not usable, regardless of evidence volume.

        Returns:
            ``True`` only for a measured report whose error is at or below the limit. An
            unmeasured report is never believable, because it has nothing to offer.
        """
        if not self.is_measured:
            return False
        return float(self.expected_calibration_error or 0.0) <= max_calibration_error

    @classmethod
    def from_metrics(
        cls,
        metrics: ModelMetrics,
        *,
        bins: tuple[ReliabilityBin, ...] = (),
    ) -> CalibrationReport:
        """Build a report from a recorded metric suite.

        Args:
            metrics: The metrics recorded for one split.
            bins: Reliability bins, when available.

        Returns:
            A report carrying whatever the split was able to measure. When the split
            could not support a calibration measurement, the result is an unmeasured
            report naming that as the reason.
        """
        if (
            metrics.expected_calibration_error is None
            or metrics.brier is None
            or metrics.n_positive is None
        ):
            return cls.unmeasured(
                f"the {metrics.split_name!r} split could not support a calibration "
                f"measurement. It holds {metrics.n_samples} observations and "
                f"{metrics.n_positive} positives. A calibration error needs both classes "
                "present, and with only one class every stated probability is trivially "
                "right or trivially wrong, so the error is undefined rather than zero. "
                "This is not a finding of good calibration; it is the absence of a "
                "measurement, and the uncertainty engine answers UNKNOWN because of it."
            )
        return cls(
            brier=float(metrics.brier),
            expected_calibration_error=float(metrics.expected_calibration_error),
            n_samples=metrics.n_samples,
            n_positive=metrics.n_positive,
            bins=bins,
        )

    @classmethod
    def unmeasured(cls, reason: str) -> CalibrationReport:
        """Build a report that explicitly holds no measurement.

        Args:
            reason: Why nothing could be measured.

        Returns:
            An unmeasured report carrying ``reason``.
        """
        return cls(reason=reason)

    @classmethod
    def from_record(
        cls,
        record: ModelRecord,
        *,
        split_name: str = "holdout",
    ) -> CalibrationReport:
        """Read calibration out of a model's own recorded evaluation.

        Reading the recorded metrics rather than recomputing them is deliberate. The
        numbers in the record are the ones the model was approved on, and a calibration
        figure recomputed at prediction time could differ from them without anything
        noticing; here the two cannot disagree, because there is only one number.

        Args:
            record: The model record carrying the recorded metrics.
            split_name: Which recorded split to read. Defaults to the holdout, which is
                the only split a model is allowed to be judged on.

        Returns:
            The report for that split, or an unmeasured report if the record holds no such
            split.
        """
        matches = tuple(metric for metric in record.metrics if metric.split_name == split_name)
        if not matches:
            available = sorted(metric.split_name for metric in record.metrics)
            return cls.unmeasured(
                f"model {record.model_id} version {record.model_version} recorded no "
                f"{split_name!r} metrics, so its calibration has never been measured. The "
                f"record holds: {available or 'no metrics at all'}. A model whose "
                "calibration is unmeasured can still be scored, but the confidence of that "
                "score cannot be stated, and the uncertainty engine answers UNKNOWN."
            )
        return cls.from_metrics(matches[0])

    @classmethod
    def from_predictions(
        cls,
        probabilities: NDArray[np.float64],
        labels: NDArray[np.int64],
        *,
        bins: int,
        brier: float,
        expected_calibration_error: float,
    ) -> CalibrationReport:
        """Build a fully populated report from raw held-out predictions.

        Args:
            probabilities: Held-out class-1 probabilities.
            labels: Held-out binary labels.
            bins: Number of equal-width reliability bins.
            brier: The already-computed Brier score.
            expected_calibration_error: The already-computed calibration error.

        Returns:
            A measured report including the reliability table.

        Raises:
            ValueError: If the sample holds no positive or no negative, in which case the
                caller's calibration error is not a measurement and the report is
                unmeasured instead.
        """
        n_samples = int(labels.shape[0])
        n_positive = int((labels == 1).sum())
        if n_samples == 0 or n_positive == 0 or n_positive == n_samples:
            return cls.unmeasured(
                f"calibration was requested for a sample of {n_samples} observations with "
                f"{n_positive} positives. With only one class present the calibration error "
                "is undefined, and treating it as zero would report a model as perfectly "
                "calibrated on the strength of a measurement that was never made."
            )
        return cls(
            brier=float(brier),
            expected_calibration_error=float(expected_calibration_error),
            n_samples=n_samples,
            n_positive=n_positive,
            bins=reliability_table(probabilities, labels, bins=bins),
        )
