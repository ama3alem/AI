"""Tests for the calibration report (Phase 9).

The load-bearing distinction in this module is between *measured badly*, *measured well*,
and *never measured*. The third is the one that disappears in a naive implementation, so
most of what follows is about keeping it visible.
"""

from __future__ import annotations

import pytest

from focus_engine.models.trainers import _expected_calibration_error
from focus_engine.schemas.versioning import ModelMetrics
from focus_engine.uncertainty.calibration import CalibrationReport, reliability_table
from tests.unit.uncertainty.scenarios import (
    labels_from,
    probabilities_around,
    train_shared_artifact,
    without_recorded_metrics,
)

pytestmark = pytest.mark.unit


def _metrics(**overrides: object) -> ModelMetrics:
    """Build a metric suite with a measured calibration unless overridden."""
    defaults: dict[str, object] = {
        "split_name": "holdout",
        "n_samples": 200,
        "n_positive": 100,
        "brier": 0.05,
        "expected_calibration_error": 0.04,
    }
    defaults.update(overrides)
    return ModelMetrics(**defaults)  # type: ignore[arg-type]


class TestReliabilityTable:
    """The binned view of what a model claimed against what happened."""

    def test_every_bin_is_retained_including_empty_ones(self) -> None:
        table = reliability_table(probabilities_around([0.05, 0.95]), labels_from([0, 1]), bins=4)
        assert len(table) == 4
        assert [bin_.n for bin_ in table] == [1, 0, 0, 1]
        assert [bin_.is_populated for bin_ in table] == [True, False, False, True]

    def test_a_populated_bin_records_claim_and_outcome(self) -> None:
        table = reliability_table(
            probabilities_around([0.10, 0.20, 0.15]), labels_from([0, 0, 1]), bins=2
        )
        first = table[0]
        assert first.n == 3
        assert first.mean_predicted == pytest.approx(0.15)
        assert first.observed_frequency == pytest.approx(1 / 3)

    def test_the_final_bin_is_closed_so_probability_one_is_counted(self) -> None:
        """A probability of exactly 1.0 must not fall outside every bin.

        If it did, the top of the confidence range would go unmeasured while the reported
        error still claimed to cover it.
        """
        table = reliability_table(probabilities_around([0.0, 1.0]), labels_from([0, 1]), bins=10)
        assert table[-1].n == 1
        assert table[-1].is_populated
        assert sum(bin_.n for bin_ in table) == 2

    def test_gap_sign_distinguishes_under_from_overconfidence(self) -> None:
        under = reliability_table(probabilities_around([0.4, 0.4]), labels_from([0, 1]), bins=2)[0]
        over = reliability_table(probabilities_around([0.4, 0.4]), labels_from([0, 0]), bins=2)[0]
        assert under.gap() > 0
        assert over.gap() < 0

    def test_fewer_than_two_bins_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two bins"):
            reliability_table(probabilities_around([0.5]), labels_from([1]), bins=1)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="disagree in length"):
            reliability_table(probabilities_around([0.1, 0.2]), labels_from([1]), bins=2)

    def test_two_dimensional_input_is_refused(self) -> None:
        import numpy as np

        with pytest.raises(ValueError, match="one-dimensional"):
            reliability_table(np.asarray([[0.1, 0.2]]), labels_from([1, 0]), bins=2)


class TestKnownAnswerCalibration:
    """Cases where the correct answer is known, so the metric can be checked directly.

    These are the tests that catch a metric which is internally consistent and entirely
    wrong. A calibration error computed against hard-prediction accuracy instead of
    observed frequency agrees with itself, is defined for every input, and reports roughly
    0.5 for a flawless model - it passes every "is it a number between zero and one"
    check while measuring the wrong thing entirely.
    """

    def test_a_perfectly_calibrated_model_scores_near_zero(self) -> None:
        """Confidence of 0.0 where nothing occurs, and 1.0 where everything does."""
        probabilities = probabilities_around([0.0] * 100 + [1.0] * 100)
        labels = labels_from([0] * 100 + [1] * 100)
        assert _expected_calibration_error(probabilities, labels) == pytest.approx(0.0)

    def test_an_inverted_model_scores_near_one(self) -> None:
        """The same confidences against flipped labels is maximal miscalibration."""
        probabilities = probabilities_around([0.0] * 100 + [1.0] * 100)
        labels = labels_from([1] * 100 + [0] * 100)
        assert _expected_calibration_error(probabilities, labels) == pytest.approx(1.0)

    def test_an_honest_middling_model_scores_near_zero(self) -> None:
        """Stating 0.3 where 30% of events occur is calibrated."""
        probabilities = probabilities_around([0.3] * 1000)
        labels = labels_from([1 if index % 10 < 3 else 0 for index in range(1000)])
        assert _expected_calibration_error(probabilities, labels) == pytest.approx(0.0, abs=0.01)

    def test_an_overconfident_model_is_penalised_proportionally(self) -> None:
        """Stating 0.3 where 90% occur is an error of 0.6."""
        probabilities = probabilities_around([0.3] * 1000)
        labels = labels_from([1 if index % 10 < 9 else 0 for index in range(1000)])
        assert _expected_calibration_error(probabilities, labels) == pytest.approx(0.6, abs=0.01)

    def test_a_flawless_model_is_not_reported_as_badly_calibrated(self) -> None:
        """The regression this whole section exists for.

        A model that predicts every holdout row correctly, with probabilities near 0 and
        1, is well calibrated. Measured against hard-prediction accuracy it would report an
        error near 0.5, which is what made a working model refuse to be trusted.
        """
        probabilities = probabilities_around([0.003] * 100 + [0.996] * 100)
        labels = labels_from([0] * 100 + [1] * 100)
        error = _expected_calibration_error(probabilities, labels)
        assert error is not None
        assert error < 0.05, f"a model with Brier near zero reported a calibration error of {error}"


class TestCalibrationReportStates:
    """The three states, kept distinct."""

    def test_a_measured_report_is_measured(self) -> None:
        report = CalibrationReport.from_metrics(_metrics())
        assert report.is_measured
        assert report.ceiling() == pytest.approx(0.96)

    def test_an_unmeasurable_split_produces_an_unmeasured_report(self) -> None:
        report = CalibrationReport.from_metrics(
            _metrics(brier=None, expected_calibration_error=None)
        )
        assert not report.is_measured
        assert report.ceiling() is None
        assert "holdout" in str(report.reason)

    def test_an_unmeasured_report_is_never_believable(self) -> None:
        """No error is not a zero error. The whole point of the distinction."""
        report = CalibrationReport.unmeasured("no holdout with both classes")
        assert not report.is_believable(0.25)
        assert not report.is_believable(1.0)

    def test_believability_follows_the_configured_limit(self) -> None:
        report = CalibrationReport.from_metrics(
            _metrics(expected_calibration_error=0.20, brier=0.20)
        )
        assert report.is_believable(0.25)
        assert not report.is_believable(0.10)

    def test_ceiling_is_clamped_into_range(self) -> None:
        assert CalibrationReport(
            expected_calibration_error=1.5, brier=0.9, n_samples=10, n_positive=5
        ).ceiling() == pytest.approx(0.0)

    def test_from_predictions_refuses_a_single_class_sample(self) -> None:
        report = CalibrationReport.from_predictions(
            probabilities_around([0.1, 0.2, 0.3]),
            labels_from([0, 0, 0]),
            bins=4,
            brier=0.05,
            expected_calibration_error=0.05,
        )
        assert not report.is_measured
        assert "undefined" in str(report.reason)

    def test_from_predictions_builds_the_table_for_a_real_sample(self) -> None:
        report = CalibrationReport.from_predictions(
            probabilities_around([0.0] * 50 + [1.0] * 50),
            labels_from([0] * 50 + [1] * 50),
            bins=5,
            brier=0.0,
            expected_calibration_error=0.0,
        )
        assert report.is_measured
        assert report.n_samples == 100
        assert report.n_positive == 50
        assert sum(bin_.n for bin_ in report.bins) == 100


class TestCalibrationReportValidation:
    """A report that contradicts itself is refused rather than stored."""

    def test_one_metric_without_the_other_is_refused(self) -> None:
        with pytest.raises(ValueError, match="present or absent together"):
            CalibrationReport(brier=0.1, expected_calibration_error=None, n_samples=10)

    def test_a_measurement_with_no_observations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            CalibrationReport(brier=0.1, expected_calibration_error=0.1, n_samples=0)

    def test_a_measured_report_may_not_also_carry_a_reason(self) -> None:
        with pytest.raises(ValueError, match="contradict each other"):
            CalibrationReport(
                brier=0.1,
                expected_calibration_error=0.1,
                n_samples=10,
                n_positive=5,
                reason="should not be here",
            )

    def test_more_positives_than_observations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exceeds n_samples"):
            CalibrationReport(brier=0.1, expected_calibration_error=0.1, n_samples=5, n_positive=9)

    def test_negative_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CalibrationReport(n_samples=-1)


class TestCalibrationFromRecord:
    """Calibration is read from the model's own record, not recomputed at serving time."""

    def test_a_real_records_holdout_calibration_is_read(self) -> None:
        artifact = train_shared_artifact()
        report = CalibrationReport.from_record(artifact.record)
        assert report.is_measured
        assert report.n_samples > 0

    def test_a_record_with_no_metrics_is_unmeasured_and_says_why(self) -> None:
        artifact = without_recorded_metrics(train_shared_artifact())
        report = CalibrationReport.from_record(artifact.record)
        assert not report.is_measured
        reason = str(report.reason)
        assert artifact.record.model_id in reason
        assert "holdout" in reason

    def test_a_record_missing_the_requested_split_names_what_it_does_have(self) -> None:
        report = CalibrationReport.from_record(
            train_shared_artifact().record, split_name="temporal_oos"
        )
        assert not report.is_measured
        assert "holdout" in str(report.reason)
