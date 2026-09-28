"""Tests for artifact serialisation and reload (Phase 8).

A reloaded model is the one this system will eventually serve, and the failure mode that
matters is a model that loads cleanly and then scores data it was never trained on. These
tests pin the properties that make a reload trustworthy: the estimator round-trips, the
provenance round-trips, the schema round-trips, and anything that would leave the artifact
describing something other than itself is refused.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest

from focus_engine.models.artifacts import ModelArtifact, load_artifact, save_artifact
from focus_engine.models.encoding import EncodingError, encode_rows
from focus_engine.models.trainers import Algorithm, TrainingConfig, train_model
from focus_engine.schemas.primitives import DataOrigin, Provenance
from tests.unit.models.scenarios import make_labelled_rows

pytestmark = pytest.mark.unit


def make_config(algorithm: Algorithm = Algorithm.LOGISTIC_REGRESSION) -> TrainingConfig:
    """Build a valid training configuration for a test."""
    return TrainingConfig(
        algorithm=algorithm,
        model_id="risk-logreg",
        model_version="RISK_MODEL_V1",
        feature_set_version="FEATURE_SET_V1",
        dataset_version="DATASET_REAL_V1",
        target_definition_version="td-1.0.0",
        label_provenance=Provenance.OBSERVED,
        data_origin=DataOrigin.REAL,
        seed=4242,
    )


def make_trained(algorithm: Algorithm = Algorithm.LOGISTIC_REGRESSION):
    """Train a model and return the outcome together with its holdout split."""
    train = encode_rows(make_labelled_rows(60, positive=20, start_index=0, overlap=True))
    holdout = encode_rows(make_labelled_rows(20, positive=6, start_index=1000, overlap=True))
    return train_model(make_config(algorithm), train, holdout), holdout


class TestRoundTrip:
    """A saved artifact loads back as the model that was saved."""

    def test_the_estimator_round_trips(self, tmp_path: Path) -> None:
        """A reloaded model predicts exactly as the original did."""
        outcome, holdout = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")

        reloaded = load_artifact(path)

        np.testing.assert_allclose(
            reloaded.estimator.predict_proba(holdout.matrix),
            outcome.artifact.estimator.predict_proba(holdout.matrix),
        )

    def test_the_record_round_trips(self, tmp_path: Path) -> None:
        """The provenance survives the round trip intact.

        A record that came back with a default status or a lost seed would leave the
        registry describing a training run that nobody can reproduce.
        """
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")

        reloaded = load_artifact(path)

        assert reloaded.record == outcome.record

    def test_the_schema_round_trips(self, tmp_path: Path) -> None:
        """The reloaded model knows the columns it was trained on."""
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")

        reloaded = load_artifact(path)

        assert reloaded.encoding_schema == outcome.artifact.encoding_schema
        assert reloaded.encoding_digest == outcome.artifact.encoding_digest
        assert reloaded.columns == outcome.artifact.columns

    def test_the_seed_and_timestamp_round_trip(self, tmp_path: Path) -> None:
        """Reproducibility needs the seed that was actually used, not just the run seed."""
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")

        reloaded = load_artifact(path)

        assert reloaded.seed == outcome.artifact.seed
        assert reloaded.trained_at == outcome.artifact.trained_at

    def test_the_algorithm_name_round_trips(self, tmp_path: Path) -> None:
        """A reader can tell what kind of model this is without the training code."""
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")

        reloaded = load_artifact(path)

        assert reloaded.algorithm_name == "sklearn.linear_model.LogisticRegression"

    def test_a_reloaded_model_can_be_resaved(self, tmp_path: Path) -> None:
        """The artifact is a fixed point, so a deployment can relocate it."""
        outcome, _ = make_trained()
        first = save_artifact(outcome.artifact, tmp_path / "first.joblib")

        reloaded = load_artifact(first)
        second = save_artifact(reloaded, tmp_path / "second.joblib")
        again = load_artifact(second)

        assert again.record == outcome.record
        assert again.encoding_digest == outcome.artifact.encoding_digest

    @pytest.mark.parametrize("algorithm", list(Algorithm))
    def test_every_algorithm_round_trips(self, algorithm: Algorithm, tmp_path: Path) -> None:
        """Each supported estimator survives serialisation, not just logistic regression."""
        outcome, holdout = make_trained(algorithm)
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")

        reloaded = load_artifact(path)

        np.testing.assert_allclose(
            reloaded.estimator.predict_proba(holdout.matrix),
            outcome.artifact.estimator.predict_proba(holdout.matrix),
        )

    def test_saving_creates_missing_directories(self, tmp_path: Path) -> None:
        """A deployment path that does not exist yet is not an error."""
        outcome, _ = make_trained()
        path = tmp_path / "nested" / "deeper" / "model.joblib"

        save_artifact(outcome.artifact, path)

        assert path.exists()


class TestRefusals:
    """An artifact that cannot be trusted is not loaded."""

    def test_a_missing_model_is_reported(self, tmp_path: Path) -> None:
        """The path does not exist, and saying so is better than an empty artifact."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_artifact(tmp_path / "absent.joblib")

    def test_a_model_without_provenance_is_refused(self, tmp_path: Path) -> None:
        """An estimator with no sidecar is not loadable.

        This is the case that matters most: the model would work, and would work on data
        whose column order nothing was left to verify.
        """
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")
        path.with_name("model.meta.joblib").unlink()

        with pytest.raises(FileNotFoundError, match="metadata"):
            load_artifact(path)

    def test_a_tampered_schema_is_refused(self, tmp_path: Path) -> None:
        """A sidecar whose digest no longer matches its schema is refused.

        Editing the recorded schema would be a way to make a model's provenance say
        something the model was never trained on, and the digest is what makes that
        visible.
        """
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")
        metadata = joblib.load(path.with_name("model.meta.joblib"))
        metadata["encoding_schema"] = {
            "columns": ["performance_recent_accuracy"],
            "requirements": [],
        }
        joblib.dump(metadata, path.with_name("model.meta.joblib"))

        with pytest.raises(EncodingError, match="does not match the schema"):
            load_artifact(path)

    def test_a_sidecar_that_is_not_metadata_is_refused(self, tmp_path: Path) -> None:
        """A file that is not an artifact metadata file is refused, not guessed at."""
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")
        joblib.dump(["not", "metadata"], path.with_name("model.meta.joblib"))

        with pytest.raises(EncodingError, match="not a recognised artefact"):
            load_artifact(path)

    def test_an_object_that_cannot_predict_is_refused(self, tmp_path: Path) -> None:
        """The loaded object must at least be a model."""
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")
        joblib.dump({"not": "an estimator"}, path)

        with pytest.raises(ValueError, match="no predict"):
            load_artifact(path)


class TestLibraryVersions:
    """A reproducibility claim is a claim about a specific environment.

    The seed and the data are not sufficient. Two runs sharing both will still disagree if
    scikit-learn was upgraded in between, and nothing about the fitted parameters would
    reveal that the difference came from the environment rather than from the data. These
    tests pin that the environment is recorded, that it survives a round trip, and that
    drift is visible rather than silent.
    """

    def test_training_records_the_library_versions(self) -> None:
        """A freshly trained artifact reports what produced it."""
        outcome, _ = make_trained()

        versions = outcome.artifact.library_versions

        assert versions, "a trained artifact must record the environment that produced it"
        assert "scikit-learn" in versions
        assert all(value for value in versions.values())

    def test_the_library_versions_round_trip(self, tmp_path: Path) -> None:
        """The environment survives serialisation, so a reloaded model can report it."""
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")

        reloaded = load_artifact(path)

        assert reloaded.library_versions == outcome.artifact.library_versions

    def test_no_drift_is_reported_for_a_freshly_saved_model(self, tmp_path: Path) -> None:
        """Saving and loading inside one environment reports nothing changed."""
        outcome, _ = make_trained()
        path = save_artifact(outcome.artifact, tmp_path / "model.joblib")

        reloaded = load_artifact(path)

        assert reloaded.version_drift() == {}

    def test_drift_is_reported_when_a_version_moved(self) -> None:
        """A recorded version that no longer matches the environment is visible.

        The failure being guarded against is a model that keeps loading cleanly and starts
        scoring differently, with the cause recorded nowhere. Reporting the drift is enough
        to make the cause investigable.
        """
        outcome, _ = make_trained()
        artifact = replace(
            outcome.artifact,
            library_versions={**outcome.artifact.library_versions, "scikit-learn": "0.0.1"},
        )

        drift = artifact.version_drift()

        assert drift == {
            "scikit-learn": ("0.0.1", outcome.artifact.library_versions["scikit-learn"])
        }
        assert drift["scikit-learn"][1] != "0.0.1"

    def test_an_artifact_with_no_recorded_versions_reports_no_drift(self) -> None:
        """Absence of a record is not reported as a change.

        A model saved before this field existed should load, and should not claim that its
        environment moved when the only fact is that nobody wrote it down.
        """
        outcome, _ = make_trained()
        artifact = replace(outcome.artifact, library_versions={})

        assert artifact.version_drift() == {}

    def test_saving_preserves_the_training_environment(self, tmp_path: Path) -> None:
        """Re-saving keeps the environment the model was fitted in.

        A model relocated from one machine to another is re-saved by the new machine. If
        the save overwrote the recorded versions, the artifact would then describe an
        environment it was never fitted in - which is the precise claim the field exists to
        support.
        """
        outcome, _ = make_trained()
        first = save_artifact(outcome.artifact, tmp_path / "first.joblib")

        reloaded = load_artifact(first)
        second = save_artifact(
            replace(reloaded, library_versions={"scikit-learn": "0.0.1"}),
            tmp_path / "second.joblib",
        )
        again = load_artifact(second)

        assert again.library_versions == {"scikit-learn": "0.0.1"}


class TestArtifactConstruction:
    """An artifact validates its own contents at construction."""

    def test_an_unfitted_or_non_estimator_is_refused(self) -> None:
        """A ModelArtifact is a claim that a model exists."""
        outcome, _ = make_trained()
        artifact = outcome.artifact

        with pytest.raises(ValueError, match="predict"):
            ModelArtifact(
                estimator=object(),
                record=artifact.record,
                encoding_schema=artifact.encoding_schema,
                encoding_digest=artifact.encoding_digest,
            )

    def test_an_artifact_without_a_schema_is_refused(self) -> None:
        """A model that cannot describe its inputs should not be constructed."""
        outcome, _ = make_trained()
        artifact = outcome.artifact

        with pytest.raises(EncodingError, match="encoding schema is empty"):
            ModelArtifact(
                estimator=artifact.estimator,
                record=artifact.record,
                encoding_schema={},
                encoding_digest="",
            )
