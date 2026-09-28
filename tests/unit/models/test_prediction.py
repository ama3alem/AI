"""Tests for scoring a feature vector with a trained artifact (Phase 8).

Prediction is where a model's assumptions meet a real vector, and it is where the
avoidable damage happens: a vector scored against the wrong schema, a missing feature
filled in with a zero, a bare probability with no record of which model produced it. These
tests pin the three refusals that prevent those, and the traceability that makes a
prediction auditable after the fact.
"""

from __future__ import annotations

import pytest

from focus_engine.features import FeatureName
from focus_engine.models.encoding import encode_rows
from focus_engine.models.predictors import (
    DEFAULT_THRESHOLD,
    Prediction,
    PredictionRefusal,
    predict_with_artifact,
    score_batch,
)
from focus_engine.models.trainers import Algorithm, TrainingConfig, train_model
from focus_engine.schemas.primitives import DataOrigin, Provenance
from tests.unit.models.scenarios import make_labelled_rows, make_vector

pytestmark = pytest.mark.unit


def make_trained():
    """Train a model and return its artifact."""
    train = encode_rows(make_labelled_rows(60, positive=20, start_index=0, overlap=True))
    holdout = encode_rows(make_labelled_rows(20, positive=6, start_index=1000, overlap=True))
    config = TrainingConfig(
        algorithm=Algorithm.LOGISTIC_REGRESSION,
        model_id="risk-logreg",
        model_version="RISK_MODEL_V1",
        feature_set_version="FEATURE_SET_V1",
        dataset_version="DATASET_REAL_V1",
        target_definition_version="td-1.0.0",
        label_provenance=Provenance.OBSERVED,
        data_origin=DataOrigin.REAL,
        seed=4242,
    )
    return train_model(config, train, holdout).artifact


class TestScoring:
    """A valid vector produces a traceable probability."""

    def test_a_complete_vector_is_scored(self) -> None:
        """The ordinary case returns a prediction, not a refusal."""
        artifact = make_trained()

        outcome = predict_with_artifact(artifact, make_vector())

        assert isinstance(outcome, Prediction)
        assert 0.0 <= outcome.probability <= 1.0

    def test_the_prediction_carries_its_provenance(self) -> None:
        """A bare probability cannot be traced back to anything.

        Six months later, nobody can tell which model produced a number, against which
        feature set, or under which threshold. Each of those fields exists to answer that
        question.
        """
        artifact = make_trained()

        outcome = predict_with_artifact(artifact, make_vector())

        assert isinstance(outcome, Prediction)
        assert outcome.model_id == "risk-logreg"
        assert outcome.model_version == "RISK_MODEL_V1"
        assert outcome.feature_set_version == "FEATURE_SET_V1"
        assert outcome.encoding_digest == artifact.encoding_digest
        assert outcome.computed_at.tzinfo is not None

    def test_the_threshold_is_reported_and_honoured(self) -> None:
        """The operating point is a stated decision, not a hidden one."""
        artifact = make_trained()
        vector = make_vector()

        low = predict_with_artifact(artifact, vector, threshold=0.0)
        high = predict_with_artifact(artifact, vector, threshold=1.0)

        assert isinstance(low, Prediction) and isinstance(high, Prediction)
        assert low.is_positive is True
        assert high.is_positive is False
        assert low.threshold == 0.0
        assert high.threshold == 1.0

    def test_the_default_threshold_is_half(self) -> None:
        """The default is visible as a name, so a caller can disagree with it."""
        assert DEFAULT_THRESHOLD == 0.5

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_an_out_of_range_threshold_is_refused(self, threshold: float) -> None:
        """A threshold outside the probability range is a caller error."""
        artifact = make_trained()

        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            predict_with_artifact(artifact, make_vector(), threshold=threshold)

    def test_the_model_sees_the_same_columns_it_was_trained_on(self) -> None:
        """Scoring agrees with encoding, which is what makes the prediction meaningful.

        Comparing against a direct call on the training matrix is what makes this a real
        check: a column reordering would pass every other test in this file.
        """
        artifact = make_trained()
        vector = make_vector(overrides={FeatureName.PERFORMANCE_RECENT_ACCURACY: 0.42})
        matrix = encode_rows([(vector, 1)])

        outcome = predict_with_artifact(artifact, vector)
        direct = artifact.estimator.predict_proba(matrix.matrix)[0, 1]

        assert isinstance(outcome, Prediction)
        assert outcome.probability == pytest.approx(direct)


class TestRefusals:
    """A vector the model was not trained for is refused, with a reason."""

    def test_a_vector_with_an_absent_feature_is_refused(self) -> None:
        """A missing measurement is not a zero.

        Imputing one would produce a confident prediction from nothing, and the confidence
        would be indistinguishable from a real one.
        """
        artifact = make_trained()
        vector = make_vector(absent=(FeatureName.PERFORMANCE_RECENT_ACCURACY,))

        outcome = predict_with_artifact(artifact, vector)

        assert isinstance(outcome, PredictionRefusal)
        assert "performance_recent_accuracy" in outcome.missing_features
        assert "performance_recent_accuracy" in outcome.reason

    def test_a_refusal_names_the_learner_and_session(self) -> None:
        """A refusal is a reportable event, so it has to identify what it refused."""
        artifact = make_trained()
        vector = make_vector(
            learner_id="learner-4242",
            session_id="session-4242",
            absent=(FeatureName.INTERVENTION_COUNT,),
        )

        outcome = predict_with_artifact(artifact, vector)

        assert isinstance(outcome, PredictionRefusal)
        assert outcome.learner_id == "learner-4242"
        assert outcome.session_id == "session-4242"
        assert outcome.computed_at == vector.computed_at

    def test_several_absent_features_are_all_named(self) -> None:
        """Reporting one at a time would hide the scale of an encoding problem."""
        artifact = make_trained()
        vector = make_vector(
            absent=(
                FeatureName.PERFORMANCE_RECENT_ACCURACY,
                FeatureName.SESSION_EVENT_COUNT,
            )
        )

        outcome = predict_with_artifact(artifact, vector)

        assert isinstance(outcome, PredictionRefusal)
        assert set(outcome.missing_features) == {
            "performance_recent_accuracy",
            "session_event_count",
        }

    def test_a_vector_from_another_feature_set_is_refused(self) -> None:
        """Same column names, different quantities behind them.

        A version check has to come before any value check, because the values can be the
        right types and still not mean what the model learned.
        """
        artifact = make_trained()
        vector = make_vector()
        relabelled = vector.model_copy(update={"feature_set_version": "FEATURE_SET_V2"})

        outcome = predict_with_artifact(artifact, relabelled)

        assert isinstance(outcome, PredictionRefusal)
        assert "FEATURE_SET_V2" in outcome.reason
        assert "FEATURE_SET_V1" in outcome.reason

    def test_an_unattributed_vector_is_refused(self) -> None:
        """A vector nobody can attribute is not scored.

        It cannot be sliced per learner or traced back when the result is questioned, and
        accepting it would make part of the served set unaccountable.
        """
        artifact = make_trained()

        outcome = predict_with_artifact(artifact, make_vector(learner_id=None))

        assert isinstance(outcome, PredictionRefusal)
        assert "learner_id" in outcome.reason

    def test_an_unrecognised_category_is_refused(self) -> None:
        """A value outside the declared vocabulary has no column to go in."""
        artifact = make_trained()
        vector = make_vector(overrides={FeatureName.TRAJECTORY_ENCODED: "exploding"})

        with pytest.raises(Exception, match="outside its declared vocabulary"):
            predict_with_artifact(artifact, vector)


class TestBatchScoring:
    """A batch reports its refusals rather than hiding them."""

    def test_a_clean_batch_returns_only_predictions(self) -> None:
        """The ordinary case."""
        artifact = make_trained()
        vectors = tuple(make_vector(learner_id=f"learner-{i:04d}") for i in range(4))

        predictions, refusals = score_batch(artifact, vectors)

        assert len(predictions) == 4
        assert refusals == []

    def test_refusals_are_separated_from_predictions(self) -> None:
        """A batch with some unusable vectors reports both, in input order.

        Dropping the refusals silently would make a shrinking served population look like
        a working one.
        """
        artifact = make_trained()
        vectors = (
            make_vector(learner_id="learner-0001"),
            make_vector(
                learner_id="learner-0002",
                absent=(FeatureName.SESSION_EVENT_COUNT,),
            ),
            make_vector(learner_id="learner-0003"),
        )

        predictions, refusals = score_batch(artifact, vectors)

        assert len(predictions) == 2
        assert len(refusals) == 1
        assert refusals[0].learner_id == "learner-0002"

    def test_an_empty_batch_is_not_an_error(self) -> None:
        """Nothing to score is a valid, empty answer."""
        artifact = make_trained()

        predictions, refusals = score_batch(artifact, ())

        assert predictions == []
        assert refusals == []

    def test_a_batch_mixing_feature_set_versions_is_refused(self) -> None:
        """A batch spanning versions has no single schema.

        Scoring it anyway would either reject most of it or, worse, accept some rows under
        a schema they were not produced under.
        """
        artifact = make_trained()
        vectors = (
            make_vector(),
            make_vector().model_copy(update={"feature_set_version": "FEATURE_SET_V2"}),
        )

        with pytest.raises(ValueError, match="feature-set versions"):
            score_batch(artifact, vectors)

    def test_batch_predictions_match_single_predictions(self) -> None:
        """Batching is an optimisation, not a different computation.

        Without this, a batch path could quietly disagree with the single path and the
        disagreement would only show up in whichever one production happened to use.
        """
        artifact = make_trained()
        vectors = tuple(make_vector(learner_id=f"learner-{i:04d}") for i in range(3))

        batched, _ = score_batch(artifact, vectors)
        individually = [predict_with_artifact(artifact, vector) for vector in vectors]

        assert [p.probability for p in batched] == [p.probability for p in individually]
