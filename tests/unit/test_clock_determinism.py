"""Regression tests: the modelling and uncertainty layers must honour an injected clock.

Both layers used to stamp their output from ``utc_now()`` — a direct wall-clock read —
while every other time-aware layer in the package took a :class:`Clock`. The effect was
that a replay over identical events produced identical probabilities but different
timestamps, so an auditable record of a decision could not be reproduced and a replay
could not prove it had replayed the same decision.

The fix is not a new clock. It is the existing one: the prediction is dated by the
vector's own reference time, which is what the two refusal branches in the same function
already did, and the uncertainty engine takes the same ``clock`` field the temporal,
policy and outcome engines already take.

These tests assert the property the fix exists to guarantee: same inputs plus same
injected clock means identical timestamps, and a different clock means a different
timestamp. A test that only asserted the first would still pass if the layer had
stopped stamping a time at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from focus_engine.models.predictors import Prediction, predict_with_artifact
from focus_engine.uncertainty import UncertaintyEngine
from focus_engine.utils.clock import FixedClock
from tests.unit.models.scenarios import make_vector
from tests.unit.uncertainty.scenarios import evidence_at, train_shared_artifact

pytestmark = pytest.mark.unit

INJECTED = datetime(2026, 6, 1, 9, 0, 0, tzinfo=UTC)
LATER = INJECTED + timedelta(days=3650)


@pytest.fixture(scope="module")
def artifact():
    """A trained artifact, so the prediction path is the real one."""
    return train_shared_artifact()


class TestPredictionRespectsTheReferenceTime:
    """`predict_with_artifact` is dated by the vector it scored."""

    def test_two_runs_over_one_vector_agree_on_the_timestamp(self, artifact) -> None:
        """The same vector scored twice carries the same instant.

        This is the regression itself. Before the fix the two calls differed by however
        long the test took to run, and nothing else about them differed.
        """
        vector = make_vector()

        first = predict_with_artifact(artifact, vector)
        second = predict_with_artifact(artifact, vector)

        assert isinstance(first, Prediction)
        assert isinstance(second, Prediction)
        assert first.computed_at == second.computed_at

    def test_the_timestamp_is_the_vectors_own_reference_time(self, artifact) -> None:
        """The prediction is dated by its evidence, not by when scoring happened."""
        vector = make_vector()

        outcome = predict_with_artifact(artifact, vector)

        assert isinstance(outcome, Prediction)
        assert outcome.computed_at == vector.computed_at

    def test_refusals_and_predictions_agree_on_where_the_time_comes_from(self, artifact) -> None:
        """Both return paths in the function date from the same source.

        The refusal paths already used the vector's time while the success path used the
        wall clock, so the same function could report two different instants for the same
        vector depending only on whether the model had an opinion.
        """
        from focus_engine.features import FeatureName

        vector = make_vector()
        incomplete = make_vector(absent=(FeatureName.PERFORMANCE_RECENT_ACCURACY,))

        predicted = predict_with_artifact(artifact, vector)
        refused = predict_with_artifact(artifact, incomplete)

        assert isinstance(predicted, Prediction)
        assert predicted.computed_at == vector.computed_at
        assert refused.computed_at == incomplete.computed_at

    def test_moving_the_vector_moves_the_prediction_timestamp(self, artifact) -> None:
        """A later vector yields a later prediction, so the stamp tracks the evidence."""
        early = make_vector()
        late = make_vector(offset_seconds=3600.0)

        first = predict_with_artifact(artifact, early)
        second = predict_with_artifact(artifact, late)

        assert isinstance(first, Prediction)
        assert isinstance(second, Prediction)
        assert first.computed_at < second.computed_at


class TestUncertaintyRespectsTheInjectedClock:
    """`UncertaintyEngine` stamps outcomes from its clock, not the host's."""

    def _assess(self, engine: UncertaintyEngine, artifact, vector):
        """Score and assess in one step, the way a caller would."""
        prediction = predict_with_artifact(artifact, vector)
        return engine.assess(
            prediction, artifact=artifact, evidence=evidence_at(600), vector=vector
        )

    def test_two_runs_with_one_clock_agree_on_the_timestamp(self, artifact) -> None:
        """Same artifact, same vector, same clock: the same instant.

        This is the regression for the outcome side, and it needed the engine to gain a
        ``clock`` field to be expressible at all.
        """
        vector = make_vector()
        engine = UncertaintyEngine(clock=FixedClock(INJECTED))

        first = self._assess(engine, artifact, vector)
        second = self._assess(engine, artifact, vector)

        assert first.computed_at == INJECTED
        assert second.computed_at == INJECTED

    def test_a_different_clock_produces_a_different_timestamp(self, artifact) -> None:
        """The clock is actually read, rather than ignored or hardcoded.

        Without this half the suite would still pass if the field were accepted and
        discarded, which is the failure mode a clock parameter actually has in practice.
        """
        vector = make_vector()

        first = self._assess(UncertaintyEngine(clock=FixedClock(INJECTED)), artifact, vector)
        second = self._assess(UncertaintyEngine(clock=FixedClock(LATER)), artifact, vector)

        assert first.computed_at == INJECTED
        assert second.computed_at == LATER
        assert first.computed_at != second.computed_at

    def test_the_default_engine_still_reads_a_real_instant(self, artifact) -> None:
        """The default remains a working clock rather than becoming ``None``.

        The production default is ``SystemClock``, so an outcome produced with no
        configuration is still dated. This is what keeps the change from having made the
        timestamp optional in practice.
        """
        vector = make_vector()

        outcome = self._assess(UncertaintyEngine(), artifact, vector)

        assert outcome.computed_at is not None
        assert outcome.computed_at.tzinfo is not None
        assert outcome.computed_at.astimezone(UTC) <= datetime.now(UTC)

    def test_advancing_the_clock_advances_the_stamp(self, artifact) -> None:
        """A mutable clock moves the outcome forward, so a replay can be told apart."""
        vector = make_vector()
        clock = FixedClock(INJECTED)
        engine = UncertaintyEngine(clock=clock)

        first = self._assess(engine, artifact, vector)
        clock.advance(timedelta(minutes=30))
        second = self._assess(engine, artifact, vector)

        assert first.computed_at == INJECTED
        assert second.computed_at == INJECTED + timedelta(minutes=30)


class TestTheWholeInferenceIsReplayable:
    """The property the two fixes exist to deliver, end to end."""

    def test_prediction_and_outcome_come_from_one_instant(self, artifact) -> None:
        """Scoring and assessing the same vector share the injected clock's instant.

        The prediction is dated by the vector and the outcome by the clock, so for them to
        agree the two have to be the same instant. That is the invariant an auditable
        replay depends on: one decision, one time, whatever layer is asked.
        """
        vector = make_vector()
        clock = FixedClock(vector.computed_at)

        prediction = predict_with_artifact(artifact, vector)
        outcome = UncertaintyEngine(clock=clock).assess(
            prediction, artifact=artifact, evidence=evidence_at(600), vector=vector
        )

        assert isinstance(prediction, Prediction)
        assert prediction.computed_at == outcome.computed_at
