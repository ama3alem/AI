"""Integration tests for the evaluation layer (Phase 13).

These tests exercise the full chain: prediction horizon, admissible event
filtering, ground truth derivation, evaluation record construction, and
aggregation. They use the real event and temporal types and avoid mocks,
preferring deterministic synthetic data.

Constraints respected:
- Ground truth provenance is ``Provenance.OBSERVED`` (never fabricated as GROUND_TRUTH).
- ``INSUFFICIENT_DATA`` and ``NOT_ASSESSABLE`` remain distinct non-findings.
- The half-open window ``[predicted_at, predicted_at + horizon)`` is enforced.
- STATE/DIRECTION predictions can be assessed without producing binary cells.
- DECLINE predictions populate the confusion matrix only for the DECLINE target.
- Future evidence outside the horizon is ignored; evidence exactly at the end is excluded.
- Evidence exactly at the prediction instant is included.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pytest

from focus_engine.baseline.models import (
    BaselineDimension,
    BaselineMaturity,
    DeviationResult,
    InferenceBasis,
    ReferenceSource,
)
from focus_engine.configuration.thresholds import EvaluationSettings
from focus_engine.evaluation import (
    BinaryVerdict,
    EvaluationEngine,
    EvaluationVerdict,
    GroundTruth,
    GroundTruthStatus,
    PredictionHorizon,
    PredictionSnapshot,
    PredictionTarget,
    summarise,
)
from focus_engine.events.types import EventEnvelope, EventType, QuestionAnsweredPayload
from focus_engine.schemas.primitives import (
    BehavioralEngagementState,
    DataOrigin,
    Provenance,
    SyntheticDataStamp,
)
from focus_engine.temporal import TemporalEngine, TemporalObservation
from focus_engine.temporal.models import TemporalState, TrendDirection
from focus_engine.uncertainty.outcomes import Verdict
from focus_engine.utils.clock import FixedClock

pytestmark = pytest.mark.integration

PREDICTED_AT = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
HORIZON_S = 600
LEARNER = "learner-0001"
SESSION = "session-0001"
MODEL_ID = "model-hgb-v1"
MODEL_VERSION = "MODEL_HGB_V1"
FEATURE_SET_VERSION = "FEATURE_SET_V1"


def _horizon() -> PredictionHorizon:
    return PredictionHorizon(predicted_at=PREDICTED_AT, duration_seconds=HORIZON_S)


def _stamp() -> SyntheticDataStamp:
    """The in-band warning stamp every synthetic event is required to carry."""
    return SyntheticDataStamp(generator="focus_engine.evaluation.integration", seed=13)


def _qevent(
    offset_s: float,
    ordinal: int = 1,
    origin: DataOrigin = DataOrigin.SYNTHETIC,
    *,
    learner_id: str = LEARNER,
    session_id: str = SESSION,
) -> EventEnvelope:
    """Build one real ``EventEnvelope`` at a fixed offset from the prediction instant.

    Uses the repository's real event envelope, real payload, and real validation: the
    synthetic stamp is mandatory for synthetic events and ``provenance`` must not be
    ``OBSERVED`` for them, both enforced by the envelope's own validators.

    ``learner_id`` and ``session_id`` are overridable so an isolation test can build a
    foreign event through the same validated path rather than a hand-rolled literal.
    """
    return EventEnvelope(
        event_id=f"ev-{abs(int(offset_s)):06d}-{ordinal:03d}",
        learner_id=learner_id,
        session_id=session_id,
        timestamp=PREDICTED_AT + timedelta(seconds=offset_s),
        event_type=EventType.QUESTION_ANSWERED,
        payload=QuestionAnsweredPayload(
            question_id=f"question-{ordinal:04d}",
            correct=True,
            response_seconds=20.0,
        ),
        origin=origin,
        provenance=Provenance.OBSERVED if origin is DataOrigin.REAL else Provenance.SYNTHETIC_LABEL,
        synthetic_stamp=None if origin is DataOrigin.REAL else _stamp(),
    )


def _tstate(state: BehavioralEngagementState, direction: TrendDirection) -> TemporalState:
    return cast(
        TemporalState,
        SimpleNamespace(
            learner_id=LEARNER,
            reference_time=PREDICTED_AT + timedelta(seconds=300),
            state=state,
            direction=direction,
        ),
    )


def _decline_snapshot(pred_positive: bool = True) -> PredictionSnapshot:
    return PredictionSnapshot(
        prediction_id="pred-0001",
        learner_id=LEARNER,
        session_id=SESSION,
        target=PredictionTarget.DECLINE,
        horizon=_horizon(),
        predicted_positive=pred_positive,
        verdict=Verdict.RESOLVED,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        feature_set_version=FEATURE_SET_VERSION,
        target_definition_version="target-engage-v1",
        data_origin=DataOrigin.SYNTHETIC,
    )


def _state_snapshot(state: BehavioralEngagementState) -> PredictionSnapshot:
    return PredictionSnapshot(
        prediction_id="pred-s1",
        learner_id=LEARNER,
        session_id=SESSION,
        target=PredictionTarget.STATE,
        horizon=_horizon(),
        predicted_state=state,
        verdict=Verdict.RESOLVED,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        feature_set_version=FEATURE_SET_VERSION,
        target_definition_version="target-engage-v1",
        data_origin=DataOrigin.SYNTHETIC,
    )


def _dir_snapshot(direction: TrendDirection) -> PredictionSnapshot:
    return PredictionSnapshot(
        prediction_id="pred-d1",
        learner_id=LEARNER,
        session_id=SESSION,
        target=PredictionTarget.DIRECTION,
        horizon=_horizon(),
        predicted_direction=direction,
        verdict=Verdict.RESOLVED,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        feature_set_version=FEATURE_SET_VERSION,
        target_definition_version="target-engage-v1",
        data_origin=DataOrigin.SYNTHETIC,
    )


class TestFullChain:
    def test_decline_chain_binary_matrix(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        rec = eng.evaluate(snap, events, reading)
        assert rec.verdict is EvaluationVerdict.CORRECT
        assert rec.binary_verdict is BinaryVerdict.TRUE_POSITIVE
        assert rec.ground_truth_status is GroundTruthStatus.CONFIRMED
        assert rec.evidence_count == 4
        assert rec.observed_positive is True
        assert rec.observed_state is None
        assert rec.observed_direction is None
        assert rec.data_origin is DataOrigin.SYNTHETIC
        truth = eng.ground_truth(snap, events, reading)
        assert truth.provenance is Provenance.OBSERVED
        assert truth.evidence_count == 4
        s = summarise([rec])
        assert s.assessed == 1
        assert s.confirmed == 1
        assert s.confusion.true_positive == 1

    def test_state_assessable_not_binary(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _state_snapshot(BehavioralEngagementState.DECLINING)
        events = [_qevent(1.0), _qevent(2.0), _qevent(3.0), _qevent(4.0)]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.STABLE)
        rec = eng.evaluate(snap, events, reading)
        assert rec.verdict is EvaluationVerdict.CORRECT
        assert rec.binary_verdict is BinaryVerdict.NOT_ASSESSABLE
        assert rec.observed_state is BehavioralEngagementState.DECLINING
        assert rec.observed_direction is None
        assert rec.observed_positive is None
        assert rec.ground_truth_status is GroundTruthStatus.CONFIRMED

    def test_direction_assessable_not_binary(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _dir_snapshot(TrendDirection.WORSENING)
        events = [_qevent(5.0), _qevent(15.0), _qevent(25.0), _qevent(35.0)]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        rec = eng.evaluate(snap, events, reading)
        assert rec.verdict is EvaluationVerdict.CORRECT
        assert rec.binary_verdict is BinaryVerdict.NOT_ASSESSABLE
        assert rec.observed_direction is TrendDirection.WORSENING
        assert rec.observed_state is None


class TestIsolationAndProvenance:
    def test_learner_isolation_ignores_other_learner(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=False)
        events = [
            _qevent(10.0),
            _qevent(20.0),
            _qevent(
                30.0,
                learner_id="other-learner",
            ),
            _qevent(40.0),
        ]
        reading = _tstate(BehavioralEngagementState.STABLE, TrendDirection.STABLE)
        rec = eng.evaluate(snap, events, reading)
        assert rec.evidence_count == 3

    def test_session_isolation(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [
            _qevent(10.0),
            _qevent(
                20.0,
                session_id="other-session",
            ),
            _qevent(30.0),
            _qevent(40.0),
        ]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        rec = eng.evaluate(snap, events, reading)
        assert rec.evidence_count == 3

    def test_provenance_observed_on_truth_and_no_ground_truth(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        truth = eng.ground_truth(snap, events, reading)
        assert truth.provenance is Provenance.OBSERVED
        rec = eng.evaluate(snap, events, reading)
        assert rec.data_origin is DataOrigin.SYNTHETIC


class TestHalfOpenWindow:
    """The window is ``[predicted_at, predicted_at + horizon)``.

    These four cases pin the exact boundary semantics. They use the repository's
    own timestamp resolution (a single instant, not a new precision), and they
    use a one-microsecond offset from the end to prove exclusivity.
    """

    def test_event_at_prediction_instant_included(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [
            _qevent(0.0),
            _qevent(100.0),
            _qevent(200.0),
            _qevent(300.0),
        ]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        truth = eng.ground_truth(snap, events, reading)
        assert len(truth.evidence_ids) == 4

    def test_event_at_horizon_end_excluded(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [
            _qevent(10.0),
            _qevent(100.0),
            _qevent(200.0),
            _qevent(float(HORIZON_S)),
        ]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        truth = eng.ground_truth(snap, events, reading)
        assert len(truth.evidence_ids) == 3

    def test_event_one_microsecond_before_end_included(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [
            _qevent(10.0),
            _qevent(100.0),
            _qevent(200.0),
            _qevent(HORIZON_S - 1e-6),
        ]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        truth = eng.ground_truth(snap, events, reading)
        assert len(truth.evidence_ids) == 4

    def test_event_one_microsecond_after_end_excluded(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [
            _qevent(10.0),
            _qevent(100.0),
            _qevent(200.0),
            _qevent(HORIZON_S + 1e-6),
        ]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        truth = eng.ground_truth(snap, events, reading)
        assert len(truth.evidence_ids) == 3


class TestNonFindings:
    def test_insufficient_data_distinct_from_not_assessable(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)

        thin = eng.ground_truth(snap, [_qevent(10.0), _qevent(20.0)], None)
        assert thin.status is GroundTruthStatus.INSUFFICIENT_DATA
        assert thin.observed_positive is None
        assert thin.evidence_count == 2

        # Too little activity stays INSUFFICIENT_DATA even when a reading is supplied:
        # the deficit is evidence volume, not the absence of a characterisation.
        thin_with_reading = eng.ground_truth(
            snap,
            [_qevent(10.0), _qevent(20.0)],
            _tstate(BehavioralEngagementState.STABLE, TrendDirection.STABLE),
        )
        assert thin_with_reading.status is GroundTruthStatus.INSUFFICIENT_DATA
        assert thin_with_reading.observed_positive is None
        assert thin_with_reading.evidence_count == 2

        # Ample activity but no reading is NOT_ASSESSABLE: the events were not turned
        # into a characterisation. The two non-findings are never conflated.
        rich_but_no_reading = eng.ground_truth(
            snap, [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)], None
        )
        assert rich_but_no_reading.status is GroundTruthStatus.NOT_ASSESSABLE
        assert rich_but_no_reading.observed_positive is None
        assert rich_but_no_reading.evidence_count == 4

    def test_contradictory_reading_refused(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        # A reading for a different learner is contradictory and must be refused.
        wrong = cast(
            TemporalState,
            SimpleNamespace(
                learner_id="someone-else-x",
                reference_time=PREDICTED_AT + timedelta(seconds=300),
                state=BehavioralEngagementState.DECLINING,
                direction=TrendDirection.WORSENING,
            ),
        )
        truth = eng.ground_truth(snap, events, wrong)
        assert truth.status is GroundTruthStatus.NOT_ASSESSABLE
        assert truth.observed_positive is None

    def test_reading_anchored_before_prediction_refused(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        before = cast(
            TemporalState,
            SimpleNamespace(
                learner_id=LEARNER,
                reference_time=PREDICTED_AT - timedelta(seconds=1),
                state=BehavioralEngagementState.DECLINING,
                direction=TrendDirection.WORSENING,
            ),
        )
        truth = eng.ground_truth(snap, events, before)
        assert truth.status is GroundTruthStatus.NOT_ASSESSABLE

    def test_insufficient_data_reading_refused(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        weak = _tstate(BehavioralEngagementState.INSUFFICIENT_DATA, TrendDirection.STABLE)
        truth = eng.ground_truth(snap, events, weak)
        assert truth.status is GroundTruthStatus.NOT_ASSESSABLE
        assert truth.observed_positive is None


class TestConfigurationAndDeterminism:
    def test_min_evidence_floor_is_respected(self) -> None:
        eng = EvaluationEngine(
            settings=EvaluationSettings(min_evidence_for_evaluation=10),
            clock=FixedClock(PREDICTED_AT),
        )
        snap = _decline_snapshot(pred_positive=True)
        events = [_qevent(float(10 * i)) for i in range(1, 6)]
        truth = eng.ground_truth(snap, events, None)
        assert truth.status is GroundTruthStatus.INSUFFICIENT_DATA
        assert truth.evidence_count == 5

    def test_same_inputs_produce_identical_records(self) -> None:
        snap = _decline_snapshot(pred_positive=True)
        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        a = EvaluationEngine(clock=FixedClock(PREDICTED_AT)).evaluate(snap, events, reading)
        b = EvaluationEngine(clock=FixedClock(PREDICTED_AT)).evaluate(snap, events, reading)
        assert a == b
        assert a.model_dump_json() == b.model_dump_json()

    def test_evaluate_many_is_deterministic(self) -> None:
        snap = _decline_snapshot(pred_positive=True)
        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        scored = [(snap, reading), (snap, reading)]
        first = eng.evaluate_many(scored, events)
        second = eng.evaluate_many(scored, events)
        assert [r.model_dump_json() for r in first] == [r.model_dump_json() for r in second]

    def test_synthetic_evidence_never_reported_as_real(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        snap = _decline_snapshot(pred_positive=True)
        real_but_one_synthetic = [
            _qevent(10.0, origin=DataOrigin.REAL),
            _qevent(20.0, origin=DataOrigin.REAL),
            _qevent(30.0, origin=DataOrigin.REAL),
            _qevent(40.0, origin=DataOrigin.SYNTHETIC),
        ]
        reading = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        truth = eng.ground_truth(snap, real_but_one_synthetic, reading)
        assert truth.data_origin is DataOrigin.SYNTHETIC


class TestConfusionMatrix:
    """All four DECLINE cells, so the matrix is never only exercised in its happy path."""

    def test_all_four_cells_aggregate_correctly(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        declining = _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
        stable = _tstate(BehavioralEngagementState.STABLE, TrendDirection.STABLE)

        cases = [
            (True, declining, BinaryVerdict.TRUE_POSITIVE, GroundTruthStatus.CONFIRMED),
            (True, stable, BinaryVerdict.FALSE_POSITIVE, GroundTruthStatus.REFUTED),
            (False, declining, BinaryVerdict.FALSE_NEGATIVE, GroundTruthStatus.REFUTED),
            (False, stable, BinaryVerdict.TRUE_NEGATIVE, GroundTruthStatus.CONFIRMED),
        ]
        records = [
            eng.evaluate(_decline_snapshot(pred_positive=pred), events, reading)
            for pred, reading, _, _ in cases
        ]
        for rec, (_, _, expected_binary, expected_status) in zip(records, cases, strict=True):
            assert rec.binary_verdict is expected_binary
            # The status describes agreement with the prediction, not the observation:
            # a false positive is a prediction the evidence refuted.
            assert rec.ground_truth_status is expected_status
            assert rec.observed_positive is not None

        s = summarise(records)
        assert s.total_records == 4
        assert s.assessed == 4
        assert s.confirmed == 2
        assert s.refuted == 2
        c = s.confusion
        assert (c.true_positive, c.false_positive) == (1, 1)
        assert (c.false_negative, c.true_negative) == (1, 1)
        assert s.accuracy == 0.5
        assert s.precision == 0.5
        assert s.recall == 0.5


class TestMetadataIntegrity:
    def test_historical_prediction_metadata_cannot_be_rewritten(self) -> None:
        """A prediction is a historical fact; its recorded attributes are frozen."""
        snap = _decline_snapshot(pred_positive=True)
        original = snap.model_dump()
        for field, bad in (
            ("predicted_positive", False),
            ("learner_id", "someone-else-x"),
            ("predicted_at", PREDICTED_AT + timedelta(days=1)),
            ("model_version", "OTHER"),
        ):
            with pytest.raises(Exception, match="frozen"):
                setattr(snap, field, bad)
        assert snap.model_dump() == original

    def test_ground_truth_provenance_cannot_be_asserted(self) -> None:
        """Ground truth is observed, never asserted by the evaluator.

        Takes a genuine engine-produced ground truth and re-validates it with the
        provenance swapped. The model constrains the field to ``OBSERVED``
        unconditionally, so ``GROUND_TRUTH`` is unconstructable rather than merely
        discouraged: independence is not available to this repository to claim.
        """
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        truth = eng.ground_truth(
            _decline_snapshot(pred_positive=True),
            [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)],
            _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING),
        )
        assert truth.provenance is Provenance.OBSERVED

        forged = {**truth.model_dump(), "provenance": Provenance.GROUND_TRUTH}
        with pytest.raises(Exception, match="provenance|OBSERVED|observed"):
            GroundTruth.model_validate(forged)

    def test_engine_never_emits_ground_truth_provenance(self) -> None:
        eng = EvaluationEngine(clock=FixedClock(PREDICTED_AT))
        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        for snap in (
            _decline_snapshot(pred_positive=True),
            _state_snapshot(BehavioralEngagementState.DECLINING),
            _dir_snapshot(TrendDirection.WORSENING),
        ):
            truth = eng.ground_truth(
                snap, events, _tstate(BehavioralEngagementState.DECLINING, TrendDirection.WORSENING)
            )
            assert truth.provenance is Provenance.OBSERVED


class TestRealTemporalStateSeam:
    """The evaluator is handed a state the temporal layer actually produced.

    The other tests in this module supply a reading directly, which proves the
    evaluator's own logic but not that the two layers agree on a shape. This one runs
    the real ``TemporalEngine`` over real deviations until it characterises the learner,
    then scores that genuine ``TemporalState``. If the evaluator could only consume a
    stand-in, this would fail.
    """

    def test_evaluator_accepts_a_genuine_temporal_state(self) -> None:
        temporal = TemporalEngine(clock=FixedClock(PREDICTED_AT))
        state = temporal.create_state(LEARNER, PREDICTED_AT)

        # A sustained run of large negative deviations is what moves a learner out of
        # cold start into a characterised state; this is the real input path, not a
        # hand-set field. Which characterisation results is the temporal layer's call,
        # not this test's, so the assertion below checks that the state is a real
        # characterisation rather than pinning a particular one.
        for step in range(8):
            state = temporal.ingest(
                state,
                TemporalObservation(
                    learner_id=LEARNER,
                    computed_at=PREDICTED_AT + timedelta(seconds=30 * (step + 1)),
                    deviations=(
                        DeviationResult(
                            dimension=BaselineDimension.ACCURACY,
                            value=0.1,
                            centre=0.7,
                            spread=0.15,
                            absolute=-0.6,
                            standardised=-4.0,
                            maturity=BaselineMaturity.ESTABLISHED,
                            basis=InferenceBasis.PERSONAL,
                            source=ReferenceSource.PERSONAL_HISTORY,
                            computed_at=PREDICTED_AT + timedelta(seconds=30 * (step + 1)),
                        ),
                    ),
                    origins=frozenset({DataOrigin.SYNTHETIC}),
                ),
            )

        assert state.state is not BehavioralEngagementState.INSUFFICIENT_DATA, (
            f"the real temporal engine never left cold start: {state.evidence.reason!r}. A "
            "state this layer cannot produce could never be evaluated in production either."
        )

        events = [_qevent(10.0), _qevent(20.0), _qevent(30.0), _qevent(40.0)]
        truth = EvaluationEngine(clock=FixedClock(PREDICTED_AT)).ground_truth(
            _state_snapshot(state.state), events, state
        )
        # The real state anchored at the last ingestion, which falls inside the window.
        assert truth.status is GroundTruthStatus.CONFIRMED
        assert truth.observed_state is state.state
        assert truth.provenance is Provenance.OBSERVED
