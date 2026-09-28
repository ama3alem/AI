"""Shared builders for the evaluation tests (Phase 13).

A prediction is a statement about what will happen next; an evaluation checks what actually
happened. These builders construct the inputs the evaluation engine needs: events in a
session, a temporal reading of the learner at a point in time, and a prediction wrapping a
recorded probability.

**Every default is synthetic.** Real data is marked explicitly, so a test that forgets to
choose gets the safe label rather than the dangerous one.

**Every stand-in carries only the attributes the evaluation layer reads.** The temporal
state is the same cast-from-SimpleNamespace pattern the outcome tests use: the real
TemporalState is tested in the temporal layer's own tests, and building one here would re-test
it from inside a boundary test.

The engine's own leakage test requires a prediction made *before* events arrive. Every
prediction is therefore anchored at ``PREDICTED_AT`` and the events follow it, so the window
is always well-defined.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Final

from focus_engine.evaluation.models import (
    PredictionHorizon,
    PredictionSnapshot,
    PredictionTarget,
)
from focus_engine.events.types import EventEnvelope, EventType, QuestionAnsweredPayload
from focus_engine.schemas.primitives import (
    BehavioralEngagementState,
    ConfidenceLevel,
    DataOrigin,
    InferenceBasis,
    Provenance,
    SyntheticDataStamp,
)
from focus_engine.temporal.models import TrendDirection
from focus_engine.uncertainty import ConfidenceAssessment, Constraint
from focus_engine.uncertainty.outcomes import PredictionOutcome, Verdict

#: A fixed instant in UTC, so a failure message reads as a date rather than as "now".
PREDICTED_AT: Final[datetime] = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)

LEARNER: Final[str] = "learner-0001"
SESSION: Final[str] = "session-0001"
MODEL_ID: Final[str] = "model-hgb-v1"
MODEL_VERSION: Final[str] = "MODEL_HGB_V1"
FEATURE_SET_VERSION: Final[str] = "FEATURE_SET_V1"

#: How far ahead the prediction claims to be about.
HORIZON_SECONDS: Final[int] = 600  # 10 minutes


def _assessment(value: float = 0.6) -> ConfidenceAssessment:
    return ConfidenceAssessment(
        value=value,
        level=ConfidenceLevel.MEDIUM,
        model_assertion=value,
        calibration_ceiling=1.0,
        evidence_ceiling=1.0,
        maturity_ceiling=1.0,
        binding=Constraint.MODEL_ASSERTION,
        explanation_capped=False,
    )


def horizon(duration_seconds: int = HORIZON_SECONDS) -> PredictionHorizon:
    """Build a horizon anchored at ``PREDICTED_AT``."""
    return PredictionHorizon(predicted_at=PREDICTED_AT, duration_seconds=duration_seconds)


def activity_event(
    offset_seconds: float,
    *,
    learner_id: str = LEARNER,
    session_id: str = SESSION,
    event_id: str | None = None,
    event_type: EventType = EventType.QUESTION_ANSWERED,
    origin: DataOrigin = DataOrigin.SYNTHETIC,
    ordinal: int = 1,
    **payload_kwargs: Any,
) -> EventEnvelope:
    """Build one activity event at a fixed offset from ``PREDICTED_AT``.

    Args:
        offset_seconds: Seconds relative to ``PREDICTED_AT``. Must be >= 0 to sit inside the
            default half-open window ``[PREDICTED_AT, PREDICTED_AT + horizon)``.
        learner_id: The learner the event belongs to.
        session_id: The session the event belongs to.
        event_id: Optional event id override.
        event_type: The event type.
        origin: Whether the event is real or synthetic.
        ordinal: An index used to build distinct content and question identifiers.
        payload_kwargs: Per-type field overrides.

    Returns:
        The envelope.
    """
    payload: object
    if event_type is EventType.QUESTION_ANSWERED:
        payload = QuestionAnsweredPayload(
            question_id=f"question-{ordinal:04d}",
            correct=payload_kwargs.get("correct", True),
            response_seconds=float(payload_kwargs.get("response_seconds", 20.0)),
        )
    else:
        raise NotImplementedError(f"event type {event_type!r} has no builder yet")
    resolved_id = (
        event_id or f"event-{abs(int(offset_seconds)):06d}-{event_type.value}-{ordinal:03d}"
    )
    return EventEnvelope(
        event_id=resolved_id,
        learner_id=learner_id,
        session_id=session_id,
        timestamp=PREDICTED_AT + timedelta(seconds=offset_seconds),
        event_type=event_type,
        payload=payload,
        origin=origin,
        provenance=Provenance.OBSERVED if origin is DataOrigin.REAL else Provenance.SYNTHETIC_LABEL,
        synthetic_stamp=(
            SyntheticDataStamp(generator="test-evaluation", seed=12345)
            if origin is DataOrigin.SYNTHETIC
            else None
        ),
    )


def activity_events(
    count: int,
    *,
    learner_id: str = LEARNER,
    session_id: str = SESSION,
    origin: DataOrigin = DataOrigin.SYNTHETIC,
) -> list[EventEnvelope]:
    """Build ``count`` activity events evenly spaced inside the window.

    The first event sits at 1 second after ``PREDICTED_AT`` so it is always admissible, and
    the last is well before the boundary.
    """
    if count < 1:
        return []
    end = PREDICTED_AT + timedelta(seconds=HORIZON_SECONDS)
    span = (end - PREDICTED_AT).total_seconds()
    step = span / (count + 1)
    return [
        activity_event(
            step * (i + 1),
            learner_id=learner_id,
            session_id=session_id,
            origin=origin,
            ordinal=i + 1,
        )
        for i in range(count)
    ]


def temporal_reading(
    state: BehavioralEngagementState,
    direction: TrendDirection,
    *,
    reference_time: datetime | None = None,
    learner_id: str = LEARNER,
) -> Any:
    """Build a stand-in temporal reading carrying the attributes the evaluation engine reads.

    Follows the same cast-from-SimpleNamespace convention the outcome scenarios use: the real
    TemporalState is tested in the temporal layer's own tests, and building one here would
    re-test it from inside a boundary test. Only the fields the evaluation engine touches are
    present.

    Args:
        state: The behavioural engagement state.
        direction: The temporal direction.
        reference_time: When the state speaks from. Defaults to 5 minutes after prediction.
        learner_id: The learner the state describes.

    Returns:
        A stand-in exposing ``state``, ``direction``, ``learner_id``, and ``reference_time``.
    """
    return SimpleNamespace(
        state=state,
        direction=direction,
        learner_id=learner_id,
        reference_time=(
            reference_time if reference_time is not None else PREDICTED_AT + timedelta(seconds=300)
        ),
    )


def resolved_outcome(
    is_positive: bool,
    *,
    probability: float = 0.82,
    threshold: float = 0.5,
) -> PredictionOutcome:
    """A RESOLVED prediction outcome for the DECLINE target."""
    return PredictionOutcome(
        verdict=Verdict.RESOLVED,
        probability=probability,
        threshold=threshold,
        is_positive=is_positive,
        confidence=ConfidenceLevel.MEDIUM,
        assessment=_assessment(),
        basis=InferenceBasis.PERSONAL,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        feature_set_version=FEATURE_SET_VERSION,
        target_definition_version="target-engage-v1",
        data_origin=DataOrigin.SYNTHETIC,
        computed_at=PREDICTED_AT,
    )


def snapshot_from_outcome(
    is_positive: bool,
    *,
    prediction_id: str = "pred-0001",
    probability: float = 0.82,
) -> PredictionSnapshot:
    """A snapshot wrapping a resolved DECLINE outcome."""
    return PredictionSnapshot.from_outcome(
        resolved_outcome(is_positive, probability=probability),
        prediction_id=prediction_id,
        learner_id=LEARNER,
        session_id=SESSION,
        horizon=horizon(),
    )


def snapshot_from_state(
    state: BehavioralEngagementState,
    direction: TrendDirection,
    *,
    target: PredictionTarget = PredictionTarget.STATE,
    prediction_id: str = "pred-0001",
) -> PredictionSnapshot:
    """A snapshot built directly from a temporal reading, for STATE/DIRECTION targets."""
    predicted_state = state if target is PredictionTarget.STATE else None
    predicted_direction = direction if target is PredictionTarget.DIRECTION else None
    return PredictionSnapshot(
        prediction_id=prediction_id,
        learner_id=LEARNER,
        session_id=SESSION,
        target=target,
        horizon=horizon(),
        predicted_state=predicted_state,
        predicted_direction=predicted_direction,
        verdict=Verdict.RESOLVED,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        feature_set_version=FEATURE_SET_VERSION,
        target_definition_version="target-engage-v1",
        data_origin=DataOrigin.SYNTHETIC,
    )
