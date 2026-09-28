"""Evaluation leakage and half-open window boundary tests."""

from __future__ import annotations

import pytest

from focus_engine.evaluation.engine import EvaluationEngine
from focus_engine.evaluation.models import GroundTruthStatus
from tests.unit.evaluation.scenarios import (
    HORIZON_SECONDS,
    activity_event,
    snapshot_from_outcome,
)


@pytest.mark.evaluation
def test_evaluation_rejects_pre_prediction_events_leakage() -> None:
    """Events occurring strictly before predicted_at must not be evaluated."""
    engine = EvaluationEngine()
    snapshot = snapshot_from_outcome(is_positive=True)

    # 4 events before predicted_at (negative offset)
    pre_events = [activity_event(offset_seconds=-i * 10, ordinal=i) for i in range(1, 5)]
    gt = engine.ground_truth(snapshot, pre_events, reading=None)
    assert gt.status == GroundTruthStatus.INSUFFICIENT_DATA


@pytest.mark.evaluation
def test_evaluation_half_open_window_end_is_exclusive() -> None:
    """An event exactly at predicted_at + duration is outside the window."""
    engine = EvaluationEngine()
    snapshot = snapshot_from_outcome(is_positive=True)

    # Event exactly at upper window boundary: offset_seconds = HORIZON_SECONDS
    boundary_event = activity_event(offset_seconds=HORIZON_SECONDS, ordinal=99)
    # 3 valid events inside window (need 4 for sufficient data)
    inside_events = [
        activity_event(offset_seconds=10, ordinal=1),
        activity_event(offset_seconds=20, ordinal=2),
        activity_event(offset_seconds=30, ordinal=3),
    ]
    gt = engine.ground_truth(snapshot, inside_events + [boundary_event], reading=None)
    assert gt.status == GroundTruthStatus.INSUFFICIENT_DATA
