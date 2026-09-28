"""Tests verifying that feedback cannot trigger automated model retraining."""

from __future__ import annotations

import pytest

from focus_engine.feedback.engine import FeedbackEngine


def test_retraining_from_feedback_is_prohibited() -> None:
    """Feedback engine structurally refuses any automated model retraining call."""
    engine = FeedbackEngine()
    with pytest.raises(NotImplementedError) as exc_info:
        engine.retrain_model_from_feedback()

    assert "Direct automated retraining from live feedback is architecturally prohibited" in str(
        exc_info.value
    )
