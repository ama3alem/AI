"""Simulation scenarios and archetype holdout verification tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from focus_engine.schemas.primitives import DataOrigin
from focus_engine.simulator import (
    ARCHETYPE_PROFILES,
    LearnerArchetype,
    SimulationConfig,
    generate_session,
    generate_session_events,
)

BASE_TIME = datetime(2026, 6, 15, 10, 0, 0, tzinfo=UTC)


@pytest.mark.simulation
def test_all_archetypes_generate_valid_events() -> None:
    """Verify that all simulator archetypes produce valid, synthetic-stamped events."""
    for archetype in LearnerArchetype:
        config = SimulationConfig(
            archetype=archetype,
            learner_id=f"test-learner-{archetype.value}",
            session_id=f"test-session-{archetype.value}",
            base_seed=12345,
            question_count=10,
            start_time=BASE_TIME,
        )
        session = generate_session(config)
        assert session.provenance.archetype == archetype
        assert len(session.events) >= 10
        for event in session.events:
            assert event.origin == DataOrigin.SYNTHETIC
            assert event.learner_id == config.learner_id
            assert event.session_id == config.session_id


@pytest.mark.simulation
def test_archetype_profile_completeness() -> None:
    """Verify that every archetype has a registered profile."""
    for archetype in LearnerArchetype:
        assert archetype in ARCHETYPE_PROFILES
        profile = ARCHETYPE_PROFILES[archetype]
        assert profile.signature()
        assert 0.0 <= profile.base_accuracy <= 1.0


@pytest.mark.simulation
def test_generator_deterministic_across_calls() -> None:
    """Verify exact determinism for identical seed and configuration."""
    config = SimulationConfig(
        archetype=LearnerArchetype.GRADUAL_DECLINE,
        learner_id="det-learner-0001",
        session_id="det-session-0001",
        base_seed=9999,
        question_count=8,
        start_time=BASE_TIME,
    )
    events_1 = generate_session_events(config)
    events_2 = generate_session_events(config)
    assert [e.model_dump() for e in events_1] == [e.model_dump() for e in events_2]
