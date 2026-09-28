"""Unit tests for feedback and response profile data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from focus_engine.feedback.models import (
    FEEDBACK_V1,
    InterventionResponseObservation,
    InterventionResponseProfile,
    ProfileMaturity,
    ResponseOutcomeDirection,
)
from focus_engine.schemas.primitives import DataOrigin, Provenance


def test_valid_collecting_profile_creation() -> None:
    """A profile with fewer than min_observations is collecting with None rate."""
    profile = InterventionResponseProfile(
        learner_id="test-learner-001",
        candidate_id="take_break",
        observation_count=3,
        positive_count=2,
        neutral_count=1,
        negative_count=0,
        unassessable_count=0,
        maturity=ProfileMaturity.COLLECTING,
        min_observations=5,
        effective_success_rate=None,
    )
    assert profile.feedback_version == FEEDBACK_V1
    assert profile.maturity == ProfileMaturity.COLLECTING
    assert profile.effective_success_rate is None
    assert profile.observation_count == 3


def test_collecting_profile_refuses_active_status_or_rate() -> None:
    """A profile with < min_observations cannot claim ACTIVE or a success rate."""
    with pytest.raises(ValidationError):
        InterventionResponseProfile(
            learner_id="test-learner-001",
            candidate_id="take_break",
            observation_count=2,
            positive_count=2,
            neutral_count=0,
            negative_count=0,
            unassessable_count=0,
            maturity=ProfileMaturity.ACTIVE,  # Refused: not enough observations
            min_observations=5,
            effective_success_rate=1.0,
        )

    with pytest.raises(ValidationError):
        InterventionResponseProfile(
            learner_id="test-learner-001",
            candidate_id="take_break",
            observation_count=2,
            positive_count=2,
            neutral_count=0,
            negative_count=0,
            unassessable_count=0,
            maturity=ProfileMaturity.COLLECTING,
            min_observations=5,
            effective_success_rate=1.0,  # Refused: cannot have rate while collecting
        )


def test_valid_active_profile_creation() -> None:
    """A profile with >= min_observations computes correct effective rate."""
    profile = InterventionResponseProfile(
        learner_id="test-learner-001",
        candidate_id="take_break",
        observation_count=5,
        positive_count=4,
        neutral_count=1,
        negative_count=0,
        unassessable_count=0,
        maturity=ProfileMaturity.ACTIVE,
        min_observations=5,
        effective_success_rate=0.8,
    )
    assert profile.maturity == ProfileMaturity.ACTIVE
    assert profile.effective_success_rate == 0.8


def test_active_profile_refuses_inconsistent_rate() -> None:
    """An active profile with mismatched rate raises ValidationError."""
    with pytest.raises(ValidationError):
        InterventionResponseProfile(
            learner_id="test-learner-001",
            candidate_id="take_break",
            observation_count=5,
            positive_count=4,
            neutral_count=1,
            negative_count=0,
            unassessable_count=0,
            maturity=ProfileMaturity.ACTIVE,
            min_observations=5,
            effective_success_rate=0.5,  # Inconsistent: 4/5 = 0.8, not 0.5
        )


def test_observation_model_fields_and_defaults() -> None:
    """Observation carries correct immutable fields and origin stamps."""
    obs = InterventionResponseObservation(
        learner_id="test-learner-001",
        candidate_id="take_break",
        session_id="test-session-001",
        direction=ResponseOutcomeDirection.POSITIVE,
    )
    assert obs.data_origin == DataOrigin.SYNTHETIC
    assert obs.provenance == Provenance.OBSERVED
    assert obs.context_key == "global"
    assert obs.direction == ResponseOutcomeDirection.POSITIVE
