"""Unit tests for :mod:`focus_engine.schemas.primitives`.

Covers pseudonymous identifier validation, timezone enforcement, probability range
checks, and the provenance/confidence vocabularies that prevent a synthetic value from
being representable as a real observation.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import BaseModel, ValidationError

from focus_engine.schemas.primitives import (
    BEHAVIORAL_ENGAGEMENT_STATES,
    BehavioralEngagementState,
    ConfidenceLevel,
    DataOrigin,
    InferenceBasis,
    LearnerId,
    Probability,
    Provenance,
    SyntheticDataStamp,
    Timestamp,
    non_empty_text,
    utc_now,
)

pytestmark = pytest.mark.unit


class _LearnerRef(BaseModel):
    """Minimal consumer model used to exercise the identifier types."""

    learner_id: LearnerId


class _Moment(BaseModel):
    """Minimal consumer model used to exercise the timestamp type."""

    at: Timestamp


class _Rate(BaseModel):
    """Minimal consumer model used to exercise the probability type."""

    rate: Probability


class _Label(BaseModel):
    """Minimal consumer model used to exercise the text type."""

    text: non_empty_text


# --------------------------------------------------------------------------------------
# Pseudonymous identifiers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier",
    [
        "learner-0001",
        "LEARNER_0001",
        "abc12345",
        "a" * 128,
    ],
)
def test_valid_pseudonymous_identifiers_are_accepted(identifier: str) -> None:
    assert _LearnerRef(learner_id=identifier).learner_id == identifier


@pytest.mark.parametrize(
    "identifier",
    [
        "short1",  # below the minimum length
        "a" * 129,  # above the maximum length
        "learner 0001",  # whitespace
        "learner.0001",  # dot: could be a domain or filename
        "learner@example",  # email-like
        "C:\\learners\\0001",  # path traversal
        "learner/0001",  # path separator
        "learner\n0001",  # newline: log-injection vector
    ],
)
def test_non_pseudonymous_identifiers_are_rejected(identifier: str) -> None:
    with pytest.raises(ValidationError):
        _LearnerRef(learner_id=identifier)


def test_short_identifier_rejection_explains_reidentification_risk() -> None:
    """The rejection message must state the reason, not merely fail.

    A future contributor who needs a short id should learn why it is refused.
    """
    with pytest.raises(ValidationError, match="re-identification"):
        _LearnerRef(learner_id="abc")


def test_identifier_is_surrounding_whitespace_stripped() -> None:
    assert _LearnerRef(learner_id="  learner-0001  ").learner_id == "learner-0001"


def test_learner_identifier_field_carries_no_pii_semantics() -> None:
    """The identifier type must be documented as pseudonymous at the schema level."""
    field = _LearnerRef.model_fields["learner_id"]
    assert "Pseudonymous" in (field.description or "")


# --------------------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------------------


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _Moment(at=datetime(2026, 9, 26, 10, 0, 0))


def test_aware_datetime_is_normalised_to_utc() -> None:
    tz = timezone(timedelta(hours=5, minutes=30))
    moment = _Moment(at=datetime(2026, 9, 26, 15, 30, 0, tzinfo=tz))
    assert moment.at.tzinfo is UTC
    assert moment.at.utcoffset() == timedelta(0)
    assert moment.at.hour == 10


def test_utc_datetime_round_trips() -> None:
    moment = _Moment(at=datetime(2026, 9, 26, 10, 0, 0, tzinfo=UTC))
    assert moment.at == datetime(2026, 9, 26, 10, 0, 0, tzinfo=UTC)


def test_utc_now_returns_aware_utc() -> None:
    now = utc_now()
    assert now.tzinfo is UTC
    assert now.utcoffset() == timedelta(0)


# --------------------------------------------------------------------------------------
# Probabilities
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 1e-9])
def test_probabilities_within_unit_interval_are_accepted(value: float) -> None:
    assert _Rate(rate=value).rate == value


@pytest.mark.parametrize("value", [-0.0001, 1.0001, -1.0, 2.0])
def test_probabilities_outside_unit_interval_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError, match=r"\[0.0, 1.0\]"):
        _Rate(rate=value)


def test_nan_probability_is_rejected() -> None:
    with pytest.raises(ValidationError, match="NaN"):
        _Rate(rate=float("nan"))


def test_probability_cannot_exceed_one_even_by_one_ulp() -> None:
    """Guards against a computed value one unit-in-last-place above 1.0 slipping through.

    ``nextafter`` is used because a literal such as ``1.0 + 1e-16`` rounds back to exactly
    ``1.0`` in float64 and would test nothing.
    """
    just_above_one = math.nextafter(1.0, 2.0)
    assert just_above_one > 1.0
    with pytest.raises(ValidationError):
        _Rate(rate=just_above_one)


def test_probability_at_exactly_one_is_accepted() -> None:
    assert _Rate(rate=1.0).rate == 1.0


# --------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValidationError, match="whitespace-only"):
        _Label(text="   ")


def test_text_is_stripped() -> None:
    assert _Label(text="  engagement  ").text == "engagement"


# --------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------


def test_synthetic_label_is_distinct_from_ground_truth() -> None:
    """The central anti-hallucination invariant, asserted directly.

    If these two ever collapse into the same value, a simulator-derived label could be
    reported as a real measurement.
    """
    assert Provenance.SYNTHETIC_LABEL != Provenance.GROUND_TRUTH
    assert Provenance.SYNTHETIC_LABEL.value == "synthetic_label"
    assert Provenance.GROUND_TRUTH.value == "ground_truth"


def test_inference_basis_distinguishes_population_from_personal() -> None:
    assert InferenceBasis.POPULATION_PRIOR != InferenceBasis.PERSONAL
    assert InferenceBasis.PARTIAL_PERSONAL != InferenceBasis.PERSONAL


def test_unknown_and_insufficient_data_are_distinct_confidence_states() -> None:
    """They require different remedies: more data versus a better model."""
    assert ConfidenceLevel.UNKNOWN != ConfidenceLevel.INSUFFICIENT_DATA


def test_engagement_states_contain_no_internal_mental_state_terms() -> None:
    """State names must describe observable behaviour only.

    A term implying cognition or attention would overstate what interaction telemetry can
    support, which is precisely the claim discipline this project enforces.
    """
    forbidden = {
        "focus",
        "attention",
        "distracted",
        "distraction",
        "aware",
        "conscious",
        "engaged_mind",
        "bored",
        "motivation",
        "emotion",
    }
    for state in BEHAVIORAL_ENGAGEMENT_STATES:
        for term in forbidden:
            assert term not in state, f"state {state!r} implies an unsupported internal claim"


def test_engagement_states_tuple_matches_enum() -> None:
    assert set(BEHAVIORAL_ENGAGEMENT_STATES) == {s.value for s in BehavioralEngagementState}


def test_every_state_is_lowercase_and_non_empty() -> None:
    for state in BehavioralEngagementState:
        assert state.value == state.value.lower()
        assert state.value.strip()


# --------------------------------------------------------------------------------------
# Synthetic data stamp
# --------------------------------------------------------------------------------------


def test_synthetic_stamp_defaults_to_synthetic_origin_and_warning() -> None:
    stamp = SyntheticDataStamp(generator="simulator_v1", seed=7)
    assert stamp.origin is DataOrigin.SYNTHETIC
    assert stamp.warning == "SYNTHETIC DATA — NOT REAL STUDENT DATA"


def test_synthetic_stamp_can_be_relabelled_as_real_origin() -> None:
    """The field is not literally frozen to SYNTHETIC.

    Phase 15 must ensure that relabelling requires an explicit, recorded act rather than
    a default. Until then this test documents the gap rather than pretending it is
    closed.
    """
    stamp = SyntheticDataStamp(origin=DataOrigin.REAL, generator="g", seed=1)
    assert stamp.origin is DataOrigin.REAL


def test_assert_not_real_passes_for_synthetic_stamp() -> None:
    SyntheticDataStamp(generator="g", seed=1).assert_not_real()


def test_assert_not_real_raises_for_real_origin() -> None:
    with pytest.raises(ValueError, match="expected a synthetic origin"):
        SyntheticDataStamp(origin=DataOrigin.REAL, generator="g", seed=1).assert_not_real()


def test_stamp_is_immutable() -> None:
    stamp = SyntheticDataStamp(generator="g", seed=1)
    with pytest.raises(ValidationError):
        stamp.seed = 2  # type: ignore[misc]
