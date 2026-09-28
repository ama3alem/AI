"""Tests for the feature engine (Phase 5).

Covers the phase exit criteria: versioned immutable definitions, deterministic output,
missing data as an explicit marker rather than an invented value, no future information
in any window, and refusal to serve an unregistered feature set.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import pytest
from pydantic import ValidationError

from focus_engine.configuration.thresholds import FeatureSettings
from focus_engine.context.engine import ContextEngine
from focus_engine.context.models import (
    ContentContext,
    ContextModel,
    PerformanceContext,
    SessionContext,
)
from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    InterventionStartedPayload,
    QuestionAnsweredPayload,
    QuestionStartedPayload,
    SessionEndedPayload,
    SessionStartedPayload,
)
from focus_engine.features import (
    DEFAULT_REGISTRY,
    FEATURE_SET_V1,
    FEATURE_SPECS_V1,
    INSUFFICIENT_DATA_MARKER,
    FeatureAvailability,
    FeatureCategory,
    FeatureEngine,
    FeatureName,
    FeatureRegistry,
    FeatureSpec,
    FeatureValue,
    FeatureValueType,
    UnsupportedFeatureSetError,
    compute_features,
)
from focus_engine.features.models import ContextKey
from focus_engine.schemas.primitives import (
    DataOrigin,
    LearnerId,
    Provenance,
    SessionId,
    SyntheticDataStamp,
)
from focus_engine.utils.clock import FixedClock

pytestmark = pytest.mark.unit

BASE_TIME = datetime(2025, 9, 27, 10, 0, 0, tzinfo=UTC)
LEARNER_ID = "learner-0001"
SESSION_ID = "session-0001"
STAMP = SyntheticDataStamp(generator="test-features", seed=0)


class _Event(NamedTuple):
    """A synthetic event to be materialised by the scenario builder."""

    index: int
    offset: float
    event_type: EventType
    payload: object
    origin: DataOrigin = DataOrigin.SYNTHETIC


def _build(events: list[_Event]) -> tuple[EventEnvelope, ...]:
    """Materialise a list of event descriptors into envelopes."""
    return tuple(
        EventEnvelope(
            event_id=f"evt-{descriptor.index:04d}-{descriptor.event_type.value}",
            learner_id=LearnerId(LEARNER_ID),
            session_id=SessionId(SESSION_ID),
            timestamp=BASE_TIME + timedelta(seconds=descriptor.offset),
            event_type=descriptor.event_type,
            payload=descriptor.payload,
            origin=descriptor.origin,
            provenance=(
                Provenance.SYNTHETIC_LABEL
                if descriptor.origin is DataOrigin.SYNTHETIC
                else Provenance.OBSERVED
            ),
            synthetic_stamp=STAMP if descriptor.origin is DataOrigin.SYNTHETIC else None,
        )
        for descriptor in events
    )


def _context(events: tuple[EventEnvelope, ...]) -> ContextModel:
    """Derive context at the reference time, pinned to the last event.

    The clock is fixed rather than the system clock so that the empty-stream case, which
    the context engine resolves against the clock, is still deterministic.
    """
    engine = ContextEngine(clock=FixedClock(BASE_TIME), cooldown_seconds=300.0)
    return engine.process(events)


def _answer(index: int, offset: float, correct: bool, seconds: float) -> _Event:
    """A answered-question event descriptor."""
    return _Event(
        index=index,
        offset=offset,
        event_type=EventType.QUESTION_ANSWERED,
        payload=QuestionAnsweredPayload(
            question_id=f"question-{index:04d}", correct=correct, response_seconds=seconds
        ),
    )


def _question_start(index: int, offset: float, difficulty: float) -> _Event:
    """A question-start event descriptor carrying difficulty metadata."""
    return _Event(
        index=index,
        offset=offset,
        event_type=EventType.QUESTION_STARTED,
        payload=QuestionStartedPayload(question_id=f"question-{index:04d}", difficulty=difficulty),
    )


def _full_session() -> tuple[EventEnvelope, ...]:
    """A session with content, answers, and an explicit end."""
    return _build(
        [
            _Event(0, 0.0, EventType.SESSION_STARTED, SessionStartedPayload()),
            _question_start(1, 5.0, 0.4),
            _answer(2, 20.0, correct=True, seconds=15.0),
            _question_start(3, 25.0, 0.7),
            _answer(4, 50.0, correct=False, seconds=25.0),
            _question_start(5, 55.0, 0.5),
            _answer(6, 80.0, correct=True, seconds=20.0),
            _Event(7, 90.0, EventType.SESSION_ENDED, SessionEndedPayload()),
        ]
    )


def _two_windows(include_future: bool = False) -> tuple[EventEnvelope, ...]:
    """Three incorrect answers, then three correct ones far enough apart to split windows.

    The reference time is the last answer of the recent group, so the short window holds
    the recent group and the comparison window holds the earlier one. The optional future
    group would visibly change the trend if it were allowed into the short window.
    """
    descriptors = [
        _Event(0, 0.0, EventType.SESSION_STARTED, SessionStartedPayload()),
        _answer(1, 50.0, correct=False, seconds=40.0),
        _answer(2, 100.0, correct=False, seconds=40.0),
        _answer(3, 200.0, correct=False, seconds=40.0),
        _answer(4, 1000.0, correct=True, seconds=10.0),
        _answer(5, 1100.0, correct=True, seconds=10.0),
        _answer(6, 1200.0, correct=True, seconds=10.0),
    ]
    if include_future:
        descriptors += [
            _answer(7, 1300.0, correct=False, seconds=5.0),
            _answer(8, 1400.0, correct=False, seconds=5.0),
            _answer(9, 1500.0, correct=False, seconds=5.0),
        ]
    return _build(descriptors)


# Registry and versioning
# --------------------------------------------------------------------------------------


def test_every_feature_name_has_a_v1_specification() -> None:
    names = {spec.name for spec in FEATURE_SPECS_V1}
    assert names == set(FeatureName)


def test_every_v1_specification_has_a_calculator() -> None:
    for spec in FEATURE_SPECS_V1:
        computed = compute_features((), _context(())).as_dict[spec.name.value]
        assert computed == INSUFFICIENT_DATA_MARKER, (
            f"{spec.name.value} fell through to 'not implemented'; every spec must be backed "
            "by a registered calculation"
        )


def test_v1_specs_have_unique_names() -> None:
    names = [spec.name for spec in FEATURE_SPECS_V1]
    assert len(names) == len(set(names))


def test_specs_declare_required_context_keys_as_known_keys() -> None:
    for spec in FEATURE_SPECS_V1:
        assert spec.required_context_keys <= set(ContextKey)


def test_registry_refuses_to_reregister_a_version() -> None:
    registry = FeatureRegistry({FEATURE_SET_V1: FEATURE_SPECS_V1})
    with pytest.raises(ValueError, match="append-only"):
        registry.register(FEATURE_SET_V1, FEATURE_SPECS_V1)


def test_a_fresh_registry_accepts_the_same_version() -> None:
    """The refusal is about editing a registered version, not about the version name."""
    FeatureRegistry().register(FEATURE_SET_V1, FEATURE_SPECS_V1)


def test_registry_refuses_duplicate_feature_names() -> None:
    spec = FEATURE_SPECS_V1[0]
    with pytest.raises(ValueError, match="more than once"):
        FeatureRegistry().register("FEATURE_SET_V9", [spec, spec])


def test_registry_refuses_an_empty_feature_set() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        FeatureRegistry().register("FEATURE_SET_V9", [])


def test_registry_refuses_a_malformed_version() -> None:
    with pytest.raises(ValueError, match="invalid version identifier"):
        FeatureRegistry().register("feature-set-1", FEATURE_SPECS_V1[:1])


def test_registry_supports_a_new_version_without_touching_history() -> None:
    spec = FeatureSpec(
        name=FeatureName.SESSION_EVENT_COUNT,
        version="2.0.0",
        category=FeatureCategory.SESSION,
        data_source="context.session.event_count",
        definition="Event count, redefined for a new feature-set version.",
        required_context_keys=frozenset({ContextKey.SESSION}),
        value_type=FeatureValueType.INT,
        valid_range=(0.0, 100000.0),
    )
    registry = FeatureRegistry({FEATURE_SET_V1: FEATURE_SPECS_V1})
    registry.register("FEATURE_SET_V2", [spec])

    assert registry.versions() == (FEATURE_SET_V1, "FEATURE_SET_V2")
    assert len(registry) == 2
    assert registry.get_spec(FeatureName.SESSION_EVENT_COUNT, FEATURE_SET_V1).version == "1.0.0"
    assert registry.get_spec(FeatureName.SESSION_EVENT_COUNT, "FEATURE_SET_V2").version == "2.0.0"
    assert registry.get_spec(FeatureName.CONTENT_DIFFICULTY, "FEATURE_SET_V2") is None


def test_engine_refuses_an_unregistered_version_at_construction() -> None:
    with pytest.raises(UnsupportedFeatureSetError, match="FEATURE_SET_V2"):
        FeatureEngine(feature_set_version="FEATURE_SET_V2")


def test_compute_features_refuses_an_unregistered_version() -> None:
    events = _full_session()
    with pytest.raises(UnsupportedFeatureSetError):
        compute_features(events, _context(events), feature_set_version="FEATURE_SET_V2")


def test_default_registry_exposes_v1() -> None:
    assert FEATURE_SET_V1 in DEFAULT_REGISTRY
    assert DEFAULT_REGISTRY.list_features(FEATURE_SET_V1) == FEATURE_SPECS_V1


def test_unknown_version_lookup_returns_nothing_rather_than_raising() -> None:
    assert DEFAULT_REGISTRY.get_spec(FeatureName.SESSION_POSITION, "FEATURE_SET_V77") is None
    assert DEFAULT_REGISTRY.list_features("FEATURE_SET_V77") == ()


# Determinism
# --------------------------------------------------------------------------------------


def test_identical_input_produces_identical_output() -> None:
    events = _full_session()
    context = _context(events)
    first = compute_features(events, context)
    second = compute_features(events, context)
    assert first.model_dump_json() == second.model_dump_json()


def test_replaying_the_same_stream_reproduces_the_vector() -> None:
    events = _full_session()
    first = compute_features(events, _context(events))
    second = compute_features(events, _context(events))
    assert first.as_dict == second.as_dict
    assert first.computed_at == second.computed_at


def test_vector_records_the_feature_set_version() -> None:
    events = _full_session()
    vector = compute_features(events, _context(events))
    assert vector.feature_set_version == FEATURE_SET_V1
    assert {entry.feature_set_version for entry in vector.values} == {FEATURE_SET_V1}


def test_engine_is_stateless_across_calls() -> None:
    events = _full_session()
    context = _context(events)
    engine = FeatureEngine()
    before = engine.compute(events, context)
    compute_features((), _context(()))
    after = engine.compute(events, context)
    assert before.model_dump_json() == after.model_dump_json()


def test_computed_at_matches_the_context_reference_time() -> None:
    events = _full_session()
    context = _context(events)
    vector = compute_features(events, context)
    assert vector.computed_at == context.reference_time
    assert all(entry.computed_at == context.reference_time for entry in vector.values)


# Missing data
# --------------------------------------------------------------------------------------


def test_no_events_leaves_every_feature_inadequate() -> None:
    vector = compute_features((), _context(()))
    assert vector.available_count == 0
    assert vector.insufficient_count == len(FEATURE_SPECS_V1)
    assert set(vector.as_dict.values()) == {INSUFFICIENT_DATA_MARKER}


def test_missing_data_states_a_reason_for_every_gap() -> None:
    vector = compute_features((), _context(()))
    for entry in vector.values:
        assert entry.availability is FeatureAvailability.INSUFFICIENT_DATA
        assert entry.reason


def test_absent_difficulty_is_not_replaced_by_a_midpoint() -> None:
    events = _build(
        [
            _Event(0, 0.0, EventType.SESSION_STARTED, SessionStartedPayload()),
            _Event(
                1,
                5.0,
                EventType.QUESTION_STARTED,
                QuestionStartedPayload(question_id="question-0001"),
            ),
            _answer(2, 20.0, correct=True, seconds=12.0),
        ]
    )
    vector = compute_features(events, _context(events))
    assert vector.as_dict[FeatureName.CONTENT_DIFFICULTY.value] == INSUFFICIENT_DATA_MARKER
    assert 0.5 not in [value for value in vector.as_dict.values() if isinstance(value, float)]


def test_never_intervened_is_distinguishable_from_just_intervened() -> None:
    without = _build(
        [
            _Event(0, 0.0, EventType.SESSION_STARTED, SessionStartedPayload()),
            _answer(1, 20.0, correct=True, seconds=12.0),
        ]
    )
    with_intervention = _build(
        [
            _Event(0, 0.0, EventType.SESSION_STARTED, SessionStartedPayload()),
            _Event(
                1,
                10.0,
                EventType.INTERVENTION_STARTED,
                InterventionStartedPayload(
                    intervention_id="intervention-0001",
                    intervention_type="prompt",
                    trigger_probability=0.5,
                    trigger_state="test",
                ),
            ),
            _answer(2, 20.0, correct=True, seconds=12.0),
        ]
    )
    absent = compute_features(without, _context(without))
    present = compute_features(with_intervention, _context(with_intervention))

    assert absent.as_dict[FeatureName.INTERVENTION_SECONDS_SINCE_LAST.value] == (
        INSUFFICIENT_DATA_MARKER
    )
    assert present.as_dict[FeatureName.INTERVENTION_SECONDS_SINCE_LAST.value] == 10.0


def test_availability_counts_partition_the_vector() -> None:
    events = _full_session()
    vector = compute_features(events, _context(events))
    assert vector.available_count + vector.insufficient_count == len(vector.values)


def test_state_features_are_inadequate_until_their_engines_exist() -> None:
    events = _full_session()
    vector = compute_features(events, _context(events))
    for name in (FeatureName.TRAJECTORY_ENCODED, FeatureName.BASELINE_MATURITY_ENCODED):
        entry = next(e for e in vector.values if e.name is name)
        assert entry.availability is FeatureAvailability.INSUFFICIENT_DATA
        assert name.value.rsplit("_encoded", 1)[0] in (entry.reason or "")


def test_absent_identifiers_stay_absent_rather_than_being_invented() -> None:
    vector = compute_features((), _context(()))
    assert vector.learner_id is None
    assert vector.session_id is None


def test_observed_identifiers_are_propagated() -> None:
    events = _full_session()
    vector = compute_features(events, _context(events))
    assert vector.learner_id == LEARNER_ID
    assert vector.session_id == SESSION_ID


# No future information
# --------------------------------------------------------------------------------------


def test_events_after_the_reference_time_do_not_change_the_vector() -> None:
    history = _two_windows()
    context = _context(history)
    baseline = compute_features(history, context)

    with_future = compute_features(_two_windows(include_future=True), context)

    assert baseline.as_dict[FeatureName.PERFORMANCE_ACCURACY_TREND.value] == pytest.approx(1.0)
    assert baseline.model_dump_json() == with_future.model_dump_json()


def test_future_answers_do_not_enter_the_short_window() -> None:
    """Appending answers after the reference time must not move the trend."""
    context = _context(_two_windows())
    without = compute_features(_two_windows(), context)
    with_future = compute_features(_two_windows(include_future=True), context)

    assert without.as_dict[FeatureName.PERFORMANCE_ACCURACY_TREND.value] == pytest.approx(1.0)
    assert with_future.as_dict[FeatureName.PERFORMANCE_ACCURACY_TREND.value] == pytest.approx(1.0)
    assert with_future.as_dict[FeatureName.PERFORMANCE_RESPONSE_TREND.value] == pytest.approx(-30.0)


def test_context_derived_from_a_prefix_bounds_the_vector() -> None:
    """A context built from a prefix must not be widened by a longer event tuple."""
    history = _two_windows()
    prefix_context = _context(history)
    assert prefix_context.reference_time == BASE_TIME + timedelta(seconds=1200)

    assert compute_features(_two_windows(include_future=True), prefix_context).as_dict == (
        compute_features(history, prefix_context).as_dict
    )


def test_short_window_start_is_inclusive() -> None:
    """An answer exactly on the short-window boundary counts as recent."""
    events = _build(
        [
            _Event(0, 0.0, EventType.SESSION_STARTED, SessionStartedPayload()),
            _answer(1, 50.0, correct=False, seconds=40.0),
            _answer(2, 100.0, correct=False, seconds=40.0),
            _answer(3, 200.0, correct=False, seconds=40.0),
            _answer(4, 300.0, correct=False, seconds=5.0),
            _answer(5, 1000.0, correct=True, seconds=10.0),
            _answer(6, 1100.0, correct=True, seconds=10.0),
            _answer(7, 1200.0, correct=True, seconds=10.0),
        ]
    )
    vector = compute_features(events, _context(events))
    # Short window holds four answers (three correct, one incorrect) once the boundary
    # answer is included; excluding it would give 1.0 instead of 0.75.
    assert vector.as_dict[FeatureName.PERFORMANCE_ACCURACY_TREND.value] == pytest.approx(0.75)


def test_trend_reflects_a_change_between_the_two_windows() -> None:
    events = _two_windows()
    vector = compute_features(events, _context(events))
    assert vector.as_dict[FeatureName.PERFORMANCE_ACCURACY_TREND.value] == pytest.approx(1.0)
    assert vector.as_dict[FeatureName.PERFORMANCE_RESPONSE_TREND.value] == pytest.approx(-30.0)


def test_trend_is_inadequate_below_the_configured_minimum() -> None:
    events = _build(
        [
            _Event(0, 0.0, EventType.SESSION_STARTED, SessionStartedPayload()),
            _answer(1, 100.0, correct=True, seconds=20.0),
            _answer(2, 200.0, correct=True, seconds=20.0),
            _answer(3, 1100.0, correct=True, seconds=20.0),
            _answer(4, 1200.0, correct=True, seconds=20.0),
        ]
    )
    vector = compute_features(events, _context(events))
    assert vector.as_dict[FeatureName.PERFORMANCE_ACCURACY_TREND.value] == INSUFFICIENT_DATA_MARKER
    assert vector.as_dict[FeatureName.PERFORMANCE_RESPONSE_TREND.value] == INSUFFICIENT_DATA_MARKER


def test_trend_threshold_follows_the_configured_setting() -> None:
    events = _two_windows()
    context = _context(events)
    strict = compute_features(events, context, settings=FeatureSettings(min_events_for_window=4))
    lax = compute_features(events, context, settings=FeatureSettings(min_events_for_window=3))

    assert strict.as_dict[FeatureName.PERFORMANCE_ACCURACY_TREND.value] == INSUFFICIENT_DATA_MARKER
    assert lax.as_dict[FeatureName.PERFORMANCE_ACCURACY_TREND.value] == pytest.approx(1.0)


# Range handling
# --------------------------------------------------------------------------------------


def test_out_of_range_result_is_reported_rather_than_clamped() -> None:
    context = ContextModel(
        reference_time=BASE_TIME,
        content=ContentContext(content_type="question", difficulty=5.0),
    )
    engine = FeatureEngine()
    spec = DEFAULT_REGISTRY.require(FEATURE_SET_V1)[3]
    result = engine.compute_one(spec, (), context)

    assert result.availability is FeatureAvailability.INSUFFICIENT_DATA
    assert "outside the valid range" in (result.reason or "")
    assert "has not been clamped" in (result.reason or "")


def test_in_range_result_is_returned_unchanged() -> None:
    context = ContextModel(
        reference_time=BASE_TIME,
        content=ContentContext(content_type="question", difficulty=0.25),
    )
    engine = FeatureEngine()
    spec = DEFAULT_REGISTRY.require(FEATURE_SET_V1)[3]
    result = engine.compute_one(spec, (), context)

    assert result.availability is FeatureAvailability.AVAILABLE
    assert result.value == 0.25


# Provenance
# --------------------------------------------------------------------------------------


def test_synthetic_only_stream_marks_computed_features_as_synthetic() -> None:
    events = _full_session()
    vector = compute_features(events, _context(events))

    assert vector.is_synthetic_only
    assert vector.origin_label == "synthetic"
    computed = [
        e for e in vector.values if e.availability is not FeatureAvailability.INSUFFICIENT_DATA
    ]
    assert computed
    assert all(e.availability is FeatureAvailability.SYNTHETIC_ONLY for e in computed)


def test_real_stream_marks_computed_features_as_available() -> None:
    events = _full_session()
    real = tuple(
        EventEnvelope(
            event_id=event.event_id,
            learner_id=event.learner_id,
            session_id=event.session_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            payload=event.payload,
            origin=DataOrigin.REAL,
            provenance=Provenance.OBSERVED,
        )
        for event in events
    )
    vector = compute_features(real, _context(real))

    assert vector.origin_label == "real"
    computed = [
        e for e in vector.values if e.availability is not FeatureAvailability.INSUFFICIENT_DATA
    ]
    assert all(e.availability is FeatureAvailability.AVAILABLE for e in computed)


def test_mixed_provenance_is_recorded_without_being_called_real() -> None:
    events = _full_session()
    context = _context(events)
    context_with_mixed = context.model_copy(
        update={"origins": frozenset({DataOrigin.REAL, DataOrigin.SYNTHETIC})}
    )
    vector = compute_features(events, context_with_mixed)

    assert vector.origins == frozenset({DataOrigin.REAL, DataOrigin.SYNTHETIC})
    assert vector.origin_label == "mixed"
    assert vector.is_synthetic_only is False


def test_empty_origin_set_is_not_reported_as_observed() -> None:
    assert (
        FeatureValue(
            feature_set_version=FEATURE_SET_V1, computed_at=BASE_TIME, values=()
        ).origin_label
        == "unknown"
    )


# Model-level guarantees
# --------------------------------------------------------------------------------------


def test_marker_and_availability_cannot_disagree() -> None:
    context = _context(_full_session())
    entry = compute_features(_full_session(), context).values[0]
    with pytest.raises(ValidationError, match="carries the value"):
        entry.model_copy(
            update={"availability": FeatureAvailability.INSUFFICIENT_DATA}
        ).model_validate(
            {
                **entry.model_dump(),
                "availability": FeatureAvailability.INSUFFICIENT_DATA,
                "reason": "forced",
            }
        )


def test_computed_value_may_not_carry_an_insufficiency_reason() -> None:
    entry = compute_features(_full_session(), _context(_full_session())).values[0]
    with pytest.raises(ValidationError, match="must not carry an insufficiency reason"):
        entry.model_validate(
            {
                **entry.model_dump(),
                "name": FeatureName.SESSION_ELAPSED_SECONDS,
                "availability": FeatureAvailability.AVAILABLE,
                "value": 90.0,
                "reason": "contradictory",
            }
        )


def test_repeated_feature_names_are_rejected() -> None:
    entry = compute_features(_full_session(), _context(_full_session())).values[0]
    with pytest.raises(ValidationError, match="repeats feature names"):
        FeatureValue(
            feature_set_version=FEATURE_SET_V1,
            computed_at=BASE_TIME,
            values=(entry, entry),
        )


def test_synthetic_origins_forbid_claiming_a_feature_is_available() -> None:
    entry = compute_features(_full_session(), _context(_full_session())).values[0]
    payload = entry.model_dump()
    payload["availability"] = FeatureAvailability.AVAILABLE
    available = entry.model_validate(payload)
    with pytest.raises(ValidationError, match="purely synthetic"):
        FeatureValue(
            feature_set_version=FEATURE_SET_V1,
            computed_at=BASE_TIME,
            values=(available,),
            origins=frozenset({DataOrigin.SYNTHETIC}),
        )


def test_vector_refuses_a_naive_reference_time() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        FeatureValue(
            feature_set_version=FEATURE_SET_V1,
            computed_at=datetime(2025, 9, 27, 10, 0, 0),
            values=(),
        )


def test_spec_rejects_an_unordered_range() -> None:
    with pytest.raises(ValidationError, match="exceeds upper bound"):
        FeatureSpec(
            name=FeatureName.SESSION_POSITION,
            version="1.0.0",
            category=FeatureCategory.SESSION,
            data_source="context.session.position_in_session",
            definition="An invalid range used to prove the check runs.",
            required_context_keys=frozenset({ContextKey.POSITION_IN_SESSION}),
            value_type=FeatureValueType.FLOAT,
            valid_range=(1.0, 0.0),
        )


def test_spec_rejects_a_malformed_version() -> None:
    with pytest.raises(ValidationError, match="1.0.0"):
        FeatureSpec(
            name=FeatureName.SESSION_POSITION,
            version="v1",
            category=FeatureCategory.SESSION,
            data_source="context.session.position_in_session",
            definition="A malformed version used to prove the check runs.",
            required_context_keys=frozenset({ContextKey.POSITION_IN_SESSION}),
            value_type=FeatureValueType.FLOAT,
        )


# Encodings
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "encoded"),
    [("question", "question"), ("video", "video"), ("article", "reading"), ("podcast", "other")],
)
def test_content_type_encoding_is_total(raw: str, encoded: str) -> None:
    context = ContextModel(reference_time=BASE_TIME, content=ContentContext(content_type=raw))
    engine = FeatureEngine()
    spec = DEFAULT_REGISTRY.get_spec(FeatureName.CONTENT_TYPE_ENCODED, FEATURE_SET_V1)
    assert spec is not None
    assert engine.compute_one(spec, (), context).value == encoded


def test_performance_features_read_the_context_window() -> None:
    context = ContextModel(
        reference_time=BASE_TIME,
        session=SessionContext(
            session_id=SessionId(SESSION_ID),
            learner_id=LearnerId(LEARNER_ID),
            started_at=BASE_TIME,
            elapsed_seconds=60.0,
            event_count=4,
            position_in_session=0.1,
        ),
        performance=PerformanceContext(
            window_size=10,
            recent_questions=4,
            recent_correct=3,
            recent_accuracy=0.75,
            average_response_seconds=11.5,
        ),
    )
    vector = compute_features((), context)
    assert vector.as_dict[FeatureName.SESSION_ELAPSED_SECONDS.value] == 60.0
    assert vector.as_dict[FeatureName.SESSION_EVENT_COUNT.value] == 4
    assert vector.as_dict[FeatureName.PERFORMANCE_RECENT_ACCURACY.value] == 0.75
    assert vector.as_dict[FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS.value] == 11.5


def test_integer_features_are_reported_as_integers() -> None:
    events = _full_session()
    vector = compute_features(events, _context(events))
    count = vector.as_dict[FeatureName.SESSION_EVENT_COUNT.value]
    assert isinstance(count, int)
    assert not isinstance(count, bool)


def test_summary_dict_is_serialisable() -> None:
    import json

    events = _full_session()
    summary = compute_features(events, _context(events)).to_summary_dict()
    assert json.loads(json.dumps(summary))["origin_label"] == "synthetic"
    assert summary["available_count"] + summary["insufficient_count"] == len(FEATURE_SPECS_V1)
