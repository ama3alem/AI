"""Tests for the synthetic learner simulator (Phase 4).

Covers the phase exit criteria: reproducible output from a seed, measurable archetype
distinctness, non-degenerate session structure, strictly monotonic timestamps, in-band
synthetic warning on every artefact, and internally consistent payloads.

The archetype-behaviour tests assert *directional* relationships across many seeds rather
than exact numbers. Exact values would make the suite a change detector for the generator's
constants; what matters is that the configured coefficient actually produces the described
pattern.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from focus_engine.events.types import EventType, QuestionAnsweredPayload, QuestionStartedPayload
from focus_engine.events.validation import validate_event
from focus_engine.schemas.primitives import DataOrigin, Provenance
from focus_engine.simulator import (
    ARCHETYPE_PROFILES,
    SIMULATOR_VERSION,
    SYNTHETIC_WARNING,
    LatencyShape,
    LearnerArchetype,
    SimulationConfig,
    generate_session,
    generate_session_events,
    session_seed,
)
from focus_engine.simulator.config import MAX_QUESTIONS_PER_SESSION, MIN_QUESTIONS_PER_SESSION

BASE_TIME = datetime(2025, 9, 27, 10, 0, 0, tzinfo=UTC)

ALL_ARCHETYPES = tuple(LearnerArchetype)


def make_config(**overrides: object) -> SimulationConfig:
    """Build a valid default config, applying overrides.

    Args:
        **overrides: Field values to override on the default configuration.

    Returns:
        A validated :class:`SimulationConfig`.
    """
    defaults: dict[str, object] = {
        "archetype": LearnerArchetype.STABLE,
        "learner_id": "synthetic-learner-0001",
        "session_id": "synthetic-session-0001",
        "base_seed": 4242,
        "question_count": 12,
        "start_time": BASE_TIME,
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)  # type: ignore[arg-type]


def latency_series(result_events: tuple) -> list[float]:
    """Extract per-question response latencies in order.

    Args:
        result_events: Events from a generated session.

    Returns:
        The ``response_seconds`` of each ``question_answered`` event, in order.
    """
    return [
        e.payload.response_seconds
        for e in result_events
        if e.event_type is EventType.QUESTION_ANSWERED
    ]


# --------------------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------------------


def test_config_rejects_inverted_intervals() -> None:
    """An inverted sampling interval is a misconfiguration, not a runtime surprise."""
    with pytest.raises(ValueError, match="inter_event_gap_max"):
        make_config(
            inter_event_gap_min=timedelta(seconds=10),
            inter_event_gap_max=timedelta(seconds=1),
        )


def test_config_rejects_zero_gap_to_preserve_monotonicity() -> None:
    """A zero gap would permit repeated timestamps, so it is rejected up front."""
    with pytest.raises(ValueError, match="inter_event_gap_min must be positive"):
        make_config(inter_event_gap_min=timedelta(0), inter_event_gap_max=timedelta(seconds=5))


def test_config_rejects_intervention_with_no_subsequent_question() -> None:
    """An intervention with nothing after it could not affect any behaviour."""
    with pytest.raises(ValueError, match="intervention_after_answers"):
        make_config(
            question_count=6,
            include_intervention=True,
            intervention_after_answers=6,
        )


def test_config_rejects_dropout_floor_above_question_count() -> None:
    """Dropout cannot truncate below its own floor."""
    with pytest.raises(ValueError, match="dropout_min_questions"):
        make_config(question_count=5, dropout_min_questions=6)


def test_config_rejects_question_count_outside_bounds() -> None:
    """Question counts are bounded so a run cannot emit an unbounded stream."""
    with pytest.raises(ValueError):
        make_config(question_count=MIN_QUESTIONS_PER_SESSION - 1)
    with pytest.raises(ValueError):
        make_config(question_count=MAX_QUESTIONS_PER_SESSION + 1)


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def test_same_config_produces_byte_identical_events() -> None:
    """Identical config and seed must serialise to identical bytes."""
    cfg = make_config()
    first = generate_session(cfg)
    second = generate_session(cfg)

    assert first.provenance.derived_seed == second.provenance.derived_seed
    assert len(first.events) == len(second.events)
    for a, b in zip(first.events, second.events, strict=True):
        assert a.model_dump_json() == b.model_dump_json()


def test_derived_seed_is_pure_function_of_identifying_fields() -> None:
    """Seed derivation depends on identity, not on timing or dropout settings."""
    base = make_config()
    same_identity = make_config(
        question_count=6,
        inter_event_gap_min=timedelta(seconds=2),
        enable_dropout=True,
    )
    different_learner = make_config(learner_id="synthetic-learner-0002")
    different_archetype = make_config(archetype=LearnerArchetype.NOISY)

    assert session_seed(base) == session_seed(same_identity)
    assert session_seed(base) != session_seed(different_learner)
    assert session_seed(base) != session_seed(different_archetype)


@pytest.mark.parametrize("archetype", ALL_ARCHETYPES)
def test_different_seeds_produce_different_streams(archetype: LearnerArchetype) -> None:
    """Every archetype's output must actually depend on the seed."""
    a = generate_session(make_config(archetype=archetype, base_seed=1))
    b = generate_session(make_config(archetype=archetype, base_seed=2))
    assert [e.model_dump_json() for e in a.events] != [e.model_dump_json() for e in b.events]


def test_start_time_is_the_only_source_of_session_timing() -> None:
    """Generation must not read wall-clock time, so shifting the start shifts everything."""
    shifted = BASE_TIME + timedelta(days=365)
    a = generate_session(make_config(start_time=BASE_TIME))
    b = generate_session(make_config(start_time=shifted))
    assert (b.events[0].timestamp - a.events[0].timestamp) == shifted - BASE_TIME


# --------------------------------------------------------------------------------------
# Monotonicity and internal consistency
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("archetype", ALL_ARCHETYPES)
def test_timestamps_strictly_increase_within_session(archetype: LearnerArchetype) -> None:
    """Every session's timestamps must be strictly increasing."""
    result = generate_session(
        make_config(
            archetype=archetype,
            question_count=16,
            include_intervention=True,
            intervention_after_answers=5,
        )
    )
    timestamps = [e.timestamp for e in result.events]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:], strict=False))


@pytest.mark.parametrize("archetype", ALL_ARCHETYPES)
def test_response_seconds_matches_timestamp_interval(archetype: LearnerArchetype) -> None:
    """A question's recorded latency must equal its observed timestamp interval."""
    events = generate_session(make_config(archetype=archetype, question_count=14)).events
    started: dict[str, datetime] = {}
    checked = 0
    for event in events:
        if event.event_type is EventType.QUESTION_STARTED:
            assert isinstance(event.payload, QuestionStartedPayload)
            started[event.payload.question_id] = event.timestamp
        elif event.event_type is EventType.QUESTION_ANSWERED:
            assert isinstance(event.payload, QuestionAnsweredPayload)
            begin = started[event.payload.question_id]
            observed = (event.timestamp - begin).total_seconds()
            assert observed == pytest.approx(event.payload.response_seconds, abs=1e-3)
            checked += 1
    assert checked == 14


def test_session_ended_reports_true_duration() -> None:
    """The reported session duration must match the event timestamps."""
    events = generate_session(make_config(question_count=8)).events
    start = events[0].timestamp
    ended = events[-1]
    assert ended.event_type is EventType.SESSION_ENDED
    assert ended.payload.duration_seconds == pytest.approx(
        (ended.timestamp - start).total_seconds(), abs=1e-3
    )


def test_response_latency_respects_minimum_floor() -> None:
    """Jitter must not produce zero or negative latencies."""
    for seed in range(20):
        events = generate_session(
            make_config(archetype=LearnerArchetype.NOISY, base_seed=seed, question_count=20)
        ).events
        for latency in latency_series(events):
            assert latency >= 0.5


# --------------------------------------------------------------------------------------
# Provenance and warning
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("archetype", ALL_ARCHETYPES)
def test_every_artifact_carries_exact_synthetic_warning(archetype: LearnerArchetype) -> None:
    """Events and provenance must both carry the literal warning string."""
    result = generate_session(make_config(archetype=archetype, question_count=10))
    assert result.provenance.warning == SYNTHETIC_WARNING
    assert result.provenance.warning == "SYNTHETIC DATA — NOT REAL STUDENT DATA"
    for event in result.events:
        assert event.origin is DataOrigin.SYNTHETIC
        assert event.provenance is Provenance.SYNTHETIC_LABEL
        assert event.synthetic_stamp is not None
        assert event.synthetic_stamp.warning == SYNTHETIC_WARNING
        assert event.synthetic_stamp.seed == result.provenance.derived_seed
        assert event.synthetic_stamp.generator == SIMULATOR_VERSION


@pytest.mark.parametrize("archetype", ALL_ARCHETYPES)
def test_generated_events_pass_the_real_event_validator(archetype: LearnerArchetype) -> None:
    """Simulator output must survive the same ingest path as real captured data."""
    events = generate_session(make_config(archetype=archetype, question_count=8)).events
    assert events
    for event in events:
        revalidated = validate_event(event.model_dump(mode="json"))
        assert revalidated == event


def test_synthetic_warning_survives_json_round_trip() -> None:
    """The warning must remain in-band after serialisation, not only in memory."""
    result = generate_session(make_config(question_count=5))
    reloaded = json.loads(result.events[0].model_dump_json())
    assert reloaded["synthetic_stamp"]["warning"] == SYNTHETIC_WARNING
    assert reloaded["origin"] == "synthetic"
    assert reloaded["provenance"] == "synthetic_label"


# --------------------------------------------------------------------------------------
# Non-degenerate structure
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("archetype", ALL_ARCHETYPES)
def test_session_structure_is_complete_and_non_degenerate(archetype: LearnerArchetype) -> None:
    """Sessions must bracket their questions and never repeat an event identifier."""
    result = generate_session(make_config(archetype=archetype, question_count=10))
    types = [e.event_type for e in result.events]
    assert types[0] is EventType.SESSION_STARTED
    assert types[-1] is EventType.SESSION_ENDED
    assert types.count(EventType.QUESTION_STARTED) == 10
    assert types.count(EventType.QUESTION_ANSWERED) == 10
    assert result.provenance.emitted_question_count == 10
    assert result.provenance.planned_question_count == 10
    identifiers = [e.event_id for e in result.events]
    assert len(identifiers) == len(set(identifiers))
    assert result.events[-1].payload.reason == "completed"


def test_question_count_configures_output_length() -> None:
    """Output length must follow the configured question count, not a fixed fixture size."""
    lengths = {n: len(generate_session(make_config(question_count=n)).events) for n in (3, 7, 15)}
    assert lengths == {3: 8, 7: 16, 15: 32}
    assert lengths[3] < lengths[7] < lengths[15]


def test_questions_are_not_all_identical() -> None:
    """A session of identical rows would be a degenerate fixture."""
    result = generate_session(make_config(archetype=LearnerArchetype.STABLE, question_count=14))
    answers = [e for e in result.events if e.event_type is EventType.QUESTION_ANSWERED]
    latencies = {a.payload.response_seconds for a in answers}
    outcomes = {a.payload.correct for a in answers}
    question_ids = {a.payload.question_id for a in answers}
    assert len(latencies) > 1, "latency must vary across questions"
    assert len(outcomes) == 2, "both correct and incorrect answers must appear"
    assert len(question_ids) == 14


def test_every_archetype_generates_a_full_session_for_its_minimum_length() -> None:
    """Each archetype must be generable at the length it declares as its minimum."""
    for archetype, profile in ARCHETYPE_PROFILES.items():
        result = generate_session(
            make_config(archetype=archetype, question_count=profile.min_questions)
        )
        assert result.provenance.emitted_question_count == profile.min_questions
        assert result.provenance.pattern_measurable is True


def test_sessions_shorter_than_minimum_are_flagged_unmeasurable() -> None:
    """Provenance must warn consumers when a session is too short to support a claim."""
    profile = ARCHETYPE_PROFILES[LearnerArchetype.RECOVERY]
    short_count = max(MIN_QUESTIONS_PER_SESSION, profile.min_questions - 1)
    result = generate_session(
        make_config(archetype=LearnerArchetype.RECOVERY, question_count=short_count)
    )
    assert result.provenance.emitted_question_count < profile.min_questions
    assert result.provenance.pattern_measurable is False
    assert profile.pattern_is_measurable(result.provenance.emitted_question_count) is False


# --------------------------------------------------------------------------------------
# Dropout
# --------------------------------------------------------------------------------------


def test_dropout_is_disabled_by_default_so_length_is_exact() -> None:
    """Default runs must not silently truncate, or structural tests become flaky."""
    result = generate_session(make_config(archetype=LearnerArchetype.NOISY, question_count=12))
    assert result.provenance.emitted_question_count == 12
    assert result.events[-1].payload.reason == "completed"


def test_dropout_truncates_and_records_a_distinct_reason() -> None:
    """With dropout enabled, low-completion archetypes should produce early endings."""
    truncated = 0
    for seed in range(40):
        result = generate_session(
            make_config(
                archetype=LearnerArchetype.FAST_BUT_INACCURATE,
                base_seed=seed,
                question_count=20,
                enable_dropout=True,
                dropout_min_questions=3,
            )
        )
        emitted = result.provenance.emitted_question_count
        assert emitted <= 20
        if emitted < 20:
            truncated += 1
            assert result.events[-1].payload.reason == "timeout"
            assert emitted >= 3
        else:
            assert result.events[-1].payload.reason == "completed"
    assert truncated > 0, "dropout never triggered across 40 seeds"


def test_dropout_never_truncates_below_its_floor() -> None:
    """The floor protects against degenerate one-question sessions."""
    for seed in range(25):
        result = generate_session(
            make_config(
                archetype=LearnerArchetype.NOISY,
                base_seed=seed,
                question_count=10,
                enable_dropout=True,
                dropout_min_questions=7,
            )
        )
        assert result.provenance.emitted_question_count >= 7


# --------------------------------------------------------------------------------------
# Intervention
# --------------------------------------------------------------------------------------


def test_intervention_emits_a_matched_started_completed_pair() -> None:
    """An intervention must be a pair sharing one identifier, in the right order."""
    cfg = make_config(
        question_count=10,
        include_intervention=True,
        intervention_after_answers=4,
    )
    events = generate_session(cfg).events
    started = [e for e in events if e.event_type is EventType.INTERVENTION_STARTED]
    completed = [e for e in events if e.event_type is EventType.INTERVENTION_COMPLETED]
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0].payload.intervention_id == completed[0].payload.intervention_id
    assert completed[0].timestamp > started[0].timestamp
    assert started[0].timestamp > events[8].timestamp  # after the fourth answer
    assert 0.0 <= started[0].payload.trigger_probability <= 1.0


def test_intervention_absent_unless_requested() -> None:
    """Default runs must not contain intervention events."""
    types = [e.event_type for e in generate_session(make_config()).events]
    assert EventType.INTERVENTION_STARTED not in types
    assert EventType.INTERVENTION_COMPLETED not in types


# --------------------------------------------------------------------------------------
# Archetype distinctness
# --------------------------------------------------------------------------------------


def test_all_archetype_signatures_are_pairwise_distinct() -> None:
    """No two archetypes may share a coefficient vector, or they are unlabelled noise."""
    signatures = {a: p.signature() for a, p in ARCHETYPE_PROFILES.items()}
    assert len(signatures) == len(ALL_ARCHETYPES)
    assert len(set(signatures.values())) == len(ALL_ARCHETYPES)


def test_every_archetype_is_configured() -> None:
    """The profile table must cover the closed archetype enum exactly."""
    assert set(ARCHETYPE_PROFILES) == set(ALL_ARCHETYPES)


def test_latency_and_accuracy_levels_separate_the_known_pairs() -> None:
    """The named fast/slow and accurate/inaccurate pairs must differ where named."""
    stable = ARCHETYPE_PROFILES[LearnerArchetype.STABLE]
    fast = ARCHETYPE_PROFILES[LearnerArchetype.FAST_BUT_INACCURATE]
    slow = ARCHETYPE_PROFILES[LearnerArchetype.SLOW_BUT_ENGAGED]
    assert fast.base_response_seconds < stable.base_response_seconds < slow.base_response_seconds
    assert fast.base_accuracy < stable.base_accuracy < slow.base_accuracy


def test_stable_archetype_shows_no_index_dependent_drift() -> None:
    """A flat archetype's difficulty-adjusted latency must not trend with question index."""
    events = generate_session(
        make_config(archetype=LearnerArchetype.STABLE, question_count=20)
    ).events
    latencies = latency_series(events)
    assert len(latencies) == 20
    first_half = sum(latencies[:10]) / 10
    second_half = sum(latencies[10:]) / 10
    assert first_half == pytest.approx(second_half, rel=0.15)


def test_gradual_decline_archetype_trends_upward() -> None:
    """Linear decline must actually produce rising latency across many seeds."""
    rises = 0
    for seed in range(12):
        latencies = latency_series(
            generate_session(
                make_config(
                    archetype=LearnerArchetype.GRADUAL_DECLINE, base_seed=seed, question_count=20
                )
            ).events
        )
        if sum(latencies[10:]) / 10 > sum(latencies[:10]) / 10 * 1.15:
            rises += 1
    assert rises >= 10, f"expected a clear upward trend, saw it in {rises}/12 seeds"


def test_recovery_archetype_returns_toward_baseline() -> None:
    """A dip-and-recover shape must end near its own start, having risen in between.

    Windowed means are compared rather than single questions, because a single question
    carries the archetype's full jitter and would make this a coin flip.
    """
    returned_near_start = 0
    peak_above_both_ends = 0
    seeds = 12
    for seed in range(seeds):
        latencies = latency_series(
            generate_session(
                make_config(archetype=LearnerArchetype.RECOVERY, base_seed=seed, question_count=16)
            ).events
        )
        start = sum(latencies[:3]) / 3
        middle = sum(latencies[5:9]) / 4
        end = sum(latencies[-3:]) / 3
        if middle > start and middle > end:
            peak_above_both_ends += 1
        if abs(end - start) < middle - end:
            returned_near_start += 1
    assert peak_above_both_ends >= seeds - 1, (
        f"the dip must dominate both ends in nearly every seed, saw it in "
        f"{peak_above_both_ends}/{seeds}"
    )
    assert returned_near_start >= seeds - 1, (
        f"the session must end nearer its start than its dip in nearly every seed, saw it "
        f"in {returned_near_start}/{seeds}"
    )


def test_recovery_shape_is_dip_and_others_are_not() -> None:
    """The shape assignment itself must match the archetype definitions."""
    assert (
        ARCHETYPE_PROFILES[LearnerArchetype.RECOVERY].latency_shape is LatencyShape.DIP_AND_RECOVER
    )
    assert (
        ARCHETYPE_PROFILES[LearnerArchetype.INTERVENTION_RESISTANT].latency_shape
        is LatencyShape.LINEAR_DECLINE
    )
    assert ARCHETYPE_PROFILES[LearnerArchetype.STABLE].latency_shape is LatencyShape.FLAT


def test_intervention_resistant_archetype_ignores_interventions() -> None:
    """By definition this archetype's drift must be unchanged after an intervention."""
    profile = ARCHETYPE_PROFILES[LearnerArchetype.INTERVENTION_RESISTANT]
    assert profile.response_to_intervention == 1.0
    assert (
        profile.response_to_intervention
        > ARCHETYPE_PROFILES[LearnerArchetype.RECOVERY].response_to_intervention
    )


def test_context_sensitive_archetype_tracks_item_difficulty_most_strongly() -> None:
    """Difficulty sensitivity must be the dominant differentiator for this archetype."""
    sensitivities = {a: p.difficulty_sensitivity for a, p in ARCHETYPE_PROFILES.items()}
    assert sensitivities[LearnerArchetype.CONTEXT_SENSITIVE] == max(sensitivities.values())
    assert (
        sensitivities[LearnerArchetype.CONTEXT_SENSITIVE]
        > 2 * sensitivities[LearnerArchetype.STABLE]
    )


def test_context_sensitive_latency_correlates_with_recorded_difficulty() -> None:
    """Latency must respond to difficulty, and the correlation must be the expected sign."""
    events = generate_session(
        make_config(archetype=LearnerArchetype.CONTEXT_SENSITIVE, question_count=25)
    ).events
    pairs: list[tuple[float, float]] = []
    started: dict[str, float] = {}
    for event in events:
        if event.event_type is EventType.QUESTION_STARTED:
            assert isinstance(event.payload, QuestionStartedPayload)
            started[event.payload.question_id] = event.payload.difficulty or 0.0
        elif event.event_type is EventType.QUESTION_ANSWERED:
            assert isinstance(event.payload, QuestionAnsweredPayload)
            pairs.append((started[event.payload.question_id], event.payload.response_seconds))

    assert len(pairs) == 25
    n = len(pairs)
    mean_d = sum(d for d, _ in pairs) / n
    mean_r = sum(r for _, r in pairs) / n
    covariance = sum((d - mean_d) * (r - mean_r) for d, r in pairs)
    var_d = sum((d - mean_d) ** 2 for d, _ in pairs)
    var_r = sum((r - mean_r) ** 2 for _, r in pairs)
    correlation = covariance / (var_d**0.5 * var_r**0.5)
    assert correlation > 0.3, (
        f"expected a positive difficulty-latency relationship, got {correlation}"
    )


def test_fixed_difficulty_removes_the_difficulty_response() -> None:
    """With difficulty pinned, difficulty must stop explaining any latency variation."""
    events = generate_session(
        make_config(
            archetype=LearnerArchetype.CONTEXT_SENSITIVE,
            question_count=20,
            difficulty_min=0.5,
            difficulty_max=0.5,
        )
    ).events
    difficulties = {
        e.payload.difficulty for e in events if e.event_type is EventType.QUESTION_STARTED
    }
    assert difficulties == {0.5}


def test_noisy_archetype_has_wider_spread_than_stable() -> None:
    """The defining property of NOISY is dispersion, so it must be measurably wider."""

    def coefficient_of_variation(archetype: LearnerArchetype) -> float:
        """Return mean-normalised dispersion of latencies for an archetype.

        Args:
            archetype: The archetype to sample.

        Returns:
            Latency standard deviation divided by the mean.
        """
        pooled: list[float] = []
        for seed in range(8):
            pooled.extend(
                latency_series(
                    generate_session(
                        make_config(archetype=archetype, base_seed=seed, question_count=15)
                    ).events
                )
            )
        mean = sum(pooled) / len(pooled)
        variance = sum((x - mean) ** 2 for x in pooled) / len(pooled)
        return variance**0.5 / mean

    assert coefficient_of_variation(LearnerArchetype.NOISY) > 1.5 * coefficient_of_variation(
        LearnerArchetype.STABLE
    )


# --------------------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------------------


def test_streaming_iterator_matches_materialised_result() -> None:
    """The iterator and the tuple-returning API must agree exactly."""
    cfg = make_config(question_count=9, include_intervention=True, intervention_after_answers=3)
    streamed = list(generate_session_events(cfg))
    materialised = generate_session(cfg).events
    assert [e.model_dump_json() for e in streamed] == [e.model_dump_json() for e in materialised]


def test_generator_source_contains_no_wall_clock_reads() -> None:
    """The generator must never read the host clock.

    This is a static check because the failure it guards against is invisible to a
    behavioural test: a wall-clock read only shows up as irreproducible output on a
    different day. The project's rule is that time enters the engine through
    ``SimulationConfig.start_time`` and nowhere else.
    """
    import inspect

    import focus_engine.simulator.generator as generator_module

    source = inspect.getsource(generator_module)
    for forbidden in ("utc_now", "datetime.now", "time.time", "SystemClock"):
        assert forbidden not in source, f"generator must not reference {forbidden}"
