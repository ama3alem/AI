"""Deterministic synthetic learner event generator.

Generates a sequence of typed events for one synthetic session from a
:class:`~focus_engine.simulator.config.SimulationConfig`. The properties this module exists
to guarantee:

* **Deterministic by construction.** The seed is derived from ``(base_seed, archetype,
  learner_id, session_id)`` via :func:`~focus_engine.utils.determinism.derive_seed`, and the
  session start time comes from the config rather than the wall clock. The same
  configuration therefore produces a byte-identical event stream regardless of when or
  where it runs.
* **Identifiers that cannot collide within a run.** Event and question identifiers are
  derived by hashing a per-session seed with a unique index, so uniqueness is structural
  rather than probabilistic.
* **Events only.** Output is :class:`~focus_engine.events.types.EventEnvelope` using the
  closed 18-type taxonomy. No derived metrics, no interpretations, no labels about
  internal state.
* **Provenance in-band.** Every event carries ``origin=SYNTHETIC``,
  ``provenance=SYNTHETIC_LABEL``, and a
  :class:`~focus_engine.schemas.primitives.SyntheticDataStamp` holding the literal warning
  ``SYNTHETIC DATA — NOT REAL STUDENT DATA``. The stamp travels inside the data so the
  artefacts stay self-identifying after being copied or reloaded by a tool that never read
  the documentation.
* **Strictly monotonic timestamps.** Within a session every timestamp is strictly greater
  than the one before it.
* **Internally consistent payloads.** A ``question_answered`` payload's
  ``response_seconds`` equals the interval between its ``question_started`` and
  ``question_answered`` timestamps, and ``session_ended`` carries the true session
  duration. Downstream code can therefore trust the payload instead of recomputing it.

The generated stream is deliberately question-centric. Richer event mixes (video,
navigation, inactivity) are a later concern; adding them here would widen the taxonomy's
usage without evidence that any consumer needs them yet.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import numpy as np

from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    InterventionCompletedPayload,
    InterventionStartedPayload,
    QuestionAnsweredPayload,
    QuestionStartedPayload,
    SessionEndedPayload,
    SessionStartedPayload,
)
from focus_engine.schemas.primitives import (
    DataOrigin,
    Provenance,
    SyntheticDataStamp,
)
from focus_engine.simulator.archetypes import (
    ARCHETYPE_PROFILES,
    ArchetypeProfile,
    LatencyShape,
)
from focus_engine.simulator.config import (
    MAX_SESSION_EVENTS,
    SIMULATOR_VERSION,
    SYNTHETIC_WARNING,
    SimulationConfig,
    SimulatorProvenance,
)
from focus_engine.utils.determinism import derive_seed, stable_hash_int

__all__ = [
    "MIN_RESPONSE_SECONDS",
    "SimulatorResult",
    "generate_session",
    "generate_session_events",
    "session_seed",
]


#: Floor on a generated response latency, in seconds. Without it, a heavily jittered
#: draw could produce a sub-second or negative latency and make the event's
#: ``response_seconds`` inconsistent with its timestamps.
MIN_RESPONSE_SECONDS: Final[float] = 0.5

#: Fraction of a session at which a ``DIP_AND_RECOVER`` archetype reaches peak drift.
_DIP_POSITION: Final[float] = 0.4

#: How much of the difficulty axis translates into accuracy, scaled by the archetype's
#: sensitivity. At full sensitivity, the hardest item costs 25 accuracy points.
_DIFFICULTY_ACCURACY_WEIGHT: Final[float] = 0.25

#: The payload types this generator emits. Narrower than the envelope's full union, and
#: declared here so the emission helper is type-checked rather than silenced.
_EmittedPayload = (
    SessionEndedPayload
    | SessionStartedPayload
    | QuestionStartedPayload
    | QuestionAnsweredPayload
    | InterventionStartedPayload
    | InterventionCompletedPayload
)


def session_seed(cfg: SimulationConfig) -> int:
    """Derive the generator seed for a session.

    The seed is a pure function of the configuration's identifying fields, so two
    configurations that differ only in timing bounds share a random stream while
    configurations for different learners, sessions, or archetypes do not.

    Args:
        cfg: Simulation configuration.

    Returns:
        A seed in ``[0, 2**32)``.
    """
    return derive_seed(
        cfg.base_seed,
        "simulator",
        cfg.archetype.value,
        str(cfg.learner_id),
        str(cfg.session_id),
    )


def _event_id(seed: int, index: int) -> str:
    """Build a collision-free event identifier for a position in the session.

    Args:
        seed: The session's derived seed.
        index: Zero-based position of the event within the session.

    Returns:
        An opaque event identifier.
    """
    digest = stable_hash_int("focus_engine/simulator/event", str(seed), str(index), bits=64)
    return f"sev-{digest:016x}"


def _question_id(cfg: SimulationConfig, seed: int, index: int) -> str:
    """Build an opaque question identifier.

    The index makes an identifier traceable within a session for debugging without
    encoding any semantic content about the question.

    Args:
        cfg: Simulation configuration, supplying the id prefix.
        seed: The session's derived seed.
        index: One-based question number.

    Returns:
        An opaque question identifier.
    """
    digest = stable_hash_int("focus_engine/simulator/question", str(seed), str(index), bits=32)
    return f"{cfg.question_id_prefix}-{index:03d}-{digest:08x}"


def _clamp01(value: float) -> float:
    """Clamp a probability into ``[0, 1]``.

    Args:
        value: The raw probability.

    Returns:
        The clamped probability.
    """
    return min(1.0, max(0.0, value))


def _drift_progress(shape: LatencyShape, index: int, total: int) -> float:
    """Return how far a session has progressed along an archetype's degradation curve.

    Args:
        shape: The archetype's drift shape.
        index: One-based question number.
        total: Total questions in the session.

    Returns:
        A non-negative progress value. ``0.0`` means no drift, larger means more. For
        ``DIP_AND_RECOVER`` the value rises to a peak partway through the session and
        returns to ``0.0`` at the final question, so the archetype ends at baseline.
    """
    if shape is LatencyShape.FLAT:
        return 0.0
    if shape is LatencyShape.LINEAR_DECLINE:
        return float(index)

    last_index = max(total - 1, 1)
    peak = max(1, round(_DIP_POSITION * last_index))
    if index <= peak:
        return float(index)
    tail = last_index - peak
    if tail <= 0:
        return float(index)
    return peak * (1.0 - (index - peak) / tail)


def _sample_delay(rng: np.random.Generator, lo: timedelta, hi: timedelta) -> timedelta:
    """Sample a delay uniformly from ``[lo, hi]``.

    Args:
        rng: The session's seeded generator.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        The sampled delay, at least ``lo``.
    """
    lo_seconds = lo.total_seconds()
    hi_seconds = hi.total_seconds()
    if hi_seconds <= lo_seconds:
        return lo
    return timedelta(seconds=float(rng.uniform(lo_seconds, hi_seconds)))


def _resolve_emitted_count(
    cfg: SimulationConfig, profile: ArchetypeProfile, rng: np.random.Generator
) -> int:
    """Decide how many questions the session will actually emit.

    Dropout is a configuration decision, not an implicit one. With ``enable_dropout`` off
    the answer is always ``question_count``, which makes structural assertions in tests
    exact. With it on, completion is drawn once from the archetype's
    ``session_completion_rate`` and, on failure, a truncation point is drawn so the session
    ends early with a distinct ``session_ended`` reason.

    Args:
        cfg: Simulation configuration.
        profile: The archetype's coefficients.
        rng: The session's seeded generator.

    Returns:
        The number of questions to emit, between 1 and ``question_count``.
    """
    if not cfg.enable_dropout:
        return cfg.question_count
    if cfg.question_count <= cfg.dropout_min_questions:
        return cfg.question_count
    if float(rng.random()) < profile.session_completion_rate:
        return cfg.question_count
    lowest = cfg.dropout_min_questions
    highest = cfg.question_count - 1
    return int(rng.integers(lowest, highest + 1))


@dataclass(frozen=True, slots=True)
class SimulatorResult:
    """A complete generated session and its provenance.

    Attributes:
        events: Events in temporal order.
        provenance: Reproducible record of how the session was generated.
    """

    events: tuple[EventEnvelope, ...]
    provenance: SimulatorProvenance


def _generate(cfg: SimulationConfig) -> tuple[tuple[EventEnvelope, ...], int, int]:
    """Generate a session's events.

    Args:
        cfg: Simulation configuration.

    Returns:
        A tuple of the events in order, the derived seed, and the emitted question count.
    """
    profile = ARCHETYPE_PROFILES[cfg.archetype]
    seed = session_seed(cfg)
    rng = np.random.default_rng(seed)
    stamp = SyntheticDataStamp(generator=SIMULATOR_VERSION, seed=seed)

    emitted_questions = _resolve_emitted_count(cfg, profile, rng)

    def envelope(
        index: int,
        event_type: EventType,
        timestamp: datetime,
        payload: _EmittedPayload,
    ) -> EventEnvelope:
        """Build one stamped, provenance-tagged envelope.

        Args:
            index: Zero-based position in the session.
            event_type: The event type.
            timestamp: The event timestamp.
            payload: The type-specific payload.

        Returns:
            A validated envelope marked as simulated.
        """
        return EventEnvelope(
            event_id=_event_id(seed, index),
            learner_id=cfg.learner_id,
            session_id=cfg.session_id,
            timestamp=timestamp,
            event_type=event_type,
            payload=payload,
            origin=DataOrigin.SYNTHETIC,
            provenance=Provenance.SYNTHETIC_LABEL,
            synthetic_stamp=stamp,
        )

    events: list[EventEnvelope] = []
    cursor = cfg.start_time
    index = 0

    events.append(
        envelope(
            index,
            EventType.SESSION_STARTED,
            cursor,
            SessionStartedPayload(platform="synthetic", entry_point="synthetic/session"),
        )
    )
    index += 1

    intervention_emitted = False

    for question_number in range(1, emitted_questions + 1):
        started_at = cursor + _sample_delay(rng, cfg.inter_event_gap_min, cfg.inter_event_gap_max)
        difficulty = float(rng.uniform(cfg.difficulty_min, cfg.difficulty_max))
        question_id = _question_id(cfg, seed, question_number)

        events.append(
            envelope(
                index,
                EventType.QUESTION_STARTED,
                started_at,
                QuestionStartedPayload(
                    question_id=question_id,
                    question_type=cfg.question_type,
                    difficulty=round(difficulty, 3),
                ),
            )
        )
        index += 1

        progress = _drift_progress(profile.latency_shape, question_number, emitted_questions)
        if intervention_emitted:
            progress *= profile.response_to_intervention

        difficulty_latency = 1.0 + profile.difficulty_sensitivity * (2.0 * difficulty - 1.0)
        difficulty_accuracy = (
            profile.difficulty_sensitivity * _DIFFICULTY_ACCURACY_WEIGHT * (2.0 * difficulty - 1.0)
        )

        expected_latency = (
            profile.base_response_seconds
            * difficulty_latency
            * (1.0 + profile.latency_drift_per_question * progress)
        )
        jitter = float(rng.normal(0.0, profile.noise_scale * expected_latency))
        response_seconds = round(max(expected_latency + jitter, MIN_RESPONSE_SECONDS), 3)

        answered_at = started_at + timedelta(seconds=response_seconds)
        accuracy = _clamp01(
            profile.base_accuracy
            + profile.accuracy_drift_per_question * progress
            - difficulty_accuracy
            + float(rng.normal(0.0, profile.accuracy_noise))
        )
        correct = bool(rng.random() < accuracy)

        events.append(
            envelope(
                index,
                EventType.QUESTION_ANSWERED,
                answered_at,
                QuestionAnsweredPayload(
                    question_id=question_id,
                    correct=correct,
                    response_seconds=response_seconds,
                ),
            )
        )
        index += 1
        cursor = answered_at

        if (
            cfg.include_intervention
            and not intervention_emitted
            and question_number == cfg.intervention_after_answers
        ):
            intervention_started_at = cursor + cfg.intervention_lag
            duration = _sample_delay(
                rng, cfg.intervention_duration_min, cfg.intervention_duration_max
            )
            intervention_completed_at = intervention_started_at + duration

            events.append(
                envelope(
                    index,
                    EventType.INTERVENTION_STARTED,
                    intervention_started_at,
                    InterventionStartedPayload(
                        intervention_id=cfg.intervention_id,
                        intervention_type=cfg.intervention_type,
                        trigger_probability=round(float(rng.uniform(0.4, 0.95)), 3),
                        trigger_state="increasing_deviation",
                    ),
                )
            )
            index += 1
            events.append(
                envelope(
                    index,
                    EventType.INTERVENTION_COMPLETED,
                    intervention_completed_at,
                    InterventionCompletedPayload(
                        intervention_id=cfg.intervention_id,
                        outcome=cfg.intervention_outcome,
                        response_seconds=duration.total_seconds(),
                    ),
                )
            )
            index += 1
            intervention_emitted = True
            cursor = intervention_completed_at

    completed = emitted_questions == cfg.question_count
    ended_at = cursor + _sample_delay(rng, cfg.session_end_delay_min, cfg.session_end_delay_max)
    events.append(
        envelope(
            index,
            EventType.SESSION_ENDED,
            ended_at,
            SessionEndedPayload(
                duration_seconds=(ended_at - cfg.start_time).total_seconds(),
                reason=cfg.session_end_reason if completed else cfg.dropout_reason,
            ),
        )
    )

    return tuple(events), seed, emitted_questions


def generate_session_events(cfg: SimulationConfig) -> Iterator[EventEnvelope]:
    """Generate a session's events as a streaming iterator.

    Args:
        cfg: Simulation configuration.

    Yields:
        Event envelopes in strictly increasing timestamp order.

    Raises:
        ValueError: If the configuration is invalid or the session would exceed
            :data:`~focus_engine.simulator.config.MAX_SESSION_EVENTS`.
    """
    events, _, _ = _generate(cfg)
    if len(events) > MAX_SESSION_EVENTS:  # pragma: no cover - guarded by config bounds
        raise ValueError(
            f"session would emit {len(events)} events, exceeding the limit of {MAX_SESSION_EVENTS}"
        )
    return iter(events)


def generate_session(cfg: SimulationConfig) -> SimulatorResult:
    """Generate a complete synthetic session with its provenance.

    The returned stream is checked against the invariants downstream code is entitled to
    assume: strictly increasing timestamps, unique event identifiers, non-empty content,
    and a synthetic stamp on every event.

    Args:
        cfg: Simulation configuration.

    Returns:
        The generated events and their provenance.

    Raises:
        ValueError: If generation violates an invariant. A violation here is a simulator
            bug, and failing loudly is preferable to emitting a stream that quietly breaks
            a consumer's assumptions.
    """
    events, seed, emitted_questions = _generate(cfg)
    profile = ARCHETYPE_PROFILES[cfg.archetype]

    if not events:
        raise ValueError("generated session produced no events")

    previous: datetime | None = None
    seen_ids: set[str] = set()
    for event in events:
        if previous is not None and event.timestamp <= previous:
            raise ValueError(
                f"non-monotonic timestamps in session {cfg.session_id}: "
                f"{previous.isoformat()} -> {event.timestamp.isoformat()}"
            )
        previous = event.timestamp
        if event.event_id in seen_ids:
            raise ValueError(f"duplicate event id generated: {event.event_id}")
        seen_ids.add(event.event_id)
        if event.origin is not DataOrigin.SYNTHETIC:
            raise ValueError(f"event {event.event_id} is not marked synthetic")
        if event.synthetic_stamp is None or event.synthetic_stamp.warning != SYNTHETIC_WARNING:
            raise ValueError(f"event {event.event_id} is missing the synthetic warning stamp")

    return SimulatorResult(
        events=events,
        provenance=SimulatorProvenance(
            archetype=cfg.archetype,
            learner_id=cfg.learner_id,
            session_id=cfg.session_id,
            base_seed=cfg.base_seed,
            derived_seed=seed,
            planned_question_count=cfg.question_count,
            emitted_question_count=emitted_questions,
            pattern_measurable=profile.pattern_is_measurable(emitted_questions),
            warning=SYNTHETIC_WARNING,
        ),
    )
