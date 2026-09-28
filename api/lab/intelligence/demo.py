"""Phase 15 demonstration scenarios: deterministic, topic-tagged learner streams.

The canonical store stays exactly five learners and is byte-identical (tests pin the
count, the id tuple, and the roster). The Phase 15 demonstrations therefore live in a
*separate* store built from their own streams. Canonical and demo stores are disjoint:
no demo learner appears in ``known_learner_ids()`` and no canonical id is reused.

**Why topic metadata at all?** The engine never emits a topic, so the canonical corpus is
subject-blind and the canonical gap producer correctly says so. The demo streams
deterministically re-tag every ``question_started`` envelope after generation — the same
way a real deployment would attach content metadata at ingest — which is what makes
subject-scoped reasoning observable at all in this repository.

**Determinism.** Every stream is a pure function of fixed seeds, ids, and anchors. The
same call twice returns the same events, and rebuilding the store is byte-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Final

from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    QuestionAnsweredPayload,
    QuestionStartedPayload,
    SessionEndedPayload,
    SessionStartedPayload,
)
from focus_engine.schemas.primitives import DataOrigin, Provenance, SyntheticDataStamp
from focus_engine.simulator import LearnerArchetype, SimulationConfig, generate_session

__all__ = [
    "INTELLIGENCE_DEMO_IDS",
    "IntelligenceDemoScenario",
    "demo_learner_ids",
    "get_demo_events",
    "get_demo_scenario",
]

#: One fixed instant past which nothing in a demo stream is scheduled, kept consistent with
#: the lab's deterministic clock so staleness and recency read sensibly.
_DEMO_ANCHOR: Final[datetime] = datetime(2026, 6, 14, 23, 30, tzinfo=UTC)

_BASE_BASELINE_TOPICS: Final[tuple[str, ...]] = ("algebra", "reading", "geometry", "reading")
_ALGEBRA: Final[str] = "algebra"
_READING: Final[str] = "reading"
_GEOMETRY: Final[str] = "geometry"


class IntelligenceDemoScenario:
    """One Phase 15 demonstration learner and its deterministic stream."""

    __slots__ = ("id", "learner_id", "title", "archetype", "description", "build")

    def __init__(
        self,
        *,
        id: str,
        learner_id: str,
        title: str,
        description: str,
        build: Callable[[], tuple[EventEnvelope, ...]],
    ) -> None:
        self.id = id
        self.learner_id = learner_id
        self.title = title
        self.archetype = "demo"
        self.description = description
        self.build = build

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"IntelligenceDemoScenario(id={self.id!r})"


def _topic_retag(events: Sequence[EventEnvelope], topic: str) -> tuple[EventEnvelope, ...]:
    """Attach one topic label to every question in a generated session.

    Args:
        events: A generated session's envelopes.
        topic: The topic every ``question_started`` in the session carries.

    Returns:
        A new session tuple with the topic injected on each question.
    """

    def retag(event: EventEnvelope) -> EventEnvelope:
        payload = event.payload
        if event.event_type is EventType.QUESTION_STARTED and isinstance(
            payload, QuestionStartedPayload
        ):
            return event.model_copy(update={"payload": payload.model_copy(update={"topic": topic})})
        return event

    return tuple(retag(event) for event in events)


def _plan_generated_sessions(
    *,
    learner_id: str,
    prefix: str,
    count: int,
    archetype: LearnerArchetype,
    anchor: datetime,
    per_day: int,
    difficulty_min: float | Callable[[int], float],
    difficulty_max: float | Callable[[int], float],
    topic_for: Callable[[int], str],
    base_seed: int,
    question_count: int = 20,
) -> tuple[EventEnvelope, ...]:
    """Generate ``count`` deterministic sessions tagged with per-session topics."""
    out: list[EventEnvelope] = []
    for index in range(count):
        day = index // per_day
        slot = index % per_day
        dmin = difficulty_min(index) if callable(difficulty_min) else difficulty_min
        dmax = difficulty_max(index) if callable(difficulty_max) else difficulty_max
        session_id = f"{prefix}{index:03d}"
        config = SimulationConfig(
            archetype=archetype,
            learner_id=learner_id,
            session_id=session_id,
            base_seed=base_seed + index * 37,
            question_count=question_count,
            start_time=anchor + timedelta(days=day) + timedelta(hours=8 + slot),
            question_type="multiple_choice",
            difficulty_min=dmin,
            difficulty_max=dmax,
            inter_event_gap_min=timedelta(seconds=45),
            inter_event_gap_max=timedelta(seconds=90),
            session_end_delay_min=timedelta(seconds=2),
            session_end_delay_max=timedelta(seconds=10),
        )
        events = generate_session(config).events
        out.extend(_topic_retag(events, topic_for(index)))
    return tuple(out)


def _authored_session(
    *,
    learner_id: str,
    session_id: str,
    start: datetime,
    questions: Sequence[tuple[str, float, bool, float]],
    seed: int,
    gap_seconds: float = 1.5,
) -> tuple[EventEnvelope, ...]:
    """Build the decision session by hand: exact topics, difficulty, and timings.

    Eight answers precede the cut and eight follow it, so the outcome and evaluation
    stages have real post-decision evidence to read without the simulation's noise.

    Args:
        learner_id: The pseudonymous learner.
        session_id: The decision session's id.
        start: The session's start instant.
        questions: ``(topic, difficulty, correct, response_seconds)`` per question.
        seed: Seed recorded on this session's synthetic stamps.
        gap_seconds: Fixed inter-event gap, in seconds, for a dense post-cut stream.

    Returns:
        The decision session's envelopes, strictly increasing in time.
    """
    events: list[EventEnvelope] = []
    cursor = start
    sequence = 0

    def put(timestamp: datetime, event_type: EventType, payload: object) -> None:
        nonlocal sequence
        events.append(
            EventEnvelope(
                event_id=f"ev-{learner_id}-{session_id}-{sequence:04d}",
                learner_id=learner_id,
                session_id=session_id,
                timestamp=timestamp,
                event_type=event_type,
                payload=payload,
                origin=DataOrigin.SYNTHETIC,
                provenance=Provenance.SYNTHETIC_LABEL,
                synthetic_stamp=SyntheticDataStamp(generator="focus_engine.lab.demo/v1", seed=seed),
            )
        )
        sequence += 1

    put(
        cursor,
        EventType.SESSION_STARTED,
        SessionStartedPayload(platform="synthetic", entry_point="synthetic/session"),
    )
    cursor += timedelta(seconds=gap_seconds)
    for index, (topic, difficulty, correct, response) in enumerate(questions):
        question_id = f"iq-{learner_id[-4:]}-{index:04d}"
        started_at = cursor
        put(
            started_at,
            EventType.QUESTION_STARTED,
            QuestionStartedPayload(
                question_id=question_id,
                question_type="multiple_choice",
                difficulty=difficulty,
                topic=topic,
            ),
        )
        answered_at = started_at + timedelta(seconds=response)
        put(
            answered_at,
            EventType.QUESTION_ANSWERED,
            QuestionAnsweredPayload(
                question_id=question_id,
                correct=correct,
                response_seconds=response,
                attempt_number=1,
            ),
        )
        cursor = answered_at + timedelta(seconds=gap_seconds)
    ended_at = cursor + timedelta(seconds=gap_seconds)
    put(
        ended_at,
        EventType.SESSION_ENDED,
        SessionEndedPayload(
            duration_seconds=(ended_at - start).total_seconds(), reason="completed"
        ),
    )
    return tuple(events)


# --------------------------------------------------------------------------------------
# Decision-session question plans
# --------------------------------------------------------------------------------------

#: The full-lifecycle learner's decision session: algebra is failing and slow before the cut;
#: the decline continues after it, while reading still answers correctly and fast.
#:
#: The pacing is deliberate, not cosmetic. Prediction features are drawn from the decision
#: session alone, and the trend feature needs at least three answers to fall *before* the
#: recent 15-minute window. So the session opens with three fast, correct reading answers
#: (the healthy early phase), then spends six slow, wrong algebra answers descending across
#: more than fifteen minutes before the cut, which the runner places at the session's
#: midpoint during the slow phase.
_FULL_DECISION_QUESTIONS: Final[tuple[tuple[str, float, bool, float], ...]] = (
    (_READING, 0.55, True, 12.0),
    (_READING, 0.58, True, 15.0),
    (_READING, 0.56, True, 13.0),
    (_ALGEBRA, 0.85, False, 200.0),
    (_ALGEBRA, 0.87, False, 200.0),
    (_ALGEBRA, 0.84, False, 200.0),
    (_ALGEBRA, 0.86, False, 200.0),
    (_ALGEBRA, 0.83, False, 200.0),
    (_ALGEBRA, 0.85, False, 200.0),
    (_ALGEBRA, 0.84, False, 220.0),
    (_ALGEBRA, 0.86, False, 220.0),
    (_ALGEBRA, 0.85, False, 220.0),
    (_ALGEBRA, 0.87, False, 220.0),
    (_ALGEBRA, 0.83, False, 220.0),
    (_ALGEBRA, 0.88, False, 220.0),
    (_ALGEBRA, 0.85, False, 220.0),
)

#: The conflict learner's decision session mirrors the contradiction present in its run:
#: reading is calm and correct while algebra fails and slows. The same long pre-cut pacing
#: guarantees the trend feature has a full window on each side of the cut.
_CONFLICT_DECISION_QUESTIONS: Final[tuple[tuple[str, float, bool, float], ...]] = (
    (_READING, 0.58, True, 12.0),
    (_READING, 0.60, True, 14.0),
    (_READING, 0.59, True, 13.0),
    (_ALGEBRA, 0.85, False, 200.0),
    (_ALGEBRA, 0.84, False, 200.0),
    (_ALGEBRA, 0.86, False, 200.0),
    (_ALGEBRA, 0.83, False, 200.0),
    (_ALGEBRA, 0.87, False, 200.0),
    (_ALGEBRA, 0.85, False, 200.0),
    (_ALGEBRA, 0.86, False, 118.0),
    (_READING, 0.61, True, 26.0),
    (_ALGEBRA, 0.84, False, 120.0),
    (_READING, 0.60, True, 25.0),
    (_ALGEBRA, 0.85, False, 118.0),
    (_READING, 0.62, True, 24.0),
    (_ALGEBRA, 0.87, False, 110.0),
)

#: The recovery learner's decision session is the inversion of the decline: a slow, failing
#: opening gives way to a long, steady, mostly-correct stretch before the cut and after it.
#: Both trend windows therefore resolve -- the early window is the bad phase, the recent
#: window the recovered one -- and a "no decline" probability reading is the point.
_RECOVERY_DECISION_QUESTIONS: Final[tuple[tuple[str, float, bool, float], ...]] = (
    (_ALGEBRA, 0.84, False, 300.0),
    (_ALGEBRA, 0.86, False, 300.0),
    (_ALGEBRA, 0.83, False, 300.0),
    (_ALGEBRA, 0.55, True, 200.0),
    (_READING, 0.52, True, 200.0),
    (_ALGEBRA, 0.54, True, 200.0),
    (_READING, 0.55, True, 200.0),
    (_ALGEBRA, 0.53, True, 200.0),
    (_READING, 0.54, True, 200.0),
    (_ALGEBRA, 0.55, True, 45.0),
    (_READING, 0.53, True, 45.0),
    (_ALGEBRA, 0.56, True, 45.0),
    (_READING, 0.55, True, 45.0),
    (_ALGEBRA, 0.54, True, 45.0),
    (_READING, 0.56, True, 45.0),
)


def _build_full() -> tuple[EventEnvelope, ...]:
    anchor = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    baseline = _plan_generated_sessions(
        learner_id="intel-full-0001",
        prefix="intel-full-s-",
        count=96,
        archetype=LearnerArchetype.STABLE,
        anchor=anchor,
        per_day=8,
        difficulty_min=0.22,
        difficulty_max=0.44,
        topic_for=lambda index: _BASE_BASELINE_TOPICS[index % len(_BASE_BASELINE_TOPICS)],
        base_seed=900_001,
    )
    decline = _plan_generated_sessions(
        learner_id="intel-full-0001",
        prefix="intel-full-d-",
        count=12,
        archetype=LearnerArchetype.GRADUAL_DECLINE,
        anchor=datetime(2026, 6, 13, 8, 0, tzinfo=UTC),
        per_day=6,
        difficulty_min=lambda index: 0.55 + index * 0.02,
        difficulty_max=lambda index: 0.75 + index * 0.02,
        topic_for=lambda _index: _ALGEBRA,
        base_seed=900_101,
    )
    authored = _authored_session(
        learner_id="intel-full-0001",
        session_id="intel-full-dcs",
        start=_DEMO_ANCHOR,
        questions=_FULL_DECISION_QUESTIONS,
        seed=900_201,
    )
    return (*baseline, *decline, *authored)


def _build_conflict() -> tuple[EventEnvelope, ...]:
    anchor = datetime(2026, 5, 20, 8, 0, tzinfo=UTC)
    algebra: list[EventEnvelope] = []
    reading: list[EventEnvelope] = []
    for pair in range(24):
        day = anchor + timedelta(days=pair)
        algebra.extend(
            _plan_generated_sessions(
                learner_id="intel-conflict-0001",
                prefix=f"intel-conflict-a{pair:02d}-",
                count=1,
                archetype=LearnerArchetype.GRADUAL_DECLINE,
                anchor=day,
                per_day=1,
                difficulty_min=0.55 + pair * 0.008,
                difficulty_max=0.78,
                topic_for=lambda _index: _ALGEBRA,
                base_seed=910_001 + pair * 13,
            )
        )
        reading.extend(
            _plan_generated_sessions(
                learner_id="intel-conflict-0001",
                prefix=f"intel-conflict-r{pair:02d}-",
                count=1,
                archetype=LearnerArchetype.STABLE,
                anchor=day + timedelta(seconds=25200),
                per_day=1,
                difficulty_min=max(0.52 - pair * 0.012, 0.16),
                difficulty_max=max(0.74 - pair * 0.012, 0.34),
                topic_for=lambda _index: _READING,
                base_seed=920_001 + pair * 17,
            )
        )
    authored = _authored_session(
        learner_id="intel-conflict-0001",
        session_id="intel-conflict-dcs",
        start=_DEMO_ANCHOR,
        questions=_CONFLICT_DECISION_QUESTIONS,
        seed=930_001,
    )
    return (*algebra, *reading, *authored)


def _build_recovery() -> tuple[EventEnvelope, ...]:
    anchor = datetime(2026, 5, 24, 8, 0, tzinfo=UTC)
    decline = _plan_generated_sessions(
        learner_id="intel-recovery-0001",
        prefix="intel-recovery-d-",
        count=20,
        archetype=LearnerArchetype.GRADUAL_DECLINE,
        anchor=anchor,
        per_day=4,
        difficulty_min=0.55,
        difficulty_max=0.80,
        topic_for=lambda index: _BASELINE_TOPICS_ALT[index % len(_BASELINE_TOPICS_ALT)],
        base_seed=940_001,
    )
    improving = _plan_generated_sessions(
        learner_id="intel-recovery-0001",
        prefix="intel-recovery-i-",
        count=40,
        archetype=LearnerArchetype.RECOVERY,
        anchor=anchor + timedelta(days=5),
        per_day=4,
        difficulty_min=0.28,
        difficulty_max=0.45,
        topic_for=lambda index: _BASELINE_TOPICS_ALT[index % len(_BASELINE_TOPICS_ALT)],
        base_seed=950_001,
    )
    authored = _authored_session(
        learner_id="intel-recovery-0001",
        session_id="intel-recovery-dcs",
        start=_DEMO_ANCHOR,
        questions=_RECOVERY_DECISION_QUESTIONS,
        seed=960_001,
    )
    return (*decline, *improving, *authored)


def _build_thin() -> tuple[EventEnvelope, ...]:
    anchor = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    topics: tuple[str, ...] = (_ALGEBRA, _READING, _GEOMETRY)
    return _plan_generated_sessions(
        learner_id="intel-thin-0001",
        prefix="intel-thin-s-",
        count=3,
        archetype=LearnerArchetype.STABLE,
        anchor=anchor,
        per_day=1,
        difficulty_min=0.5,
        difficulty_max=0.62,
        topic_for=lambda index: topics[index % len(topics)],
        base_seed=970_001,
        question_count=4,
    )


#: Alternate topic rotation used by the recovery learner.
_BASELINE_TOPICS_ALT: Final[tuple[str, ...]] = ("algebra", "reading", "geometry")

#: The four Phase 15 demonstration scenarios, in a fixed, stable order.
DEMO_SCENARIOS: Final[tuple[IntelligenceDemoScenario, ...]] = (
    IntelligenceDemoScenario(
        id="intel-full",
        learner_id="intel-full-0001",
        title=(
            "INTELLIGENCE DEMO 1 — FULL ENTRYPOINT: MULTIDIMENSIONAL DECLINE, "
            "SUBJECT SPLIT, DELIVERED INTERVENTION"
        ),
        description=(
            "A mature learner who declines in algebra while reading holds steady; the "
            "decline coincides with a rise in content difficulty and a delivered "
            "intervention produces a measurable outcome and an evaluated prediction."
        ),
        build=_build_full,
    ),
    IntelligenceDemoScenario(
        id="intel-thin",
        learner_id="intel-thin-0001",
        title="INTELLIGENCE DEMO 2 — SPARSE DATA · REFUSAL PATH",
        description=(
            "Three short sessions. Too little history to establish a baseline, so the "
            "engine refuses rather than invents, in every intent that needs one."
        ),
        build=_build_thin,
    ),
    IntelligenceDemoScenario(
        id="intel-conflict",
        learner_id="intel-conflict-0001",
        title="INTELLIGENCE DEMO 3 — CONTRADICTORY SUBJECT SIGNALS",
        description=(
            "Algebra declines while reading improves across the same weeks. The global "
            "reading is mixed, the subjects contradict each other, and the layer reports "
            "the disagreement instead of averaging it away."
        ),
        build=_build_conflict,
    ),
    IntelligenceDemoScenario(
        id="intel-recovery",
        learner_id="intel-recovery-0001",
        title="INTELLIGENCE DEMO 4 — DECLINE-THEN-RECOVERY TIMELINE",
        description=(
            "A decline that reverses: behaviour falls, then returns toward the learner's "
            "own baseline, giving a RECOVERING change state and a direction-specific "
            "reading of the trajectory."
        ),
        build=_build_recovery,
    ),
)

#: The Phase 15 demo learner ids, in demo order. Disjoint from the canonical store.
INTELLIGENCE_DEMO_IDS: Final[tuple[str, ...]] = tuple(spec.learner_id for spec in DEMO_SCENARIOS)

_CACHE: dict[str, tuple[EventEnvelope, ...]] = {}


def demo_learner_ids() -> tuple[str, ...]:
    """The Phase 15 demo learner ids, in scenario order."""
    return INTELLIGENCE_DEMO_IDS


def get_demo_scenario(learner_id: str) -> IntelligenceDemoScenario:
    """Return one scenario spec by learner id.

    Args:
        learner_id: The pseudonymous learner id.

    Returns:
        The matching scenario.

    Raises:
        ValueError: If the learner is not a demo learner. The message never lists the
            known ids, so an id probe cannot learn what ids exist.
    """
    for spec in DEMO_SCENARIOS:
        if spec.learner_id == learner_id:
            return spec
    raise ValueError(f"Unknown demo learner {learner_id!r}")


def get_demo_events(learner_id: str) -> tuple[EventEnvelope, ...]:
    """Return a demo learner's deterministic event stream, cached per process.

    Args:
        learner_id: The pseudonymous learner id.

    Returns:
        The flat, ordered event stream.

    Raises:
        ValueError: If the learner is not a demo learner.
    """
    spec = get_demo_scenario(learner_id)
    cached = _CACHE.get(learner_id)
    if cached is None:
        cached = spec.build()
        _CACHE[learner_id] = cached
    return cached
