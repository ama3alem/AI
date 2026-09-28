"""Phase B: five deterministic scenarios built on the real simulator.

Every event here comes from :func:`focus_engine.simulator.generate_session` — the same
generator the integration tests use. A scenario fixes *inputs* (archetype, seed, session
count, question count, spacing) and never fixes an *output*. What the engine decides is
whatever the engine decides; the scenario name says what was attempted, not what is
guaranteed.

**Why a scenario is several sessions, not one long one.** The feature engine emits one
vector per session, so the temporal engine is designed to receive one observation per
session — the pattern in ``tests/integration/test_temporal_integration.py``. Handing it
forty observations from a single sitting instead makes its run-length logic track
within-session noise, and every archetype collapses onto the same state. So each scenario
below is a short sequence of separate sessions for one learner, which is also the shape
real data would have.

**Honest archetype mapping.** The simulator offers eight archetypes and none of them is
called "rapid decline", so scenario 3 uses the gradual-decline generator over fewer
questions and tighter spacing. The compression is a property of the input, not a
different kind of learner, and the description says so.

All timestamps are UTC. Inter-event gaps are wide (40-90s) so a session's short window has
a populated comparison window by its later questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from focus_engine.events.types import EventEnvelope
from focus_engine.simulator import (
    MIN_QUESTIONS_PER_SESSION,
    LearnerArchetype,
    SimulationConfig,
    generate_session,
)

__all__ = [
    "SCENARIO_DEFS",
    "SCENARIO_IDS",
    "DEFAULT_SAMPLE_SEED",
    "ScenarioDef",
    "canonical_example",
    "generate_sample_session",
    "generate_scenario",
]

#: Ordered for the UI's scenario picker.
SCENARIO_IDS: Final[tuple[str, ...]] = (
    "stable",
    "gradual_decline",
    "rapid_decline",
    "recovery",
    "insufficient_data",
)

_ANCHOR: Final[datetime] = datetime(2026, 6, 1, 8, 0, 0, tzinfo=UTC)
_ONE_DAY: Final[timedelta] = timedelta(days=1)


@dataclass(frozen=True, slots=True)
class ScenarioDef:
    """A scenario is a specification, not an expectation."""

    id: str
    title: str
    archetype: LearnerArchetype
    description: str
    expected_behavior: str
    learner_id: str
    sessions: int
    questions_per_session: int
    session_stride: timedelta
    base_seed: int
    gap_min: timedelta = timedelta(seconds=45)
    gap_max: timedelta = timedelta(seconds=90)

    def session_configs(self) -> tuple[SimulationConfig, ...]:
        """One config per session, same learner, advancing seeds and clock."""
        return tuple(
            SimulationConfig(
                archetype=self.archetype,
                learner_id=self.learner_id,
                session_id=f"{self.learner_id}-s{index + 1:02d}",
                base_seed=self.base_seed + index,
                question_count=self.questions_per_session,
                start_time=_ANCHOR + index * self.session_stride,
                inter_event_gap_min=self.gap_min,
                inter_event_gap_max=self.gap_max,
            )
            for index in range(self.sessions)
        )

    def build(self) -> tuple[tuple[EventEnvelope, ...], ...]:
        """Generate every session. Returns events per session, in session order."""
        return tuple(generate_session(config).events for config in self.session_configs())


SCENARIO_DEFS: Final[tuple[ScenarioDef, ...]] = (
    ScenarioDef(
        id="stable",
        title="SCENARIO 1 — STABLE LEARNER",
        archetype=LearnerArchetype.STABLE,
        description=(
            "Behaviour holds around baseline across eight separate sessions. Accuracy and "
            "response time stay roughly where they started, so each session is close to "
            "the learner's own recent self."
        ),
        expected_behavior=(
            "A stable or flat temporal state and a low deterioration signal. The policy "
            "should normally decline to intervene, and the reason it prints should say "
            "which constraint stopped it."
        ),
        learner_id="lab-stable-0001",
        sessions=24,
        questions_per_session=20,
        session_stride=_ONE_DAY,
        base_seed=900_001,
    ),
    ScenarioDef(
        id="gradual_decline",
        title="SCENARIO 2 — GRADUAL DECLINE",
        archetype=LearnerArchetype.GRADUAL_DECLINE,
        description=(
            "Behaviour progressively worsens across eight sessions. Accuracy falls and "
            "response time lengthens relative to the baseline the learner established in "
            "their earliest sessions."
        ),
        expected_behavior=(
            "A worsening direction building into a declining state once the evidence "
            "supports it. The probability should respond to the trend and uncertainty "
            "should report which constraint is binding. Whether the policy acts is the "
            "engine's decision, not this scenario's."
        ),
        learner_id="lab-decline-0001",
        sessions=24,
        questions_per_session=20,
        session_stride=_ONE_DAY,
        base_seed=900_002,
    ),
    ScenarioDef(
        id="rapid_decline",
        title="SCENARIO 3 — RAPID DECLINE",
        archetype=LearnerArchetype.GRADUAL_DECLINE,
        description=(
            "The same decline generator as scenario 2, compressed into six shorter "
            "sessions spaced six hours apart. There is no dedicated rapid-decline "
            "archetype in the simulator, so the speed here is a property of the input, "
            "not a different kind of learner."
        ),
        expected_behavior=(
            "Less evidence per session, so the maturity ladder and the evidence ceiling "
            "bind sooner. Watch whether the engine declines to claim more than the data "
            "supports — that restraint is the point of this scenario."
        ),
        learner_id="lab-rapid-0001",
        sessions=22,
        questions_per_session=16,
        session_stride=timedelta(hours=6),
        base_seed=900_003,
        gap_min=timedelta(seconds=40),
        gap_max=timedelta(seconds=80),
    ),
    ScenarioDef(
        id="recovery",
        title="SCENARIO 4 — RECOVERY",
        archetype=LearnerArchetype.RECOVERY,
        description=(
            "The learner deteriorates and then improves across ten sessions. Early "
            "sessions sit below their own baseline; later ones climb back toward it."
        ),
        expected_behavior=(
            "The direction should follow the evidence and change sign as the learner "
            "recovers, rather than latching onto the worst earlier session. A state that "
            "stays 'declining' at the end would be worth questioning."
        ),
        learner_id="lab-recovery-0001",
        sessions=26,
        questions_per_session=20,
        session_stride=_ONE_DAY,
        base_seed=900_004,
    ),
    ScenarioDef(
        id="insufficient_data",
        title="SCENARIO 5 — INSUFFICIENT DATA",
        archetype=LearnerArchetype.STABLE,
        description=(
            "Two sessions of three questions each. Far too little to establish a baseline, "
            "mature the evidence, or produce the trend features a prediction needs."
        ),
        expected_behavior=(
            "Refusal rather than a weak guess. The model should decline to score, "
            "uncertainty should name the missing features, and the policy should decline "
            "for want of a resolved verdict. A probability printed here would be the bug."
        ),
        learner_id="lab-thin-0001",
        sessions=2,
        questions_per_session=3,
        session_stride=_ONE_DAY,
        base_seed=900_005,
    ),
)

_BY_ID: Final[dict[str, ScenarioDef]] = {spec.id: spec for spec in SCENARIO_DEFS}


def get_scenario(scenario_id: str) -> ScenarioDef:
    spec = _BY_ID.get(scenario_id)
    if spec is None:
        raise ValueError(f"Unknown scenario {scenario_id!r}. Available: {', '.join(SCENARIO_IDS)}.")
    return spec


def generate_scenario(
    scenario_id: str,
) -> tuple[ScenarioDef, tuple[tuple[EventEnvelope, ...], ...]]:
    """Run the real simulator and return the spec with its sessions of events.

    The events are the generator's own :class:`EventEnvelope` objects, unprojected and
    unfiltered, exactly as the integration tests receive them.
    """
    spec = get_scenario(scenario_id)
    return spec, spec.build()


def flatten(sessions: tuple[tuple[EventEnvelope, ...], ...]) -> tuple[EventEnvelope, ...]:
    """Concatenate sessions into one stream, preserving order."""
    events: list[EventEnvelope] = []
    for session in sessions:
        events.extend(session)
    return tuple(events)


#: A minimal, human-readable session, written as a literal and then *proved* valid.
#: Every event here is passed through :class:`EventEnvelope` before this module finishes
#: importing, so the example the UI offers cannot silently drift out of step with the
#: schemas: if the engine tightens a field, the lab fails at startup rather than shipping a
#: template the engine would reject.
_CANONICAL_LITERAL: Final[dict[str, Any]] = {
    "events": [
        {
            "event_id": "evt-00000001",
            "learner_id": "lab-sample-0001",
            "session_id": "lab-sample-0001-s01",
            "timestamp": "2026-06-01T09:00:00Z",
            "event_type": "session_started",
            "payload": {"platform": "web", "entry_point": "course/lesson/1"},
        },
        {
            "event_id": "evt-00000002",
            "learner_id": "lab-sample-0001",
            "session_id": "lab-sample-0001-s01",
            "timestamp": "2026-06-01T09:00:05Z",
            "event_type": "question_started",
            "payload": {"question_id": "qst-000001", "difficulty": 0.5, "topic": "algebra"},
        },
        {
            "event_id": "evt-00000003",
            "learner_id": "lab-sample-0001",
            "session_id": "lab-sample-0001-s01",
            "timestamp": "2026-06-01T09:00:17Z",
            "event_type": "question_answered",
            "payload": {"question_id": "qst-000001", "correct": True, "response_seconds": 12.5},
        },
        {
            "event_id": "evt-00000004",
            "learner_id": "lab-sample-0001",
            "session_id": "lab-sample-0001-s01",
            "timestamp": "2026-06-01T09:00:22Z",
            "event_type": "session_ended",
            "payload": {"duration_seconds": 22.0, "reason": "logout"},
        },
    ]
}


def canonical_example() -> dict[str, Any]:
    """The smallest valid native session, verified against the engine on every call.

    This exists so the UI can offer something a user can read and edit by hand. It is
    validated here rather than merely being written carefully, so "valid" is a checked
    property of the response and not a claim in a comment.

    Returns:
        The ``{"events": [...]}`` document plus the identity it declares.
    """
    events = [
        EventEnvelope.model_validate(event).model_dump(mode="json", exclude_none=False)
        for event in _CANONICAL_LITERAL["events"]
    ]
    return {
        "events": events,
        "learner_id": "lab-sample-0001",
        "session_id": "lab-sample-0001-s01",
        "event_count": len(events),
    }


def describe(scenario_id: str) -> dict[str, Any]:
    """A JSON-serialisable summary for the UI's scenario cards."""
    spec = get_scenario(scenario_id)
    return {
        "id": spec.id,
        "title": spec.title,
        "archetype": spec.archetype.value,
        "description": spec.description,
        "expected_behavior": spec.expected_behavior,
        "learner_id": spec.learner_id,
        "sessions": spec.sessions,
        "questions_per_session": spec.questions_per_session,
        "session_stride_hours": spec.session_stride.total_seconds() / 3600.0,
    }


#: The seed used when the UI's "Generate Valid Synthetic Session" button is pressed without
#: an explicit one. Fixed, so pressing the button twice produces byte-identical JSON.
DEFAULT_SAMPLE_SEED: Final[int] = 424_242

#: Anchored start instant, matching the scenarios. A sample session must not read the wall
#: clock: a "random" example that changes every time it is regenerated cannot be pasted,
#: diffed, or reasoned about.
_SAMPLE_ANCHOR: Final[datetime] = datetime(2026, 6, 1, 9, 0, 0, tzinfo=UTC)


def generate_sample_session(seed: int = DEFAULT_SAMPLE_SEED, questions: int = 4) -> dict[str, Any]:
    """Generate one short, valid, native Focus Engine session.

    The UI needs a payload a user can actually paste, and it must be one the engine accepts
    without editing. Rather than hand-writing a template — which drifts from the schemas the
    moment the engine changes — this runs the repository's own simulator through the same
    :class:`SimulationConfig` path the scenarios use and serialises what it emits. The
    result is therefore valid by construction: it went through
    :class:`focus_engine.events.EventEnvelope` on the way out.

    Determinism is a requirement, not a nicety. The seed is folded into
    :class:`SimulationConfig.base_seed` and the start instant is fixed, so the same seed
    yields the same events on every call and in every process. A user who regenerates to
    compare against a previous run gets the same data, and the Lab's replay guarantee holds
    for sample input as well as generated scenarios.

    Args:
        seed: Non-negative seed. Folded into the simulator's own base seed.
        questions: Questions to attempt. The simulator requires at least
            ``MIN_QUESTIONS_PER_SESSION``; a session of four emits ten native events.

    Returns:
        A dict with the canonical ``{"events": [...]}`` document to paste, the learner and
        session ids it declares, the seed used, and the serialised event count.
    """
    if questions < MIN_QUESTIONS_PER_SESSION:
        raise ValueError(
            f"questions must be at least {MIN_QUESTIONS_PER_SESSION}; got {questions}. The "
            "simulator's own floor, not a lab restriction."
        )
    if seed < 0:
        raise ValueError(f"seed must be non-negative; got {seed}.")

    learner_id = "lab-sample-0001"
    session_id = f"{learner_id}-s01"
    config = SimulationConfig(
        archetype=LearnerArchetype.STABLE,
        learner_id=learner_id,
        session_id=session_id,
        base_seed=seed,
        question_count=questions,
        start_time=_SAMPLE_ANCHOR,
    )
    result = generate_session(config)
    events = [event.model_dump(mode="json", exclude_none=False) for event in result.events]
    return {
        "events": events,
        "learner_id": learner_id,
        "session_id": session_id,
        "seed": seed,
        "event_count": len(events),
        "archetype": LearnerArchetype.STABLE.value,
    }
