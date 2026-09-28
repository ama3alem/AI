"""Configuration for the synthetic learner simulator.

The simulator is **configurable by construction**, not a set of static fixtures. Every
parameter that can change the generated event sequence is captured here, which gives three
properties that the phase depends on:

* *Reproducibility* — the same config and seed produce a byte-identical event stream.
* *Visible distinctness* — differences between archetypes are expressed as parameters, so
  they can be inspected and compared rather than taken on trust.
* *Documented provenance* — a generation run can state exactly what it was configured to do.

No field here encodes a mental state, and no field is a real-learner parameter. All values
describe how synthetic events are emitted.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.schemas.primitives import LearnerId, SessionId, Timestamp
from focus_engine.simulator.archetypes import LearnerArchetype

__all__ = [
    "MAX_QUESTIONS_PER_SESSION",
    "MIN_QUESTIONS_PER_SESSION",
    "SYNTHETIC_WARNING",
    "SIMULATOR_VERSION",
    "SimulationConfig",
    "SimulatorProvenance",
]


#: The literal warning carried in-band on every synthetic artefact. Defined once here so
#: the generator, the provenance record, and the tests cannot drift apart.
SYNTHETIC_WARNING: Final[str] = "SYNTHETIC DATA — NOT REAL STUDENT DATA"

#: Version of the generation logic. Bump when output would change for an unchanged config.
SIMULATOR_VERSION: Final[str] = "simulator-v1"

MIN_QUESTIONS_PER_SESSION: Final[int] = 3
MAX_QUESTIONS_PER_SESSION: Final[int] = 40

#: Hard ceiling on emitted events per session, guarding against a misconfigured loop.
#: The largest legal session emits ``2 + 2 * MAX_QUESTIONS_PER_SESSION + 2`` events.
MAX_SESSION_EVENTS: Final[int] = 2 + 2 * MAX_QUESTIONS_PER_SESSION + 2


class SimulatorProvenance(BaseModel):
    """Reproducible provenance for one generated session.

    Carries enough to regenerate the session exactly: archetype, identifiers, seeds,
    planned and emitted question counts, and the simulation version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    archetype: LearnerArchetype
    """Behavioural archetype used to parameterise this run."""

    learner_id: LearnerId
    """Pseudonymous learner identifier for the simulated session."""

    session_id: SessionId
    """Pseudonymous session identifier for the simulated session."""

    base_seed: int
    """Seed supplied by the caller."""

    derived_seed: int
    """Seed actually used by the generator. A pure function of base seed, archetype,
    learner id, and session id."""

    planned_question_count: int
    """Questions the configuration asked for."""

    emitted_question_count: int
    """Questions actually emitted, which is lower than planned when dropout occurs."""

    pattern_measurable: bool
    """Whether the emitted length meets the archetype's ``min_questions``. A session that
    is too short to separate the archetype's pattern must not be used for pattern-based
    claims."""

    simulation_version: str = SIMULATOR_VERSION
    """Version of the generation logic that produced this artefact."""

    warning: str = SYNTHETIC_WARNING
    """Non-negotiable in-band warning string."""


class SimulationConfig(BaseModel):
    """Full configuration for generating one synthetic session.

    Timing bounds are intervals; the generator samples within them from the session's
    seeded generator. All intervals are validated here so a misconfigured run fails fast
    rather than producing degenerate or non-reproducible output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    archetype: LearnerArchetype
    """Archetype to simulate."""

    learner_id: LearnerId
    """Pseudonymous learner identifier."""

    session_id: SessionId
    """Pseudonymous session identifier."""

    base_seed: int = Field(default=12345, ge=0, le=2**63 - 1)
    """Caller-supplied base seed. Non-negative so seed derivation is total."""

    question_count: int = Field(
        default=12, ge=MIN_QUESTIONS_PER_SESSION, le=MAX_QUESTIONS_PER_SESSION
    )
    """Questions to attempt when dropout is disabled."""

    start_time: Timestamp
    """Session start instant. This is the injected clock: generation never reads wall-clock
    time, so a run is reproducible regardless of when it executes."""

    question_type: str = Field(default="short_answer", min_length=1, max_length=64)
    """Question type recorded on ``question_started`` events."""

    question_id_prefix: str = Field(default="sq", min_length=1, max_length=8)
    """Prefix for generated opaque question identifiers. Carries no semantic content."""

    difficulty_min: float = Field(default=0.0, ge=0.0, le=1.0)
    """Lower bound for sampled per-question difficulty."""

    difficulty_max: float = Field(default=1.0, ge=0.0, le=1.0)
    """Upper bound for sampled per-question difficulty."""

    inter_event_gap_min: timedelta = Field(default=timedelta(seconds=1))
    """Lower bound for the gap between consecutive events, including the gap from
    ``session_started`` to the first ``question_started``."""

    inter_event_gap_max: timedelta = Field(default=timedelta(seconds=15))
    """Upper bound for the gap between consecutive events."""

    session_end_delay_min: timedelta = Field(default=timedelta(seconds=2))
    """Lower bound for the delay from the last event to ``session_ended``."""

    session_end_delay_max: timedelta = Field(default=timedelta(seconds=10))
    """Upper bound for the delay from the last event to ``session_ended``."""

    enable_dropout: bool = False
    """Whether the session may end before the requested question count. When ``False``
    (the default) the emitted length is exactly ``question_count``, which keeps structural
    tests deterministic. When ``True``, dropout is sampled once using the archetype's
    ``session_completion_rate``."""

    dropout_min_questions: int = Field(default=2, ge=1, le=MAX_QUESTIONS_PER_SESSION)
    """Fewest questions that may be emitted before dropout when dropout is enabled."""

    include_intervention: bool = False
    """Whether to emit an ``intervention_started`` / ``intervention_completed`` pair."""

    intervention_after_answers: int = Field(default=4, ge=1, le=MAX_QUESTIONS_PER_SESSION)
    """Emit the intervention immediately after this many answers, if it is fewer than
    ``question_count``."""

    intervention_id: str = Field(
        default="synthetic-intervention-0001", min_length=8, max_length=128
    )
    """Opaque intervention identifier. Must be long enough for the payload schema."""

    intervention_type: str = Field(default="micro_question", min_length=1, max_length=64)
    """Intervention type recorded on ``intervention_started``."""

    intervention_lag: timedelta = Field(default=timedelta(seconds=1))
    """Delay from the triggering answer to ``intervention_started``. Positive by
    construction, so the session stays strictly monotonic."""

    intervention_duration_min: timedelta = Field(default=timedelta(seconds=3))
    """Lower bound for the intervention's duration."""

    intervention_duration_max: timedelta = Field(default=timedelta(seconds=12))
    """Upper bound for the intervention's duration."""

    intervention_outcome: str = Field(default="accepted", min_length=1, max_length=64)
    """Outcome recorded on ``intervention_completed``. This is a generated label, not
    evidence that any intervention worked."""

    session_end_reason: str = Field(default="completed", min_length=1, max_length=64)
    """Reason recorded on a normally completed ``session_ended``."""

    dropout_reason: str = Field(default="timeout", min_length=1, max_length=64)
    """Reason recorded on ``session_ended`` when dropout truncated the session."""

    @model_validator(mode="after")
    def _validate_intervals(self) -> Self:
        """Check that every sampling interval is ordered and within question bounds.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If an interval is inverted, a duration is non-positive, or the
                dropout floor exceeds the requested question count.
        """
        if self.inter_event_gap_max < self.inter_event_gap_min:
            raise ValueError("inter_event_gap_max must be >= inter_event_gap_min")
        if self.inter_event_gap_min <= timedelta(0):
            raise ValueError(
                "inter_event_gap_min must be positive so session timestamps stay strictly "
                "increasing"
            )
        if self.session_end_delay_max < self.session_end_delay_min:
            raise ValueError("session_end_delay_max must be >= session_end_delay_min")
        if self.session_end_delay_min <= timedelta(0):
            raise ValueError("session_end_delay_min must be positive")
        if self.difficulty_max < self.difficulty_min:
            raise ValueError("difficulty_max must be >= difficulty_min")
        if self.intervention_duration_max < self.intervention_duration_min:
            raise ValueError("intervention_duration_max must be >= intervention_duration_min")
        if self.intervention_duration_min <= timedelta(0):
            raise ValueError("intervention_duration_min must be positive")
        if self.intervention_lag <= timedelta(0):
            raise ValueError("intervention_lag must be positive")
        if self.dropout_min_questions > self.question_count:
            raise ValueError("dropout_min_questions must be <= question_count")
        if self.include_intervention and self.intervention_after_answers >= self.question_count:
            raise ValueError(
                "intervention_after_answers must be < question_count so the intervention "
                "has at least one subsequent question to affect"
            )
        return self
