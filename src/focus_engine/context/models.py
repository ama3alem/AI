"""Context data model.

These are the structures downstream layers receive. Each context section is a frozen
snapshot of what is known at one point in time; nothing is mutated in place, and nothing
is inferred by this module. Derivation lives in :mod:`focus_engine.context.engine`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.schemas.primitives import DataOrigin, LearnerId, SessionId

__all__ = [
    "BLOCKING_CONTEXT_KEYS",
    "ContentContext",
    "ContextKey",
    "ContextModel",
    "ContextVersion",
    "InterventionContext",
    "PerformanceContext",
    "SessionContext",
]


class ContextVersion(StrEnum):
    """Version of the context derivation logic.

    Changing derivation logic mints a new version rather than editing in place, so an
    experiment recorded against one derivation cannot be silently reinterpreted by a
    later one.
    """

    V1 = "v1"
    """Initial derivation: session position, current content, recent performance,
    intervention history, and cooldown status."""


class ContextKey(StrEnum):
    """Names of the individual context fields a downstream layer may depend on.

    These names appear in :attr:`ContextModel.missing`, so "the context layer had no
    accuracy information" is a checkable statement rather than an inference from a
    ``None`` scattered through a dict.
    """

    SESSION = "session"
    ELAPSED_SECONDS = "elapsed_seconds"
    POSITION_IN_SESSION = "position_in_session"
    CONTENT_TYPE = "content_type"
    DIFFICULTY = "difficulty"
    RECENT_ACCURACY = "recent_accuracy"
    AVERAGE_RESPONSE_SECONDS = "average_response_seconds"
    BASELINE_MATURITY = "baseline_maturity"
    TRAJECTORY = "trajectory"
    INTERVENTION_HISTORY = "intervention_history"
    COOLDOWN_STATUS = "cooldown_status"


#: Context that must be present before any layer is entitled to map a signal to a
#: behavioural state. These are the fields whose absence makes interpretation unsound
#: rather than merely less precise.
#:
#: ``recent_accuracy`` is blocking because a latency or accuracy signal cannot be read
#: without any performance evidence. ``baseline_maturity`` is blocking because
#: Separation 2 requires a population estimate to be distinguishable from a personal
#: one; with no maturity recorded, the basis of the value is unknown. ``difficulty`` and
#: ``trajectory`` are deliberately *not* blocking: their absence reduces confidence,
#: which the uncertainty layer is responsible for expressing.
BLOCKING_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
    {
        ContextKey.SESSION,
        ContextKey.ELAPSED_SECONDS,
        ContextKey.POSITION_IN_SESSION,
        ContextKey.RECENT_ACCURACY,
        ContextKey.BASELINE_MATURITY,
    }
)


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Session-level context.

    Attributes:
        session_id: The session this context describes.
        learner_id: The learner owning this session.
        started_at: First instant attributable to this session.
        start_inferred: True when the start came from the first observed event rather
            than an explicit ``session_started`` event. The engine must be able to
            tolerate a stream that begins mid-session, but it must also say so.
        elapsed_seconds: Seconds between session start and the reference time.
        event_count: Number of events contributing to this session.
        position_in_session: Normalised position in ``[0, 1]`` against
            ``expected_session_seconds``. ``1.0`` means the session has met or
            exceeded the expected length; it does not mean the session is over.
        has_ended: Whether a ``session_ended`` event has been observed.
    """

    session_id: SessionId
    learner_id: LearnerId
    started_at: datetime
    start_inferred: bool = False
    elapsed_seconds: float = 0.0
    event_count: int = 0
    position_in_session: float = 0.0
    has_ended: bool = False


@dataclass(frozen=True, slots=True)
class ContentContext:
    """The content currently being interacted with.

    The engine owns no content database, so this is limited to what the event stream
    carries: the kind of content, its opaque identifier, and any difficulty or topic
    metadata attached to the event.

    Attributes:
        content_type: Inferred content type, e.g. ``question``, ``video``.
        content_id: Opaque identifier of the current content, if any.
        difficulty: Normalised difficulty from a question event, if present.
        topic: Topic metadata from a question event, if present.
        video_duration_seconds: Total duration from a ``video_started`` event, if present.
    """

    content_type: str | None = None
    content_id: str | None = None
    difficulty: float | None = None
    topic: str | None = None
    video_duration_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class PerformanceContext:
    """Rolling performance over the most recent question events.

    This is an immediate-window summary, not a personal baseline. The baseline engine
    owns the long-run view; conflating a ten-question window with a learner's normal
    behaviour is the error this separation prevents.

    Attributes:
        window_size: Size of the configured window.
        recent_questions: Questions observed within the window.
        recent_correct: Of those, how many were correct.
        recent_accuracy: ``recent_correct / recent_questions``, or ``None`` if none.
        average_response_seconds: Mean response time in the window, or ``None``.
    """

    window_size: int = 10
    recent_questions: int = 0
    recent_correct: int = 0
    recent_accuracy: float | None = None
    average_response_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class InterventionContext:
    """Intervention history and cooldown status for the session.

    Attributes:
        total_interventions: Interventions started in this session.
        last_intervention_type: Type of the most recent intervention, if any.
        last_intervention_at: When the most recent intervention started, if any.
        last_intervention_outcome: Outcome of the most recent intervention, if reported.
        seconds_since_last_intervention: Elapsed time since that intervention, or
            ``None`` if there has been none.
        in_cooldown: Whether the cooldown period is still active.
        cooldown_remaining_seconds: Seconds left in the cooldown, ``0.0`` if not active.
    """

    total_interventions: int = 0
    last_intervention_type: str | None = None
    last_intervention_at: datetime | None = None
    last_intervention_outcome: str | None = None
    seconds_since_last_intervention: float | None = None
    in_cooldown: bool = False
    cooldown_remaining_seconds: float = 0.0


class ContextModel(BaseModel):
    """A frozen snapshot of everything the context layer knows at one point in time.

    Attributes:
        version: Which derivation produced this context.
        reference_time: The instant the context describes. This is the timestamp of the
            last event observed, not the wall clock, so replaying a historical session
            produces the same context it produced live.
        session: Session context, or ``None`` if no session has been observed.
        content: Content currently being interacted with.
        performance: Rolling performance window.
        intervention: Intervention history and cooldown.
        baseline_maturity: Maturity level reported by the baseline engine. ``None``
            until that engine exists, and treated as a blocking gap until then.
        trajectory: Direction reported by the temporal engine. ``None`` until that
            engine exists; not a blocking gap.
        observed_event_types: Event types contributing to this context.
        origins: Data origins of the contributing events. Recorded so a context derived
            entirely from synthetic events cannot be mistaken for one derived from
            observation.
        missing: Context keys known to be unavailable, derived rather than hand-set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: ContextVersion = ContextVersion.V1
    reference_time: datetime
    session: SessionContext | None = None
    content: ContentContext = Field(default_factory=ContentContext)
    performance: PerformanceContext = Field(default_factory=PerformanceContext)
    intervention: InterventionContext = Field(default_factory=InterventionContext)
    baseline_maturity: str | None = None
    trajectory: str | None = None
    observed_event_types: frozenset[str] = Field(default_factory=frozenset)
    origins: frozenset[DataOrigin] = Field(default_factory=frozenset)
    missing: frozenset[ContextKey] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def _derive_missing_keys(self) -> ContextModel:
        """Compute which context keys are unavailable.

        Derived rather than supplied, so a caller cannot declare a gap closed that is
        still open, and cannot leave a real gap undeclared.

        Returns:
            ``self``, with ``missing`` populated.

        Raises:
            ValueError: If ``reference_time`` carries no timezone.
        """
        if self.reference_time.tzinfo is None:
            raise ValueError("reference_time must be timezone-aware")

        missing: set[ContextKey] = set()
        if self.session is None:
            missing.add(ContextKey.SESSION)
            missing.add(ContextKey.ELAPSED_SECONDS)
            missing.add(ContextKey.POSITION_IN_SESSION)
        if self.content.content_type is None:
            missing.add(ContextKey.CONTENT_TYPE)
        if self.content.difficulty is None:
            missing.add(ContextKey.DIFFICULTY)
        if self.performance.recent_accuracy is None:
            missing.add(ContextKey.RECENT_ACCURACY)
        if self.performance.average_response_seconds is None:
            missing.add(ContextKey.AVERAGE_RESPONSE_SECONDS)
        if self.baseline_maturity is None:
            missing.add(ContextKey.BASELINE_MATURITY)
        if self.trajectory is None:
            missing.add(ContextKey.TRAJECTORY)
        if self.intervention.total_interventions == 0:
            missing.add(ContextKey.INTERVENTION_HISTORY)
        if self.intervention.last_intervention_at is None:
            missing.add(ContextKey.COOLDOWN_STATUS)

        object.__setattr__(self, "missing", frozenset(missing))
        return self

    def blocking_gaps(self) -> frozenset[ContextKey]:
        """Context gaps that make interpretation unsound rather than merely imprecise.

        Returns:
            The subset of :attr:`missing` that a downstream layer must refuse to
            interpret across.
        """
        return frozenset(self.missing & BLOCKING_CONTEXT_KEYS)

    def is_interpretable(self) -> bool:
        """Whether any layer may map a signal in this context to a behavioural state.

        Returns:
            ``True`` only when no blocking gap is present. This is the mechanical form
            of the rule that a latency signal is never read without context.
        """
        return not self.blocking_gaps()

    def is_synthetic_only(self) -> bool:
        """Whether every contributing event was synthetic.

        A context built only from generated data is a software artifact, not evidence
        about a person. Recording this explicitly means a downstream consumer cannot
        read synthetic-derived context as observational without inspecting the events.

        Returns:
            ``True`` when at least one event contributed and all of them were synthetic.
        """
        return self.origins == frozenset({DataOrigin.SYNTHETIC})

    def to_summary_dict(self) -> dict[str, Any]:
        """Render a JSON-serialisable summary for logs and debugging.

        Returns:
            A dictionary of the key fields, including the missing and blocking keys.
        """
        return {
            "version": self.version.value,
            "reference_time": self.reference_time.isoformat(),
            "session_id": self.session.session_id if self.session else None,
            "elapsed_seconds": self.session.elapsed_seconds if self.session else None,
            "position_in_session": self.session.position_in_session if self.session else None,
            "content_type": self.content.content_type,
            "difficulty": self.content.difficulty,
            "recent_accuracy": self.performance.recent_accuracy,
            "average_response_seconds": self.performance.average_response_seconds,
            "total_interventions": self.intervention.total_interventions,
            "in_cooldown": self.intervention.in_cooldown,
            "baseline_maturity": self.baseline_maturity,
            "trajectory": self.trajectory,
            "origins": sorted(origin.value for origin in self.origins),
            "synthetic_only": self.is_synthetic_only(),
            "missing": sorted(key.value for key in self.missing),
            "blocking_gaps": sorted(key.value for key in self.blocking_gaps()),
        }
