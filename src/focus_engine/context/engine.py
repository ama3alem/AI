"""Context derivation from behavioural events.

A signal is never interpreted in isolation. This module answers the only question the
context layer is allowed to answer: what situation did these events occur in? It
derives session position, current content, recent performance, and intervention history
from the event stream, and hands the result to downstream layers.

Two rules shape the implementation.

**The event stream is the only input.** No content database, no profile store, no
side-channel. Given a session's events, the context is fully determined. This keeps the
engine portable and makes the layer testable without fixtures.

**Replay must reproduce the live result.** All elapsed times are measured against the
timestamp of the last observed event, not the wall clock. Replaying a recorded session
therefore produces the context that was produced when it was recorded. The injected
clock is a fallback for the empty case only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from focus_engine.context.models import (
    ContentContext,
    ContextModel,
    InterventionContext,
    PerformanceContext,
    SessionContext,
)
from focus_engine.events.types import (
    ContentOpenedPayload,
    EventEnvelope,
    EventType,
    InterventionCompletedPayload,
    InterventionStartedPayload,
    QuestionAnsweredPayload,
    QuestionStartedPayload,
    SessionEndedPayload,
    SessionStartedPayload,
    VideoStartedPayload,
)
from focus_engine.schemas.primitives import DataOrigin, LearnerId, SessionId
from focus_engine.utils.clock import Clock, SystemClock

__all__ = ["ContextEngine"]


@dataclass(slots=True)
class _SessionState:
    """Mutable per-session derivation state.

    Attributes:
        session_id: Session being tracked.
        learner_id: Learner owning the session.
        started_at: First instant attributable to the session.
        start_inferred: Whether the start came from an explicit ``session_started``
            event or from the first event seen.
        ended: Whether ``session_ended`` has been observed.
        event_count: Events applied to this session.
        last_event_at: Timestamp of the most recent event applied.
        content: Current content context.
        responses: ``(correct, response_seconds)`` pairs within the window.
        interventions: ``(timestamp, intervention_id, type, outcome)`` records.
    """

    session_id: SessionId
    learner_id: LearnerId
    started_at: datetime
    start_inferred: bool = False
    ended: bool = False
    event_count: int = 0
    last_event_at: datetime | None = None
    content: ContentContext = field(default_factory=ContentContext)
    responses: list[tuple[bool, float]] = field(default_factory=list)
    interventions: list[tuple[datetime, str, str, str | None]] = field(default_factory=list)


class ContextEngine:
    """Derives context from an event stream.

    The engine keeps derived summaries rather than events, so memory does not grow with
    the length of a session. Calling :meth:`process` with events from a different session
    starts fresh tracking for that session rather than mixing two sessions' contexts
    together.

    Attributes:
        clock: Fallback clock used only when no events have been observed.
        expected_session_seconds: Reference length used to normalise
            ``position_in_session``. This is an engineering starting point, not an
            empirical finding, and it is a parameter rather than a constant so it can be
            stated and tested rather than buried.
        performance_window_size: Number of recent question answers to summarise.
        cooldown_seconds: Intervention cooldown duration.
    """

    __slots__ = (
        "_origins",
        "_session",
        "_seen_event_types",
        "clock",
        "cooldown_seconds",
        "expected_session_seconds",
        "performance_window_size",
    )

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        expected_session_seconds: float = 3600.0,
        performance_window_size: int = 10,
        cooldown_seconds: float = 300.0,
    ) -> None:
        """Create a context engine.

        Args:
            clock: Fallback clock for the no-events case. Defaults to the system clock.
            expected_session_seconds: Reference session length for position
                normalisation. Must be positive.
            performance_window_size: Number of recent answers in the performance
                window. Must be positive.
            cooldown_seconds: Intervention cooldown duration. Must be non-negative.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if expected_session_seconds <= 0:
            raise ValueError("expected_session_seconds must be positive")
        if performance_window_size < 1:
            raise ValueError("performance_window_size must be at least 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")

        self.clock: Clock = clock or SystemClock()
        self.expected_session_seconds = expected_session_seconds
        self.performance_window_size = performance_window_size
        self.cooldown_seconds = cooldown_seconds
        self._session: _SessionState | None = None
        self._seen_event_types: set[str] = set()
        self._origins: set[DataOrigin] = set()

    def process(self, events: Sequence[EventEnvelope]) -> ContextModel:
        """Apply events in order and return the resulting context.

        Args:
            events: Events in the order they occurred. Later events must not precede
                earlier ones; out-of-order input is applied as given rather than
                reordered, because reordering would hide a producer defect.

        Returns:
            The context after all events have been applied.
        """
        for event in events:
            self._apply(event)
        return self.compute_context()

    def compute_context(self) -> ContextModel:
        """Build a context snapshot from current state.

        Returns:
            A context model. When no events have been seen, the reference time is the
            clock's current time and the session context is absent.
        """
        if self._session is None:
            return ContextModel(reference_time=self.clock.now(), observed_event_types=frozenset())

        state = self._session
        reference = state.last_event_at or state.started_at
        elapsed = (reference - state.started_at).total_seconds()

        return ContextModel(
            reference_time=reference,
            session=SessionContext(
                session_id=state.session_id,
                learner_id=state.learner_id,
                started_at=state.started_at,
                start_inferred=state.start_inferred,
                elapsed_seconds=elapsed,
                event_count=state.event_count,
                position_in_session=min(1.0, max(0.0, elapsed / self.expected_session_seconds)),
                has_ended=state.ended,
            ),
            content=state.content,
            performance=self._performance_context(state),
            intervention=self._intervention_context(state, reference),
            observed_event_types=frozenset(self._seen_event_types),
            origins=frozenset(self._origins),
        )

    def reset(self) -> None:
        """Discard all derived state.

        Used when a new session begins, and between test cases.
        """
        self._session = None
        self._seen_event_types.clear()
        self._origins.clear()

    def _apply(self, event: EventEnvelope) -> None:
        """Apply one event, starting or switching session state as needed.

        Args:
            event: The event to apply.
        """
        state = self._session

        if state is None or state.session_id != event.session_id:
            explicit = event.event_type is EventType.SESSION_STARTED
            self._session = _SessionState(
                session_id=event.session_id,
                learner_id=event.learner_id,
                started_at=event.timestamp,
                start_inferred=not explicit,
            )
            state = self._session

        self._seen_event_types.add(event.event_type.value)
        self._origins.add(event.origin)
        state.last_event_at = max(
            (state.last_event_at or event.timestamp), event.timestamp, key=lambda t: t
        )
        state.event_count += 1

        # EventEnvelope already guarantees that a payload type corresponds to its
        # event_type, so dispatching on the payload alone is sufficient and removes a
        # redundant check per branch. The isinstance tests narrow the union for mypy.
        payload = event.payload

        if isinstance(payload, SessionStartedPayload):
            state.ended = False
        elif isinstance(payload, SessionEndedPayload):
            state.ended = True
        elif isinstance(payload, QuestionStartedPayload):
            state.content = ContentContext(
                content_type="question",
                content_id=payload.question_id,
                difficulty=payload.difficulty,
                topic=payload.topic,
            )
        elif isinstance(payload, QuestionAnsweredPayload):
            state.responses.append((payload.correct, payload.response_seconds))
            if len(state.responses) > self.performance_window_size:
                del state.responses[: len(state.responses) - self.performance_window_size]
        elif isinstance(payload, VideoStartedPayload):
            state.content = ContentContext(
                content_type="video",
                content_id=payload.video_id,
                video_duration_seconds=payload.duration_seconds,
            )
        elif isinstance(payload, ContentOpenedPayload):
            state.content = ContentContext(
                content_type=payload.content_type or "content",
                content_id=payload.content_id,
            )
        elif isinstance(payload, InterventionStartedPayload):
            state.interventions.append(
                (event.timestamp, payload.intervention_id, payload.intervention_type, None)
            )
        elif isinstance(payload, InterventionCompletedPayload):
            self._attach_outcome(state, payload.intervention_id, payload.outcome)

    @staticmethod
    def _attach_outcome(state: _SessionState, intervention_id: str, outcome: str) -> None:
        """Record an outcome against the intervention with the matching identifier.

        Matching is by identifier, not by recency: a learner may have two interventions
        in flight, and completing the first must not be recorded against the second. A
        completion whose start was never observed is ignored rather than invented —
        that gap is a producer defect, and repairing it silently would hide it.

        Args:
            state: Session state to update.
            intervention_id: Identifier of the completed intervention.
            outcome: Reported outcome.
        """
        for index, (started_at, recorded_id, intervention_type, existing) in enumerate(
            state.interventions
        ):
            if recorded_id == intervention_id and existing is None:
                state.interventions[index] = (started_at, recorded_id, intervention_type, outcome)
                return

    def _performance_context(self, state: _SessionState) -> PerformanceContext:
        """Summarise the recent performance window.

        Args:
            state: Session state to summarise.

        Returns:
            Rolling accuracy and mean response time, or empty values if no answers were
            observed.
        """
        if not state.responses:
            return PerformanceContext(window_size=self.performance_window_size)

        correct = sum(1 for is_correct, _ in state.responses if is_correct)
        total_seconds = sum(seconds for _, seconds in state.responses)
        count = len(state.responses)

        return PerformanceContext(
            window_size=self.performance_window_size,
            recent_questions=count,
            recent_correct=correct,
            recent_accuracy=correct / count,
            average_response_seconds=total_seconds / count,
        )

    def _intervention_context(
        self, state: _SessionState, reference: datetime
    ) -> InterventionContext:
        """Summarise intervention history and cooldown status.

        Args:
            state: Session state to summarise.
            reference: Reference time for the cooldown calculation.

        Returns:
            Intervention context. ``in_cooldown`` is ``False`` when there has been no
            intervention, since a learner who has not been intervened with is not
            restrained.
        """
        if not state.interventions:
            return InterventionContext()

        last_at, _last_id, last_type, last_outcome = state.interventions[-1]
        since = (reference - last_at).total_seconds()
        in_cooldown = since < self.cooldown_seconds

        return InterventionContext(
            total_interventions=len(state.interventions),
            last_intervention_type=last_type,
            last_intervention_at=last_at,
            last_intervention_outcome=last_outcome,
            seconds_since_last_intervention=since,
            in_cooldown=in_cooldown,
            cooldown_remaining_seconds=max(0.0, self.cooldown_seconds - since),
        )
