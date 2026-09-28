"""The record of what has already been delivered to a learner.

The policy needs history to enforce its restraints, but it must not *own* history: an
engine that quietly accumulated per-learner state would make a decision impossible to
replay, because the state that produced it would no longer be inspectable. So the record
is a frozen value that is passed in, and the policy treats it as read-only.

Two properties are worth stating because they are the ones that are easy to get wrong:

- **Completions match by identifier, never by recency.** A completion for an unknown
  identifier is ignored rather than applied to the most recent delivery. Applying it to
  the most recent delivery looks like it works right up until two completions arrive out
  of order, at which point it attributes a response to the wrong intervention and quietly
  corrupts the retirement logic that depends on it.
- **A history is truncated at the decision instant.** History carries an optional ``up_to``
  bound, and the policy always applies it. A decision that could see interventions
  delivered *after* it was made would be non-reproducible, and the difference would only
  appear in replay, long after the live behaviour looked fine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    InterventionCompletedPayload,
    InterventionStartedPayload,
)

__all__ = [
    "NON_RESPONSE_OUTCOMES",
    "RESPONSE_OUTCOMES",
    "DeliveredIntervention",
    "InterventionHistory",
    "OutcomeClass",
    "classify_outcome",
    "history_from_events",
]

#: Outcomes that count as the learner engaging with the intervention.
RESPONSE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"accepted", "responded", "completed", "engaged"}
)

#: Outcomes that count as the learner declining or ignoring the intervention.
#:
#: These are *observed delivery outcomes*, not measurements of whether the intervention
#: worked. A learner who dismisses a prompt and then answers correctly has still
#: dismissed the prompt, and treating the dismissal as a success because the session went
#: well afterwards is exactly the causal reading this project refuses.
NON_RESPONSE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"dismissed", "ignored", "timeout", "declined", "abandoned"}
)


class OutcomeClass(StrEnum):
    """How a delivery's reported outcome is classified for restraint purposes."""

    RESPONSE = "response"
    """A recognised response. Breaks a non-response run."""

    NON_RESPONSE = "non_response"
    """A recognised non-response. Extends a non-response run."""

    UNCLASSIFIED = "unclassified"
    """A reported outcome this layer does not recognise.

    Treated as neither a response nor a non-response, and it terminates a run rather than
    being skipped over. Retirement asserts that a type was *repeatedly* declined, so it
    should rest on positive evidence; silently skipping an unrecognised label would let a
    naming change quietly manufacture a run of failures out of nothing.
    """

    NOT_REPORTED = "not_reported"
    """No completion has been observed for this delivery."""


def classify_outcome(outcome: str | None) -> OutcomeClass:
    """Classify a reported delivery outcome.

    Args:
        outcome: The outcome label, or ``None`` if no completion was observed.

    Returns:
        The classification. Unrecognised labels are returned as
        :attr:`OutcomeClass.UNCLASSIFIED` rather than being coerced into a bucket, so a
        caller can tell "we do not know" apart from "we know it was a non-response".
    """
    if outcome is None:
        return OutcomeClass.NOT_REPORTED
    normalised = outcome.strip().lower()
    if normalised in NON_RESPONSE_OUTCOMES:
        return OutcomeClass.NON_RESPONSE
    if normalised in RESPONSE_OUTCOMES:
        return OutcomeClass.RESPONSE
    return OutcomeClass.UNCLASSIFIED


@dataclass(frozen=True, slots=True)
class DeliveredIntervention:
    """One intervention that was actually delivered to a learner.

    Attributes:
        intervention_id: The delivery's identifier, which is also how its completion is
            matched.
        session_id: The session it was delivered in.
        intervention_type: The controlled-vocabulary type label.
        delivered_at: When it was delivered.
        outcome: The reported delivery outcome, or ``None`` if none has been observed.
        responded_at: When the completion was observed, or ``None``.
    """

    intervention_id: str
    session_id: str
    intervention_type: str
    delivered_at: datetime
    outcome: str | None = None
    responded_at: datetime | None = None

    @property
    def outcome_class(self) -> OutcomeClass:
        """How this delivery's outcome is classified."""
        return classify_outcome(self.outcome)

    @property
    def has_outcome(self) -> bool:
        """Whether a completion has been reported for this delivery."""
        return self.outcome is not None

    def describe(self) -> str:
        """Render the delivery as a short human-readable line.

        Returns:
            A one-line description naming the type, when it was delivered, and its
            classified outcome.
        """
        return (
            f"{self.intervention_type} ({self.intervention_id}) delivered "
            f"{self.delivered_at.isoformat()}, outcome {self.outcome!r} "
            f"[{self.outcome_class.value}]"
        )


@dataclass(frozen=True, slots=True)
class InterventionHistory:
    """An immutable, time-ordered record of interventions delivered to one learner.

    Attributes:
        learner_id: The learner this history belongs to.
        interventions: The deliveries, ordered by delivery time then identifier so that
            two events with the same timestamp still produce a single stable order.
    """

    learner_id: str
    interventions: tuple[DeliveredIntervention, ...] = ()

    def __post_init__(self) -> None:
        """Validate that every delivery belongs to this learner.

        Raises:
            ValueError: If any delivery carries a different learner. A history that mixed
                two learners would make every restraint computed from it wrong in a way
                that would look like a plausible policy decision.
        """
        for delivery in self.interventions:
            if not delivery.intervention_id:
                raise ValueError("a delivery must carry an intervention identifier")
        ordered = tuple(
            sorted(self.interventions, key=lambda item: (item.delivered_at, item.intervention_id))
        )
        if ordered != self.interventions:
            raise ValueError(
                "interventions must be ordered by delivery time, then identifier; pass a "
                "sorted history or use history_from_events, which sorts"
            )

    def through(self, moment: datetime) -> InterventionHistory:
        """Return the history as it stood at an instant.

        Args:
            moment: The instant to truncate at, inclusive.

        Returns:
            A history containing only deliveries at or before ``moment``. Deliveries with
            no timezone are rejected by the comparison, which is the intended outcome -
            an ambiguous timestamp cannot be ordered against a decision instant.
        """
        return InterventionHistory(
            learner_id=self.learner_id,
            interventions=tuple(
                delivery for delivery in self.interventions if delivery.delivered_at <= moment
            ),
        )

    def in_window(self, moment: datetime, window: timedelta) -> tuple[DeliveredIntervention, ...]:
        """Return the deliveries inside a sliding window ending at an instant.

        Args:
            moment: The end of the window, inclusive.
            window: The width of the window.

        Returns:
            The deliveries in ``(moment - window, moment]``. The lower bound is exclusive
            so that a delivery exactly one window old has aged out, which is what makes
            the cap a sliding window rather than a tumbling one.

        Raises:
            ValueError: If the window is not positive.
        """
        if window <= timedelta(0):
            raise ValueError(f"a sliding window must be positive; got {window}")
        cutoff = moment - window
        return tuple(
            delivery for delivery in self.interventions if cutoff < delivery.delivered_at <= moment
        )

    def count_in_session(self, session_id: str) -> int:
        """Count the deliveries made in a session.

        Args:
            session_id: The session to count within.

        Returns:
            How many interventions were delivered in that session.
        """
        return sum(1 for delivery in self.interventions if delivery.session_id == session_id)

    @property
    def total_delivered(self) -> int:
        """Total deliveries recorded, across all sessions."""
        return len(self.interventions)

    def last_delivery(self) -> DeliveredIntervention | None:
        """The most recent delivery.

        Returns:
            The latest delivery, or ``None`` if nothing has been delivered.
        """
        return self.interventions[-1] if self.interventions else None

    def seconds_since_last(self, moment: datetime) -> float | None:
        """Seconds between the most recent delivery and an instant.

        Args:
            moment: The instant to measure back to.

        Returns:
            The elapsed seconds, or ``None`` if the learner has never been intervened with.
            ``None`` rather than ``0.0``, because "never intervened with" and "intervened
            with just now" are different facts and a policy that cannot tell them apart
            will apply the wrong restraint.
        """
        last = self.last_delivery()
        if last is None:
            return None
        return (moment - last.delivered_at).total_seconds()

    def trailing_run_length(self, intervention_type: str) -> int:
        """How many of the most recent deliveries were of one type.

        Args:
            intervention_type: The type to count.

        Returns:
            The length of the trailing run of that type, counted over *that type's own*
            deliveries. An intervening delivery of a different type does not reset the
            run, because a learner who has just engaged with a different prompt has not
            thereby become receptive to this one.
        """
        if not self.interventions:
            return 0
        if self.interventions[-1].intervention_type != intervention_type:
            return 0
        return sum(
            1
            for delivery in reversed(self.interventions)
            if delivery.intervention_type == intervention_type
        )

    def consecutive_non_responses(self, intervention_type: str) -> int:
        """How many of a type's most recent deliveries were declined.

        Args:
            intervention_type: The type to count.

        Returns:
            The length of the trailing run of that type's deliveries that are recognised
            non-responses. A response, an unrecognised label, or a delivery with no
            reported outcome all terminate the run, so the count only ever grows on
            positive evidence of repeated decline.
        """
        count = 0
        for delivery in reversed(self.interventions):
            if delivery.intervention_type != intervention_type:
                continue
            if delivery.outcome_class is not OutcomeClass.NON_RESPONSE:
                break
            count += 1
        return count

    def measured_outcomes_for(self, intervention_type: str) -> int:
        """Count the recognised outcomes recorded for a type.

        Args:
            intervention_type: The type to count outcomes for.

        Returns:
            How many completions of that type were reported. Exposed so a caller can see
            that response data *exists*; the policy deliberately does not consult it when
            choosing, because ranking candidates by observed response is the feedback
            architecture's job and doing it here would smuggle in an unversioned
            selection rule.
        """
        return sum(
            1
            for delivery in self.interventions
            if delivery.intervention_type == intervention_type
            and delivery.outcome_class in (OutcomeClass.RESPONSE, OutcomeClass.NON_RESPONSE)
        )

    def with_delivery(self, delivery: DeliveredIntervention) -> InterventionHistory:
        """Return a new history including one more delivery.

        Args:
            delivery: The delivery to record.

        Returns:
            A new history with the delivery appended in order. The receiver is unchanged.

        Raises:
            ValueError: If the delivery would break the ordering invariant.
        """
        combined = (*self.interventions, delivery)
        return InterventionHistory(
            learner_id=self.learner_id,
            interventions=tuple(
                sorted(combined, key=lambda item: (item.delivered_at, item.intervention_id))
            ),
        )

    def with_outcome(
        self,
        intervention_id: str,
        outcome: str,
        responded_at: datetime,
    ) -> InterventionHistory:
        """Return a new history with a completion attached to a delivery.

        Args:
            intervention_id: The delivery to attach to, matched by identifier.
            outcome: The reported outcome label.
            responded_at: When the completion was observed.

        Returns:
            A new history with the completion recorded, or the receiver unchanged if the
            identifier is unknown or already carries an outcome. Unknown completions are
            ignored rather than matched by recency, and the first outcome for a delivery
            wins, because the event log is append-only and a duplicate completion is
            noise rather than a correction.
        """
        for index, delivery in enumerate(self.interventions):
            if delivery.intervention_id != intervention_id:
                continue
            if delivery.has_outcome:
                return self
            updated = DeliveredIntervention(
                intervention_id=delivery.intervention_id,
                session_id=delivery.session_id,
                intervention_type=delivery.intervention_type,
                delivered_at=delivery.delivered_at,
                outcome=outcome,
                responded_at=responded_at,
            )
            deliveries = list(self.interventions)
            deliveries[index] = updated
            return InterventionHistory(learner_id=self.learner_id, interventions=tuple(deliveries))
        return self

    def retired_types(self, abandon_after: int) -> frozenset[str]:
        """Types that have been declined enough times to stop offering.

        Args:
            abandon_after: The consecutive non-response count at which a type is retired.

        Returns:
            The types whose trailing non-response run has reached the threshold. This is a
            *suppression* rule, not a ranking one: it removes candidates that have
            repeatedly been declined, and it never promotes one type over another. That
            distinction matters, because ranking candidates by observed response is
            precisely the feedback rule this layer is not permitted to implement.
        """
        if abandon_after < 1:
            raise ValueError(f"abandon_after must be at least 1; got {abandon_after}")
        retired = {
            delivery.intervention_type
            for delivery in self.interventions
            if self.consecutive_non_responses(delivery.intervention_type) >= abandon_after
        }
        return frozenset(retired)

    def is_empty(self) -> bool:
        """Whether nothing has been delivered to this learner."""
        return not self.interventions

    def describe(self) -> str:
        """Render the history as a short human-readable block.

        Returns:
            A multi-line summary of the learner and each recorded delivery.
        """
        header = f"{self.learner_id}: {self.total_delivered} intervention(s)"
        if not self.interventions:
            return f"{header}, none"
        return "\n".join([header, *(f"  {item.describe()}" for item in self.interventions)])


def history_from_events(
    learner_id: str,
    events: Sequence[EventEnvelope],
    *,
    up_to: datetime | None = None,
) -> InterventionHistory:
    """Build a history from an event stream.

    Args:
        learner_id: The learner the history is for.
        events: Candidate events, in any order. Non-intervention events are ignored.
        up_to: If given, only events at or before this instant are considered.

    Returns:
        The learner's history, ordered by delivery time then identifier.

    Raises:
        ValueError: If an intervention event names a different learner. Events are
            filtered by type but not by learner: a cross-learner event reaching this
            function is a routing bug, and dropping it quietly would let the bug reach
            production and produce restraints computed from the wrong history.
    """
    deliveries: dict[str, DeliveredIntervention] = {}
    completions: dict[str, tuple[str, datetime]] = {}

    for event in events:
        if event.event_type not in (
            EventType.INTERVENTION_STARTED,
            EventType.INTERVENTION_COMPLETED,
        ):
            continue
        if event.learner_id != learner_id:
            raise ValueError(
                f"event {event.event_id} belongs to learner {event.learner_id}, not "
                f"{learner_id}. Refusing to build a history from another learner's events."
            )
        if up_to is not None and event.timestamp > up_to:
            continue

        if event.event_type is EventType.INTERVENTION_STARTED:
            payload = event.payload
            assert isinstance(payload, InterventionStartedPayload)
            deliveries.setdefault(
                payload.intervention_id,
                DeliveredIntervention(
                    intervention_id=payload.intervention_id,
                    session_id=event.session_id,
                    intervention_type=payload.intervention_type,
                    delivered_at=event.timestamp,
                ),
            )
        else:
            payload = event.payload
            assert isinstance(payload, InterventionCompletedPayload)
            completions.setdefault(
                payload.intervention_id,
                (payload.outcome, event.timestamp),
            )

    ordered = tuple(
        sorted(deliveries.values(), key=lambda item: (item.delivered_at, item.intervention_id))
    )
    history = InterventionHistory(learner_id=learner_id, interventions=ordered)
    for intervention_id, (outcome, responded_at) in completions.items():
        history = history.with_outcome(intervention_id, outcome, responded_at)
    return history
