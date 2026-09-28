"""Outcome engine: what happened after the prompt.

The policy layer answers *should something be delivered*. This layer answers the question
that comes after — *what happened next, and what may be said about it from the events that
exist?* — and it answers it without ever claiming the prompt caused what it read.

The separation is the whole point, and it is why this layer sits where it does. A learner
who answers the next question correctly after a recap is not thereby a learner the recap
helped; the events after the prompt would very likely have looked the same without it. So
every record produced here is an observation about an event stream, stamped
:attr:`~focus_engine.outcomes.models.OutcomeRecord.provenance` as such and unable to be
stamped as anything else. Reading a causal conclusion out of these records is Phase 12's
job, on the other side of a wall, with its own version, its own observation gate, and its
own refusal to train on them.

**The engine is stateless and holds no history.** A measurement is a pure function of a
delivery, a slice of events, and optionally a temporal state. Nothing accumulates between
calls, so a record is reproducible by replay, two concurrent measurements cannot interleave,
and there is no hidden state that would make an audit impossible. What usually happens
after a recap is a question for the feedback layer, and this layer is not interested in it.

**Both windows are anchored to the delivery instant.** The before window ends where the
delivery begins and the after window begins there, so the two meet exactly and no event can
be evidence on both sides of the intervention. The after window is deliberately *not*
anchored to the response: a response-anchored window would give a learner who engaged a
different measurement interval from one who did not, and would leave the delivery whose
outcome most needs recording — the ignored one — with no window at all.

**Nothing here decides anything.** The engine does not choose an intervention, does not
score whether one was worth delivering, and does not train. It reports what it read, marks
what it could not read, and reduces nothing to a single score: no defensible weighting of
seven measures exists without knowing which matter for which learner, and that weighting is
a policy decision rather than a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final

from focus_engine.configuration.thresholds import OutcomeSettings
from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    InterventionCompletedPayload,
)
from focus_engine.outcomes import measures
from focus_engine.outcomes.models import (
    OUTCOME_MEASURES,
    OUTCOME_V1,
    Measurement,
    OutcomeMeasure,
    OutcomeRecord,
    OutcomeWindow,
    WindowKind,
)
from focus_engine.policy.history import (
    DeliveredIntervention,
    InterventionHistory,
    OutcomeClass,
    classify_outcome,
)
from focus_engine.policy.models import POLICY_V1
from focus_engine.schemas.primitives import DataOrigin, Provenance
from focus_engine.schemas.versioning import OutcomeVersion, PolicyVersion
from focus_engine.temporal.models import TemporalState
from focus_engine.utils.clock import Clock, SystemClock
from focus_engine.utils.determinism import content_digest

__all__ = [
    "OutcomeEngine",
    "OutcomeError",
]

#: The policy definition assumed when the caller does not state one.
#:
#: A default, and the record still carries the value it actually used, so a caller working
#: under a later policy version says so rather than leaving a measurement labelled with a
#: definition it never saw. The version travels on the record precisely so that this
#: default is never load-bearing: if it were wrong, the record would at least be honestly
#: mislabelled rather than silently so.
DEFAULT_POLICY_VERSION: Final[PolicyVersion] = POLICY_V1


class OutcomeError(ValueError):
    """Raised when a record cannot be produced from the supplied inputs.

    A :class:`ValueError` because every case is a defective argument rather than a missing
    record: a delivery that was never made, an event belonging to someone else, or a
    temporal state describing a different learner. All three are mistakes a caller can fix,
    and none of them should yield a record with a plausible-looking measurement in it.
    """


class OutcomeEngine:
    """Measures what followed a delivery, deterministically and without retaining state.

    Args:
        settings: Measurement configuration. The window widths determine which events are
            inside a measurement at all, which is why they are a constructor argument and
            not an internal constant.
        clock: Time source for the record's ``computed_at``. Injected so a measurement is
            reproducible in a test without freezing the record's timestamp to nothing.
        outcome_version: The measurement definition set to compute.

    Note:
        Two engines built from equal settings and versions are interchangeable, which is
        what makes :meth:`settings_fingerprint` sufficient to identify a measurement
        configuration. The clock is the one input that is not part of the fingerprint,
        because it does not influence any reading — only the record's own timestamp.
    """

    __slots__ = ("_clock", "_outcome_version", "_settings")

    def __init__(
        self,
        settings: OutcomeSettings | None = None,
        clock: Clock | None = None,
        outcome_version: OutcomeVersion = OUTCOME_V1,
    ) -> None:
        """Build an engine.

        Args:
            settings: Measurement configuration. Defaults to the shipped
                :class:`~focus_engine.configuration.thresholds.OutcomeSettings`.
            clock: Time source. Defaults to
                :class:`~focus_engine.utils.clock.SystemClock`.
            outcome_version: The measurement definition set to compute. Defaults to
                :data:`~focus_engine.outcomes.models.OUTCOME_V1`.
        """
        self._settings = OutcomeSettings() if settings is None else settings
        self._clock = SystemClock() if clock is None else clock
        self._outcome_version = outcome_version

    @property
    def settings(self) -> OutcomeSettings:
        """The measurement configuration in force."""
        return self._settings

    @property
    def clock(self) -> Clock:
        """The time source in force."""
        return self._clock

    @property
    def outcome_version(self) -> OutcomeVersion:
        """The measurement definition set this engine computes."""
        return self._outcome_version

    def settings_fingerprint(self) -> str:
        """Fingerprint the configuration that shapes a record.

        Returns:
            A stable content digest over the version and every settings field. Stored on
            each record so that a record cannot later be read as though it had been
            produced under window widths it never saw.
        """
        return content_digest(
            {"outcome_version": self._outcome_version, "settings": repr(self._settings)}
        )

    def before_window(self, delivered_at: datetime) -> OutcomeWindow:
        """Build the before window for a delivery instant.

        Args:
            delivered_at: The instant the intervention was delivered.

        Returns:
            The half-open window ``[delivered_at - before_window_minutes, delivered_at)``.
        """
        return _window(
            WindowKind.BEFORE,
            delivered_at - timedelta(minutes=self._settings.before_window_minutes),
            delivered_at,
        )

    def after_window(self, delivered_at: datetime) -> OutcomeWindow:
        """Build the after window for a delivery instant.

        Args:
            delivered_at: The instant the intervention was delivered.

        Returns:
            The half-open window ``[delivered_at, delivered_at + after_window_minutes)``.
        """
        return _window(
            WindowKind.AFTER,
            delivered_at,
            delivered_at + timedelta(minutes=self._settings.after_window_minutes),
        )

    def immediate_window(self, delivered_at: datetime) -> OutcomeWindow:
        """Build the short window at the start of the after window.

        Args:
            delivered_at: The instant the intervention was delivered.

        Returns:
            The half-open window ``[delivered_at, delivered_at + immediate_window_minutes)``.

        Note:
            This is a prefix of the after window, not a third window: it exists so the
            immediate and continued measures can read the same behaviour at two lengths.
            The settings validator refuses a configuration in which it would not be a
            prefix, since a longer "prefix" would read events the continued measure cannot
            see and the two measures would disagree for arithmetic reasons alone.
        """
        return _window(
            WindowKind.AFTER,
            delivered_at,
            delivered_at + timedelta(minutes=self._settings.immediate_window_minutes),
        )

    def measure(
        self,
        history: InterventionHistory,
        intervention_id: str,
        events: Sequence[EventEnvelope],
        *,
        state: TemporalState | None = None,
        policy_version: PolicyVersion = DEFAULT_POLICY_VERSION,
    ) -> OutcomeRecord:
        """Measure what followed one delivered intervention.

        The delivery is located through the history rather than passed in directly,
        because the history is what proves it happened and what identifies the learner. An
        identifier with no matching delivery raises rather than yielding a record, since a
        measurement attached to an intervention that was never delivered is precisely the
        fabricated finding this layer exists to prevent.

        Args:
            history: The learner's intervention history.
            intervention_id: Which recorded delivery to measure.
            events: The learner's events. Events belonging to another learner are refused
                rather than ignored, because silently dropping them would hide a routing
                bug that would otherwise surface as a mysteriously thin measurement.
            state: The temporal state as of the end of the after window, if one was
                computed. Supplies the subsequent-trajectory reading; without it that
                measure is insufficient data.
            policy_version: The policy definition in force when the intervention was
                selected, carried onto the record so an outcome can be read against the
                rules that chose the intervention.

        Returns:
            A record covering all seven measures, in the canonical order of
            :data:`~focus_engine.outcomes.models.OUTCOME_MEASURES`.

        Raises:
            OutcomeError: If the delivery is not in the history, if any event belongs to
                another learner, or if the temporal state describes another learner.
        """
        delivery = _locate(history, intervention_id)
        _require_learner(events, history.learner_id)
        if state is not None and state.learner_id != history.learner_id:
            raise OutcomeError(
                f"temporal state belongs to learner {state.learner_id!r}, not "
                f"{history.learner_id!r}. A trajectory reading about someone else is not a "
                "gap in the data; it is a record of the wrong person."
            )

        before = self.before_window(delivery.delivered_at)
        after = self.after_window(delivery.delivered_at)
        immediate = self.immediate_window(delivery.delivered_at)
        origin = _origin_of(events)
        completion = _completion_of(events, intervention_id)
        tolerance = self._settings.min_relative_change
        floor = self._settings.min_samples_for_comparison

        readings: dict[OutcomeMeasure, Measurement] = {
            OutcomeMeasure.IMMEDIATE_INTERACTION_CHANGE: measures.measure_immediate_interaction(
                events, before, immediate, min_samples=floor, tolerance=tolerance, origin=origin
            ),
            OutcomeMeasure.RESPONSE_LATENCY: measures.measure_response_latency(
                events,
                before,
                completion,
                response_class=_response_class(delivery, completion),
                tolerance=tolerance,
                origin=origin,
            ),
            OutcomeMeasure.ACCURACY: measures.measure_accuracy(
                events, before, after, tolerance=tolerance, origin=origin
            ),
            OutcomeMeasure.CONTINUED_ACTIVITY: measures.measure_continued_activity(
                events, before, after, min_samples=floor, tolerance=tolerance, origin=origin
            ),
            OutcomeMeasure.TASK_PERSISTENCE: measures.measure_task_persistence(
                events, after, origin=origin
            ),
            OutcomeMeasure.SESSION_CONTINUATION: measures.measure_session_continuation(
                delivery.session_id, events, before, after, origin=origin
            ),
            OutcomeMeasure.SUBSEQUENT_TRAJECTORY: measures.measure_subsequent_trajectory(
                state, after, origin=origin
            ),
        }

        return OutcomeRecord(
            outcome_version=self._outcome_version,
            intervention_id=delivery.intervention_id,
            learner_id=history.learner_id,
            session_id=delivery.session_id,
            intervention_type=delivery.intervention_type,
            delivered_at=delivery.delivered_at,
            before_window=before,
            after_window=after,
            measurements=tuple(readings[key] for key in OUTCOME_MEASURES),
            response_class=_response_class(delivery, completion),
            policy_version=policy_version,
            data_origin=origin,
            provenance=Provenance.OBSERVED,
            settings_fingerprint=self.settings_fingerprint(),
            computed_at=self._clock.now(),
        )

    def measure_many(
        self,
        history: InterventionHistory,
        events: Sequence[EventEnvelope],
        *,
        state: TemporalState | None = None,
        policy_version: PolicyVersion = DEFAULT_POLICY_VERSION,
    ) -> tuple[OutcomeRecord, ...]:
        """Measure every delivery in a history, oldest first.

        Args:
            history: The learner's intervention history.
            events: The learner's events.
            state: The temporal state as of the end of the final delivery's after window.
                Supplied to every record. A caller measuring many deliveries against one
                later state is describing a single point in time, and the trajectory reading
                on each record is labelled with the state that produced it.
            policy_version: The policy definition in force for these deliveries.

        Returns:
            One record per delivery, in the history's own chronological order.

        Note:
            The order is the history's, not a sort applied here. The policy layer already
            imposes a total order on deliveries and re-sorting would risk disagreeing with
            it on a tie, which is the kind of small divergence between two layers that
            becomes a bug report much later.
        """
        return tuple(
            self.measure(
                history,
                delivery.intervention_id,
                events,
                state=state,
                policy_version=policy_version,
            )
            for delivery in history.interventions
        )


def _window(kind: WindowKind, start: datetime, end: datetime) -> OutcomeWindow:
    """Build a window with its duration derived from its bounds.

    Args:
        kind: Which side of the delivery this window is on.
        start: The inclusive start.
        end: The exclusive end.

    Returns:
        The window.
    """
    return OutcomeWindow(
        kind=kind,
        start=start,
        end=end,
        duration_seconds=(end - start).total_seconds(),
    )


def _locate(history: InterventionHistory, intervention_id: str) -> DeliveredIntervention:
    """Find a recorded delivery by identifier.

    Args:
        history: The learner's history.
        intervention_id: The delivery to find.

    Returns:
        The matching delivery.

    Raises:
        OutcomeError: If no delivery matches. The identifier is echoed in the message
            because a caller that mistypes an id would otherwise have to guess which of
            several measurements it was asking about.
    """
    for delivery in history.interventions:
        if delivery.intervention_id == intervention_id:
            return delivery
    known = ", ".join(item.intervention_id for item in history.interventions) or "none"
    raise OutcomeError(
        f"no delivery {intervention_id!r} in the history of {history.learner_id!r}. "
        f"Recorded deliveries: {known}. An outcome cannot be measured for an intervention "
        "that was not delivered."
    )


def _require_learner(events: Sequence[EventEnvelope], learner_id: str) -> None:
    """Refuse events belonging to another learner.

    Args:
        events: Candidate events.
        learner_id: The learner whose history is being measured.

    Raises:
        OutcomeError: On the first foreign event. Raised rather than filtered, because a
            foreign event in the slice is a routing bug upstream, and filtering it would
            turn a data-integrity failure into a quietly thinner measurement.
    """
    for event in events:
        if event.learner_id != learner_id:
            raise OutcomeError(
                f"event {event.event_id} belongs to learner {event.learner_id!r}, not "
                f"{learner_id!r}. Refusing to measure one learner's outcome from another's "
                "events."
            )


def _origin_of(events: Sequence[EventEnvelope]) -> DataOrigin:
    """Derive the origin of a record from the events actually read.

    Args:
        events: Candidate events.

    Returns:
        :attr:`~focus_engine.schemas.primitives.DataOrigin.SYNTHETIC` when no event is
        real, and ``REAL`` when at least one is.

    Note:
        Derived from the events rather than accepted from the caller, because a caller who
        has the events in hand can be wrong about where they came from and a provenance
        field that accepts an assertion is a field that eventually carries a false one. An
        empty slice is synthetic by this rule, which is the safe direction: an empty slice
        produces almost no measurements, and reporting them as real observations of silence
        would be the worse of the two mistakes.
    """
    if any(event.origin is DataOrigin.REAL for event in events):
        return DataOrigin.REAL
    return DataOrigin.SYNTHETIC


def _completion_of(
    events: Sequence[EventEnvelope], intervention_id: str
) -> InterventionCompletedPayload | None:
    """Find the completion payload for a delivery, if one was recorded.

    Args:
        events: Candidate events.
        intervention_id: The delivery to look for.

    Returns:
        The first matching completion payload, or ``None`` when the delivery was never
        completed.
    """
    for event in events:
        if event.event_type is not EventType.INTERVENTION_COMPLETED:
            continue
        if isinstance(event.payload, InterventionCompletedPayload) and (
            event.payload.intervention_id == intervention_id
        ):
            return event.payload
    return None


def _response_class(
    delivery: DeliveredIntervention, completion: InterventionCompletedPayload | None
) -> OutcomeClass:
    """Classify how the delivery was answered, from the completion actually observed.

    Args:
        delivery: The delivery as the history records it.
        completion: The completion payload found in the event slice, if one was.

    Returns:
        The classification of the observed completion, or the delivery's own classification
        when the slice carries no completion.

    Note:
        The completion event in the slice wins over the history's own copy of the outcome.
        Both are records of the same fact, and this layer reports what the events show, so
        when a slice carries a completion and the history disagrees, the disagreement is
        resolved towards the event. Preferring the ledger would mean a caller could hand in a
        history calling a delivery answered while the slice in front of the engine says the
        learner ignored it, and the record would report an engagement no evidence supports.

        The classification itself belongs to the policy layer: this calls
        :func:`~focus_engine.policy.history.classify_outcome` rather than deciding what counts
        as a response, so the two layers cannot come to disagree about which labels mean what.
    """
    if completion is None:
        return delivery.outcome_class
    return classify_outcome(completion.outcome)
