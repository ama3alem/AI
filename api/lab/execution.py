"""The delivery boundary. **Development-only, by construction.**

**This module is not part of ``src/focus_engine`` and must never move into it.** The engine
declares the lifecycle *schemas* — :class:`~focus_engine.events.types.InterventionStartedPayload`
and :class:`~focus_engine.events.types.InterventionCompletedPayload` — and several layers
consume them, but nothing in the engine constructs one. That gap is deliberate: an engine
that could send a prompt to a learner would need a transport, a retry policy, an escalation
path, and an approval store, and none of those exist here. Adding a delivery path to the
engine would mean inventing them badly.

So the capability lives here instead, where it can be exercised, asserted, and thrown away.
Every value that reaches the engine from this file is the engine's own type, validated by the
engine's own models. The only thing invented here is *when to send* — and the answer to
that is gated below.

**The gate is the module's real content.** A delivery adapter that sends whatever it is
handed is a prompt-firing machine, and the failure it produces is the one nobody can audit
afterwards: a learner sees an intervention nobody can explain. So delivery proceeds only when
all of the following hold, and each is a separate refusal with its own reason:

1. Policy actually selected a candidate (:attr:`~focus_engine.policy.models.PolicyDecision.selected`).
2. The final action from :class:`~focus_engine.authority.safety.SafetyGate` is ``ALLOWED``.
3. That ``ALLOWED`` does **not** carry ``requires_human_approval``.

**Point 3 deserves its own paragraph, because it looks like a bug and is not.** The safety
gate returns ``ALLOWED`` with ``requires_human_approval=True`` for a restricted action when
the escalation role is *available*. Read alone, that says "go ahead". Read with the flag, it
says "go ahead once a named human says so" — and this repository contains no approval store,
no approval artifact, and no way to prove a human said anything. An adapter that treated that
combination as permission to fire would be manufacturing a human approval that never
happened. So it reports ``SELECTED_NOT_DELIVERED`` and names ``human_approval`` as the gate.
Delivering on a real approval is a future feature that will need a real approval store, and it
is deliberately not stubbed here.

**Everything is clocked, nothing is fabricated.** All instants come from an injected
:class:`~focus_engine.utils.clock.Clock`; this file contains no ``datetime.now()`` and no
``utc_now()``. The outcome and evaluation stages measure only what the event stream already
contains — no observation is invented to make a window look populated. When evidence is thin
the engine's own ``INSUFFICIENT_DATA`` is reported rather than papered over, which is why
some runs measure fewer measures than others.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final, cast

from api.lab.contract import DeliveryStatus
from focus_engine.authority import (
    FinalAction,
    FinalActionOutcome,
)
from focus_engine.baseline import BaselineDimension, BaselineEngine, BaselineProfile
from focus_engine.context import ContextEngine
from focus_engine.evaluation import (
    PredictionHorizon,
    PredictionSnapshot,
    PredictionTarget,
)
from focus_engine.events import EventType
from focus_engine.events.types import (
    EventEnvelope,
    InterventionCompletedPayload,
    InterventionStartedPayload,
)
from focus_engine.features import FeatureValue, compute_features
from focus_engine.policy import PolicyDecision
from focus_engine.policy.history import InterventionHistory, history_from_events
from focus_engine.schemas import DataOrigin, Provenance, SyntheticDataStamp
from focus_engine.temporal import TemporalEngine, TemporalObservation, TemporalState
from focus_engine.uncertainty.outcomes import PredictionOutcome
from focus_engine.utils.clock import FixedClock

logger = logging.getLogger(__name__)

__all__ = [
    "DEV_ONLY",
    "DEV_ONLY_WARNING",
    "DELIVERY_OUTCOME_ACCEPTED",
    "DeliveryAttempt",
    "history_for",
    "intervention_id_for",
    "plan_delivery",
    "post_window_reading",
    "snapshot_for",
]

#: Marks the boundary in every payload and log line that reaches a caller. A dev-only
#: capability that does not announce itself is a dev-only capability that ships.
DEV_ONLY: Final[str] = "dev_only_lab_delivery"

DEV_ONLY_WARNING: Final[str] = (
    "DEV-ONLY DELIVERY — executed by the Lab boundary (api/lab/execution.py), not by "
    "src/focus_engine. No transport, retry, or approval store exists. The learner never saw "
    "this prompt; the events below were constructed in-process to exercise the engine's own "
    "lifecycle schemas."
)

#: The delivery-side label the policy layer classifies. The outcome layer never interprets it
#: directly; it is the delivery component's own vocabulary, so it is named here and nowhere else.
DELIVERY_OUTCOME_ACCEPTED: Final[str] = "accepted"

#: How long after the lifecycle starts that completion is reported. Fixed rather than drawn
#: from the clock so a replay cannot differ in the completion instant.
_COMPLETION_LAG: Final[timedelta] = timedelta(seconds=25)

#: Deterministic response latency on the constructed completion event. See the note above:
#: this is a constant, not a measurement, because nothing in a simulation was measured.
_RESPONSE_SECONDS: Final[float] = 9.0


def intervention_id_for(
    learner_id: str, session_id: str, intervention_type: str, decided_at: datetime
) -> str:
    """Derive a stable intervention identifier from the decision, not from a counter.

    **Why a digest rather than a UUID.** A counter or UUID makes every replay a different
    delivery, which would make the determinism guarantee untestable and would make
    ``measure()`` report a fresh intervention each time. Deriving the id from the four facts
    that define the decision means the same input always produces the same identifier, so
    ``tests/lab`` can assert that two replays are byte-identical including event ids.

    The digest is truncated to 24 hex characters. That is ample for a lab identifier while
    staying well inside the engine's minimum-length rule, and the prefix keeps it legible in
    a log without being mistaken for a learner or session id.

    Args:
        learner_id: The pseudonymous learner identifier.
        session_id: The pseudonymous session the decision was made in.
        intervention_type: The catalogue entry policy selected.
        decided_at: The instant the decision was evaluated.

    Returns:
        A deterministic identifier of the form ``intervention-<digest>``.
    """
    parts = "\x1f".join((learner_id, session_id, intervention_type, decided_at.isoformat()))
    digest = hashlib.sha256(parts.encode("utf-8")).hexdigest()[:24]
    return f"intervention-{digest}"


@dataclass(frozen=True, slots=True)
class DeliveryAttempt:
    """The outcome of one delivery decision, including the gate that produced it.

    A frozen dataclass rather than a ``dict`` because the whole Phase A/Phase B boundary rests
    on this object surviving unchanged between phases. If the status could be mutated in
    transit, the "no recomputation" guarantee would be unfalsifiable.
    """

    status: DeliveryStatus
    gate: str | None
    final_action: FinalAction
    intervention_id: str | None
    started_at: datetime | None
    delivered_at: datetime | None
    events: tuple[EventEnvelope, ...]
    requires_human_approval: bool
    reason: str

    @property
    def delivered(self) -> bool:
        """Whether a completion event exists. The only definition of "delivered" used here."""
        return self.status is DeliveryStatus.DELIVERED


def _gate_of(final_action: FinalAction) -> str:
    """Name the gate responsible for a non-delivery.

    Ordered so the most specific cause is reported. A refusal outranks a pending approval
    because refusing is a decision, while a pending approval is an absence, and the reader
    needs the decision first.
    """
    if final_action.outcome is FinalActionOutcome.REFUSED:
        return "safety"
    if final_action.outcome is FinalActionOutcome.DEFERRED:
        return "human_approval"
    return "authority"


def _refusal_reason(final_action: FinalAction) -> str:
    """Explain a non-delivery in the engine's own terms, plus the approval caveat.

    The engine's ``reason`` is quoted rather than paraphrased, because a paraphrase is where
    a refusal quietly becomes a recommendation. The approval sentence is appended only when
    the action was permitted *conditionally*, because that is the case a reader is most
    likely to misread as permission.
    """
    parts = [final_action.reason]
    if final_action.requires_human_approval:
        parts.append(
            "The gate permits this action only on a named human's approval. No approval "
            "store exists in this repository, so there is no artifact proving anyone "
            "approved it, and the adapter will not treat availability as consent. Status is "
            "SELECTED_NOT_DELIVERED rather than DELIVERED."
        )
    return " ".join(part for part in parts if part)


def _stamp() -> SyntheticDataStamp:
    """The mandatory in-band stamp for a synthetic event.

    ``EventEnvelope`` refuses a synthetic event without one, which is the engine's own
    guarantee that a constructed event cannot be smuggled in as an observation. The seed is
    fixed for the same reason the identifiers are derived rather than generated: a replay must
    produce the same bytes.
    """
    return SyntheticDataStamp(generator=DEV_ONLY, seed=0)


def _envelope(
    event_type: EventType,
    timestamp: datetime,
    learner_id: str,
    session_id: str,
    event_id: str,
    payload: object,
) -> EventEnvelope:
    """Build one native synthetic envelope, validated by the engine's own model.

    ``Provenance.SYNTHETIC_LABEL`` rather than ``OBSERVED`` is mandatory and enforced by the
    envelope: a constructed event is a label, and calling it observed would be the single
    most damaging lie this boundary could tell.
    """
    return EventEnvelope(
        event_id=event_id,
        learner_id=learner_id,
        session_id=session_id,
        timestamp=timestamp,
        event_type=event_type,
        payload=payload,
        origin=DataOrigin.SYNTHETIC,
        provenance=Provenance.SYNTHETIC_LABEL,
        synthetic_stamp=_stamp(),
    )


def _lifecycle_events(
    *,
    learner_id: str,
    session_id: str,
    intervention_type: str,
    intervention_id: str,
    trigger_probability: float,
    trigger_state: str,
    started_at: datetime,
) -> tuple[EventEnvelope, EventEnvelope]:
    """Construct the engine's own start and completion events for one delivery.

    The payload field names come from the engine's schemas unchanged. ``trigger_probability``
    and ``trigger_state`` are copied from the values the engine itself produced in Phase A, so
    the lifecycle event records why the prompt was sent rather than restating the decision
    in the adapter's own words.
    """
    started = _envelope(
        EventType.INTERVENTION_STARTED,
        started_at,
        learner_id,
        session_id,
        f"event-{intervention_id}-started",
        InterventionStartedPayload(
            intervention_id=intervention_id,
            intervention_type=intervention_type,
            trigger_probability=trigger_probability,
            trigger_state=trigger_state,
        ),
    )
    completed = _envelope(
        EventType.INTERVENTION_COMPLETED,
        started_at + _COMPLETION_LAG,
        learner_id,
        session_id,
        f"event-{intervention_id}-completed",
        InterventionCompletedPayload(
            intervention_id=intervention_id,
            outcome=DELIVERY_OUTCOME_ACCEPTED,
            response_seconds=_RESPONSE_SECONDS,
        ),
    )
    return started, completed


def plan_delivery(
    *,
    decision: PolicyDecision,
    final_action: FinalAction,
    learner_id: str,
    session_id: str,
    decided_at: datetime,
    trigger_probability: float,
    trigger_state: str,
) -> DeliveryAttempt:
    """Decide whether the selected intervention may be delivered, and deliver it if so.

    This is the final gate. It runs after policy has selected a candidate and after the
    safety gate has returned a final action, and it re-checks both rather than trusting that
    they were called. A function that assumed its inputs had already been validated is a
    function that cannot be tested for the case where they were not.

    **No event is emitted for any non-delivered status.** A start event for an intervention
    that was never sent would put a lie in the stream: every downstream consumer, including
    ``history_from_events`` and therefore ``OutcomeEngine.measure``, would see a delivery that
    did not happen. For a refused or deferred action the correct output is an empty event
    tuple, and the tests assert exactly that.

    Args:
        decision: The engine's policy decision. Its ``selected`` candidate is required.
        final_action: The safety gate's final action, consumed as-is and never recomputed.
        learner_id: The pseudonymous learner identifier.
        session_id: The pseudonymous session the decision was made in.
        decided_at: The instant the decision was evaluated, and the delivery anchor. The
            lifecycle events start at this same instant: an intervention follows the decision
            that ordered it, not a wall clock that may be a day late.
        trigger_probability: The engine's own probability, copied into the start payload.
        trigger_state: The engine's own temporal state value, copied into the start payload.

    Returns:
        A :class:`DeliveryAttempt` naming the status, the gate responsible, and any events
        emitted. Never raises for a legitimate refusal: refusing is a result, not an error.
    """
    selected = decision.selected
    if selected is None:
        return DeliveryAttempt(
            status=DeliveryStatus.NOT_SELECTED,
            gate="policy",
            final_action=final_action,
            intervention_id=None,
            started_at=None,
            delivered_at=None,
            events=(),
            requires_human_approval=final_action.requires_human_approval,
            reason=(
                f"Policy returned {decision.decision.value!r} and selected no candidate, so "
                "there was nothing for a delivery boundary to send. This is not a delivery "
                "failure and must not be reported as one."
            ),
        )

    intervention_type = selected.intervention_type
    not_delivered = DeliveryStatus.SELECTED_NOT_DELIVERED

    if final_action.outcome is not FinalActionOutcome.ALLOWED:
        return DeliveryAttempt(
            status=not_delivered,
            gate=_gate_of(final_action),
            final_action=final_action,
            intervention_id=None,
            started_at=None,
            delivered_at=None,
            events=(),
            requires_human_approval=final_action.requires_human_approval,
            reason=(
                f"Policy selected {intervention_type!r}, but the {final_action.outcome.value} "
                f"final action stopped it before any event was emitted. {_refusal_reason(final_action)}"
            ),
        )

    if final_action.requires_human_approval:
        return DeliveryAttempt(
            status=not_delivered,
            gate="human_approval",
            final_action=final_action,
            intervention_id=None,
            started_at=None,
            delivered_at=None,
            events=(),
            requires_human_approval=True,
            reason=(
                f"Policy selected {intervention_type!r} and the gate returned ALLOWED, but "
                f"conditionally: {final_action.reason} {_refusal_reason(final_action)}"
            ),
        )

    # Delivery is permitted: every gate cleared and no approval is outstanding. The delivery
    # is anchored to the decision instant itself: the decision was evaluated on the evidence
    # up to ``decided_at``, and the intervention immediately follows it. Anchoring anywhere
    # else would make the lifecycle events wander relative to the evidence they respond to —
    # a fixed clock a day late is still deterministic, but it delivers a recap for a state
    # that has already moved on, and the outcome windows fall where no learner has acted.
    intervention_id = intervention_id_for(learner_id, session_id, intervention_type, decided_at)
    started_at = decided_at
    events = _lifecycle_events(
        learner_id=learner_id,
        session_id=session_id,
        intervention_type=intervention_type,
        intervention_id=intervention_id,
        trigger_probability=trigger_probability,
        trigger_state=trigger_state,
        started_at=started_at,
    )
    logger.info(
        "%s delivered intervention %s to %s in %s",
        DEV_ONLY,
        intervention_id,
        learner_id,
        session_id,
    )
    return DeliveryAttempt(
        status=DeliveryStatus.DELIVERED,
        gate="delivered",
        final_action=final_action,
        intervention_id=intervention_id,
        started_at=started_at,
        delivered_at=events[1].timestamp,
        events=events,
        requires_human_approval=False,
        reason=(
            f"The final action was ALLOWED with no outstanding approval requirement, so the "
            f"Lab delivered {intervention_type!r} as {intervention_id!r} using the engine's own "
            f"lifecycle schemas. {DEV_ONLY_WARNING}"
        ),
    )


def history_for(
    learner_id: str, events: Sequence[EventEnvelope], delivered: DeliveryAttempt
) -> InterventionHistory:
    """Rebuild the intervention history including this delivery's own events.

    ``OutcomeEngine.measure`` locates a delivery inside the history rather than accepting one
    as an argument, so a history that predates the delivery would report "no delivery" for an
    intervention that was sent. Passing the *extended* stream is what makes the measure find
    it. The extension is the adapter's own events plus the original stream, sorted by
    timestamp so the history's ordering is a fact of the data rather than of append order.
    """
    return history_from_events(learner_id, sorted((*events, *delivered.events), key=_by_timestamp))


def _by_timestamp(event: EventEnvelope) -> datetime:
    return event.timestamp


# ---------------------------------------------------------------------------
# The post-window reading
# ---------------------------------------------------------------------------


def _chunk_for_reading(
    in_window: Sequence[EventEnvelope],
) -> tuple[tuple[EventEnvelope, ...], ...] | None:
    """Split in-window evidence into the fewest chunks that still carry features.

    **Why chunking at all.** The temporal engine needs at least
    :data:`TemporalSettings.state_min_observations` observations before it will characterise a
    learner at all, and the feature engine needs a minimum number of events before it will
    produce a vector. One mid-session slice therefore produces neither: zero observations
    worth characterising. Two or more slices satisfy both floors.

    The target is exactly two observations. That is the minimum that can characterise, and
    asking for more would silently discard evidence for no analytical gain — a reading built
    from four two-event slices is not better informed than one built from two four-event
    slices, it is just thinner per observation.

    Returns:
        The chunks, or ``None`` when the evidence is too thin to build any reading at all. A
        ``None`` here is an honest absence: the caller reports it rather than inventing one.
    """
    minimum_chunk = 3
    total = len(in_window)
    if total < minimum_chunk * 2:
        return None
    size = max(minimum_chunk, -(-total // 2))
    return tuple(
        tuple(in_window[start : start + size])
        for start in range(0, total, size)
        if in_window[start : start + size]
    )


def post_window_reading(
    *,
    learner_id: str,
    in_window: Sequence[EventEnvelope],
    window_start: datetime,
    profile: BaselineProfile,
    baseline: BaselineEngine,
    dimensions: Sequence[BaselineDimension],
    dimension_value: Callable[[FeatureValue, BaselineDimension], float],
    engine: TemporalEngine,
    feature_set_version: str,
) -> TemporalState | None:
    """Build an independent reading of the learner *inside* the post-prediction window.

    **This function exists because reusing the decision-time state is wrong, and the mistake
    is easy to make.** ``TemporalState`` is immutable and perfectly reusable, so passing the
    Phase A state here is a one-line change that type-checks, passes every test that only
    asserts a verdict was produced, and silently grades the model on the evidence it was
    given. The evaluation layer guards against it — ``_reading_refusal`` refuses a reading
    anchored before the prediction — but a decision-time state is anchored *at* the
    prediction, which the guard admits, and the resulting evaluation is meaningless.

    So the reading is rebuilt from scratch:

    * a **new** temporal state, created at the window start and advancing only through
      in-window observations, so its final ``reference_time`` lands inside the window and the
      engine's own guard accepts it;
    * **in-window events only** — never the pre-window events the prediction already saw;
    * the **Phase A baseline profile as a fixed reference**, not an updated one. This is the
      conservative choice and worth stating plainly: the reference the learner is measured
      against is exactly what it was at decision time, so the reading cannot be flattered by
      a baseline that has already absorbed the deterioration it is meant to detect.

    **The known limitation, stated rather than hidden.** Session-relative features
    (``session_elapsed_seconds``, ``session_position``) describe each *chunk*, not the whole
    session, because the feature engine sees only the chunk. Absolute features — accuracy,
    response seconds, trends — are unaffected, because they do not depend on where the chunk
    sits. The baseline dimensions used here are the absolute ones, which is why the reading
    is defensible rather than merely convenient. A production implementation would ingest
    whole sessions and anchor its final observation inside the window; this Lab cannot,
    because the window is mid-session by construction.

    Args:
        learner_id: The learner the reading is about.
        in_window: Events strictly inside the half-open observation window.
        window_start: The prediction instant; the new state's creation time.
        profile: The Phase A baseline profile, used unmodified as the reference.
        baseline: The baseline engine that computes deviations.
        dimensions: The baseline dimensions to measure.
        dimension_value: Reads a dimension's source value out of a feature vector.
        engine: The temporal engine that accumulates the state.
        feature_set_version: The engine's declared feature set identifier.

    Returns:
        A state anchored inside the window, or ``None`` when the evidence is too thin. A
        ``None`` is a legitimate answer, and the evaluation engine reports it as
        ``INSUFFICIENT_DATA`` rather than guessing.
    """
    chunks = _chunk_for_reading(in_window)
    if chunks is None:
        logger.info(
            "%s: %d in-window event(s) cannot support a reading; reporting the absence",
            DEV_ONLY,
            len(in_window),
        )
        return None

    state = engine.create_state(learner_id, window_start)
    for chunk in chunks:
        if len(chunk) < 3:
            # A trailing remainder too small to produce features is dropped rather than
            # ingested as a malformed observation. Dropping it loses evidence, which the
            # evaluation record's own evidence_count still reports.
            logger.info(
                "%s: dropping a %d-event trailing chunk that cannot produce features",
                DEV_ONLY,
                len(chunk),
            )
            continue
        context = ContextEngine(clock=FixedClock(chunk[-1].timestamp)).process(chunk)
        vector = compute_features(
            chunk, context, feature_set_version=cast(Any, feature_set_version)
        )
        # Build a profile snapshot with the same statistics (so deviations are computed
        # against the Phase A reference) but with updated_at advanced to the chunk time
        # so that deviation.computed_at matches the observation time.
        profile_at_chunk = BaselineProfile.model_validate(
            profile.model_dump() | {"updated_at": vector.computed_at}
        )
        deviations = tuple(
            baseline.deviation(profile_at_chunk, dimension, dimension_value(vector, dimension))
            for dimension in dimensions
        )
        state = engine.ingest(
            state,
            TemporalObservation(
                learner_id=learner_id,
                computed_at=vector.computed_at,
                deviations=deviations,
                origins=frozenset({DataOrigin.SYNTHETIC}),
            ),
        )
    if state.evidence.observations < 2:
        logger.info(
            "%s: reading reached only %d observation(s); the temporal engine will not "
            "characterise the learner on this",
            DEV_ONLY,
            state.evidence.observations,
        )
    return state


def snapshot_for(
    *,
    prediction_id: str,
    learner_id: str,
    session_id: str,
    outcome: PredictionOutcome,
    predicted_at: datetime,
    horizon_seconds: float,
) -> PredictionSnapshot:
    """Build the engine's own ``PredictionSnapshot`` from the engine's own ``PredictionOutcome``.

    Every field is copied rather than supplied. ``model_id``, ``model_version``,
    ``feature_set_version``, ``target_definition_version``, ``data_origin``, ``probability``,
    ``is_positive``, and ``verdict`` all originate in the uncertainty layer's output, so the
    snapshot cannot disagree with the prediction that is being scored. The only values chosen
    here are the two identifiers and the horizon, which is the caller's to define.

    A ``PredictionRefusal`` cannot produce a usable snapshot: it has no probability and its
    verdict is ``refused``, so scoring it would grade a model for declining to answer. The
    caller must not build one in that case.
    """
    return PredictionSnapshot(
        prediction_id=prediction_id,
        learner_id=learner_id,
        session_id=session_id,
        target=PredictionTarget.DECLINE,
        horizon=PredictionHorizon(predicted_at=predicted_at, duration_seconds=horizon_seconds),
        verdict=outcome.verdict,
        model_id=outcome.model_id or "unknown",
        model_version=outcome.model_version or "unknown",
        feature_set_version=outcome.feature_set_version or "unknown",
        target_definition_version=outcome.target_definition_version or "unknown",
        probability=outcome.probability,
        predicted_positive=outcome.is_positive,
        data_origin=outcome.data_origin,
    )
