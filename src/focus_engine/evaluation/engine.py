"""Turning a recorded prediction into a reading of what followed it.

This module derives ground truths and evaluations. It derives exactly two things: which
events a prediction is permitted to be scored against, and whether the reading the caller
supplied for that window matches the claim. Everything else is read from upstream records.

**The event filter is the whole point of the layer.** A prediction made at ``t`` claiming to
be about the next ``d`` seconds may be scored only against events satisfying
``t <= timestamp < t + d``, for that learner, in that session. Evidence from before ``t`` is
the prediction's own input and using it would grade the model on its inputs; an event at
exactly ``t + d`` belongs to the next prediction. The filter is applied here rather than
trusted to the caller, because a leaky evaluation is arithmetically indistinguishable from a
correct one in its results -- it simply looks better.

**A supplied reading is consumed, never recomputed.** Determining what a learner's
engagement was at an instant is the temporal layer's job, and re-deriving it here would
create a second implementation of the persistence rule that drifts from the first. So the
caller passes the :class:`~focus_engine.temporal.models.TemporalState` the prediction was
made from, and this layer checks that it is *admissible*: same learner, and a
``reference_time`` inside the window. A reading anchored outside the window is refused rather
than used, because a state computed at a different instant answers a different question.

**Unscoreable inputs are refused by returning, not by raising.** A prediction that never
resolved, a reading for the wrong learner, a reading anchored before the prediction: each
produces a record with a reason rather than an exception. An evaluation run over a historical
corpus will meet all of these, and a run that aborts on the first refusal reports a corpus
of predictions that happened to be clean rather than a measure of the system.

The refusal reasons name the constraint, not the consequence. "the reading is anchored
before the prediction" tells a reader what to fix; "invalid evaluation" would not.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from focus_engine.configuration.thresholds import EvaluationSettings
from focus_engine.evaluation.models import (
    EVALUATION_V1,
    BinaryVerdict,
    EvaluationRecord,
    EvaluationVerdict,
    GroundTruth,
    GroundTruthStatus,
    InterventionResponse,
    ObservationWindow,
    PredictionSnapshot,
    PredictionTarget,
    claim_value,
    is_declining,
    response_from_outcome,
)
from focus_engine.events.types import EventEnvelope
from focus_engine.outcomes import ACTIVITY_EVENTS, OutcomeRecord
from focus_engine.schemas.primitives import (
    BehavioralEngagementState,
    DataOrigin,
    Provenance,
)
from focus_engine.schemas.versioning import EvaluationVersion
from focus_engine.temporal.models import TemporalState, TrendDirection
from focus_engine.utils.clock import Clock, SystemClock

__all__ = ["EvaluationEngine"]

#: Evaluation definitions in force. Named as a constant so a stored record states the ruleset
#: that judged it, rather than inheriting whatever the current defaults happened to be.
DEFAULT_EVALUATION_VERSION: Final[EvaluationVersion] = EVALUATION_V1

#: State and direction members that mean "the reading could not characterise the learner".
#: Compared against, never asserted as a valid claim: an observed INSUFFICIENT_DATA is a
#: non-finding that makes the evaluation unassessable, never a direction that happened to be
#: predicted correctly.
_UNCHARACTERISED_STATES: Final[frozenset[str]] = frozenset({"insufficient_data"})


def _is_uncharacterised(target: PredictionTarget, state: TemporalState) -> bool:
    """Whether a supplied reading declined to characterise the learner.

    Args:
        target: The prediction target being scored.
        state: The temporal reading supplied for the window.

    Returns:
        ``True`` when the reading's own value for this target is its vocabulary's
        ``INSUFFICIENT_DATA`` member.
    """
    if target is PredictionTarget.STATE:
        return state.state.value in _UNCHARACTERISED_STATES
    if target is PredictionTarget.DIRECTION:
        return state.direction.value in _UNCHARACTERISED_STATES
    return state.state.value in _UNCHARACTERISED_STATES


def _origin_of(
    snapshot: PredictionSnapshot,
    admissible: Sequence[EventEnvelope],
) -> DataOrigin:
    """Resolve the data origin of a ground truth.

    Synthetic evidence makes the whole record synthetic, never the reverse. Reporting a
    record built from simulator output as a real measurement is the one provenance error this
    project exists to prevent, and the resolution has to be in the conservative direction
    because the failure is asymmetric: a real-marked record hiding synthetic evidence is
    worse than the opposite, which is merely over-cautious.

    Args:
        snapshot: The prediction the record is for, supplying the fallback origin when the
            window holds no events.
        admissible: The events that passed the window filter.

    Returns:
        :attr:`~focus_engine.schemas.primitives.DataOrigin.SYNTHETIC` when any admissible
        event is synthetic, otherwise the snapshot's own origin.
    """
    if any(event.origin is DataOrigin.SYNTHETIC for event in admissible):
        return DataOrigin.SYNTHETIC
    return snapshot.data_origin


class EvaluationEngine:
    """Scores recorded predictions against the events that followed them.

    Constructed with the evidence floor, the clock, and the definitions in force. Everything
    the engine needs is injected, so two runs over the same inputs and the same clock produce
    byte-identical records; nothing here reads the wall clock, the network, or a random
    source.

    The engine holds no mutable state. :meth:`evaluate` may be called concurrently over
    disjoint predictions without locking, because the only thing carried between calls is
    configuration.
    """

    __slots__ = ("_evaluation_version", "_settings", "_clock")

    def __init__(
        self,
        settings: EvaluationSettings | None = None,
        clock: Clock | None = None,
        evaluation_version: EvaluationVersion = DEFAULT_EVALUATION_VERSION,
    ) -> None:
        """Initialise the engine.

        Args:
            settings: The evidence floor and split thresholds in force. Defaults to
                :class:`~focus_engine.configuration.thresholds.EvaluationSettings` with its
                documented values, of which only the evidence floor affects scoring here.
            clock: Source of the ``computed_at`` instant on produced records. Defaults to the
                system clock; tests inject a
                :class:`~focus_engine.utils.clock.FixedClock` so records are deterministic.
            evaluation_version: The evaluation definitions in force, recorded on every
                produced record.
        """
        self._settings = settings if settings is not None else EvaluationSettings()
        self._clock = clock if clock is not None else SystemClock()
        self._evaluation_version = evaluation_version

    @property
    def settings(self) -> EvaluationSettings:
        """The configuration in force."""
        return self._settings

    @property
    def evaluation_version(self) -> EvaluationVersion:
        """The evaluation definitions in force."""
        return self._evaluation_version

    def _admissible_events(
        self,
        snapshot: PredictionSnapshot,
        events: Sequence[EventEnvelope],
    ) -> list[EventEnvelope]:
        """Select the events a prediction may be scored against.

        The window is half-open, so the prediction instant is admissible and the horizon
        boundary is not. Events for another learner or another session are excluded even
        inside the window, because a session is the unit the prediction was made about and
        another session's activity is not evidence about this one.

        Args:
            snapshot: The prediction whose horizon and identifiers define the selection.
            events: The event stream to filter.

        Returns:
            The admissible events, in the order supplied, so a record's evidence list is
            reproducible from the same input sequence.
        """
        window = snapshot.horizon.observation_window()
        return [
            event
            for event in events
            if event.learner_id == snapshot.learner_id
            and event.session_id == snapshot.session_id
            and event.event_type in ACTIVITY_EVENTS
            and window.contains(event.timestamp)
        ]

    def _reading_refusal(self, snapshot: PredictionSnapshot, state: TemporalState) -> str | None:
        """Whether a supplied reading may be used to score this prediction.

        Args:
            snapshot: The prediction being scored.
            state: The temporal reading supplied for the window.

        Returns:
            ``None`` when the reading is admissible, otherwise the reason it is not.
        """
        if state.learner_id != snapshot.learner_id:
            return (
                f"the reading is for learner {state.learner_id}, but the prediction is about "
                f"{snapshot.learner_id}"
            )
        reference = state.reference_time
        if reference < snapshot.horizon.predicted_at:
            return (
                f"the reading is anchored at {reference.isoformat()}, before the prediction at "
                f"{snapshot.horizon.predicted_at.isoformat()}. A state computed before the "
                "prediction is the prediction's own input, so scoring against it would grade the "
                "model on the evidence it was given."
            )
        if not snapshot.horizon.contains(reference):
            return (
                f"the reading is anchored at {reference.isoformat()}, outside the observation "
                f"window {snapshot.horizon.describe()}. A state computed at a different instant "
                "answers a different question, and the horizon is the only thing that decides "
                "which instants are in scope."
            )
        return None

    def ground_truth(
        self,
        snapshot: PredictionSnapshot,
        events: Sequence[EventEnvelope],
        reading: TemporalState | None,
    ) -> GroundTruth:
        """Derive what the post-prediction window shows.

        The evidence gate is applied first and unconditionally: a window below
        ``min_evidence_for_evaluation`` is reported as ``INSUFFICIENT_DATA`` even when a
        reading was supplied, because a reading derived from two events is not a
        characterisation of a learner regardless of how confidently it is labelled.

        Args:
            snapshot: The prediction to read the window for.
            events: The event stream to filter. The engine filters it itself; the caller
                passes the whole stream rather than a pre-filtered subset, because a
                caller-supplied subset is exactly where a leak would enter.
            reading: The temporal state describing the learner inside the window, or ``None``
                when the caller has none. ``None`` yields ``NOT_ASSESSABLE`` once the
                evidence gate is passed.

        Returns:
            A frozen ground truth. Every failure mode returns a record with a reason rather
            than raising, so a corpus-wide run is not aborted by its first unscoreable
            prediction.
        """
        window = snapshot.horizon.observation_window()
        admissible = self._admissible_events(snapshot, events)
        evidence_ids = tuple(event.event_id for event in admissible)
        evidence_count = len(admissible)
        floor = self._settings.min_evidence_for_evaluation

        if not snapshot.is_resolved:
            return self._unscored_ground_truth(
                snapshot,
                window,
                admissible,
                GroundTruthStatus.NOT_ASSESSABLE,
                (
                    f"the prediction reached the {snapshot.verdict.value!r} verdict and issued no "
                    "claim, so there is nothing to confirm or refute. A refused prediction is a "
                    "record of an answer the engine declined to give, and evaluating it would "
                    "grade a claim that was never made."
                ),
            )
        if evidence_count < floor:
            return self._unscored_ground_truth(
                snapshot,
                window,
                admissible,
                GroundTruthStatus.INSUFFICIENT_DATA,
                (
                    f"the observation window holds {evidence_count} activity event(s) against a "
                    f"floor of {floor}. A window this thin cannot support a reading about a "
                    "learner, and the events that are present are more consistent with a pause "
                    "than with a behavioural state."
                ),
            )
        if reading is None:
            return self._unscored_ground_truth(
                snapshot,
                window,
                admissible,
                GroundTruthStatus.NOT_ASSESSABLE,
                (
                    f"the window holds {evidence_count} admissible event(s), but no reading of "
                    "the learner inside it was supplied. The events are present; what they mean "
                    "is not derivable here, and this layer does not derive it."
                ),
            )
        refusal = self._reading_refusal(snapshot, reading)
        if refusal is not None:
            return self._unscored_ground_truth(
                snapshot,
                window,
                admissible,
                GroundTruthStatus.NOT_ASSESSABLE,
                f"{refusal[0].upper()}{refusal[1:]}.",
            )
        if _is_uncharacterised(snapshot.target, reading):
            return self._unscored_ground_truth(
                snapshot,
                window,
                admissible,
                GroundTruthStatus.NOT_ASSESSABLE,
                (
                    f"the supplied reading reports {reading.state.value!r}, which is its own "
                    f"vocabulary's insufficient-data member rather than a characterisation of the "
                    "learner. The events are present and were not enough to read; the remedy is a "
                    "better reading, not more activity."
                ),
            )

        observed_state: BehavioralEngagementState | None = None
        observed_direction: TrendDirection | None = None
        observed_positive: bool | None = None
        if snapshot.target is PredictionTarget.STATE:
            observed_state = reading.state
        elif snapshot.target is PredictionTarget.DIRECTION:
            observed_direction = reading.direction
        else:
            observed_positive = is_declining(reading.state)

        claimed = claim_value(
            snapshot.target,
            snapshot.predicted_state,
            snapshot.predicted_direction,
            snapshot.predicted_positive,
        )
        observed = claim_value(
            snapshot.target,
            observed_state,
            observed_direction,
            observed_positive,
        )
        confirmed = claimed == observed
        return GroundTruth(
            ground_truth_id=f"GT-{snapshot.prediction_id}",
            prediction_id=snapshot.prediction_id,
            learner_id=snapshot.learner_id,
            session_id=snapshot.session_id,
            target=snapshot.target,
            horizon=snapshot.horizon,
            observation_window=window,
            predicted_state=snapshot.predicted_state,
            predicted_direction=snapshot.predicted_direction,
            predicted_positive=snapshot.predicted_positive,
            observed_state=observed_state,
            observed_direction=observed_direction,
            observed_positive=observed_positive,
            status=(GroundTruthStatus.CONFIRMED if confirmed else GroundTruthStatus.REFUTED),
            reason=(
                f"the observation window holds {evidence_count} admissible event(s) and the "
                f"reading at {reading.reference_time.isoformat()} is "
                f"{observed.value if isinstance(observed, StrEnum) else observed}, which "
                f"{'matches' if confirmed else 'does not match'} the predicted "
                f"{claimed.value if isinstance(claimed, StrEnum) else claimed}"
            ),
            evidence_ids=evidence_ids,
            evidence_count=evidence_count,
            data_origin=_origin_of(snapshot, admissible),
            provenance=Provenance.OBSERVED,
            computed_at=self._clock.now(),
            evaluation_version=self._evaluation_version,
        )

    def _unscored_ground_truth(
        self,
        snapshot: PredictionSnapshot,
        window: ObservationWindow,
        admissible: Sequence[EventEnvelope],
        status: GroundTruthStatus,
        reason: str,
    ) -> GroundTruth:
        """Build a ground truth that states an absence rather than a finding.

        Args:
            snapshot: The prediction the record is for.
            window: The horizon's own window.
            admissible: The admissible events, which are still reported even when they are
                too few or too uninformative to support a reading. Recording how far short
                the evidence fell is what makes the absence diagnosable.
            status: Which non-finding to report.
            reason: Why the reading could not be made.

        Returns:
            A ground truth with every ``observed_*`` field ``None``, which its own validator
            enforces.
        """
        return GroundTruth(
            ground_truth_id=f"GT-{snapshot.prediction_id}",
            prediction_id=snapshot.prediction_id,
            learner_id=snapshot.learner_id,
            session_id=snapshot.session_id,
            target=snapshot.target,
            horizon=snapshot.horizon,
            observation_window=window,
            predicted_state=snapshot.predicted_state,
            predicted_direction=snapshot.predicted_direction,
            predicted_positive=snapshot.predicted_positive,
            status=status,
            reason=reason,
            evidence_ids=tuple(event.event_id for event in admissible),
            evidence_count=len(admissible),
            data_origin=_origin_of(snapshot, admissible),
            provenance=Provenance.OBSERVED,
            computed_at=self._clock.now(),
            evaluation_version=self._evaluation_version,
        )

    @staticmethod
    def _binary_verdict(
        target: PredictionTarget,
        predicted_positive: bool | None,
        observed_positive: bool | None,
    ) -> BinaryVerdict:
        """Place a confirmed or refuted claim in the confusion matrix.

        Args:
            target: The prediction target.
            predicted_positive: The predicted class, for a ``DECLINE`` target.
            observed_positive: The observed class, for a ``DECLINE`` target.

        Returns:
            The cell the pair occupies, or :attr:`BinaryVerdict.NOT_ASSESSABLE` for a
            non-binary target, where no positive class exists to place in a corner.
        """
        if target is not PredictionTarget.DECLINE:
            return BinaryVerdict.NOT_ASSESSABLE
        if predicted_positive is None or observed_positive is None:
            return BinaryVerdict.NOT_ASSESSABLE
        if predicted_positive:
            return (
                BinaryVerdict.TRUE_POSITIVE if observed_positive else BinaryVerdict.FALSE_POSITIVE
            )
        return BinaryVerdict.FALSE_NEGATIVE if observed_positive else BinaryVerdict.TRUE_NEGATIVE

    def evaluate(
        self,
        snapshot: PredictionSnapshot,
        events: Sequence[EventEnvelope],
        reading: TemporalState | None,
        outcome: OutcomeRecord | None = None,
    ) -> EvaluationRecord:
        """Score one prediction and, when one is supplied, describe what followed a delivery.

        The intervention response is derived from an upstream outcome record rather than
        computed here, and only when the caller supplies one. Evaluation does not select
        interventions, does not re-measure a delivery, and does not infer a counterfactual: a
        delivery that improved is recorded as an improvement *observed after* the delivery, and
        nothing in this layer says the delivery did it.

        Args:
            snapshot: The prediction to score.
            events: The event stream to filter.
            reading: The temporal state describing the learner inside the window, or ``None``.
            outcome: The outcome record for a delivery associated with this prediction, or
                ``None`` when the prediction is not being read as an intervention result.

        Returns:
            A frozen evaluation record. When the ground truth is a non-finding the verdict is
            :attr:`~focus_engine.evaluation.models.EvaluationVerdict.NOT_ASSESSABLE` and the
            record is excluded from every rate, because a gap in a log is not a mistake by the
            model.
        """
        truth = self.ground_truth(snapshot, events, reading)
        response: InterventionResponse | None = None
        intervention_id: str | None = None
        if outcome is not None:
            response = response_from_outcome(outcome)
            intervention_id = outcome.intervention_id

        if truth.status is GroundTruthStatus.CONFIRMED:
            verdict = EvaluationVerdict.CORRECT
        elif truth.status is GroundTruthStatus.REFUTED:
            verdict = EvaluationVerdict.INCORRECT
        else:
            verdict = EvaluationVerdict.NOT_ASSESSABLE

        return EvaluationRecord(
            evaluation_id=f"EVAL-{snapshot.prediction_id}",
            evaluation_version=self._evaluation_version,
            prediction_id=snapshot.prediction_id,
            ground_truth_id=truth.ground_truth_id,
            learner_id=snapshot.learner_id,
            session_id=snapshot.session_id,
            target=snapshot.target,
            verdict=verdict,
            binary_verdict=self._binary_verdict(
                snapshot.target,
                snapshot.predicted_positive,
                truth.observed_positive,
            ),
            ground_truth_status=truth.status,
            observed_state=truth.observed_state,
            observed_direction=truth.observed_direction,
            observed_positive=truth.observed_positive,
            intervention_response=response,
            intervention_id=intervention_id,
            evidence_count=truth.evidence_count,
            observation_window=truth.observation_window,
            predicted_at=snapshot.horizon.predicted_at,
            reason=truth.reason,
            model_id=snapshot.model_id,
            model_version=snapshot.model_version,
            evaluator_version=self._evaluation_version,
            data_origin=truth.data_origin,
            computed_at=truth.computed_at,
        )

    def evaluate_many(
        self,
        scored: Sequence[tuple[PredictionSnapshot, TemporalState | None]],
        events: Sequence[EventEnvelope],
        outcomes: dict[str, OutcomeRecord] | None = None,
    ) -> tuple[EvaluationRecord, ...]:
        """Score a corpus of predictions in the order supplied.

        Deterministic by construction: the same inputs in the same order produce the same
        tuple, and every record's ``computed_at`` comes from the injected clock rather than
        the wall clock. A run that is not reproducible cannot be used to argue that a change
        to the rules changed the result, because the timestamps would differ as well.

        Args:
            scored: Each prediction with the reading to score it against, or ``None``.
            events: The whole event stream, filtered per prediction.
            outcomes: Outcome records keyed by the prediction identifier the caller has
                associated with each delivery, attached where one exists. Optional, because
                most predictions are not tied to a delivery.

        Returns:
            One record per prediction, in the order supplied. Predictions that could not be
            scored are present as ``NOT_ASSESSABLE`` records rather than omitted, so a corpus
            summary can report how much of it was measurable instead of silently scoring only
            the part that was clean.
        """
        linked = outcomes if outcomes is not None else {}
        return tuple(
            self.evaluate(snapshot, events, reading, linked.get(snapshot.prediction_id))
            for snapshot, reading in scored
        )
