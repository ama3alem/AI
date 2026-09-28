"""The vocabulary of an evaluated prediction.

This module holds the vocabulary for the question that comes after *what did the engine
predict*: **was the claim borne out, and what may be said about that from the events that
actually followed it.**

A prediction is a historical fact, and this layer cannot edit it. A
:class:`PredictionSnapshot` is frozen at construction, and no operation in this package
revises a belief in the light of what came after. A system that can update a prediction once
the outcome is known has no record of what it believed, and an evaluation built on the
revised belief is grading an answer against itself.

**Ground truth is what the post-prediction window actually shows.** A
:class:`PredictionHorizon` is the *only* thing that decides which events a ground truth may
read, and it reads none from before the prediction. An event at the prediction instant is
admissible; an event at the horizon boundary is not. The half-open form is the only shape in
which a single boundary rule holds: two adjacent horizons tile without overlap, so no event
can be evidence for two predictions made a moment apart.

**Absence of evidence is not a negative answer, and there are two kinds of it.**
:class:`GroundTruthStatus` keeps ``INSUFFICIENT_DATA`` and ``NOT_ASSESSABLE`` as separate
members because they are separate findings. Too few events in the window is a gap in a log;
a reading that could not characterise the learner even with events present is a gap in what
the log can support. Collapsing them would let a corpus of unmeasurable windows be reported
as a corpus of learners who did not decline, which is the failure this layer exists to
prevent. A :class:`GroundTruth` carrying a non-finding is structurally *unable* to carry an
observed value, so no code path can turn "nothing was observed" into "the learner was fine".

**No second state vocabulary.** Predicted and observed values use the vocabularies that
already exist -- :class:`~focus_engine.schemas.primitives.BehavioralEngagementState` and
:class:`~focus_engine.temporal.models.TrendDirection` -- and the binary decline question
reuses the policy layer's existing :data:`~focus_engine.policy.models.ACTIONABLE_STATES`
rather than reminting a notion of "bad". A layer with its own seven states would drift from
the ones every other layer reads, and a disagreement between two state enums is
indistinguishable from two learners in two different states.

**An evaluation is not a causal finding.** Every ground truth carries
:attr:`Provenance.OBSERVED`, enforced by the constructor. :attr:`Provenance.GROUND_TRUTH` is
refused here for the reason its own docstring gives: it requires an identified source
external to the model, and this repository has none. An event stream that follows a
prediction is not that source.

The module holds vocabulary and structure only. Derivation lives in
:mod:`focus_engine.evaluation.engine`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.outcomes import OutcomeRecord
from focus_engine.policy.models import ACTIONABLE_STATES
from focus_engine.schemas.primitives import (
    BehavioralEngagementState,
    ConfidenceLevel,
    DataOrigin,
    Probability,
    Provenance,
    Timestamp,
    non_empty_text,
)
from focus_engine.schemas.versioning import (
    EvaluationVersion,
    FeatureSetVersion,
    ModelVersion,
    TemporalVersion,
)
from focus_engine.temporal.models import TrendDirection
from focus_engine.uncertainty.outcomes import PredictionOutcome, Verdict

__all__ = [
    "EVALUATION_V1",
    "BinaryVerdict",
    "EvaluationRecord",
    "EvaluationVerdict",
    "GroundTruth",
    "GroundTruthStatus",
    "InterventionResponse",
    "ObservationWindow",
    "PredictionHorizon",
    "PredictionSnapshot",
    "PredictionTarget",
    "claim_value",
    "is_declining",
    "response_from_outcome",
]


EVALUATION_V1: Final[EvaluationVersion] = "EVALUATION_V1"
"""The initial evaluation definition set.

Freezes the target vocabulary, the horizon boundary rule, the evidence gate, and the split
between an unmeasurable window and an uncharacterisable learner. Changing any of them mints
``EVALUATION_V2`` rather than silently reinterpreting every evaluation already recorded,
because each changes what a stored record *means*: a window widened past its old boundary
would admit an event that the store of ground truths had excluded, and those two corpora are
not comparable.
"""


class PredictionTarget(StrEnum):
    """Which question a prediction is being scored against.

    Three targets, because the three things this project predicts are not the same shape and
    pretending otherwise would force a binary verdict onto a five-way one.
    """

    DECLINE = "decline"
    """Binary. Was the learner in an actionable state at the prediction instant?

    The positive class is :data:`~focus_engine.policy.models.ACTIONABLE_STATES` -- declining,
    increasing deviation, high deviation -- read from the policy layer rather than
    redefined. This is the only target a confusion matrix applies to.
    """

    DIRECTION = "direction"
    """Multi-state over :class:`~focus_engine.temporal.models.TrendDirection`.

    Four members, one of which is ``INSUFFICIENT_DATA`` and is therefore a non-finding rather
    than a direction.
    """

    STATE = "state"
    """Multi-state over :class:`~focus_engine.schemas.primitives.BehavioralEngagementState`.

    Seven members, one of which is ``INSUFFICIENT_DATA`` and is likewise a non-finding. An
    observed ``INSUFFICIENT_DATA`` never compares equal to a predicted one as a *finding*; it
    makes the evaluation unassessable instead.
    """


class GroundTruthStatus(StrEnum):
    """What the post-prediction window established, if anything.

    The first two members are findings and the last two are not. The split is the point:
    ``CONFIRMED`` and ``REFUTED`` are statements about a learner, and the other two are
    statements about a log.
    """

    CONFIRMED = "confirmed"
    """The window showed what the prediction claimed."""

    REFUTED = "refuted"
    """The window showed something else.

    A finding, not a failure: a refuted prediction is the measurement this layer exists to
    produce.
    """

    INSUFFICIENT_DATA = "insufficient_data"
    """Too few admissible events to support any reading at all.

    A gap in the evidence. The remedy is more activity, and nothing about the prediction is
    implicated. Distinct from ``NOT_ASSESSABLE`` because the two point at different things.
    """

    NOT_ASSESSABLE = "not_assessable"
    """The events were present but could not be read as this target at all.

    Either no reading was supplied for the window, or the supplied reading was itself an
    ``INSUFFICIENT_DATA`` member of its own vocabulary. The remedy is a better reading rather
    than more events, and reporting this as ``INSUFFICIENT_DATA`` would send an operator to
    collect data they already have.
    """


class EvaluationVerdict(StrEnum):
    """Whether the prediction matched what followed it."""

    CORRECT = "correct"
    """The ground truth confirmed the claim."""

    INCORRECT = "incorrect"
    """The ground truth refuted the claim."""

    NOT_ASSESSABLE = "not_assessable"
    """No reading was produced, so the prediction was not scored. Not a failure."""


class BinaryVerdict(StrEnum):
    """A binary prediction scored against an observed binary outcome.

    Meaningful only for :attr:`PredictionTarget.DECLINE`. Every other target carries
    :attr:`NOT_ASSESSABLE`, enforced by :class:`EvaluationRecord` rather than left for a
    consumer to notice.
    """

    TRUE_POSITIVE = "true_positive"
    """Predicted actionable, and the learner was."""

    FALSE_POSITIVE = "false_positive"
    """Predicted actionable, and the learner was not.

    The most expensive cell in the table for this project: a prompt delivered to a learner who
    did not need it spent their attention and bought nothing.
    """

    TRUE_NEGATIVE = "true_negative"
    """Predicted not actionable, and the learner was not."""

    FALSE_NEGATIVE = "false_negative"
    """Predicted not actionable, and the learner was."""

    NOT_ASSESSABLE = "not_assessable"
    """No binary reading was available, or the target was not binary."""


class InterventionResponse(StrEnum):
    """What was measured either side of a delivery, in observational language.

    Every member names a *window* and none names a cause. The phase entitled to say a
    delivery caused anything does not exist yet, and a vocabulary containing causal words
    invites a reader to supply the causal reading the events cannot support. A learner who
    answers the next question correctly after a recap is not thereby a learner the recap
    helped.
    """

    POST_INTERVENTION_IMPROVEMENT_OBSERVED = "post_intervention_improvement_observed"
    """One or more measures moved favourably and none moved unfavourably."""

    NO_MEASURABLE_CHANGE = "no_measurable_change"
    """Measures were read and none exceeded the configured tolerance.

    A positive claim of stability, and deliberately distinct from :attr:`INSUFFICIENT_DATA`:
    "we looked and nothing moved" is a result, not ignorance.
    """

    POST_INTERVENTION_DETERIORATION_OBSERVED = "post_intervention_deterioration_observed"
    """One or more measures moved unfavourably and none moved favourably."""

    MIXED_MOVEMENT_OBSERVED = "mixed_movement_observed"
    """Some measures moved favourably and others unfavourably.

    Separate from the two single-direction members because the seven measures are not one
    quantity. A response latency that improved alongside an accuracy that fell is a real and
    frequent reading, and forcing it into either single-direction member would report a
    summary the record does not support.
    """

    INSUFFICIENT_DATA = "insufficient_data"
    """No measure produced a reading, so no movement can be reported.

    Unreachable for a record with a measured measure: an absence of readings cannot be
    constructed as stability, and cannot be constructed as deterioration.
    """


_CLAIM_FIELDS: Final[tuple[str, ...]] = ("state", "direction", "positive")


def _claim_suffix(target: PredictionTarget) -> str:
    """Return the claim field name a target uses.

    Args:
        target: The prediction target.

    Returns:
        ``"state"``, ``"direction"``, or ``"positive"``.
    """
    if target is PredictionTarget.STATE:
        return "state"
    if target is PredictionTarget.DIRECTION:
        return "direction"
    return "positive"


def _require_exact_claim(
    model_name: str,
    target: PredictionTarget,
    prefix: str,
    state: object | None,
    direction: object | None,
    positive: object | None,
) -> None:
    """Require that exactly the one claim field the target names is populated.

    Args:
        model_name: The record's name, for the error message.
        target: The prediction target, which selects the one permitted field.
        prefix: The field prefix, ``"predicted"`` or ``"observed"``.
        state: The state value, if any.
        direction: The direction value, if any.
        positive: The binary value, if any.

    Raises:
        ValueError: If the populated fields are not exactly the target's one field. A record
            carrying two claims has no single reading, and one carrying none cannot be
            compared to anything.
    """
    expected = f"{prefix}_{_claim_suffix(target)}"
    present = [
        f"{prefix}_{name}"
        for name, value in zip(_CLAIM_FIELDS, (state, direction, positive), strict=True)
        if value is not None
    ]
    if present != [expected]:
        raise ValueError(
            f"{model_name} targets {target.value!r}, so it must set exactly {expected}; it set "
            f"{present or ['nothing']}. A record carrying two claims has no single reading, and "
            "one carrying none cannot be compared against anything."
        )


def claim_value(
    target: PredictionTarget,
    state: object | None,
    direction: object | None,
    positive: object | None,
) -> object:
    """Return the one claim value a target uses.

    Args:
        target: The prediction target.
        state: The state value, if any.
        direction: The direction value, if any.
        positive: The binary value, if any.

    Returns:
        The value belonging to ``target``.

    Raises:
        ValueError: If the field ``target`` names is not populated.
    """
    if target is PredictionTarget.STATE:
        if state is None:
            raise ValueError("a STATE claim requires a state value")
        return state
    if target is PredictionTarget.DIRECTION:
        if direction is None:
            raise ValueError("a DIRECTION claim requires a direction value")
        return direction
    if positive is None:
        raise ValueError("a DECLINE claim requires a positive value")
    return positive


def is_declining(state: BehavioralEngagementState) -> bool:
    """Whether a state is actionable, read from the policy layer's existing set.

    Args:
        state: The behavioural state to test.

    Returns:
        ``True`` when the state is in
        :data:`~focus_engine.policy.models.ACTIONABLE_STATES`.
    """
    return state in ACTIONABLE_STATES


def response_from_outcome(outcome: OutcomeRecord) -> InterventionResponse:
    """Derive an observational response from an outcome record's own directions.

    Reads the direction vocabulary the outcome layer already computed. No measure is
    recomputed here, and no window is re-derived: this function reads a finding and names it.

    Args:
        outcome: The outcome record describing one delivery.

    Returns:
        The response member the record's directions support.

    Note:
        A record with no measured measure yields :attr:`InterventionResponse.INSUFFICIENT_DATA`
        rather than :attr:`InterventionResponse.NO_MEASURABLE_CHANGE`. Nothing moved because
        nothing was read, and a log that stops is not a learner whose behaviour was stable.
    """
    if not outcome.measured:
        return InterventionResponse.INSUFFICIENT_DATA
    improved = bool(outcome.improved)
    deteriorated = bool(outcome.deteriorated)
    if improved and deteriorated:
        return InterventionResponse.MIXED_MOVEMENT_OBSERVED
    if improved:
        return InterventionResponse.POST_INTERVENTION_IMPROVEMENT_OBSERVED
    if deteriorated:
        return InterventionResponse.POST_INTERVENTION_DETERIORATION_OBSERVED
    return InterventionResponse.NO_MEASURABLE_CHANGE


class ObservationWindow(BaseModel):
    """The half-open instant range a ground truth is permitted to read.

    Half-open means the start is inclusive and the end is exclusive:
    ``[predicted_at, predicted_at + duration)``. An event timestamped at the prediction
    instant is admissible evidence, because it is the first event that followed the
    prediction. An event timestamped at the horizon boundary is not, because it belongs to
    whatever prediction was made next.

    This type deliberately does not reuse
    :class:`~focus_engine.outcomes.models.OutcomeWindow`. That type is a pair of windows
    meeting at a *delivery* instant, and its ``kind`` discriminator admits only ``BEFORE`` and
    ``AFTER``. A prediction horizon is one window anchored to a prediction rather than a
    delivery, and borrowing the type would mean asserting a side of a delivery that did not
    happen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: Timestamp
    end: Timestamp
    duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        """Check the window is a forward interval consistent with its stated duration.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the end precedes the start, or if the recorded duration disagrees
                with the interval. The duration is a convenience field, and a stale copy
                would misreport how wide a measurement a record claims to have taken.
        """
        if self.end < self.start:
            raise ValueError(
                f"observation window ends at {self.end.isoformat()}, before it starts at "
                f"{self.start.isoformat()}"
            )
        actual = (self.end - self.start).total_seconds()
        if abs(actual - self.duration_seconds) > 1e-6:
            raise ValueError(
                f"observation window spans {actual:g} seconds but records a duration of "
                f"{self.duration_seconds:g}; the recorded width must match the interval, or a "
                "reader cannot tell how wide the measurement actually was"
            )
        return self

    def contains(self, instant: datetime) -> bool:
        """Whether an instant lies inside this window.

        Args:
            instant: The instant to test.

        Returns:
            ``True`` when ``start <= instant < end``.
        """
        return self.start <= instant < self.end


class PredictionHorizon(BaseModel):
    """When a prediction was made, and how far ahead it was claiming to be about.

    The window is derived by :meth:`observation_window` rather than stored. A stored window
    could disagree with the horizon, and where they disagreed nothing would say which one the
    evidence was actually filtered by.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    predicted_at: Timestamp
    duration_seconds: float = Field(gt=0.0)

    @property
    def ends_at(self) -> datetime:
        """The exclusive end of the observation window."""
        return self.predicted_at + timedelta(seconds=self.duration_seconds)

    def observation_window(self) -> ObservationWindow:
        """Derive the half-open window ``[predicted_at, ends_at)``.

        Returns:
            The window this horizon licenses evidence to be read from.
        """
        return ObservationWindow(
            start=self.predicted_at,
            end=self.ends_at,
            duration_seconds=self.duration_seconds,
        )

    def contains(self, instant: datetime) -> bool:
        """Whether an instant is admissible evidence for this prediction.

        Args:
            instant: The instant to test.

        Returns:
            ``True`` when the instant falls inside the half-open window.
        """
        return self.observation_window().contains(instant)

    def describe(self) -> str:
        """Render the horizon as a single human-readable line.

        Returns:
            A description naming the span and marking the end as exclusive.
        """
        return (
            f"{self.duration_seconds:g}s from {self.predicted_at.isoformat()} to "
            f"{self.ends_at.isoformat()} (end exclusive)"
        )


class PredictionSnapshot(BaseModel):
    """A prediction, frozen as it was made, together with what it was made about.

    Exactly one of ``predicted_state``, ``predicted_direction``, and ``predicted_positive`` is
    set, selected by :attr:`target`; a validator refuses any other combination, because a
    snapshot holding two claims is not a record of a single prediction and one holding none
    cannot be scored at all.

    The upstream :class:`~focus_engine.uncertainty.outcomes.PredictionOutcome` carries no
    learner, no session, and no horizon, so this type is where that information is attached.
    It is attached *to* the outcome and never derived from it: the snapshot is a wrapper that
    cannot revise the claim inside.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_id: non_empty_text
    learner_id: non_empty_text
    session_id: non_empty_text
    target: PredictionTarget
    horizon: PredictionHorizon
    predicted_state: BehavioralEngagementState | None = None
    predicted_direction: TrendDirection | None = None
    predicted_positive: bool | None = None
    probability: Probability | None = None
    confidence: Probability | None = None
    confidence_level: ConfidenceLevel | None = None
    threshold: Probability | None = None
    verdict: Verdict
    """The upstream uncertainty verdict, carried so a prediction that was never issued can be
    recorded as unassessable rather than quietly scored against a claim it never made."""

    model_id: non_empty_text
    model_version: ModelVersion
    feature_set_version: FeatureSetVersion
    target_definition_version: non_empty_text
    temporal_version: TemporalVersion | None = None
    data_origin: DataOrigin

    @model_validator(mode="after")
    def _validate_claim_shape(self) -> Self:
        """Check the snapshot carries exactly the one claim its target names.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the claim fields do not match the target, if a prediction that
                declined carries a probability, or if a ``DECLINE`` prediction that resolved
                does not record whether the threshold was met.
        """
        _require_exact_claim(
            "prediction snapshot",
            self.target,
            "predicted",
            self.predicted_state,
            self.predicted_direction,
            self.predicted_positive,
        )
        if self.verdict is not Verdict.RESOLVED:
            if self.probability is not None:
                raise ValueError(
                    f"a {self.verdict.value} prediction must not carry a probability; it "
                    "declined to issue one, and recording a number here would let a consumer "
                    "read a claim the engine never made"
                )
            if self.predicted_positive is not None:
                raise ValueError(
                    f"a {self.verdict.value} prediction must not record a positive class; it "
                    "declined to decide, and a recorded decision would be one the engine never "
                    "made"
                )
        return self

    @property
    def is_resolved(self) -> bool:
        """Whether the upstream engine actually issued a claim."""
        return self.verdict is Verdict.RESOLVED

    @classmethod
    def from_outcome(
        cls,
        outcome: PredictionOutcome,
        *,
        prediction_id: str,
        learner_id: str,
        session_id: str,
        horizon: PredictionHorizon,
    ) -> PredictionSnapshot:
        """Wrap an upstream outcome as a frozen ``DECLINE`` snapshot.

        Args:
            outcome: The uncertainty engine's answer. It must be ``RESOLVED`` and must carry
                ``is_positive``; the ``STATE`` and ``DIRECTION`` targets are not derivable
                from a probability and must be constructed directly from the temporal state
                the prediction was actually made from.
            prediction_id: Identifier for this prediction, since the outcome carries none.
            learner_id: The learner the prediction is about.
            session_id: The session the prediction was made in.
            horizon: When the prediction was made and how far it claimed to reach.

        Returns:
            A frozen snapshot wrapping ``outcome`` without altering it.

        Raises:
            ValueError: If the outcome did not resolve, or if it carries no ``computed_at``
                instant. A prediction with no timestamp has no horizon, and a horizon with
                no anchor is not a window.
        """
        if outcome.verdict is not Verdict.RESOLVED:
            raise ValueError(
                f"cannot snapshot a {outcome.verdict.value} outcome: it issued no claim to "
                "record. A snapshot of a refusal would be a record of a prediction nobody made."
            )
        if outcome.computed_at is None:
            raise ValueError(
                "cannot snapshot an outcome with no computed_at instant: a prediction with no "
                "timestamp has no horizon, and a horizon with no anchor is not a window"
            )
        return cls(
            prediction_id=prediction_id,
            learner_id=learner_id,
            session_id=session_id,
            target=PredictionTarget.DECLINE,
            horizon=horizon,
            predicted_positive=outcome.is_positive,
            probability=outcome.probability,
            confidence=outcome.assessment.value if outcome.assessment is not None else None,
            confidence_level=outcome.confidence,
            threshold=outcome.threshold,
            verdict=outcome.verdict,
            model_id=outcome.model_id,
            model_version=outcome.model_version,
            feature_set_version=outcome.feature_set_version,
            target_definition_version=outcome.target_definition_version,
            data_origin=outcome.data_origin,
        )

    def describe(self) -> str:
        """Render the snapshot as a single human-readable line.

        Returns:
            A description naming the prediction, its target, its claim, and its horizon.
        """
        value = claim_value(
            self.target,
            self.predicted_state,
            self.predicted_direction,
            self.predicted_positive,
        )
        rendered = value.value if isinstance(value, StrEnum) else str(value)
        return (
            f"{self.prediction_id} [{self.target.value}] predicted {rendered} for "
            f"{self.learner_id} over {self.horizon.describe()}"
        )

    def to_summary_dict(self) -> dict[str, object]:
        """Reduce the snapshot to flat, countable fields.

        Returns:
            A mapping with no nested containers, suitable for a counter or a metrics frame.
        """
        return {
            "prediction_id": self.prediction_id,
            "learner_id": self.learner_id,
            "session_id": self.session_id,
            "target": self.target.value,
            "predicted_at": self.horizon.predicted_at.isoformat(),
            "horizon_seconds": self.horizon.duration_seconds,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "feature_set_version": self.feature_set_version,
            "target_definition_version": self.target_definition_version,
            "verdict": self.verdict.value,
            "data_origin": self.data_origin.value,
        }


class GroundTruth(BaseModel):
    """What the post-prediction window actually shows.

    When :attr:`status` is ``INSUFFICIENT_DATA`` or ``NOT_ASSESSABLE``, every ``observed_*``
    field is ``None``, enforced by the constructor rather than by convention, so the
    dangerous value is the one that cannot be built. When the status is ``CONFIRMED`` or
    ``REFUTED``, exactly the claim field the target names is populated.

    :attr:`provenance` is constrained to :attr:`Provenance.OBSERVED` unconditionally. The
    events that follow a prediction are not an external source of truth, and
    :attr:`Provenance.GROUND_TRUTH` requires one, so a ground truth stamped with it would
    claim an independence this repository cannot supply.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ground_truth_id: non_empty_text
    prediction_id: non_empty_text
    learner_id: non_empty_text
    session_id: non_empty_text
    target: PredictionTarget
    horizon: PredictionHorizon
    observation_window: ObservationWindow
    predicted_state: BehavioralEngagementState | None = None
    predicted_direction: TrendDirection | None = None
    predicted_positive: bool | None = None
    observed_state: BehavioralEngagementState | None = None
    observed_direction: TrendDirection | None = None
    observed_positive: bool | None = None
    status: GroundTruthStatus
    reason: non_empty_text
    evidence_ids: tuple[str, ...] = ()
    evidence_count: int = Field(default=0, ge=0)
    data_origin: DataOrigin
    provenance: Provenance
    computed_at: Timestamp
    evaluation_version: EvaluationVersion

    @model_validator(mode="after")
    def _validate_ground_truth(self) -> Self:
        """Check the finding, the claim, the evidence, and the provenance agree.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the claim fields do not match the target, if a non-finding carries
                an observed value, if the window is not the horizon's own, if the evidence
                count disagrees with the identifiers listed, or if the provenance is anything
                but observed.
        """
        _require_exact_claim(
            "ground truth",
            self.target,
            "predicted",
            self.predicted_state,
            self.predicted_direction,
            self.predicted_positive,
        )
        assessed = self.status in (GroundTruthStatus.CONFIRMED, GroundTruthStatus.REFUTED)
        if assessed:
            _require_exact_claim(
                "ground truth",
                self.target,
                "observed",
                self.observed_state,
                self.observed_direction,
                self.observed_positive,
            )
        else:
            for name, value in (
                ("observed_state", self.observed_state),
                ("observed_direction", self.observed_direction),
                ("observed_positive", self.observed_positive),
            ):
                if value is not None:
                    raise ValueError(
                        f"ground truth is {self.status.value!r} but carries {name}={value!r}. An "
                        "unmeasurable window must be structurally unable to state an "
                        "observation, because 'nothing was observed' and 'the learner did not "
                        "decline' are different claims and only the second one is a finding."
                    )
        if self.horizon.observation_window() != self.observation_window:
            raise ValueError(
                "observation_window does not match the horizon's own window. The horizon is the "
                "only thing that decides which events were admissible, so a record carrying a "
                "different window is reporting a measurement that was never taken."
            )
        if len(self.evidence_ids) != self.evidence_count:
            raise ValueError(
                f"ground truth reports {self.evidence_count} admissible event(s) but lists "
                f"{len(self.evidence_ids)} identifier(s). The count is what distinguishes a "
                "measured reading from an absence, so it cannot disagree with the evidence."
            )
        if self.provenance is not Provenance.OBSERVED:
            raise ValueError(
                f"ground truth carries provenance {self.provenance.value!r}. A reading of the "
                "events that followed a prediction is an observation. Provenance "
                "'ground_truth' additionally requires an identified source external to the "
                "model, and this repository has none, so stamping it would claim an "
                "independence the events do not provide."
            )
        return self

    @property
    def is_assessed(self) -> bool:
        """Whether this ground truth states a finding rather than an absence."""
        return self.status in (GroundTruthStatus.CONFIRMED, GroundTruthStatus.REFUTED)

    @property
    def observed_value(self) -> object | None:
        """The observed claim value, or ``None`` for a non-finding."""
        if not self.is_assessed:
            return None
        return claim_value(
            self.target,
            self.observed_state,
            self.observed_direction,
            self.observed_positive,
        )

    def describe(self) -> str:
        """Render the ground truth as a single human-readable line.

        Returns:
            A description naming the prediction, the status, and the evidence count.
        """
        return (
            f"GT {self.ground_truth_id} for {self.prediction_id}: {self.status.value} "
            f"({self.reason}) from {self.evidence_count} event(s)"
        )

    def to_summary_dict(self) -> dict[str, object]:
        """Reduce the ground truth to flat, countable fields.

        Returns:
            A mapping with no nested containers.
        """
        return {
            "evaluation_version": self.evaluation_version,
            "ground_truth_id": self.ground_truth_id,
            "prediction_id": self.prediction_id,
            "learner_id": self.learner_id,
            "session_id": self.session_id,
            "target": self.target.value,
            "status": self.status.value,
            "reason": self.reason,
            "evidence_count": self.evidence_count,
            "horizon_seconds": self.horizon.duration_seconds,
            "provenance": self.provenance.value,
            "data_origin": self.data_origin.value,
        }


class EvaluationRecord(BaseModel):
    """One prediction scored against the evidence that followed it.

    Carries the prediction's own :attr:`model_version` beside the :attr:`evaluator_version`
    that produced the verdict. The two are separate because they answer separate questions:
    the model version says which artifact made the claim, and the evaluator version says which
    rules judged it. Collapsing them would mean a rule change could not be distinguished from
    a model change, and a regression would have no identifiable cause.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_id: non_empty_text
    evaluation_version: EvaluationVersion
    prediction_id: non_empty_text
    ground_truth_id: non_empty_text
    learner_id: non_empty_text
    session_id: non_empty_text
    target: PredictionTarget
    verdict: EvaluationVerdict
    binary_verdict: BinaryVerdict
    ground_truth_status: GroundTruthStatus
    observed_state: BehavioralEngagementState | None = None
    observed_direction: TrendDirection | None = None
    observed_positive: bool | None = None
    intervention_response: InterventionResponse | None = None
    intervention_id: str | None = None
    evidence_count: int = Field(default=0, ge=0)
    observation_window: ObservationWindow
    predicted_at: Timestamp
    reason: non_empty_text
    model_id: non_empty_text
    model_version: ModelVersion
    evaluator_version: EvaluationVersion
    data_origin: DataOrigin
    computed_at: Timestamp

    @model_validator(mode="after")
    def _validate_record(self) -> Self:
        """Check the verdict, the ground truth, and the binary cell agree.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If an unassessable record carries a binary cell or an observation, if
                an assessable record does not carry exactly the observed field its target
                names, if a non-``DECLINE`` target carries a binary cell, or if the two version
                coordinates disagree.
        """
        assessable = self.verdict is not EvaluationVerdict.NOT_ASSESSABLE
        binary = self.target is PredictionTarget.DECLINE
        if binary and assessable and self.binary_verdict is BinaryVerdict.NOT_ASSESSABLE:
            raise ValueError(
                f"verdict is {self.verdict.value!r} but binary_verdict is not_assessable. A "
                "scored prediction must land in a cell of the confusion matrix, and a record "
                "that is scored but uncounted is a reading nobody can aggregate."
            )
        if not assessable and self.binary_verdict is not BinaryVerdict.NOT_ASSESSABLE:
            raise ValueError(
                "verdict is not_assessable, so binary_verdict must be too. Placing an unassessable "
                "record in a cell would count a gap in a log as though it were a finding about a "
                "learner."
            )
        if not binary and self.binary_verdict is not BinaryVerdict.NOT_ASSESSABLE:
            raise ValueError(
                f"target is {self.target.value!r}, so binary_verdict must be not_assessable. A "
                "confusion matrix requires the DECLINE target, because a five-way or four-way "
                "claim has no positive class to place in the top-left cell."
            )
        assessed = self.ground_truth_status in (
            GroundTruthStatus.CONFIRMED,
            GroundTruthStatus.REFUTED,
        )
        if assessable != assessed:
            raise ValueError(
                f"verdict {self.verdict.value!r} and ground truth status "
                f"{self.ground_truth_status.value!r} disagree about whether a reading was made"
            )
        if assessable:
            _require_exact_claim(
                "evaluation record",
                self.target,
                "observed",
                self.observed_state,
                self.observed_direction,
                self.observed_positive,
            )
        else:
            for name, value in (
                ("observed_state", self.observed_state),
                ("observed_direction", self.observed_direction),
                ("observed_positive", self.observed_positive),
            ):
                if value is not None:
                    raise ValueError(
                        f"an unassessable record carries {name}={value!r}; it scored nothing, so "
                        "it observed nothing"
                    )
        if self.evaluation_version != self.evaluator_version:
            raise ValueError(
                f"evaluation_version {self.evaluation_version} disagrees with evaluator_version "
                f"{self.evaluator_version}. They name the same ruleset, and a record that "
                "disagrees with itself about which rules judged it cannot be re-scored "
                "consistently."
            )
        return self

    @property
    def is_correct(self) -> bool | None:
        """Whether the prediction was borne out, or ``None`` when it was not scored."""
        if self.verdict is EvaluationVerdict.NOT_ASSESSABLE:
            return None
        return self.verdict is EvaluationVerdict.CORRECT

    def describe(self) -> str:
        """Render the record as a single human-readable line.

        Returns:
            A description naming the prediction, the verdict, the binary cell, and the learner.
        """
        return (
            f"EVAL {self.evaluation_id}: {self.prediction_id} -> {self.verdict.value} "
            f"[{self.binary_verdict.value}] for {self.learner_id}"
        )

    def to_summary_dict(self) -> dict[str, object]:
        """Reduce the record to flat, countable fields.

        Returns:
            A mapping with no nested containers, suitable for a counter or a metrics frame.
        """
        return {
            "evaluation_version": self.evaluation_version,
            "evaluation_id": self.evaluation_id,
            "prediction_id": self.prediction_id,
            "ground_truth_id": self.ground_truth_id,
            "learner_id": self.learner_id,
            "session_id": self.session_id,
            "target": self.target.value,
            "verdict": self.verdict.value,
            "binary_verdict": self.binary_verdict.value,
            "ground_truth_status": self.ground_truth_status.value,
            "intervention_response": (
                self.intervention_response.value if self.intervention_response else None
            ),
            "intervention_id": self.intervention_id,
            "evidence_count": self.evidence_count,
            "horizon_seconds": self.observation_window.duration_seconds,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "evaluator_version": self.evaluator_version,
            "data_origin": self.data_origin.value,
        }
