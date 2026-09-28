"""The thin adapter: events in, structured trace out. Two phases, and the seam between them.

**This is the only module in ``api/`` that imports :mod:`focus_engine` for analysis.**
It imports public surfaces only. No layer internals, no private constructors, no
monkey-patching. A reviewer scanning this file can confirm the engine is called the
way the integration tests call it and nothing else. The delivery boundary lives in
:mod:`api.lab.execution`, which is development-only and must never move into the engine.

**Why two phases.** The pipeline is split at the point where an intervention would be sent,
because everything before that point is *analysis* and everything after it is *action*, and
conflating them produces a specific and hard-to-detect error: a system that quietly re-runs
the analysis after acting, then reports the second run's numbers as if they were the ones
that justified the action. So:

* :func:`prepare_analysis` (Phase A) reads events, runs context, features, baseline, temporal,
  prediction, uncertainty, authority, policy, and the safety gate, and returns a frozen
  :class:`AnalysisPhase`. It performs no action and emits no lifecycle event.
* :func:`execute_decision` (Phase B) takes that frozen result and does the only thing Phase A
  did not: it may deliver, then measure, then evaluate. It reads Phase A's values; it never
  recomputes them.

The split is testable rather than asserted. ``tests/lab/test_delivery.py`` freezes the engines
Phase A used and calls Phase B, so any recomputation inside Phase B raises immediately instead
of quietly producing a different number.

**The decision cut point.** Phase A analyses only the events at or before a cut instant, while
Phase B retains the whole stream. That asymmetry is the point: an intervention is delivered
*into* a session that is still running, and the events after the cut are the evidence about
what it did. Analysing the full stream and then delivering would let the prediction see the
answer, and the resulting outcome measurement would be meaningless. When the decision session
is too short to leave any evidence behind, the cut lands at its end and the honest result is
that the outcome and evaluation stages report insufficient data — which is reported, not
hidden.

**Determinism (Phase L).** Every engine is constructed with an injected
:class:`~focus_engine.utils.clock.Clock`, and the prediction is dated by the vector's
own ``computed_at`` (the Phase 9 defect fix). Delivery instants, intervention identifiers,
and lifecycle event ids are all derived rather than generated, so two calls with the same
events produce byte-identical responses including the delivery events.

**No fabrication.** If a layer reports ``insufficient_data``, that is what the stage log
shows. If the policy declines, that is what the stage log shows. If a gate stops delivery, the
gate is named. The adapter never pads an empty trace with invented values, never drops a
negative finding because it reads worse without it, and never reports a delivery that did not
happen.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from api.lab.contract import (
    AnalysisResponse,
    AnalysisResult,
    DeliveryStatus,
    InterventionStatus,
    SyntheticDataNotice,
)
from api.lab.diagnosis import diagnose
from api.lab.execution import (
    DEV_ONLY_WARNING,
    DeliveryAttempt,
    history_for,
    plan_delivery,
    post_window_reading,
    snapshot_for,
)
from focus_engine.authority import (
    AuthorityAssessment,
    AuthorityEngine,
    AuthorityLevel,
    ContextDescriptor,
    ContextRisk,
    FinalAction,
    HumanAvailability,
)
from focus_engine.baseline import (
    BASELINE_DIMENSIONS,
    DIMENSION_SOURCE_FEATURES,
    BaselineDimension,
    BaselineEngine,
    BaselineProfile,
    maturity_for,
)
from focus_engine.configuration.thresholds import BaselineSettings
from focus_engine.context import BLOCKING_CONTEXT_KEYS, ContextEngine
from focus_engine.evaluation import (
    EVALUATION_V1,
    EvaluationEngine,
    EvaluationRecord,
    PredictionHorizon,
)
from focus_engine.events import EventType
from focus_engine.events.types import EventEnvelope, QuestionAnsweredPayload
from focus_engine.features import (
    FEATURE_SET_V1,
    FeatureName,
    FeatureValue,
    compute_features,
)
from focus_engine.models import (
    Algorithm,
    ModelArtifact,
    Prediction,
    TrainingConfig,
    predict_with_artifact,
    train_model,
)
from focus_engine.models.encoding import (
    CATEGORICAL_VOCABULARIES,
    DesignMatrix,
    FeatureRequirement,
    FeatureValueType,
    encode_rows,
)
from focus_engine.outcomes import OUTCOME_V1, OutcomeEngine, OutcomeRecord
from focus_engine.policy import (
    InterventionPolicy,
    PolicyDecision,
    PolicyDecisionType,
    history_from_events,
)
from focus_engine.schemas import DataOrigin, Provenance
from focus_engine.simulator import LearnerArchetype, SimulationConfig, generate_session
from focus_engine.temporal import TemporalEngine, TemporalObservation
from focus_engine.temporal.models import TemporalState
from focus_engine.uncertainty import EvidenceVolume, UncertaintyEngine
from focus_engine.uncertainty.outcomes import PredictionOutcome
from focus_engine.utils.clock import Clock, FixedClock, SystemClock

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisPhase",
    "LabBundle",
    "ModelBundle",
    "execute_decision",
    "get_cached_artifact",
    "prepare_analysis",
    "run_analysis",
]

#: The eight features that are (a) measurable from a session with no intervention history
#: and (b) inside the model's declared vocabularies.
TRAIN_FEATURES: Final[tuple[FeatureName, ...]] = (
    FeatureName.SESSION_ELAPSED_SECONDS,
    FeatureName.SESSION_POSITION,
    FeatureName.SESSION_EVENT_COUNT,
    FeatureName.CONTENT_DIFFICULTY,
    FeatureName.PERFORMANCE_RECENT_ACCURACY,
    FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS,
    FeatureName.PERFORMANCE_ACCURACY_TREND,
    FeatureName.PERFORMANCE_RESPONSE_TREND,
)

_INT_FEATURES: Final[frozenset[FeatureName]] = frozenset({FeatureName.SESSION_EVENT_COUNT})

_BASELINE_SETTINGS: Final[BaselineSettings] = BaselineSettings()

#: Operating threshold handed to the predictor and echoed into the response.
_THRESHOLD: Final[float] = 0.5

#: Corpus composition for the one-time model fit. The counts are arbitrary; what matters is
#: that the corpus spans more than one archetype and that the label is derived from a
#: forward window, never from the same observation the model is shown.
_TRAIN_ARCHETYPES: Final[tuple[tuple[LearnerArchetype, int], ...]] = (
    (LearnerArchetype.STABLE, 16),
    (LearnerArchetype.GRADUAL_DECLINE, 16),
    (LearnerArchetype.RECOVERY, 8),
)

_LABEL_LOOKBACK: Final[int] = 3
_LABEL_LOOKAHEAD: Final[int] = 3
_LABEL_DROP: Final[float] = 0.10
_LABEL_SKIP_HEAD: Final[int] = 10

#: The learning situation a Lab decision is taken in. The authority layer's entire notion of
#: context is this string, so it is declared once here rather than being reconstructed from
#: event types at each call site. ``free_practice`` is deliberately the *least* consequential
#: of the catalogue's contexts: a recap prompt in free practice is reversible and carries no
#: stakes beyond the session, so a Lab run is not quietly granted authority that a graded
#: assessment would demand.
_LAB_CONTEXT: Final[ContextDescriptor] = ContextDescriptor(
    context_type="free_practice",
    risk=ContextRisk.LOW,
    is_reversible=True,
    is_high_stakes=False,
    description="A development Lab run over simulated events.",
)

#: No human is present during a Lab run, and the honest value for that is ``unknown`` rather
#: than ``unavailable``: the Lab does not know whether an approver exists, because there is
#: no approver service to ask. This matters because ``unknown`` defers a restricted action
#: rather than allowing it, so a Lab can never deliver something a real deployment would have
#: held for a person. Using ``unavailable`` would produce the same deferral, but it would
#: assert a fact about the world — that no approver exists — that the Lab does not know.
_LAB_HUMAN_AVAILABILITY: Final[HumanAvailability] = HumanAvailability.UNKNOWN

#: The post-prediction observation window used to score the prediction. Matched to the
#: outcome engine's default after-window so the outcome measurement and the evaluation score
#: read the same span of evidence; a reader comparing the two stages should not have to
#: reconcile two different windows.
_EVALUATION_HORIZON_MINUTES: Final[float] = 15.0

#: Answered questions that must remain *after* the cut for the outcome and evaluation stages
#: to have anything to measure. Below this, the cut is placed at the end of the decision
#: session and the downstream stages report insufficient data rather than a thin reading.
_EVIDENCE_AFTER_CUT: Final[int] = 4


def _train_requirements() -> tuple[FeatureRequirement, ...]:
    """Declare exactly the features the model is allowed to see.

    This is the load-bearing constraint of the whole lab. ``encode_rows`` rejects any row
    whose *required* features are absent, and when called without requirements it requires
    the entire ``FEATURE_SET_V1`` schema — including ``intervention_count``,
    ``trajectory_encoded``, and ``baseline_maturity_encoded``, none of which exist without
    an intervention history and a fitted baseline. Requiring them would reject every row.

    So the requirement set is declared here, explicitly, and it is exactly
    :data:`TRAIN_FEATURES`. That set was chosen because every member is computable from
    events plus context alone, which means the model can be scored at prediction time with
    the same call that produced its training rows. A model that needed features unavailable
    at inference time would be a model that could never run.
    """
    return tuple(
        FeatureRequirement(
            name=name,
            value_type=(
                FeatureValueType.STR
                if name in CATEGORICAL_VOCABULARIES
                else FeatureValueType.INT
                if name in _INT_FEATURES
                else FeatureValueType.FLOAT
            ),
            vocabulary=CATEGORICAL_VOCABULARIES.get(name),
        )
        for name in TRAIN_FEATURES
    )


_SYNTHETIC_NOTICE: Final[SyntheticDataNotice] = SyntheticDataNotice(
    data_origin=DataOrigin.SYNTHETIC.value,
    provenance=Provenance.OBSERVED.value,
    is_synthetic=True,
    warning="SYNTHETIC DATA — NOT REAL STUDENT DATA",
)


# ---------------------------------------------------------------------------
# One-time model fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """The trained artifact plus the provenance the UI must display about it."""

    artifact: ModelArtifact
    corpus_sessions: int
    train_rows: int
    holdout_rows: int
    holdout_brier: float | None
    holdout_ece: float | None

    def to_summary(self) -> dict[str, Any]:
        record = self.artifact.record
        return {
            "model_id": record.model_id,
            "model_version": record.model_version,
            "algorithm": str(getattr(record.algorithm, "value", record.algorithm)),
            "feature_set": str(
                getattr(record.feature_set_version, "value", record.feature_set_version)
            ),
            "target_definition": record.target_definition_version,
            "label_provenance": str(
                getattr(record.label_provenance, "value", record.label_provenance)
            ),
            "corpus_sessions": self.corpus_sessions,
            "train_rows": self.train_rows,
            "holdout_rows": self.holdout_rows,
            "holdout_brier": self.holdout_brier,
            "holdout_ece": self.holdout_ece,
            "caveat": (
                "Fit on a synthetic corpus for lab demonstration only. These metrics "
                "describe the simulator corpus, not any real learner's performance and "
                "not any production model."
            ),
        }


_CACHED_MODEL: ModelBundle | None = None


def _answered(events: Sequence[EventEnvelope]) -> list[EventEnvelope]:
    return [e for e in events if e.event_type is EventType.QUESTION_ANSWERED]


def _answer_payloads(events: Sequence[EventEnvelope]) -> list[QuestionAnsweredPayload]:
    return [
        e.payload
        for e in events
        if e.event_type is EventType.QUESTION_ANSWERED
        and isinstance(e.payload, QuestionAnsweredPayload)
    ]


def _group_sessions(
    events: Sequence[EventEnvelope],
) -> tuple[tuple[str, tuple[EventEnvelope, ...]], ...]:
    """Split a flat event stream into sessions, ordered by first event.

    The feature engine emits one vector per session, so this grouping is what makes the
    temporal engine's observation count mean something. It is also why a scenario is
    several sessions: one long session would produce one observation, and a temporal
    state built from a single observation cannot have a direction.
    """
    buckets: dict[str, list[EventEnvelope]] = {}
    for event in events:
        buckets.setdefault(event.session_id, []).append(event)
    return tuple(
        sorted(
            ((session_id, tuple(group)) for session_id, group in buckets.items()),
            key=lambda item: item[1][0].timestamp,
        )
    )


def _instant_for(events: Sequence[EventEnvelope], index: int) -> datetime:
    """The timestamp of the ``index``-th answered question, in timestamp order."""
    instants = sorted(e.timestamp for e in _answered(events))
    return instants[index]


def _vector_at(events: Sequence[EventEnvelope], instant: datetime) -> FeatureValue:
    """Run the real context + feature layers over the prefix ending at ``instant``."""
    prefix = tuple(e for e in events if e.timestamp <= instant)
    context = ContextEngine(clock=FixedClock(instant)).process(prefix)
    return compute_features(prefix, context)


def _session_vector(
    session_events: Sequence[EventEnvelope],
) -> tuple[ContextEngine, Any, FeatureValue]:
    """Context and features for one session, clocked to that session's last event.

    Returns the engine, the derived context, and the feature vector, so the caller can
    report the context without recomputing it.
    """
    ordered = tuple(session_events)
    last = ordered[-1].timestamp
    engine = ContextEngine(clock=FixedClock(last))
    context = engine.process(ordered)
    return engine, context, compute_features(ordered, context)


def _dimension_value(vector: FeatureValue, dimension: BaselineDimension) -> float:
    """Read the feature a baseline dimension is measured from.

    The mapping comes from the engine's own ``DIMENSION_SOURCE_FEATURES``. Re-deriving it
    here would be a second opinion about which feature stands for which dimension, and a
    disagreement here would silently misreport every deviation.
    """
    name = DIMENSION_SOURCE_FEATURES[dimension]
    for entry in vector.values:
        if entry.name is name:
            return float(entry.value)
    raise KeyError(f"Feature {name.value} absent from vector; cannot measure {dimension.value}.")


def _build_corpus_rows() -> tuple[DesignMatrix, int]:
    """Derive labelled rows from real simulator sessions.

    The label compares a trailing accuracy window to a forward window at the same
    question. It is a *definition of the target*, not a discovery: the point is to give
    the lab a fitted artifact, not to claim the target is the right one.
    """
    rows: list[tuple[FeatureValue, int]] = []
    session_index = 0

    for archetype, repeats in _TRAIN_ARCHETYPES:
        for repeat in range(repeats):
            session_index += 1
            config = SimulationConfig(
                archetype=archetype,
                learner_id=f"corpus-{archetype.value[:3]}-{repeat:03d}",
                session_id=f"corpus-s-{session_index:04d}",
                base_seed=700_000 + session_index,
                question_count=24,
                start_time=datetime(2026, 6, 1, tzinfo=UTC) - timedelta(days=40 - session_index),
                inter_event_gap_min=timedelta(seconds=45),
                inter_event_gap_max=timedelta(seconds=90),
            )
            events = generate_session(config).events
            flags = [p.correct for p in _answer_payloads(events)]
            count = len(flags)
            for index in range(count):
                if index < _LABEL_SKIP_HEAD or index + _LABEL_LOOKAHEAD >= count:
                    continue
                now = flags[index - _LABEL_LOOKBACK : index]
                soon = flags[index + 1 : index + 1 + _LABEL_LOOKAHEAD]
                if len(now) < _LABEL_LOOKBACK or len(soon) < _LABEL_LOOKAHEAD:
                    continue
                now_accuracy = sum(now) / len(now)
                soon_accuracy = sum(soon) / len(soon)
                label = 1 if soon_accuracy < now_accuracy - _LABEL_DROP else 0
                rows.append((_vector_at(events, _instant_for(events, index)), label))

    matrix = encode_rows(rows, requirements=_train_requirements())
    if len(matrix.labels) == 0:
        reasons = "; ".join(sorted({row.reason for row in matrix.rejected})[:3])
        raise RuntimeError(
            "The lab corpus produced no encodable rows. Every row was rejected by the "
            f"engine's own encoder. First reasons: {reasons}"
        )
    return matrix, session_index


def _split_by_learner(matrix: DesignMatrix) -> tuple[DesignMatrix, DesignMatrix]:
    """Hold out every fourth learner so the holdout is a learner split, not a row split."""
    ordered: list[str] = []
    for learner_id in matrix.learner_ids:
        if learner_id not in ordered:
            ordered.append(learner_id)
    holdout = set(ordered[::4])

    train_idx = [i for i, lid in enumerate(matrix.learner_ids) if lid not in holdout]
    holdout_idx = [i for i, lid in enumerate(matrix.learner_ids) if lid in holdout]

    def subset(indices: Sequence[int]) -> DesignMatrix:
        return DesignMatrix(
            matrix=matrix.matrix[list(indices)],
            labels=matrix.labels[list(indices)],
            columns=matrix.columns,
            requirements=matrix.requirements,
            learner_ids=tuple(matrix.learner_ids[i] for i in indices),
            rejected=matrix.rejected,
        )

    return subset(train_idx), subset(holdout_idx)


def _train_once() -> ModelBundle:
    matrix, sessions = _build_corpus_rows()
    train, holdout = _split_by_learner(matrix)

    config = TrainingConfig(
        algorithm=Algorithm.LOGISTIC_REGRESSION,
        model_id="lab-demo-model",
        model_version="LAB_V1",
        feature_set_version=FEATURE_SET_V1,
        dataset_version="LAB_SYNTHETIC_CORPUS_V1",
        target_definition_version="TARGET_NEXT3_ACCURACY_DROP_10PP_V1",
        label_provenance=Provenance.SYNTHETIC_LABEL,
        data_origin=DataOrigin.SYNTHETIC,
        seed=4242,
    )
    outcome = train_model(config, train, holdout)
    holdout_metric = outcome.metrics[0] if outcome.metrics else None
    return ModelBundle(
        artifact=outcome.artifact,
        corpus_sessions=sessions,
        train_rows=len(train.labels),
        holdout_rows=len(holdout.labels),
        holdout_brier=None if holdout_metric is None else holdout_metric.brier,
        holdout_ece=None if holdout_metric is None else holdout_metric.expected_calibration_error,
    )


def get_cached_artifact() -> ModelBundle:
    """Return the cached model, fitting it on first use."""
    global _CACHED_MODEL  # noqa: PLW0603
    if _CACHED_MODEL is None:
        logger.info("Fitting lab model on synthetic corpus (first call only).")
        _CACHED_MODEL = _train_once()
        logger.info("Lab model ready: %s", _CACHED_MODEL.artifact.record.model_id)
    return _CACHED_MODEL


# ---------------------------------------------------------------------------
# Stage log helpers
# ---------------------------------------------------------------------------


def _stage(
    stage: str,
    title: str,
    detail: dict[str, Any],
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"stage": stage, "title": title, "detail": detail}
    if entries is not None:
        record["entries"] = entries
    return record


def _observation_count(profile: BaselineProfile) -> int:
    """How many observations the profile retains, read from the engine's own statistics.

    Read from ``profile.statistics`` rather than ``engine.reference(...)``: the reference
    reports the observations behind whichever source it used, and with no population prior
    configured that is zero even when the profile is full. Using it made every scenario
    report ``new`` maturity and no observations, which is not what the engine holds.
    """
    if not profile.statistics:
        return 0
    return max(statistic.observations for statistic in profile.statistics)


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabBundle:
    """One completed run: the contract response plus the log the UI animates."""

    response: AnalysisResponse
    model_summary: dict[str, Any]
    question_count: int


@dataclass(frozen=True, slots=True)
class AnalysisPhase:
    """Phase A's entire output, frozen. Phase B reads this and may not recompute it.

    **Frozen because the "no recomputation" guarantee is otherwise unfalsifiable.** Every
    field the decision depended on is carried across the seam, so ``execute_decision`` has
    nothing to recompute even if it wanted to. ``tests/lab/test_delivery.py`` enforces this by
    constructing this object directly and calling Phase B with the engines it names already
    exhausted.

    The ``events``/``cut_at`` pair is the asymmetry the two phases exist to preserve:
    ``events`` is the *whole* stream, retained so Phase B can measure what happened after the
    decision, while everything else in this object was derived from the prefix at or before
    ``cut_at``. Phase B never re-derives a Phase A value; it only reads forward.
    """

    events: tuple[EventEnvelope, ...]
    cut_at: datetime
    learner_id: str
    session_id: str
    question_count: int
    events_after_cut: int
    state: TemporalState | None
    vector: FeatureValue | None
    uncertainty: PredictionOutcome | None
    assessment: AuthorityAssessment | None
    final_action: FinalAction | None
    decision: PolicyDecision | None
    profile: BaselineProfile | None
    model_summary: dict[str, Any]
    log: tuple[dict[str, Any], ...]
    cut_reason: str


def _decision_cut(
    session_events: Sequence[EventEnvelope], requested: datetime | None
) -> tuple[datetime, str]:
    """Choose the instant the decision is made, and the evidence left for what follows it.

    **Why the cut is not simply the end of the stream.** An intervention is delivered into a
    session that is still running. If the decision is taken at the last event, the
    post-delivery window is empty by construction and the outcome and evaluation stages can
    only ever report ``INSUFFICIENT_DATA`` — a pipeline that cannot demonstrate itself. So the
    cut is placed earlier, leaving the remainder of the decision session as evidence.

    **Why the cut is not early.** The prediction is made from a mid-session position, which
    is a real operational position but a harder one: the learner has less history behind them
    than at the end. Placing the cut too early would report a thin prediction as though it
    were a representative one. The midpoint of the answered questions is the compromise, and
    the number of remaining events is reported in the stage log so a reader can judge it.

    **The degenerate case is left alone rather than fixed.** If the session is too short to
    leave :data:`_EVIDENCE_AFTER_CUT` questions behind, the cut goes to the session's end and
    the downstream stages report what they actually find. Manufacturing a session long enough
    to look good would be the dishonest fix.

    Args:
        session_events: The decision session's events.
        requested: An explicit cut instant, or ``None`` to derive one.

    Returns:
        The cut instant and a human-readable statement of why it was placed there.
    """
    last = session_events[-1].timestamp
    if requested is not None:
        return requested, "Supplied by the caller as an explicit decision cut point."
    answered = _answered(session_events)
    if len(answered) < _EVIDENCE_AFTER_CUT * 2:
        return (
            last,
            f"The decision session holds {len(answered)} answered question(s), too few to "
            f"leave {_EVIDENCE_AFTER_CUT} behind a mid-session cut. The cut was placed at "
            "the end of the session, so the outcome and evaluation stages will report what "
            "evidence actually exists rather than a reading derived from a synthetic split.",
        )
    midpoint = answered[len(answered) // 2].timestamp
    return (
        midpoint,
        f"Placed at the midpoint of the decision session's {len(answered)} answered "
        f"questions, leaving {len(answered) - len(answered) // 2} question(s) after the "
        "intervention for the outcome and evaluation stages to measure.",
    )


def prepare_analysis(
    events: Sequence[EventEnvelope],
    *,
    learner_id: str,
    session_id: str,
    clock: Clock | None = None,
    decision_cut: datetime | None = None,
) -> AnalysisPhase:
    """Phase A. Read events, decide, and return a frozen result. Performs no action.

    Runs context, features, baseline, temporal, prediction, uncertainty, authority, policy, and
    the safety gate — the last two being the gates :func:`execute_decision` will consult. It
    emits no lifecycle event and calls nothing that could reach a learner, so this phase is
    safe to run repeatedly and safe to freeze.

    Args:
        events: The learner's events. Used verbatim; never reordered or filtered. The prefix
            at or before the cut is what gets analysed.
        learner_id: The pseudonymous learner identifier.
        session_id: The pseudonymous session identifier.
        clock: Injected into every layer that takes one. Defaults to
            :class:`~focus_engine.utils.clock.SystemClock`; tests pass a
            :class:`~focus_engine.utils.clock.FixedClock` to make the run byte-identical.
        decision_cut: An explicit decision instant, or ``None`` to derive one from the
            decision session.

    Returns:
        A frozen :class:`AnalysisPhase`. Fields are ``None`` only when the input held no
        answered question, which is the one case where no analysis was possible at all.
    """
    model = get_cached_artifact()
    active_clock: Clock = SystemClock() if clock is None else clock
    frozen = _EngineClocks(active_clock)

    log: list[dict[str, Any]] = []
    ordered = tuple(events)
    full_sessions = _group_sessions(ordered)
    question_count = sum(len(_answered(session)) for _, session in full_sessions)

    if question_count == 0:
        return AnalysisPhase(
            events=ordered,
            cut_at=ordered[-1].timestamp if ordered else active_clock.now(),
            learner_id=learner_id,
            session_id=session_id,
            question_count=0,
            events_after_cut=0,
            state=None,
            vector=None,
            uncertainty=None,
            assessment=None,
            final_action=None,
            decision=None,
            profile=None,
            model_summary=model.to_summary(),
            log=(),
            cut_reason="No answered question in the input, so no analysis was possible.",
        )

    # --- THE DECISION CUT ---------------------------------------------------------
    # Placed before the session grouping, because it decides which events the rest of
    # Phase A is allowed to see. Everything below analyses the prefix; the full stream is
    # carried forward untouched for Phase B.
    decision_session_id, decision_session = full_sessions[-1]
    cut_at, cut_reason = _decision_cut(decision_session, decision_cut)
    prefix = tuple(event for event in ordered if event.timestamp <= cut_at)
    sessions = _group_sessions(prefix)
    events_after_cut = sum(
        1
        for event in ordered
        if event.timestamp > cut_at and event.event_type is EventType.QUESTION_ANSWERED
    )

    # --- EVENTS -------------------------------------------------------------------
    # Reports the full stream and, separately, what Phase A was allowed to analyse. Both
    # counts appear because a reader who sees "24 events" and a reader who sees "12 events"
    # are looking at different things, and the difference is the whole point of the cut.
    log.append(
        _stage(
            "events",
            "EVENTS",
            {
                "total_events": len(ordered),
                "events_analysed": len(prefix),
                "answered_questions": question_count,
                "answered_before_cut": sum(len(_answered(session)) for _, session in sessions),
                "session_count": len(sessions),
                "learner_id": learner_id,
                "sessions": [
                    {
                        "session_id": group_session_id,
                        "events": len(session),
                        "answered": len(_answered(session)),
                        "started": session[0].timestamp.isoformat(),
                        "ended": session[-1].timestamp.isoformat(),
                    }
                    for group_session_id, session in full_sessions
                ],
                "source": "focus_engine.simulator.generate_session (synthetic)",
            },
            entries=[
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "session_id": e.session_id,
                    "timestamp": e.timestamp.isoformat(),
                    "analysed": e.timestamp <= cut_at,
                }
                for e in ordered
            ],
        )
    )

    log.append(
        _stage(
            "cut",
            "DECISION POINT",
            {
                "cut_at": cut_at.isoformat(),
                "reason": cut_reason,
                "events_total": len(ordered),
                "events_analysed": len(prefix),
                "events_after_cut": events_after_cut,
                "decision_session_id": decision_session_id,
                "note": (
                    "Phase A analysed only the events at or before the cut. The events "
                    "after it are the evidence about what a delivered intervention did, "
                    "and are held back deliberately: a prediction made while seeing them "
                    "could not be scored against them."
                ),
            },
        )
    )

    # --- CONTEXT + FEATURES (the decision session) --------------------------------
    # Context and features are reported for the *last analysed* session, because that
    # session is the one the prediction is made from. Earlier sessions contributed to the
    # baseline and the temporal state, not to this vector.
    decision_session_id, decision_session = sessions[-1]
    _, context, vector = _session_vector(decision_session)
    missing = sorted(key.value for key in context.missing)
    blocking = sorted(key.value for key in context.missing if key in BLOCKING_CONTEXT_KEYS)
    log.append(
        _stage(
            "context",
            "CONTEXT",
            {
                "session_id": decision_session_id,
                "derivation": (
                    "Derived from the decision session only. Earlier sessions are not "
                    "concatenated into it, because the feature engine computes session "
                    "features per session."
                ),
                "reference_time": context.reference_time.isoformat(),
                "elapsed_seconds": context.session.elapsed_seconds,
                "position_in_session": context.session.position_in_session,
                "event_count": context.session.event_count,
                "recent_accuracy": context.performance.recent_accuracy,
                "average_response_seconds": context.performance.average_response_seconds,
                "recent_questions": context.performance.recent_questions,
                "missing": missing,
                "blocking_missing": blocking,
                "note_on_blocking": (
                    "A blocking key that is absent means the feature layer cannot "
                    "produce the features a prediction needs."
                ),
            },
        )
    )

    log.append(
        _stage(
            "features",
            "FEATURES",
            {
                "session_id": decision_session_id,
                "feature_set": vector.feature_set_version,
                "computed_at": vector.computed_at.isoformat(),
                "values": [
                    {
                        "name": entry.name.value,
                        "value": entry.value,
                        "availability": entry.availability.value,
                        "in_model": entry.name in TRAIN_FEATURES,
                        "reason": entry.reason,
                    }
                    for entry in vector.values
                ],
            },
        )
    )

    # --- BASELINE + TEMPORAL ------------------------------------------------------
    # One temporal observation per session, in the update-then-measure order used by
    # tests/integration/test_temporal_integration.py.
    #
    # Two decisions here are load-bearing, and both were arrived at by watching the
    # engine disagree with the scenario names:
    #
    # 1. Observation granularity is per session, not per question. The feature engine
    #    emits one vector per session. Feeding it forty vectors from a single sitting
    #    makes the temporal run-length track within-session noise, and every archetype
    #    collapsed onto the same "recovering" state.
    #
    # 2. `update` runs before `deviation`. The comparison is "this session versus this
    #    learner's recent self", not "this session versus a frozen prefix". Freezing the
    #    profile instead makes `session_elapsed_seconds` diverge without bound, because
    #    that dimension is a monotonic clock, and it reports a stable learner as
    #    worsening by two standard deviations. That is a false positive manufactured by
    #    the harness, not a finding about a learner.
    baseline = BaselineEngine(_BASELINE_SETTINGS)
    temporal = frozen.temporal()
    # Created at the earliest event, because ingest requires each observation to be
    # strictly later than the state's reference time.
    session_start = sessions[0][1][0].timestamp
    profile = baseline.create_profile(learner_id, session_start)
    state = temporal.create_state(learner_id, session_start)

    temporal_entries: list[dict[str, Any]] = []
    decision_at = session_start
    decision_vector: FeatureValue = vector
    centre_first: dict[str, float] = {}

    for index, (group_session_id, group_events) in enumerate(sessions):
        _, _, session_vector = _session_vector(group_events)
        profile = baseline.update(profile, session_vector)
        # Record where each centre stood at its first accepted observation. The gap between
        # that and the centre at the end is the honest measure of how far the baseline
        # travelled with the learner, and it is what makes baseline absorption visible
        # instead of merely arguable.
        for dimension in BASELINE_DIMENSIONS:
            if dimension.value in centre_first:
                continue
            for item in profile.statistics:
                if item.dimension is dimension and item.centre is not None:
                    centre_first[dimension.value] = float(item.centre)
                    break
        deviations = tuple(
            baseline.deviation(profile, dimension, _dimension_value(session_vector, dimension))
            for dimension in BASELINE_DIMENSIONS
        )
        state = temporal.ingest(
            state,
            TemporalObservation(
                learner_id=learner_id,
                computed_at=session_vector.computed_at,
                deviations=deviations,
                origins=frozenset({DataOrigin.SYNTHETIC}),
            ),
        )
        if index == len(sessions) - 1:
            decision_at = state.reference_time
            decision_vector = session_vector
        temporal_entries.append(
            {
                "session": index + 1,
                "session_id": group_session_id,
                "answered": len(_answered(group_events)),
                "timestamp": session_vector.computed_at.isoformat(),
                "deviations": {
                    deviation.dimension.value: deviation.standardised for deviation in deviations
                },
                "state": state.state.value,
                "direction": state.direction.value,
                "consecutive": state.evidence.consecutive,
                "observations": state.evidence.observations,
            }
        )

    observations = _observation_count(profile)
    maturity = maturity_for(observations, _BASELINE_SETTINGS)
    centres: dict[str, Any] = {}
    for dimension in BASELINE_DIMENSIONS:
        reference = baseline.reference(profile, dimension)
        first = centre_first.get(dimension.value)
        centres[dimension.value] = {
            "centre": reference.centre,
            "spread": reference.spread,
            "observations": reference.observations,
            "maturity": reference.maturity.value,
            "source": reference.source.value,
            "centre_first": first,
            "centre_drift": (
                None
                if first is None or reference.centre is None
                else float(reference.centre) - first
            ),
        }

    log.append(
        _stage(
            "baseline",
            "BASELINE",
            {
                "sessions_folded_in": len(sessions),
                "observations": observations,
                "maturity": maturity.value,
                "settings_fingerprint": profile.settings_fingerprint,
                "dimensions": centres,
                "note": (
                    "A rolling robust statistic (median and MAD) over every session so "
                    "far. Each session is measured against the learner's recent self. "
                    "Absence is reported as absence: a dimension whose source feature was "
                    "insufficient is left untouched rather than counted as an observation. "
                    "centre_first and centre_drift show how far each centre travelled while "
                    "sessions were folded in, which is the mechanism behind most "
                    "scenario-label/observed-state disagreements."
                ),
            },
        )
    )

    log.append(
        _stage(
            "temporal",
            "TEMPORAL STATE",
            {
                "learner_id": state.learner_id,
                "reference_time": state.reference_time.isoformat(),
                "state": state.state.value,
                "direction": state.direction.value,
                "character": state.character.value,
                "change": state.change.value,
                "run_length": state.run,
                "evidence_observations": state.evidence.observations,
                "evidence_consecutive": state.evidence.consecutive,
                "evidence_maturity": state.evidence.maturity.value,
                "mean_standardised": state.evidence.mean_standardised,
                "dispersion": state.evidence.dispersion,
                "settings_fingerprint": state.settings_fingerprint,
            },
            entries=temporal_entries,
        )
    )

    # --- PREDICTION ---------------------------------------------------------------
    prediction = predict_with_artifact(model.artifact, decision_vector, threshold=_THRESHOLD)
    if isinstance(prediction, Prediction):
        prediction_detail: dict[str, Any] = {
            "outcome": "Prediction",
            "probability": prediction.probability,
            "is_positive": prediction.is_positive,
            "threshold": prediction.threshold,
            "model_id": prediction.model_id,
            "model_version": prediction.model_version,
            "feature_set": prediction.feature_set_version,
            "computed_at": prediction.computed_at.isoformat(),
            "model": model.to_summary(),
        }
    else:
        prediction_detail = {
            "outcome": "PredictionRefusal",
            "reason": prediction.reason,
            "missing_features": list(prediction.missing_features),
            "computed_at": prediction.computed_at.isoformat(),
            "note": "A refusal is the model declining to score. It is not a zero.",
        }
    log.append(_stage("prediction", "PREDICTION", prediction_detail))

    # --- UNCERTAINTY --------------------------------------------------------------
    evidence = EvidenceVolume.from_observation_count(observations, baseline=_BASELINE_SETTINGS)
    uncertainty_engine = frozen.uncertainty()
    outcome = uncertainty_engine.assess(
        prediction,
        artifact=model.artifact,
        evidence=evidence,
        vector=decision_vector,
        threshold=_THRESHOLD,
    )

    uncertainty_detail: dict[str, Any] = {
        "verdict": outcome.verdict.value,
        "probability": outcome.probability,
        "is_positive": outcome.is_positive,
        "threshold": outcome.threshold,
        "confidence": None if outcome.confidence is None else outcome.confidence.value,
        "remedy": outcome.remedy.value,
        "basis": None if outcome.basis is None else outcome.basis.value,
        "maturity": None if outcome.maturity is None else outcome.maturity.value,
        "evidence_units": outcome.evidence_units,
        "refusal_reason": outcome.refusal_reason,
        "missing_features": list(outcome.missing_features),
        "computed_at": None if outcome.computed_at is None else outcome.computed_at.isoformat(),
        "explanation": None,
    }
    if outcome.explanation is not None:
        uncertainty_detail["explanation"] = {
            "available": outcome.explanation.available,
            "base_probability": outcome.explanation.base_probability,
            "min_contribution": outcome.explanation.min_contribution,
            "omitted": list(outcome.explanation.omitted),
            "reason": outcome.explanation.reason,
            "top_contributions": [
                {
                    "feature": contribution.feature,
                    "column": contribution.column,
                    "contribution": contribution.contribution,
                }
                for contribution in outcome.explanation.top(4)
            ],
        }
    if outcome.assessment is not None:
        uncertainty_detail["confidence_assessment"] = {
            "value": outcome.assessment.value,
            "level": outcome.assessment.level.value,
            "binding_constraint": outcome.assessment.binding.value,
            "model_assertion": outcome.assessment.model_assertion,
            "calibration_ceiling": outcome.assessment.calibration_ceiling,
            "evidence_ceiling": outcome.assessment.evidence_ceiling,
            "maturity_ceiling": outcome.assessment.maturity_ceiling,
            "explanation_capped": outcome.assessment.explanation_capped,
        }
    log.append(_stage("uncertainty", "UNCERTAINTY", uncertainty_detail))

    # --- POLICY -------------------------------------------------------------------
    policy_events = tuple(e for e in ordered if e.timestamp <= decision_at)
    history = history_from_events(learner_id, policy_events, up_to=decision_at)
    policy = frozen.policy()
    decision = policy.decide(
        outcome,
        history,
        learner_id=learner_id,
        session_id=session_id,
        state=state.state,
        now=decision_at,
    )

    selected_type = None if decision.selected is None else decision.selected.intervention_type
    log.append(
        _stage(
            "policy",
            "POLICY",
            {
                "decision": decision.decision.value,
                # An INTERVENE decision carries no non-action reason, by the engine's own
                # validator. Reading `.value` unconditionally crashed the pipeline on the
                # one run that succeeded, which is the worst possible place to fail.
                "reason": None if decision.reason is None else decision.reason.value,
                "detail": decision.detail,
                "selected": decision.selected is not None,
                "intervention_type": selected_type,
                "evaluated_at": decision.evaluated_at.isoformat(),
                "policy_version": decision.policy_version,
                "settings_fingerprint": decision.settings_fingerprint,
                "upstream_verdict": (
                    None if decision.upstream_verdict is None else decision.upstream_verdict.value
                ),
                "restraints": [
                    {
                        "kind": restraint.kind.value,
                        "description": restraint.describe(),
                        "limit": restraint.limit,
                        "observed": restraint.observed,
                        "headroom": restraint.headroom,
                        "binding": restraint.binding,
                    }
                    for restraint in decision.restraints
                ],
                "excluded": list(decision.excluded),
            },
        )
    )

    # --- AUTHORITY ----------------------------------------------------------------
    # Runs *before* policy and produces the envelope, not a verdict. The distinction is
    # load-bearing: the envelope says which actions this learner, in this context, at this
    # confidence, may have — and policy then chooses among them. Running authority after
    # policy would invert that into "choose, then justify", which is how a permission check
    # becomes a rubber stamp.
    authority = frozen.authority()
    assessment = authority.authorize(
        learner_id,
        _LAB_CONTEXT,
        outcome.confidence,
    )
    envelope = assessment.envelope
    log.append(
        _stage(
            "authority",
            "AUTHORITY",
            {
                "context_type": _LAB_CONTEXT.context_type,
                "risk": _LAB_CONTEXT.risk.value,
                "is_reversible": _LAB_CONTEXT.is_reversible,
                "is_high_stakes": _LAB_CONTEXT.is_high_stakes,
                "authority_level": envelope.authority_level.value,
                "permission": assessment.permission,
                "permitted_actions": sorted(str(item) for item in envelope.permitted_actions),
                # RestrictedAction is a closed StrEnum of action names, not a record per
                # action, so it is listed as names. The approval roles those names require
                # travel separately in ``required_approvals`` and are not re-derived here.
                "restricted_actions": sorted(str(item) for item in envelope.restricted_actions),
                "required_approvals": [str(item) for item in envelope.required_approvals],
                "escalation_path": list(envelope.escalation_path),
                "confidence_weight": envelope.confidence_weight,
                "confidence_band": None if outcome.confidence is None else outcome.confidence.value,
                "confidence_ceiling": assessment.decay.ceiling,
                "autonomy_min_weight": authority.settings.autonomy_min_weight,
                "meets_autonomy_threshold": (
                    envelope.authority_level is AuthorityLevel.ALLOWED
                    and envelope.confidence_weight >= authority.settings.autonomy_min_weight
                ),
                "calculated_at": envelope.calculated_at.isoformat(),
                "expires_at": envelope.expires_at.isoformat(),
                "basis": envelope.basis,
                "restrictions": list(envelope.restrictions),
                "memory_entries_counted": assessment.memory_entries_counted,
                "decay": {
                    "weight": assessment.decay.weight,
                    "base": assessment.decay.base,
                    "decayed": assessment.decay.decayed,
                    "penalty_total": assessment.decay.penalty_total,
                    "credit_total": assessment.decay.credit_total,
                    "binding": assessment.decay.binding,
                    "applied_penalties": [
                        {"kind": item.kind.value, "amount": item.amount}
                        for item in assessment.decay.applied_penalties
                    ],
                    "applied_credits": [
                        {"kind": item.kind.value, "amount": item.amount}
                        for item in assessment.decay.applied_credits
                    ],
                },
                "human_availability": _LAB_HUMAN_AVAILABILITY.value,
                "note": (
                    "The envelope precedes the policy decision on purpose. It states what "
                    "may be done; policy chooses what to do; the safety gate below confirms "
                    "the choice was permitted. A Lab run reports human availability as "
                    "'unknown' because there is no approver service to ask, and 'unknown' "
                    "defers a restricted action rather than allowing it."
                ),
            },
        )
    )

    # --- SAFETY (the final gate) --------------------------------------------------
    # Consumes the envelope *and* the policy decision, so it is the last thing that runs
    # before delivery may occur. It is computed here, in Phase A, and merely *consulted* in
    # Phase B — so a run that is frozen mid-pipeline still carries the gate's verdict.
    final_action = authority.finalise(decision, envelope, _LAB_HUMAN_AVAILABILITY)
    log.append(
        _stage(
            "safety",
            "SAFETY GOVERNOR",
            {
                "outcome": final_action.outcome.value,
                "action": final_action.action,
                "intervention_type": final_action.intervention_type,
                "reason": final_action.reason,
                "authority_level": final_action.authority_level.value,
                "confidence_weight": final_action.confidence_weight,
                "envelope_expired": final_action.envelope_expired,
                "requires_human_approval": final_action.requires_human_approval,
                "escalation_path": list(final_action.escalation_path),
                "available_human": None
                if final_action.available_human is None
                else final_action.available_human.value,
                "human_availability": _LAB_HUMAN_AVAILABILITY.value,
                "selected_by_policy": decision.decision is PolicyDecisionType.INTERVENE,
                "note": (
                    "This is the final gate. Delivery may proceed only on an 'allowed' "
                    "outcome, and an 'allowed' outcome that still requires a named human's "
                    "approval is not permission to send: no approval store exists here, so "
                    "no artifact could prove anyone approved. Such a case is reported as "
                    "SELECTED_NOT_DELIVERED rather than delivered."
                ),
            },
        )
    )

    return AnalysisPhase(
        events=ordered,
        cut_at=decision_at,
        learner_id=learner_id,
        session_id=session_id,
        question_count=question_count,
        events_after_cut=events_after_cut,
        state=state,
        vector=decision_vector,
        uncertainty=outcome,
        assessment=assessment,
        final_action=final_action,
        decision=decision,
        profile=profile,
        model_summary=model.to_summary(),
        log=tuple(log),
        cut_reason=cut_reason,
    )


# ---------------------------------------------------------------------------
# Phase B: execute, measure, evaluate
# ---------------------------------------------------------------------------

#: Why a stage produced nothing. Each names the *upstream fact* responsible, because
#: "no outcome" is never actionable on its own — the reader needs to know whether nothing was
#: delivered, nothing was measurable, or nothing was scoreable, because those are three
#: different bugs and one of them is expected.
_NO_ANALYSIS_REASON = (
    "The input held no answered question, so Phase A produced no decision and there was "
    "nothing for a delivery boundary to act on."
)
_NOT_DELIVERED_REASON = (
    "No delivery occurred, so there is no anchor for OutcomeEngine.measure. The engine "
    "refuses to measure an intervention that was not delivered, and the adapter does not "
    "invent one."
)
_NO_EVIDENCE_REASON = (
    "The delivered intervention has fewer events in its measurement window than the engine's "
    "minimum, so the measures report insufficient data rather than a reading."
)
_NO_SNAPSHOT_REASON = (
    "EvaluationEngine scores a PredictionSnapshot against admissible in-window evidence. "
    "The prediction was refused, so there is no scored prediction to evaluate — grading a "
    "model for declining to answer would be meaningless."
)
_NO_READING_REASON = (
    "The in-window evidence is too thin to rebuild a temporal reading of the learner. The "
    "engine's own floor is two observations, and each needs enough events to produce "
    "features. The absence is reported rather than papered over with the decision-time "
    "state, which would grade the model on the evidence it was given."
)


def _intervention_stage(attempt: DeliveryAttempt) -> dict[str, Any]:
    """The intervention stage, reporting the delivery status and every lifecycle event."""
    return _stage(
        "intervention",
        "INTERVENTION",
        {
            "dev_only": True,
            "warning": DEV_ONLY_WARNING,
            "delivered": attempt.delivered,
            "status": attempt.status.value,
            "status_label": attempt.status.name.replace("_", " "),
            "gate": attempt.gate,
            "intervention_id": attempt.intervention_id,
            "started_at": None if attempt.started_at is None else attempt.started_at.isoformat(),
            "delivered_at": (
                None if attempt.delivered_at is None else attempt.delivered_at.isoformat()
            ),
            "final_action": attempt.final_action.outcome.value,
            "requires_human_approval": attempt.requires_human_approval,
            "reason": attempt.reason,
            "lifecycle_event_count": len(attempt.events),
            "lifecycle_events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "payload": event.payload.model_dump(mode="json"),
                    "origin": event.origin.value,
                    "provenance": event.provenance.value,
                }
                for event in attempt.events
            ],
        },
    )


def _outcome_stage(record: OutcomeRecord | None, reason: str) -> dict[str, Any]:
    """The outcome stage: the measurement window, what it read, and what it could not.

    Reports the per-measure status rather than only a roll-up. A summary that showed "3 of 7
    measures" without saying which four declined would present absences as though they were
    stable readings, which is the specific misreading the outcome layer exists to prevent.
    """
    if record is None:
        return _stage(
            "outcome",
            "OUTCOME",
            {"measured": False, "status": "INSUFFICIENT_DATA", "reason": reason},
        )

    before = record.before_window
    after = record.after_window
    measured = [item for item in record.measurements if item.status.value == "measured"]
    return _stage(
        "outcome",
        "OUTCOME",
        {
            "measured": True,
            "status": "MEASURED",
            "outcome_version": record.outcome_version,
            "intervention_id": record.intervention_id,
            "intervention_type": record.intervention_type,
            "learner_id": record.learner_id,
            "session_id": record.session_id,
            "delivered_at": record.delivered_at.isoformat(),
            "response_class": record.response_class.value,
            "policy_version": record.policy_version,
            "data_origin": record.data_origin.value,
            "is_synthetic_only": record.is_synthetic_only,
            "is_complete": record.is_complete,
            "settings_fingerprint": record.settings_fingerprint,
            # The engine's own flat summary, passed through rather than recomputed, so the
            # counts a reader sees here are the engine's counts and not a second tally.
            "summary": record.to_summary_dict(),
            "window": {
                "before": {
                    "kind": before.kind.value,
                    "start": before.start.isoformat(),
                    "end": before.end.isoformat(),
                    "duration_seconds": before.duration_seconds,
                },
                "after": {
                    "kind": after.kind.value,
                    "start": after.start.isoformat(),
                    "end": after.end.isoformat(),
                    "duration_seconds": after.duration_seconds,
                },
            },
            "measures_measured": len(measured),
            "measures_total": len(record.measurements),
            # ``before_samples``/``after_samples`` are the evidence each measure was read
            # from. Reporting them per measure is the only honest way to show how much
            # evidence existed, because the record carries no single event count.
            "measurements": [
                {
                    "measure": item.measure.value,
                    "status": item.status.value,
                    "unit": item.unit,
                    "before_value": item.before_value,
                    "after_value": item.after_value,
                    "before_samples": item.before_samples,
                    "after_samples": item.after_samples,
                    "direction": item.direction.value,
                    "data_origin": item.data_origin.value,
                    "reason": item.reason,
                }
                for item in record.measurements
            ],
            "limitation": (
                None
                if record.is_complete
                else f"{len(record.measurements) - len(measured)} of "
                f"{len(record.measurements)} measures were not read. A measure with no "
                "evidence is reported as such rather than inferred."
            ),
            "note": (
                "Anchored on delivered_at. These readings describe what the events show "
                "after the prompt. They do not establish that the prompt caused them: the "
                "events cannot distinguish a learner who improved because of the recap from "
                "one who improved anyway."
            ),
        },
    )


def _evaluation_stage(record: EvaluationRecord | None, reason: str) -> dict[str, Any]:
    """The evaluation stage: the verdict, and the boundary that keeps it leakage-safe."""
    if record is None:
        return _stage(
            "evaluation",
            "EVALUATION",
            {"assessed": False, "status": "INSUFFICIENT_DATA", "reason": reason},
        )

    window = record.observation_window
    return _stage(
        "evaluation",
        "EVALUATION",
        {
            "assessed": True,
            "status": record.verdict.value,
            "evaluation_version": record.evaluation_version,
            "verdict": record.verdict.value,
            "binary_verdict": record.binary_verdict.value,
            "ground_truth_status": record.ground_truth_status.value,
            "observed_state": None
            if record.observed_state is None
            else record.observed_state.value,
            "observed_direction": (
                None if record.observed_direction is None else record.observed_direction.value
            ),
            "observed_positive": record.observed_positive,
            "intervention_response": (
                None if record.intervention_response is None else record.intervention_response.value
            ),
            "intervention_id": record.intervention_id,
            "evidence_count": record.evidence_count,
            "reason": record.reason,
            "model_id": record.model_id,
            "model_version": record.model_version,
            "evaluator_version": record.evaluator_version,
            "data_origin": record.data_origin.value,
            "horizon": {
                "predicted_at": record.predicted_at.isoformat(),
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
                "duration_seconds": window.duration_seconds,
                "boundary": (
                    "Half-open [predicted_at, predicted_at + horizon). An event exactly at "
                    "the end is excluded and one exactly at the prediction instant is "
                    "included."
                ),
            },
            "reading": {
                "anchored_at": None
                if record.observation_window is None
                else window.end.isoformat(),
                "note": (
                    "The reading was rebuilt from in-window evidence only. The decision-time "
                    "temporal state is deliberately not reused, because it is anchored at the "
                    "prediction and would grade the model on its own input."
                ),
            },
        },
    )


def execute_decision(phase: AnalysisPhase, *, clock: Clock | None = None) -> LabBundle:
    """Phase B. Act on Phase A's frozen decision, then measure and evaluate the result.

    This is the only function in the pipeline that can emit a lifecycle event, and the only
    one that may touch :mod:`api.lab.execution`. It reads Phase A's values; it never
    recomputes them, and it never revisits the decision. A run whose Phase A reported
    ``NOT_SELECTED`` cannot deliver here, because the attempt is built from the same frozen
    decision rather than from a fresh one.

    **The order of operations is the whole point.** Deliver, then measure, then evaluate. Each
    step reads only what the previous one produced, so an outcome can never be attributed to
    an intervention that was not sent, and an evaluation can never score a prediction against
    evidence it was not blind to.

    Args:
        phase: The frozen Phase A result.
        clock: The clock for the outcome and evaluation engines. Defaults to the system
            clock; tests pass a :class:`~focus_engine.utils.clock.FixedClock` to make the
            run byte-identical. The delivery itself is anchored to the decision instant and
            takes no clock, so a late wall clock cannot make the lifecycle events drift
            away from the evidence they responded to.

    Returns:
        A :class:`LabBundle` whose stage log now includes the intervention, outcome, and
        evaluation stages, each reporting its real status.
    """
    log: list[dict[str, Any]] = list(phase.log)
    active_clock: Clock = SystemClock() if clock is None else clock
    model_summary = phase.model_summary

    # --- NO ANALYSIS AT ALL --------------------------------------------------------
    if phase.decision is None or phase.final_action is None:
        log.append(
            _stage(
                "intervention",
                "INTERVENTION",
                {
                    "dev_only": True,
                    "delivered": False,
                    "status": DeliveryStatus.NOT_SELECTED.value,
                    "status_label": "NOT SELECTED",
                    "gate": "policy",
                    "intervention_id": None,
                    "started_at": None,
                    "delivered_at": None,
                    "final_action": None,
                    "requires_human_approval": False,
                    "reason": _NO_ANALYSIS_REASON,
                    "lifecycle_event_count": 0,
                    "lifecycle_events": [],
                },
            )
        )
        log.append(_outcome_stage(None, _NOT_DELIVERED_REASON))
        log.append(_evaluation_stage(None, _NO_SNAPSHOT_REASON))
        return LabBundle(
            response=_envelope([], log, phase.question_count),
            model_summary=model_summary,
            question_count=phase.question_count,
        )

    # --- DELIVERY -----------------------------------------------------------------
    # The attempt is built from the frozen decision and the frozen final action. Both were
    # computed in Phase A, and neither is consulted from anywhere else.
    assert phase.uncertainty is not None  # noqa: S101 - the guard above covers every None
    assert phase.state is not None  # noqa: S101 - a decision can only exist after a state
    attempt = plan_delivery(
        decision=phase.decision,
        final_action=phase.final_action,
        learner_id=phase.learner_id,
        session_id=phase.session_id,
        decided_at=phase.cut_at,
        trigger_probability=(
            0.0 if phase.uncertainty.probability is None else phase.uncertainty.probability
        ),
        trigger_state=phase.state.state.value,
    )
    log.append(_intervention_stage(attempt))

    intervention = InterventionStatus(
        selected_by_policy=phase.decision.selected is not None,
        intervention_type=(
            None if phase.decision.selected is None else phase.decision.selected.intervention_type
        ),
        delivery_status=attempt.status,
        gate=attempt.gate,
        final_action=attempt.final_action.outcome.value,
        requires_human_approval=attempt.requires_human_approval,
        intervention_id=attempt.intervention_id,
        started_at=attempt.started_at,
        delivered_at=attempt.delivered_at,
        lifecycle_events=list(attempt.events),
        explanation=attempt.reason,
    )

    # --- OUTCOME + EVALUATION -----------------------------------------------------
    outcome_record: OutcomeRecord | None = None
    evaluation_record: EvaluationRecord | None = None
    outcome_reason = _NOT_DELIVERED_REASON
    evaluation_reason = _NO_SNAPSHOT_REASON

    if attempt.delivered and attempt.intervention_id is not None:
        # The stream the outcome engine sees includes this delivery's own events, because
        # measure() locates the delivery inside the history rather than taking it as an
        # argument. A history that predates the delivery would report "no delivery" for an
        # intervention that was in fact sent.
        stream = sorted((*phase.events, *attempt.events), key=lambda item: item.timestamp)
        engine = OutcomeEngine(clock=active_clock, outcome_version=OUTCOME_V1)
        outcome_record = engine.measure(
            history_for(phase.learner_id, phase.events, attempt),
            attempt.intervention_id,
            stream,
            state=phase.state,
        )
        outcome_reason = _NO_EVIDENCE_REASON

        # Evaluation is scored against the prediction snapshot Phase A produced, with a
        # reading rebuilt from the window rather than reused from the decision. The horizon
        # starts at the cut, so the window is blind to everything the prediction saw.
        horizon = PredictionHorizon(
            predicted_at=phase.cut_at,
            duration_seconds=_EVALUATION_HORIZON_MINUTES * 60.0,
        )
        in_window = [
            event
            for event in stream
            if event.learner_id == phase.learner_id
            and event.session_id == phase.session_id
            and horizon.contains(event.timestamp)
        ]
        assert phase.profile is not None  # noqa: S101 - a delivery implies an analysis
        reading = post_window_reading(
            learner_id=phase.learner_id,
            in_window=in_window,
            window_start=phase.cut_at,
            profile=phase.profile,
            baseline=BaselineEngine(_BASELINE_SETTINGS),
            dimensions=BASELINE_DIMENSIONS,
            dimension_value=_dimension_value,
            engine=TemporalEngine(clock=FixedClock(phase.cut_at)),
            feature_set_version=FEATURE_SET_V1,
        )
        if reading is None:
            evaluation_reason = _NO_READING_REASON
        else:
            evaluator = EvaluationEngine(clock=active_clock, evaluation_version=EVALUATION_V1)
            snapshot = snapshot_for(
                prediction_id=f"prediction-{phase.learner_id}-{phase.session_id}",
                learner_id=phase.learner_id,
                session_id=phase.session_id,
                outcome=phase.uncertainty,
                predicted_at=phase.cut_at,
                horizon_seconds=_EVALUATION_HORIZON_MINUTES * 60.0,
            )
            evaluation_record = evaluator.evaluate(
                snapshot,
                stream,
                reading,
                outcome=outcome_record,
            )
            evaluation_reason = _NO_SNAPSHOT_REASON

    log.append(_outcome_stage(outcome_record, outcome_reason))
    log.append(_evaluation_stage(evaluation_record, evaluation_reason))

    result = AnalysisResult(
        learner_id=phase.learner_id,
        session_id=phase.session_id,
        decided_at=phase.cut_at,
        temporal_state=phase.state,
        prediction_outcome=phase.uncertainty,
        policy_decision=phase.decision,
        outcome_record=outcome_record,
        evaluation_record=evaluation_record,
        intervention=intervention,
    )
    return LabBundle(
        response=_envelope([result], log, phase.question_count),
        model_summary=model_summary,
        question_count=phase.question_count,
    )


def run_analysis(
    events: Sequence[EventEnvelope],
    *,
    learner_id: str,
    session_id: str,
    clock: Clock | None = None,
    decision_cut: datetime | None = None,
) -> LabBundle:
    """Run both phases and return the finished bundle. A convenience over the explicit pair.

    Provided for callers that do not care about the seam. The two functions remain
    separately importable so that anything which *does* care — a test asserting Phase B does
    not recompute, a caller that wants to inspect the decision before acting on it — can use
    them directly.

    Args:
        events: The learner's events.
        learner_id: The pseudonymous learner identifier.
        session_id: The pseudonymous session identifier.
        clock: Injected into every layer that takes one.
        decision_cut: An explicit decision instant, or ``None`` to derive one.

    Returns:
        A :class:`LabBundle` with the contract response, the model provenance summary, and
        the number of answered questions.
    """
    active_clock: Clock = SystemClock() if clock is None else clock
    phase = prepare_analysis(
        events,
        learner_id=learner_id,
        session_id=session_id,
        clock=active_clock,
        decision_cut=decision_cut,
    )
    return execute_decision(phase, clock=active_clock)


def _envelope(
    results: list[AnalysisResult], log: list[dict[str, Any]], question_count: int
) -> AnalysisResponse:
    return AnalysisResponse(
        synthetic=True,
        notice=_SYNTHETIC_NOTICE,
        results=results,
        stage_log=log,
        diagnosis=diagnose(log, question_count=question_count),
    )


class _EngineClocks:
    """Builds every clock-taking engine with the caller's clock.

    Injecting one clock everywhere is what makes a run reproducible. A layer that
    defaulted to the system clock would stamp its own ``computed_at``, and the same
    events would then produce two different responses.
    """

    __slots__ = ("_clock",)

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def context(self) -> ContextEngine:
        return ContextEngine(clock=self._clock)

    def temporal(self) -> TemporalEngine:
        return TemporalEngine(clock=self._clock)

    def uncertainty(self) -> UncertaintyEngine:
        return UncertaintyEngine(clock=self._clock, baseline=_BASELINE_SETTINGS)

    def policy(self) -> InterventionPolicy:
        return InterventionPolicy(clock=self._clock)

    def authority(self) -> AuthorityEngine:
        return AuthorityEngine(clock=self._clock, data_origin=DataOrigin.SYNTHETIC)
