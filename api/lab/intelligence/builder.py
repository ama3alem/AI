"""Builds canonical learner profiles by running the real pipeline, never by hand.

The load-bearing decision of this module is *how a profile is produced*: it is the finished
:func:`api.lab.adapter.run_analysis` execution over the real simulator, frozen, then read.
No value in a profile is typed in here. The evidence statements quote numbers that came out
of :class:`~focus_engine.outcomes.models.OutcomeRecord` and the other typed results, and an
assertion in tests that re-runs the pipeline and compares would fail the moment a statement
drifted from its source.

**One store, built once, cached.** The five scenarios are the only learners this
repository has, and they are fully deterministic under the fixed clock. The store is built
lazily on first access and cached for the process, so the layer is cheap to query and
byte-identical across replays.

**A profile is not a snapshot of code; it is a reading of evidence.** The engine versions
that produced a profile travel with it (``engine_versions``) and the model provenance
travels beside it (``model_summary``), so a reader can always tell *which* ruleset produced
the statements they are looking at.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from api.lab.adapter import LabBundle, run_analysis
from api.lab.contract import SyntheticDataNotice
from api.lab.intelligence import demo as demo_module
from api.lab.intelligence import reason
from api.lab.intelligence.models import (
    GAP_PRIORITY,
    EvidenceClass,
    EvidenceKind,
    EvidenceRecord,
    GapRecord,
    GapType,
    ReliabilityGrade,
    StudentIntelligenceProfile,
)
from api.lab.scenarios import SCENARIO_DEFS, flatten
from focus_engine.evaluation.models import EvaluationRecord
from focus_engine.events.types import EventEnvelope
from focus_engine.outcomes.models import OUTCOME_MEASURES, OutcomeRecord
from focus_engine.schemas.primitives import DataOrigin, Provenance
from focus_engine.uncertainty.outcomes import PredictionOutcome, Verdict
from focus_engine.utils.clock import FixedClock

__all__ = [
    "build_demo_store",
    "build_store",
    "demo_learner_ids",
    "get_demo_profile",
    "get_demo_store",
    "get_profile",
    "get_store",
    "known_learner_ids",
]

#: The same deterministic instant the lab server uses, so an intelligence answer and the
#: pipeline view it describes are produced under the same clock and replay byte-identically.
_DETERMINISTIC_CLOCK: Final[FixedClock] = FixedClock(datetime(2026, 6, 15, tzinfo=UTC))

_SYNTHETIC_NOTICE: Final[SyntheticDataNotice] = SyntheticDataNotice(
    data_origin=DataOrigin.SYNTHETIC.value,
    provenance=Provenance.OBSERVED.value,
    is_synthetic=True,
    warning="SYNTHETIC DATA — NOT REAL STUDENT DATA",
)

_store: dict[str, StudentIntelligenceProfile] | None = None


# ---------------------------------------------------------------------------
# Evidence record builders
# ---------------------------------------------------------------------------


def _events_records(learner_id: str, detail: dict[str, Any]) -> list[EvidenceRecord]:
    first_session = detail.get("sessions")[0] if detail.get("sessions") else None
    return [
        EvidenceRecord(
            evidence_id=f"ev-{learner_id}-events",
            learner_id=learner_id,
            kind=EvidenceKind.EVENT,
            evidence_class=EvidenceClass.OBSERVED,
            reliability=ReliabilityGrade.HIGH,
            statement=(
                f"{detail['answered_questions']} answered questions across "
                f"{detail['session_count']} sessions ({detail['total_events']} events) exist "
                "for this learner."
            ),
            provenance=Provenance.OBSERVED.value,
            data_origin=DataOrigin.SYNTHETIC.value,
            source="api.lab.adapter.prepare_analysis[events].detail",
            version="FOCUS_CONTRACT_V1",
            timestamp=(datetime.fromisoformat(first_session["started"]) if first_session else None),
        ),
        EvidenceRecord(
            evidence_id=f"ev-{learner_id}-origin",
            learner_id=learner_id,
            kind=EvidenceKind.PROVENANCE,
            evidence_class=EvidenceClass.SYSTEM_GENERATED,
            reliability=ReliabilityGrade.HIGH,
            statement=(
                "This learner's record is synthetic: every event was produced by "
                "focus_engine.simulator.generate_session and is stamped "
                f"data_origin={DataOrigin.SYNTHETIC.value}. It is not a real student."
            ),
            provenance=Provenance.SYNTHETIC_LABEL.value,
            data_origin=DataOrigin.SYNTHETIC.value,
            source="api.lab.adapter._SYNTHETIC_NOTICE",
            version="FOCUS_CONTRACT_V1",
        ),
    ]


def _context_record(learner_id: str, detail: dict[str, Any]) -> EvidenceRecord | None:
    accuracy = detail.get("recent_accuracy")
    response = detail.get("average_response_seconds")
    parts = [f"Decision session {detail.get('session_id')!r}"]
    if accuracy is not None:
        parts.append(f"recent accuracy {accuracy:.4f}")
    if response is not None:
        parts.append(f"average response {response:.2f}s")
    parts.append(f"elapsed {detail.get('elapsed_seconds', 0.0):.1f}s")
    note = (
        "Context is derived from a single session only, never concatenated across sessions."
        if accuracy is not None
        else "Recent accuracy absent from context for this learner."
    )
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-context",
        learner_id=learner_id,
        kind=EvidenceKind.CONTEXT,
        evidence_class=EvidenceClass.DERIVED,
        reliability=ReliabilityGrade.MEDIUM,
        statement="Per-session context: " + "; ".join(parts) + ".",
        provenance=Provenance.OBSERVED.value,
        data_origin=DataOrigin.SYNTHETIC.value,
        source="api.lab.adapter.prepare_analysis[context].detail",
        note=note,
    )


def _features_record(learner_id: str, detail: dict[str, Any]) -> EvidenceRecord:
    values = detail.get("values") or []
    in_model = sum(1 for value in values if value.get("in_model"))
    unavailable = sum(1 for value in values if value.get("availability") != "available")
    statements = []
    statements.append(f"Feature set {detail.get('feature_set')} computed")
    statements.append(f"{len(values)} features of which {in_model} enter the lab model")
    if unavailable:
        statements.append(f"{unavailable} unavailable at decision time")
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-features",
        learner_id=learner_id,
        kind=EvidenceKind.FEATURE,
        evidence_class=EvidenceClass.DERIVED,
        reliability=(ReliabilityGrade.HIGH if unavailable == 0 else ReliabilityGrade.MEDIUM),
        statement="; ".join(statements) + ".",
        provenance=Provenance.OBSERVED.value,
        data_origin=DataOrigin.SYNTHETIC.value,
        source="api.lab.adapter.prepare_analysis[features].detail",
        version=str(detail.get("feature_set")) or None,
    )


def _baseline_record(learner_id: str, detail: dict[str, Any]) -> EvidenceRecord:
    dimensions = detail.get("dimensions") or {}
    established = sum(1 for value in dimensions.values() if value.get("maturity") == "established")
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-baseline",
        learner_id=learner_id,
        kind=EvidenceKind.BASELINE,
        evidence_class=EvidenceClass.DERIVED,
        reliability=ReliabilityGrade.HIGH,
        statement=(
            f"Personal baseline maturity {detail.get('maturity')} from "
            f"{detail.get('observations')} observations across "
            f"{detail.get('sessions_folded_in')} sessions; {established} of "
            f"{len(dimensions)} dimensions established."
        ),
        provenance=Provenance.OBSERVED.value,
        data_origin=DataOrigin.SYNTHETIC.value,
        source="api.lab.adapter.prepare_analysis[baseline].detail",
    )


def _temporal_record(learner_id: str, detail: dict[str, Any]) -> EvidenceRecord:
    mean = detail.get("mean_standardised")
    mean_text = "n/a" if mean is None else f"{mean:.4f}"
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-temporal",
        learner_id=learner_id,
        kind=EvidenceKind.TEMPORAL,
        evidence_class=EvidenceClass.DERIVED,
        reliability=(
            ReliabilityGrade.HIGH
            if detail.get("evidence_maturity") == "mature"
            else ReliabilityGrade.MEDIUM
        ),
        statement=(
            f"Behavioural state {detail.get('state')} ({detail.get('direction')}, "
            f"character {detail.get('character')}), run length {detail.get('run_length')}, "
            f"mean standardised deviation {mean_text} against the learner's own baseline."
        ),
        provenance=Provenance.OBSERVED.value,
        data_origin=DataOrigin.SYNTHETIC.value,
        source="api.lab.adapter.prepare_analysis[temporal].detail",
        version=str(detail.get("temporal_version")) or None,
    )


def _prediction_record(learner_id: str, outcome: PredictionOutcome | None) -> EvidenceRecord | None:
    if outcome is None:
        return None
    if outcome.verdict is Verdict.RESOLVED:
        return EvidenceRecord(
            evidence_id=f"ev-{learner_id}-prediction",
            learner_id=learner_id,
            kind=EvidenceKind.PREDICTION,
            evidence_class=EvidenceClass.PREDICTED,
            reliability=ReliabilityGrade.LOW,
            statement=(
                f"Model predicted probability {outcome.probability:.4f} against threshold "
                f"{outcome.threshold:.2f} -> {'at risk' if outcome.is_positive else 'stable'} "
                f"(model {outcome.model_id} v{outcome.model_version})."
            ),
            provenance=Provenance.MODEL_PREDICTION.value,
            data_origin=outcome.data_origin.value,
            source="focus_engine.uncertainty.outcomes.PredictionOutcome (RESOLVED)",
            version=str(outcome.model_version),
            timestamp=outcome.computed_at,
            note="A model output, not a measurement.",
        )
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-prediction",
        learner_id=learner_id,
        kind=EvidenceKind.PREDICTION,
        evidence_class=EvidenceClass.PREDICTED,
        reliability=ReliabilityGrade.HIGH,
        statement=(
            f"Prediction refused: {outcome.refusal_reason or 'no reason stated'}"
            + (
                f" Missing features: {', '.join(outcome.missing_features)}."
                if outcome.missing_features
                else ""
            )
        ),
        provenance=Provenance.MODEL_PREDICTION.value,
        data_origin=outcome.data_origin.value,
        source="focus_engine.uncertainty.outcomes.PredictionOutcome (refusal)",
        version=str(outcome.model_version),
        timestamp=outcome.computed_at,
        note="A refusal is the model declining to score; it is not a zero.",
    )


def _uncertainty_record(
    learner_id: str, outcome: PredictionOutcome | None
) -> EvidenceRecord | None:
    if outcome is None:
        return None
    binding = None
    if outcome.assessment is not None:
        binding = outcome.assessment.binding.value
    parts = [f"Uncertainty verdict {outcome.verdict.value}"]
    if outcome.confidence is not None:
        parts.append(f"confidence {outcome.confidence.value}")
    if outcome.maturity is not None:
        parts.append(f"baseline maturity {outcome.maturity.value}")
    parts.append(f"{outcome.evidence_units} evidence units")
    parts.append(f"remedy {outcome.remedy.value}")
    if binding is not None:
        parts.append(f"binding constraint {binding}")
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-uncertainty",
        learner_id=learner_id,
        kind=EvidenceKind.UNCERTAINTY,
        evidence_class=EvidenceClass.PREDICTED,
        reliability=ReliabilityGrade.MEDIUM,
        statement="; ".join(parts) + ".",
        provenance=(
            Provenance.MODEL_PREDICTION.value
            if outcome.verdict is Verdict.RESOLVED
            else Provenance.OBSERVED.value
        ),
        data_origin=outcome.data_origin.value,
        source="focus_engine.uncertainty.outcomes.PredictionOutcome",
        version=str(outcome.model_version),
        timestamp=outcome.computed_at,
    )


def _policy_record(learner_id: str, detail: dict[str, Any]) -> EvidenceRecord:
    selected = detail.get("intervention_type")
    selected_text = selected or "no intervention selected"
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-policy",
        learner_id=learner_id,
        kind=EvidenceKind.POLICY,
        evidence_class=EvidenceClass.DERIVED,
        reliability=ReliabilityGrade.HIGH,
        statement=(
            f"Policy decision {detail.get('decision')}: {selected_text}."
            + (f" Reason: {detail.get('reason')}." if detail.get("reason") else "")
        ),
        provenance=Provenance.OBSERVED.value,
        data_origin=DataOrigin.SYNTHETIC.value,
        source="api.lab.adapter.prepare_analysis[policy].detail",
        version=str(detail.get("policy_version")) or None,
    )


def _authority_record(learner_id: str, detail: dict[str, Any]) -> EvidenceRecord:
    meets = "meets" if detail.get("meets_autonomy_threshold") else "does not meet"
    restricted = detail.get("restricted_actions") or []
    restricted_text = ", ".join(restricted) if restricted else "none"
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-authority",
        learner_id=learner_id,
        kind=EvidenceKind.AUTHORITY,
        evidence_class=EvidenceClass.SYSTEM_GENERATED,
        reliability=ReliabilityGrade.HIGH,
        statement=(
            f"Authority envelope for context {detail.get('context_type')}: level "
            f"{detail.get('authority_level')}, confidence weight "
            f"{detail.get('confidence_weight', 'n/a')} against autonomy minimum "
            f"{detail.get('autonomy_min_weight', 'n/a')}; {meets} the autonomy threshold. "
            f"Restricted actions: {restricted_text}. Human availability "
            f"{detail.get('human_availability')}."
        ),
        provenance=Provenance.OBSERVED.value,
        data_origin=DataOrigin.SYNTHETIC.value,
        source="api.lab.adapter.prepare_analysis[authority].detail",
        timestamp=(
            datetime.fromisoformat(detail["calculated_at"]) if detail.get("calculated_at") else None
        ),
    )


def _intervention_record(learner_id: str, intervention: Any) -> EvidenceRecord:
    statement = (
        f"Delivery status {intervention.delivery_status.value}"
        f" (selected by policy: {intervention.selected_by_policy})"
    )
    if intervention.gate is not None:
        statement += f"; stopped at gate {intervention.gate}"
    statement += f"; requires human approval: {intervention.requires_human_approval}"
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-intervention",
        learner_id=learner_id,
        kind=EvidenceKind.INTERVENTION,
        evidence_class=EvidenceClass.SYSTEM_GENERATED,
        reliability=ReliabilityGrade.HIGH,
        statement=statement + ".",
        provenance=Provenance.OBSERVED.value,
        data_origin=DataOrigin.SYNTHETIC.value,
        source="focus_engine.outcomes.models.OutcomeRecord / api.lab.execution",
        timestamp=intervention.delivered_at,
    )


def _outcome_record(
    learner_id: str, outcome: OutcomeRecord | None, detail: dict[str, Any]
) -> EvidenceRecord | None:
    if outcome is None:
        return EvidenceRecord(
            evidence_id=f"ev-{learner_id}-outcome",
            learner_id=learner_id,
            kind=EvidenceKind.OUTCOME,
            evidence_class=EvidenceClass.UNAVAILABLE,
            reliability=ReliabilityGrade.MEDIUM,
            statement=(
                "No intervention outcome was measured: "
                + (str(detail.get("status")) or "no delivery to measure")
                + "."
            ),
            provenance=Provenance.OBSERVED.value,
            data_origin=DataOrigin.SYNTHETIC.value,
            source="api.lab.adapter._outcome_stage",
            note="An absent outcome reading is reported as an absence, never as a learner "
            "who did not move.",
        )
    measured = [item for item in outcome.measurements if str(item.status.value) == "measured"]
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-outcome",
        learner_id=learner_id,
        kind=EvidenceKind.OUTCOME,
        evidence_class=EvidenceClass.OBSERVED,
        reliability=(
            ReliabilityGrade.HIGH
            if len(measured) == len(OUTCOME_MEASURES)
            else ReliabilityGrade.MEDIUM
        ),
        statement=(
            f"Intervention outcome measured {len(measured)} of {len(outcome.measurements)} "
            f"measures; response class {outcome.response_class.value}; complete "
            f"{outcome.is_complete}."
        ),
        provenance=Provenance.OBSERVED.value,
        data_origin=outcome.data_origin.value,
        source="focus_engine.outcomes.models.OutcomeRecord",
        version=str(outcome.outcome_version),
        timestamp=outcome.computed_at,
        note="These readings describe what the events show after the prompt. They do not "
        "establish that the prompt caused them.",
    )


def _evaluation_record(
    learner_id: str, evaluation: EvaluationRecord | None, detail: dict[str, Any]
) -> EvidenceRecord | None:
    if evaluation is None:
        return EvidenceRecord(
            evidence_id=f"ev-{learner_id}-evaluation",
            learner_id=learner_id,
            kind=EvidenceKind.EVALUATION,
            evidence_class=EvidenceClass.UNAVAILABLE,
            reliability=ReliabilityGrade.MEDIUM,
            statement=(
                "No prediction evaluation exists: "
                + (str(detail.get("status")) or "no scored prediction")
                + "."
            ),
            provenance=Provenance.OBSERVED.value,
            data_origin=DataOrigin.SYNTHETIC.value,
            source="api.lab.adapter._evaluation_stage",
        )
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-evaluation",
        learner_id=learner_id,
        kind=EvidenceKind.EVALUATION,
        evidence_class=EvidenceClass.DERIVED,
        reliability=ReliabilityGrade.HIGH,
        statement=(
            f"Prediction evaluated {evaluation.verdict.value} "
            f"[{evaluation.binary_verdict.value}] from {evaluation.evidence_count} events "
            f"(evaluator {evaluation.evaluator_version})."
        ),
        provenance=Provenance.OBSERVED.value,
        data_origin=evaluation.data_origin.value,
        source="focus_engine.evaluation.models.EvaluationRecord",
        version=str(evaluation.evaluation_version),
        timestamp=evaluation.computed_at,
    )


# ---------------------------------------------------------------------------
# Gap producers
# ---------------------------------------------------------------------------


def _data_gaps(
    learner_id: str,
    outcome: PredictionOutcome | None,
    detail: dict[str, Any],
    *,
    subject_metadata: bool = False,
) -> tuple[GapRecord, ...]:
    gaps: list[GapRecord] = []

    prediction = detail.get("prediction") or {}
    if outcome is not None and outcome.verdict is not Verdict.RESOLVED:
        gaps.append(
            GapRecord(
                gap_id=f"gap-{learner_id}-low-coverage",
                learner_id=learner_id,
                gap_type=GapType.LOW_COVERAGE,
                description=(
                    f"The model declined to predict for this learner ({outcome.refusal_reason})."
                    + (
                        f" Missing features: {', '.join(outcome.missing_features)}."
                        if outcome.missing_features
                        else ""
                    )
                ),
                priority=GAP_PRIORITY[GapType.LOW_COVERAGE],
                remedy=outcome.remedy.value,
            )
        )
    elif prediction.get("outcome") == "PredictionRefusal":
        gaps.append(
            GapRecord(
                gap_id=f"gap-{learner_id}-low-coverage",
                learner_id=learner_id,
                gap_type=GapType.LOW_COVERAGE,
                description="Prediction refused: " + str(prediction.get("reason")),
                priority=GAP_PRIORITY[GapType.LOW_COVERAGE],
                remedy="The engine's own remedy for the refusal applies.",
            )
        )

    baseline = detail.get("baseline") or {}
    maturity = baseline.get("maturity")
    if maturity in ("new", "early", "developing"):
        gaps.append(
            GapRecord(
                gap_id=f"gap-{learner_id}-baseline-unvalidated",
                learner_id=learner_id,
                gap_type=GapType.UNVALIDATED,
                description=(
                    f"Personal baseline is only {maturity} ({baseline.get('observations')} "
                    "observations); claims about this learner's recent self are weakly "
                    "anchored."
                ),
                priority=GAP_PRIORITY[GapType.UNVALIDATED],
                remedy="More sessions folded in would mature the baseline.",
            )
        )

    outcome_stage = detail.get("outcome") or {}
    if not outcome_stage.get("measured"):
        gaps.append(
            GapRecord(
                gap_id=f"gap-{learner_id}-outcome-missing",
                learner_id=learner_id,
                gap_type=GapType.MISSING,
                description="No intervention outcome was measured; nothing after a prompt "
                "was read, so no before/after measurement exists.",
                priority=GAP_PRIORITY[GapType.MISSING],
                remedy="A delivered intervention with enough post-delivery evidence.",
            )
        )

    evaluation_stage = detail.get("evaluation") or {}
    if not evaluation_stage.get("assessed"):
        gaps.append(
            GapRecord(
                gap_id=f"gap-{learner_id}-evaluation-missing",
                learner_id=learner_id,
                gap_type=GapType.MISSING,
                description="No prediction evaluation exists; the model's claim was never "
                "scored against post-prediction evidence.",
                priority=GAP_PRIORITY[GapType.MISSING],
                remedy="Admissible post-prediction evidence in the evaluation window.",
            )
        )

    if not subject_metadata:
        gaps.append(
            GapRecord(
                gap_id=f"gap-{learner_id}-subject-na",
                learner_id=learner_id,
                gap_type=GapType.NOT_APPLICABLE,
                description=(
                    "No subject-scoped mastery evidence exists: question events in the "
                    "synthetic corpus carry no topic metadata, so nothing can be said "
                    "per-subject."
                ),
                priority=GAP_PRIORITY[GapType.NOT_APPLICABLE],
            )
        )
    gaps.append(
        GapRecord(
            gap_id=f"gap-{learner_id}-eduflow-na",
            learner_id=learner_id,
            gap_type=GapType.NOT_APPLICABLE,
            description=(
                "EduFlow-native attributes (school, grade, class, cohort, teacher notes, "
                "official assessments, attendance) are UNAVAILABLE by construction. Only "
                "pseudonymous behavioural events exist in this repository."
            ),
            priority=GAP_PRIORITY[GapType.NOT_APPLICABLE],
        )
    )
    return tuple(sorted(gaps, key=lambda gap: gap.priority))


def _engine_versions(result: Any, detail: dict[str, Any]) -> dict[str, str]:
    versions: dict[str, str] = {"contract": "FOCUS_CONTRACT_V1"}
    outcome: PredictionOutcome | None = result.prediction_outcome
    if outcome is not None:
        versions["model"] = str(outcome.model_version)
        versions["feature_set"] = str(outcome.feature_set_version)
        versions["target_definition"] = str(outcome.target_definition_version)
    if result.outcome_record is not None:
        versions["outcome"] = str(result.outcome_record.outcome_version)
    if result.evaluation_record is not None:
        versions["evaluation"] = str(result.evaluation_record.evaluation_version)
    policy = detail.get("policy") or {}
    if policy.get("policy_version"):
        versions["policy"] = str(policy["policy_version"])
    return versions


def _assemble(
    learner_id: str, bundle: LabBundle
) -> tuple[list[EvidenceRecord], tuple[GapRecord, ...], dict[str, str]]:
    """Build the core records, gaps, and versions from a finished pipeline bundle."""
    result = bundle.response.results[0]
    detail = {stage["stage"]: stage["detail"] for stage in bundle.response.stage_log}

    records: list[EvidenceRecord] = []
    records.extend(_events_records(learner_id, detail.get("events") or {}))
    context = _context_record(learner_id, detail.get("context") or {})
    if context is not None:
        records.append(context)
    records.append(_features_record(learner_id, detail.get("features") or {}))
    records.append(_baseline_record(learner_id, detail.get("baseline") or {}))
    records.append(_temporal_record(learner_id, detail.get("temporal") or {}))
    prediction = _prediction_record(learner_id, result.prediction_outcome)
    if prediction is not None:
        records.append(prediction)
    uncertainty = _uncertainty_record(learner_id, result.prediction_outcome)
    if uncertainty is not None:
        records.append(uncertainty)
    records.append(_policy_record(learner_id, detail.get("policy") or {}))
    records.append(_authority_record(learner_id, detail.get("authority") or {}))
    records.append(_intervention_record(learner_id, result.intervention))
    outcome_rec = _outcome_record(learner_id, result.outcome_record, detail.get("outcome") or {})
    if outcome_rec is not None:
        records.append(outcome_rec)
    evaluation_rec = _evaluation_record(
        learner_id, result.evaluation_record, detail.get("evaluation") or {}
    )
    if evaluation_rec is not None:
        records.append(evaluation_rec)

    gaps = _data_gaps(learner_id, result.prediction_outcome, detail)
    versions = _engine_versions(result, detail)
    return records, gaps, versions


def _build_profile(learner_id: str) -> StudentIntelligenceProfile:
    """Run the real pipeline for one scenario learner and assemble the profile.

    Args:
        learner_id: The scenario's learner id.

    Returns:
        The canonical :class:`StudentIntelligenceProfile`.

    Raises:
        ValueError: If ``learner_id`` is not one of the known scenario learners.
    """
    spec = next((s for s in SCENARIO_DEFS if s.learner_id == learner_id), None)
    if spec is None:
        raise ValueError(f"Unknown learner {learner_id!r}")
    sessions = spec.build()
    events = flatten(sessions)
    session_id = sessions[-1][0].session_id
    bundle: LabBundle = run_analysis(
        events,
        learner_id=spec.learner_id,
        session_id=session_id,
        clock=_DETERMINISTIC_CLOCK,
    )
    records, gaps, versions = _assemble(spec.learner_id, bundle)

    return StudentIntelligenceProfile(
        learner_id=learner_id,
        scenario_id=spec.id,
        title=spec.title,
        archetype=spec.archetype.value,
        evidence=tuple(records),
        gaps=gaps,
        engine_versions=versions,
        model_summary=bundle.model_summary,
        notice=_SYNTHETIC_NOTICE,
        built_at=_DETERMINISTIC_CLOCK.now(),
    )


def build_store() -> dict[str, StudentIntelligenceProfile]:
    """Build the canonical store for every scenario learner.

    Returns:
        A fresh mapping from learner id to profile. Deterministic under the fixed clock.
    """
    return {spec.learner_id: _build_profile(spec.learner_id) for spec in SCENARIO_DEFS}


def get_store() -> dict[str, StudentIntelligenceProfile]:
    """Return the cached store, building it on first use.

    Returns:
        The process-wide store.
    """
    global _store  # noqa: PLW0603
    if _store is None:
        _store = build_store()
    return _store


def get_profile(learner_id: str) -> StudentIntelligenceProfile:
    """Return one learner's canonical profile.

    Args:
        learner_id: The pseudonymous learner id.

    Returns:
        The profile.

    Raises:
        ValueError: If the learner is not known. The message names the learner but never
            the known set, so an id probe cannot learn what ids exist.
    """
    profile = get_store().get(learner_id)
    if profile is None:
        raise ValueError(f"Unknown learner {learner_id!r}")
    return profile


def known_learner_ids() -> tuple[str, ...]:
    """The known learner ids, for listing.

    Returns:
        The learner ids of every scenario, in scenario order.
    """
    return tuple(spec.learner_id for spec in SCENARIO_DEFS)


# ---------------------------------------------------------------------------
# Phase 15 demo store
# ---------------------------------------------------------------------------


def _last_session_id(events: Sequence[EventEnvelope]) -> str:
    """The session id of the session containing the final event of a stream."""
    buckets: dict[str, list[EventEnvelope]] = {}
    for event in events:
        buckets.setdefault(event.session_id, []).append(event)
    ordered = sorted(
        ((session_id, tuple(group)) for session_id, group in buckets.items()),
        key=lambda item: item[1][0].timestamp,
    )
    return ordered[-1][0]


_demo_store: dict[str, StudentIntelligenceProfile] | None = None


def _build_demo_profile(learner_id: str) -> StudentIntelligenceProfile:
    """Assemble one Phase 15 demo profile: pipeline reading plus deep-evidence reasoning.

    Args:
        learner_id: The pseudonymous demo learner id.

    Returns:
        The demo :class:`StudentIntelligenceProfile`. It extends the same core records a
        canonical learner has with subject roll-ups, a change-state reading, and a fusion
        record, plus the gaps that reasoning exposes.

    Raises:
        ValueError: If the learner is not a demo learner.
    """
    spec = demo_module.get_demo_scenario(learner_id)
    events = demo_module.get_demo_events(learner_id)
    session_id = _last_session_id(events)
    bundle: LabBundle = run_analysis(
        events,
        learner_id=learner_id,
        session_id=session_id,
        clock=_DETERMINISTIC_CLOCK,
    )
    result = bundle.response.results[0]
    detail = {stage["stage"]: stage["detail"] for stage in bundle.response.stage_log}

    records, _, _ = _assemble(learner_id, bundle)
    rollups = reason.subject_rollups(events, now=_DETERMINISTIC_CLOCK.now())
    records.extend(reason.build_subject_records(learner_id, rollups))
    records.append(reason.build_change_record(events, learner_id))
    fusion = reason.build_fusion_record(learner_id, rollups)
    records.append(fusion)

    gaps: list[GapRecord] = list(
        _data_gaps(learner_id, result.prediction_outcome, detail, subject_metadata=True)
    )
    gaps.extend(reason.build_subject_gaps(learner_id, rollups))
    conflict = reason.build_conflict_gap(learner_id, fusion)
    if conflict is not None:
        gaps.append(conflict)

    versions = _engine_versions(result, detail)
    versions["reasoning"] = "FOCUS_REASON_V1"

    return StudentIntelligenceProfile(
        learner_id=spec.learner_id,
        scenario_id=spec.id,
        title=spec.title,
        archetype=spec.archetype,
        evidence=tuple(records),
        gaps=tuple(sorted(gaps, key=lambda gap: (gap.priority, gap.gap_id))),
        engine_versions=versions,
        model_summary=bundle.model_summary,
        notice=_SYNTHETIC_NOTICE,
        built_at=_DETERMINISTIC_CLOCK.now(),
    )


def build_demo_store() -> dict[str, StudentIntelligenceProfile]:
    """Build the Phase 15 demo store for every demonstration learner.

    Returns:
        A fresh mapping from learner id to profile. Deterministic under the fixed clock.
    """
    return {learner_id: _build_demo_profile(learner_id) for learner_id in demo_learner_ids()}


def get_demo_store() -> dict[str, StudentIntelligenceProfile]:
    """Return the cached demo store, building it on first use.

    Returns:
        The process-wide demo store.
    """
    global _demo_store  # noqa: PLW0603
    if _demo_store is None:
        _demo_store = build_demo_store()
    return _demo_store


def get_demo_profile(learner_id: str) -> StudentIntelligenceProfile:
    """Return one Phase 15 demo learner's profile.

    Args:
        learner_id: The pseudonymous demo learner id.

    Returns:
        The profile.

    Raises:
        ValueError: If the learner is not a demo learner. The message names the learner
            but never the known set, so an id probe cannot learn what ids exist.
    """
    profile = get_demo_store().get(learner_id)
    if profile is None:
        raise ValueError(f"Unknown learner {learner_id!r}")
    return profile


def demo_learner_ids() -> tuple[str, ...]:
    """The Phase 15 demo learner ids, in scenario order.

    Returns:
        The demo learner ids. Disjoint from :func:`known_learner_ids`.
    """
    return demo_module.demo_learner_ids()
