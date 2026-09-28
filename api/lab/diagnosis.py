"""Why did the pipeline produce this result? Attribution over real engine output.

**A disagreement is not a defect, and it is not an excuse either.** When a scenario named
``rapid_decline`` is reported as ``recovering``, the useful answer is not "the scenario is
wrong" and not "the engine is broken" — it is *which layer produced the surprising number,
and what did it read*. A lab that only prints the final state teaches the reader nothing
except distrust.

So this module walks the stage log the adapter already produced and reports, for each layer
that bore on the outcome, one of two kinds of statement:

``CONSTRAINT``
    Something limited what the pipeline could assert. The pipeline would have gone further
    without it.

``CAVEAT``
    Something that did not block but does shape how the result should be read. A rolling
    baseline that has followed the learner is in this class: it is always true, and it is
    the single most common reason a generated trajectory and an observed state disagree.

Every statement is *derived from a value the engine produced* and carries that value in its
``evidence`` field. Nothing here re-decides anything, and nothing here has an opinion about
whether an outcome is correct. A statement that cannot be backed by a number in the stage
log is not emitted.

The categories deliberately separate the four things a reader must not confuse: the input
was rejected (:attr:`DiagnosisCategory.INPUT_REJECTED`), the evidence was too thin
(:attr:`DiagnosisCategory.INSUFFICIENT_DATA`), the model declined to score
(:attr:`DiagnosisCategory.PREDICTION_REFUSAL`), and the estimate was produced but bounded
(:attr:`DiagnosisCategory.UNCERTAINTY_LIMIT`). A "failure" that could be any of those is
four different bugs for whoever has to fix it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from api.lab.contract import DiagnosisCategory, DiagnosisKind, Diagnostic

__all__ = ["NO_RESTRICTION_SUMMARY", "diagnose"]

#: Emitted when nothing bounded the result, so the reader is not left to infer that an
#: absence of constraints means an absence of evidence. It names the numbers that were
#: actually produced instead.
NO_RESTRICTION_SUMMARY: Final[str] = (
    "No stage reported a constraint. The pipeline produced a scored prediction and a policy "
    "decision on the evidence available; read the restraint list for the bounds that were "
    "checked and cleared rather than for a limit that bound."
)


def _stage_map(stage_log: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(entry.get("stage")): dict(entry.get("detail") or {}) for entry in stage_log}


def _evidence(**values: Any) -> dict[str, Any]:
    """Drop ``None`` values so an evidence block never implies a measurement it lacks."""
    return {key: value for key, value in values.items() if value is not None}


def _baseline_adaptation(baseline: dict[str, Any], sessions_folded: int) -> list[Diagnostic]:
    """Report how far the personal centre travelled while sessions were folded in.

    A baseline is a rolling estimate, not a fixed reference. This states the consequence as
    a number: the centre for a dimension at its first accepted observation, the centre now,
    and the difference. When that difference is material, a learner who changed gradually
    has been partly followed, so a later session can sit close to the centre while the
    underlying trajectory was a steady decline. That is the mechanism behind most
    scenario-label/observed-state disagreements, and it is invisible unless stated.
    """
    dimensions = baseline.get("dimensions") or {}
    moved: list[dict[str, Any]] = []
    for name, stats in dimensions.items():
        first = stats.get("centre_first")
        last = stats.get("centre")
        if first is None or last is None:
            continue
        drift = float(last) - float(first)
        scale = abs(float(first))
        # Relative drift, so an accuracy centre and a seconds centre are comparable.
        relative = abs(drift) / scale if scale > 1e-12 else None
        moved.append(
            _evidence(
                dimension=name,
                centre_first=float(first),
                centre=float(last),
                drift=drift,
                relative_drift=relative,
            )
        )
    if not moved:
        return []

    largest = max(moved, key=lambda item: item.get("relative_drift") or 0.0)
    relative = largest.get("relative_drift")
    summary = (
        f"The personal baseline is a rolling estimate, not a fixed reference: across "
        f"{sessions_folded} session(s) the centre for {largest['dimension']} moved from "
        f"{largest['centre_first']:.4f} to {largest['centre']:.4f}"
        + (f" ({relative * 100:.1f}% of its starting value). " if relative is not None else ". ")
        + "Each session is therefore measured against a centre that moved with the learner, "
        "so a gradual change is partly absorbed rather than accumulated. A generated "
        "trajectory and an observed deviation from a moving baseline are different "
        "quantities, and this run is a case where they disagree."
    )
    return [
        Diagnostic(
            category=DiagnosisCategory.BASELINE_ADAPTATION,
            kind=DiagnosisKind.CAVEAT,
            stage="baseline",
            summary=summary,
            evidence=_evidence(
                sessions_folded_in=sessions_folded,
                dimensions=moved,
            ),
        )
    ]


def diagnose(
    stage_log: Sequence[dict[str, Any]],
    *,
    question_count: int,
) -> list[Diagnostic]:
    """Attribute this run's outcome to the layers that produced it.

    Args:
        stage_log: The adapter's ordered stage log. Read, never modified.
        question_count: How many questions were answered in the submitted stream.

    Returns:
        Constraints first, then caveats, then a terminal
        :attr:`DiagnosisCategory.NO_RESTRICTION` entry when nothing bound the result. An
        empty list is never returned, so a caller always has something to show.
    """
    stages = _stage_map(stage_log)
    events = stages.get("events", {})
    baseline = stages.get("baseline", {})
    temporal = stages.get("temporal", {})
    prediction = stages.get("prediction", {})
    uncertainty = stages.get("uncertainty", {})
    policy = stages.get("policy", {})

    constraints: list[Diagnostic] = []
    caveats: list[Diagnostic] = []

    # --- A. the input never became events ------------------------------------
    if question_count == 0:
        constraints.append(
            Diagnostic(
                category=DiagnosisCategory.INPUT_REJECTED,
                kind=DiagnosisKind.CONSTRAINT,
                stage="events",
                summary=(
                    "No answered question reached the engine. Nothing downstream ran, so "
                    "every later stage is reported as unavailable rather than as a result."
                ),
                evidence=_evidence(
                    total_events=events.get("total_events"),
                    answered_questions=events.get("answered_questions"),
                    session_count=events.get("session_count"),
                ),
            )
        )

    # --- B. evidence too thin -------------------------------------------------
    maturity = baseline.get("maturity")
    observations = baseline.get("observations")
    if maturity == "new" or temporal.get("state") == "insufficient_data":
        constraints.append(
            Diagnostic(
                category=DiagnosisCategory.INSUFFICIENT_DATA,
                kind=DiagnosisKind.CONSTRAINT,
                stage="baseline",
                summary=(
                    "The personal baseline never left the 'new' maturity level, so no "
                    "personal centre was available and no standardised deviation could be "
                    "measured. This is the evidence floor, not a modelling failure."
                ),
                evidence=_evidence(
                    baseline_maturity=maturity,
                    baseline_observations=observations,
                    sessions_folded_in=baseline.get("sessions_folded_in"),
                    temporal_state=temporal.get("state"),
                    temporal_observations=temporal.get("evidence_observations"),
                ),
            )
        )

    # --- C. the baseline followed the learner --------------------------------
    if maturity not in (None, "new") and question_count:
        caveats.extend(_baseline_adaptation(baseline, int(baseline.get("sessions_folded_in") or 0)))

    # --- D. the temporal claim rests on a short or isolated run -------------
    run_length = temporal.get("run_length")
    character = temporal.get("character")
    if (
        question_count
        and character in ("isolated", "flat")
        and temporal.get("state")
        not in (
            None,
            "insufficient_data",
        )
    ):
        caveats.append(
            Diagnostic(
                category=DiagnosisCategory.TEMPORAL_STATE_LOGIC,
                kind=DiagnosisKind.CAVEAT,
                stage="temporal",
                summary=(
                    f"The temporal layer classified the evidence as '{character}' rather "
                    f"than persistent (current run length {run_length}). A short or "
                    "non-repeating run is read as a moment, so the state below should be "
                    "read as provisional."
                ),
                evidence=_evidence(
                    character=character,
                    run_length=run_length,
                    state=temporal.get("state"),
                    direction=temporal.get("direction"),
                    change=temporal.get("change"),
                ),
            )
        )

    # --- E. the model declined to score --------------------------------------
    if prediction.get("outcome") == "PredictionRefusal":
        constraints.append(
            Diagnostic(
                category=DiagnosisCategory.PREDICTION_REFUSAL,
                kind=DiagnosisKind.CONSTRAINT,
                stage="prediction",
                summary=(
                    "The model refused to score. A refusal is the model declining to "
                    "answer, which is different from a low probability; there is no number "
                    "to interpret."
                ),
                evidence=_evidence(
                    missing_features=prediction.get("missing_features"),
                    reason=prediction.get("reason"),
                ),
            )
        )

    # --- F. the estimate exists but is bounded -------------------------------
    assessment = uncertainty.get("confidence_assessment")
    if uncertainty.get("verdict") == "resolved" and isinstance(assessment, dict):
        binding = assessment.get("binding_constraint")
        if binding and binding != "none":
            ceiling = {
                "calibration": assessment.get("calibration_ceiling"),
                "evidence": assessment.get("evidence_ceiling"),
                "maturity": assessment.get("maturity_ceiling"),
            }
            binding_value = next((value for value in ceiling.values() if value is not None), None)
            if binding_value is not None and binding_value < 1.0:
                constraints.append(
                    Diagnostic(
                        category=DiagnosisCategory.UNCERTAINTY_LIMIT,
                        kind=DiagnosisKind.CONSTRAINT,
                        stage="uncertainty",
                        summary=(
                            f"A probability was produced, but confidence is capped at "
                            f"{binding_value:.4f} by the {binding} ceiling. The number is "
                            "usable; a strong reading of it is not available."
                        ),
                        evidence=_evidence(
                            binding_constraint=binding,
                            binding_ceiling=binding_value,
                            level=assessment.get("level"),
                            model_assertion=assessment.get("model_assertion"),
                            probability=uncertainty.get("probability"),
                            ceilings=ceiling,
                        ),
                    )
                )
    elif uncertainty.get("verdict") not in (None, "resolved"):
        constraints.append(
            Diagnostic(
                category=DiagnosisCategory.UNCERTAINTY_LIMIT,
                kind=DiagnosisKind.CONSTRAINT,
                stage="uncertainty",
                summary=(
                    f"The uncertainty layer returned '{uncertainty.get('verdict')}' rather "
                    "than a resolved verdict, so the policy has nothing resolved to act on."
                ),
                evidence=_evidence(
                    verdict=uncertainty.get("verdict"),
                    remedy=uncertainty.get("remedy"),
                    refusal_reason=uncertainty.get("refusal_reason"),
                ),
            )
        )

    # --- G. the policy declined, and said which restraint bound --------------
    binding_restraints = [
        {
            "kind": item.get("kind"),
            "observed": item.get("observed"),
            "limit": item.get("limit"),
            "headroom": item.get("headroom"),
        }
        for item in (policy.get("restraints") or [])
        if item.get("binding")
    ]
    if policy.get("decision") == "no_intervention" and binding_restraints:
        constraints.append(
            Diagnostic(
                category=DiagnosisCategory.POLICY_RESTRICTION,
                kind=DiagnosisKind.CONSTRAINT,
                stage="policy",
                summary=(
                    "The policy declined to select an intervention because "
                    + ", ".join(str(item["kind"]) for item in binding_restraints)
                    + " bound. The other restraints listed were checked and cleared."
                ),
                evidence=_evidence(
                    decision=policy.get("decision"),
                    reason=policy.get("reason"),
                    binding=binding_restraints,
                ),
            )
        )

    if not constraints:
        constraints.append(
            Diagnostic(
                category=DiagnosisCategory.NO_RESTRICTION,
                kind=DiagnosisKind.CONSTRAINT,
                stage="policy",
                summary=NO_RESTRICTION_SUMMARY,
                evidence=_evidence(
                    decision=policy.get("decision"),
                    probability=prediction.get("probability"),
                    temporal_state=temporal.get("state"),
                ),
            )
        )

    return constraints + caveats
