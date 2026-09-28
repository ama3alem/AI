"""The EduFlow -> Focus Engine integration contract.

**This module is the boundary. It is deliberately thin.**

It re-declares nothing the engine already declares. Events are
:class:`focus_engine.events.types.EventEnvelope` — the engine's own closed, validated
vocabulary — not a second event schema written here. A contract that defined its own event
shape would be a second opinion about what an event is, and the two would drift.

What this module adds is transport shape and privacy shape:

* a batch request/response pair, because a transport needs envelopes around a payload;
* a PII exclusion statement enforced by *construction* — the only identifier fields in the
  contract are the engine's pseudonymous ones, so a name, email, or student number has
  nowhere to go even if a caller tries;
* a response envelope carrying the engine's own typed results plus an explicit
  ``synthetic`` flag, so a result computed over a simulator corpus can never be presented
  as a measurement of a person.

**No EduFlow database access is required or implied.** The engine receives events and
returns results. It never reads, joins, or looks up anything on EduFlow's side, which is
what allows the two systems to be deployed, versioned, and failed independently.

See ``EDUFLOW_INTEGRATION_CONTRACT.md`` for the prose contract and the rationale.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from focus_engine.evaluation.models import EvaluationRecord
from focus_engine.events.types import EventEnvelope
from focus_engine.outcomes.models import OutcomeRecord
from focus_engine.policy.models import PolicyDecision
from focus_engine.temporal.models import TemporalState
from focus_engine.uncertainty.outcomes import PredictionOutcome

__all__ = [
    "AnalysisResponse",
    "AnalysisResult",
    "ContractVersion",
    "DeliveryStatus",
    "Diagnostic",
    "DiagnosisCategory",
    "DiagnosisKind",
    "EventBatchRequest",
    "EventBatchResponse",
    "InterventionStatus",
    "SyntheticDataNotice",
    "TransportError",
    "TransportErrorDetail",
]

ContractVersion = Annotated[str, Field(pattern=r"^FOCUS_CONTRACT_V[0-9]+$")]


class DeliveryStatus(StrEnum):
    """What actually happened to the intervention, as distinct from what policy wanted.

    **The distinction this enum exists to protect is selected-versus-delivered.** A pipeline
    that printed one field for both would report ``INTERVENE`` as if a prompt had reached a
    learner, which is precisely the claim nothing in this repository can support. So
    :attr:`~InterventionStatus.selected_by_policy` stays a separate fact with its own
    vocabulary, and this enum says what the delivery boundary actually did.

    The four values are exhaustive and mutually exclusive:

    ``NOT_SELECTED``
        Policy declined. No candidate existed, so there was nothing to deliver. This is not
        a failure and must not be styled as one.

    ``SELECTED_NOT_DELIVERED``
        A real candidate existed and was *not* delivered, because a named gate stopped it.
        :attr:`~InterventionStatus.gate` says which. This is the value that carries the most
        information in the whole enum and the one a reader most needs to see.

    ``DELIVERED``
        The delivery boundary executed the candidate and emitted the engine's own native
        lifecycle events. In this repository that is reachable only from the development-only
        Lab boundary, never from ``src/focus_engine``.

    ``DELIVERY_FAILED``
        Delivery was attempted and did not succeed. No completion event exists for it, because
        the thing that would report one never happened.
    """

    NOT_SELECTED = "not_selected"
    SELECTED_NOT_DELIVERED = "selected_not_delivered"
    DELIVERED = "delivered"
    DELIVERY_FAILED = "delivery_failed"


class DiagnosisKind(StrEnum):
    """Whether a statement limited the result or only shaped how to read it.

    The distinction is the point. A ``CONSTRAINT`` says the pipeline could not go further.
    A ``CAVEAT`` says the result stands but means something narrower than it first appears
    to. Collapsing the two produces the two failure modes a reader of a lab has to avoid:
    treating a bounded estimate as a refusal, and treating a working pipeline as broken.
    """

    CONSTRAINT = "constraint"
    CAVEAT = "caveat"


class DiagnosisCategory(StrEnum):
    """Which layer explains the outcome, kept deliberately distinct.

    These are not severities. ``INPUT_REJECTED`` and ``UNCERTAINTY_LIMIT`` are both
    "nothing acted", and they call for entirely different corrections — one in the
    caller's payload, one in the evidence available to the learner. Reporting a single
    "failed" is what makes a pipeline like this impossible to debug from the outside.
    """

    INPUT_REJECTED = "input_rejected"
    INSUFFICIENT_DATA = "insufficient_data"
    BASELINE_ADAPTATION = "baseline_adaptation"
    TEMPORAL_STATE_LOGIC = "temporal_state_logic"
    PREDICTION_REFUSAL = "prediction_refusal"
    UNCERTAINTY_LIMIT = "uncertainty_limit"
    POLICY_RESTRICTION = "policy_restriction"
    AUTHORITY_RESTRICTION = "authority_restriction"
    SAFETY_RESTRICTION = "safety_restriction"
    NO_RESTRICTION = "no_restriction"


class Diagnostic(BaseModel):
    """One attributable reason for this run's outcome, with the number that proves it."""

    model_config = ConfigDict(extra="forbid")

    category: DiagnosisCategory
    kind: DiagnosisKind
    stage: str = Field(description="The pipeline stage this statement is about.")
    summary: str = Field(description="What happened, stated so it can be read without code.")
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="The engine values behind the statement. A claim with no evidence is "
        "not emitted at all, so an empty block means only that the backing values were "
        "themselves absent.",
    )


class SyntheticDataNotice(BaseModel):
    """Provenance that travels with every result, so a slice cannot be misread.

    ``data_origin`` and ``provenance`` are copied from the engine's own values rather than
    re-derived here. A contract that recomputed provenance would be a second, divergent
    opinion about where a number came from.
    """

    model_config = ConfigDict(extra="forbid")

    data_origin: str = Field(description="Engine DataOrigin value, e.g. synthetic or real.")
    provenance: str = Field(description="Engine Provenance value, e.g. observed.")
    is_synthetic: bool = Field(
        description="True when any contributing event was synthetic. A result computed "
        "over simulator output is a measurement of the simulator, not of a learner."
    )
    warning: str | None = Field(
        default=None,
        description="Present only when synthetic. The simulator's in-band warning string.",
    )


class TransportErrorDetail(BaseModel):
    """One rejected item, carrying the engine's reason verbatim."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, description="Position in the submitted batch.")
    event_id: str | None = None
    code: str
    message: str


class EventBatchRequest(BaseModel):
    """A batch of behavioural events sent by EduFlow.

    The payload is the engine's own :class:`EventEnvelope`, so validation is the engine's
    validation: a malformed event is rejected with the engine's own error, not with a
    transport-specific paraphrase of it.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: ContractVersion = "FOCUS_CONTRACT_V1"
    events: Annotated[
        list[EventEnvelope],
        Field(min_length=1, description="At least one event. An empty batch is refused."),
    ]


class EventBatchResponse(BaseModel):
    """The result of accepting a batch.

    ``accepted`` counts events that passed engine validation; ``rejected`` reports the ones
    that did not, each with the engine's own reason. Rejections are never dropped silently
    and never repaired.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: ContractVersion = "FOCUS_CONTRACT_V1"
    accepted: int = Field(ge=0)
    rejected: int = Field(ge=0)
    rejections: list[TransportErrorDetail] = Field(default_factory=list)
    notice: SyntheticDataNotice


class TransportError(BaseModel):
    """A whole-request failure. The engine is not consulted."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    detail: list[TransportErrorDetail] = Field(default_factory=list)


class InterventionStatus(BaseModel):
    """Separates *selected by policy* from *actually delivered*, and says which gate moved it.

    This distinction is the whole point of the model. ``selected_by_policy`` is a fact the
    engine can establish. ``delivery_status`` is a fact only a delivery boundary can
    establish, and the only such boundary in this repository is development-only. The two are
    separate fields because a reader who sees ``INTERVENE`` in a log needs to be able to ask
    the next question — *was it delivered?* — and get an answer rather than an assumption.

    ``gate`` exists because "not delivered" is not diagnostic on its own. A candidate stopped
    by a safety refusal and a candidate stopped by a pending human approval are the same
    status and completely different problems, so the blocking gate travels with the status.
    """

    model_config = ConfigDict(extra="forbid")

    selected_by_policy: bool
    intervention_type: str | None = Field(
        default=None, description="The catalogue entry the policy selected, if any."
    )
    delivery_status: DeliveryStatus = Field(
        description="What the delivery boundary did. Never inferred from selected_by_policy."
    )
    gate: str | None = Field(
        default=None,
        description="The gate that determined the outcome: ``policy``, ``authority``, "
        "``safety``, ``human_approval``, or ``delivery`` when the boundary itself failed. "
        "``None`` when no gate was reached.",
    )
    final_action: str | None = Field(
        default=None,
        description="The engine's own ``FinalActionOutcome`` value from the safety gate, "
        "copied verbatim rather than re-derived.",
    )
    requires_human_approval: bool = Field(
        default=False,
        description="True when the gate would allow the action only on a named human's "
        "approval. Such an action is not delivered autonomously.",
    )
    intervention_id: str | None = Field(
        default=None,
        description="Assigned only when delivery is attempted. Stable across replays of the "
        "same input, because it is derived from the decision rather than from a counter.",
    )
    started_at: datetime | None = Field(
        default=None, description="When the intervention lifecycle began, from the injected clock."
    )
    delivered_at: datetime | None = Field(
        default=None, description="When delivery completed. Null for every non-delivered status."
    )
    lifecycle_events: list[EventEnvelope] = Field(
        default_factory=list,
        description="The engine's own native lifecycle events. Empty unless delivery was "
        "attempted, so an empty list is an assertion that nothing was sent.",
    )
    explanation: str = Field(
        description="Why the two decision fields differ. Names the gate rather than leaving "
        "the reader to infer it."
    )


class AnalysisResult(BaseModel):
    """Everything one analysis produced, as the engine's own typed objects.

    These are the engine's models, not projections of them. A projection would be a place
    for a field to be dropped or renamed without anyone noticing, and the whole purpose of
    this boundary is that the engine remains the single source of truth.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    learner_id: str
    session_id: str
    decided_at: datetime
    temporal_state: TemporalState | None
    prediction_outcome: PredictionOutcome | None
    policy_decision: PolicyDecision | None
    outcome_record: OutcomeRecord | None = Field(
        default=None,
        description="Null unless a delivered intervention exists to measure. Populated by "
        "the development-only delivery boundary, and only when the measured evidence is "
        "sufficient to measure against.",
    )
    evaluation_record: EvaluationRecord | None = Field(
        default=None,
        description="Null unless there is both a prediction snapshot and admissible "
        "post-prediction evidence to score it against.",
    )
    intervention: InterventionStatus


class AnalysisResponse(BaseModel):
    """The response envelope. ``synthetic`` is a top-level field, not a footnote."""

    model_config = ConfigDict(extra="forbid")

    contract_version: ContractVersion = "FOCUS_CONTRACT_V1"
    synthetic: bool = Field(
        description="True when the analysis ran over synthetic events. The lab and the "
        "simulator always set this. It is a first-class field because a UI that renders "
        "a probability without it invites the reader to treat a simulator's output as a "
        "measurement of a person."
    )
    notice: SyntheticDataNotice
    results: list[AnalysisResult]
    stage_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered per-stage trace for the lab visualisation. Diagnostic, not "
        "part of the stable contract.",
    )
    diagnosis: list[Diagnostic] = Field(
        default_factory=list,
        description="Attribution of this run's outcome to the layer that produced it, with "
        "the supporting engine values. Diagnostic, not part of the stable contract. "
        "Diagnostic rather than authoritative because it interprets stages that are "
        "already authoritative; it never overrides one.",
    )
