"""Data models for the Student Intelligence layer.

The layer is *inspect-first*: it never re-derives an engine value. Every one of these types
either re-states an engine value verbatim (:class:`EvidenceRecord`) or aggregates several of
those records into an answer (:class:`StudentAnswer`). The one type that is genuinely new is
the :class:`EvidenceClass`, and it classifies *how a value came to exist* rather than
re-computing what the value is.

**Why a class in addition to the engine's provenance.** The engine already stamps every
value with :class:`~focus_engine.schemas.primitives.Provenance` and
:class:`~focus_engine.schemas.primitives.DataOrigin`. Those are sufficient for the pipeline,
but they do not answer the question a consumer of this layer asks first: *should I treat this
as a fact, a reading, a guess, or an absence?* :class:`EvidenceClass` is that answer. It is a
classification of the engine's own labels, never a competing label: an evidence record
carries both the engine's ``provenance``/``data_origin`` and this layer's ``evidence_class``,
and the answer contract keeps them apart.

**Abundance is never padded.** A gap is a first-class value (:class:`GapRecord`), and
:class:`GapType` enumerates eight reasons an answer may fall short. A ``MISSING`` gap is not
mysteriously "fixed" by privilege, and a ``PRIVILEGE_RESTRICTED`` gap is not reported as a
``MISSING`` one -- the two look identical in the output but demand completely different
responses.

All models are frozen and reject extra fields, matching the repository's schema discipline.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from api.lab.contract import SyntheticDataNotice
from focus_engine.schemas.primitives import ConfidenceLevel

__all__ = [
    "AnswerType",
    "ChangeState",
    "EvidenceAgreement",
    "EvidenceClass",
    "EvidenceKind",
    "EvidenceRecord",
    "GAP_PRIORITY",
    "GapRecord",
    "GapType",
    "QuestionIntent",
    "ReliabilityGrade",
    "StudentAnswer",
    "StudentIntelligenceProfile",
]


#: The six evidence classes, mirroring the capability-matrix classification in
#: ``EDUFLOW_STUDENT_DATA_CONTRACT.md``. ``UNAVAILABLE`` is a value, not an omission: a
#: field that the contract declares unavailable cannot be represented by any other class.
#: ``HUMAN_REPORTED`` exists in the vocabulary but nothing in this synthetic repository
#: produces it yet, which is itself a stated fact about the data rather than a hole.
class EvidenceClass(StrEnum):
    OBSERVED = "observed"
    """Measured directly from the behavioural event stream."""

    DERIVED = "derived"
    """Computed by an engine layer from observations (features, baseline, state)."""

    PREDICTED = "predicted"
    """Output of a trained model. Never a measurement; never a fact about a learner."""

    HUMAN_REPORTED = "human_reported"
    """A statement supplied by a person. Nothing in this repository produces this yet."""

    SYSTEM_GENERATED = "system_generated"
    """Produced by a system boundary: the delivery lifecycle, the authority envelope."""

    UNAVAILABLE = "unavailable"
    """Declared unavailable for this learner under the data contract."""


class EvidenceKind(StrEnum):
    """Which engine surface a record came from.

    Kept as its own vocabulary rather than reusing stage names so that the layer that owns
    RBAC can attach a scope to a *kind of evidence* without depending on pipeline stage
    strings, and so two kinds can never share a name by accident.
    """

    EVENT = "event"
    CONTEXT = "context"
    FEATURE = "feature"
    BASELINE = "baseline"
    TEMPORAL = "temporal"
    PREDICTION = "prediction"
    UNCERTAINTY = "uncertainty"
    POLICY = "policy"
    AUTHORITY = "authority"
    INTERVENTION = "intervention"
    OUTCOME = "outcome"
    EVALUATION = "evaluation"
    PROVENANCE = "provenance"
    SUBJECT = "subject"
    CHANGE = "change"
    FUSION = "fusion"


class ChangeState(StrEnum):
    """The layer's own reading of how observed behaviour moved across time.

    A reconstruction of the learner's own event stream, never a self-report and never a
    cause. ``INSUFFICIENT_EVIDENCE`` is the honest value when there is too little history
    to classify; it is never substituted with a guess.
    """

    STABLE = "stable"
    IMPROVING = "improving"
    DECLINING = "declining"
    VOLATILE = "volatile"
    RECOVERING = "recovering"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceAgreement(StrEnum):
    """How subject-scoped evidence lines up with a wider claim.

    Contradictory subject signals are reported, never averaged away: the fusion record
    names the agreement and the conflicting gap carries it forward.
    """

    SUPPORTING = "supporting"
    CONTRADICTORY = "contradictory"
    NEUTRAL = "neutral"
    INSUFFICIENT = "insufficient"


class ReliabilityGrade(StrEnum):
    """How much an evidence record may be leaned on when composing an answer.

    Deliberately coarse. The layer does not invent fine-grained scores for facts the engine
    itself does not grade; three grades are enough to order an answer's confidence without
    implying a precision the underlying values do not have.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapType(StrEnum):
    """Why an answer may be short of complete, in the contract's stated priority order."""

    MISSING = "missing"
    STALE = "stale"
    LOW_COVERAGE = "low_coverage"
    LOW_QUALITY = "low_quality"
    CONFLICTING = "conflicting"
    UNVALIDATED = "unvalidated"
    NOT_APPLICABLE = "not_applicable"
    PRIVILEGE_RESTRICTED = "privilege_restricted"


#: Priority per gap type, in the contract's stated order (1 = most important). Defined on
#: the models so every gap producer and every sorter agrees on what "prioritised" means.
GAP_PRIORITY: Final[dict[GapType, int]] = {
    GapType.MISSING: 1,
    GapType.STALE: 2,
    GapType.LOW_COVERAGE: 3,
    GapType.LOW_QUALITY: 4,
    GapType.CONFLICTING: 5,
    GapType.UNVALIDATED: 6,
    GapType.NOT_APPLICABLE: 7,
    GapType.PRIVILEGE_RESTRICTED: 8,
}


class QuestionIntent(StrEnum):
    """The deterministic interpretation of the author's question.

    Intent classification is rule-based by design. There is no hosted model behind this
    layer (none is packaged with this repository), and a keyword classifier over a fixed
    vocabulary cannot be silently repurposed by a prompt the way a language model can.
    An unrecognised question is reported as ``UNKNOWN`` and refused, never guessed at.
    """

    PROFILE = "profile"
    EXPLAIN = "explain"
    MISSING = "missing"
    AUTONOMY = "autonomy"
    UNKNOWN = "unknown"
    CURRENT_STATE = "current_state"
    CHANGE = "change"
    EXPLANATION = "explanation"
    SUBJECT = "subject"
    INTERVENTION = "intervention"
    OUTCOME = "outcome"
    EVIDENCE = "evidence"
    GAPS = "gaps"
    AUTHORITY = "authority"
    TIMELINE = "timeline"
    COMPARISON = "comparison"


class AnswerType(StrEnum):
    """What kind of claim an answer makes.

    The six members mirror the answer-contract vocabulary. An :class:`EvidenceRecord` of
    class ``PREDICTED`` can never stand behind an answer typed ``FACT``, and this enum is
    what makes that pairing checkable: an answer is typed from the evidence that actually
    appears in it, not from its wording.
    """

    FACT = "fact"
    DERIVED_FACT = "derived_fact"
    PREDICTION = "prediction"
    HUMAN_REPORT = "human_report"
    INTERPRETATION = "interpretation"
    RECOMMENDATION = "recommendation"


class EvidenceRecord(BaseModel):
    """One re-stated engine value, with everything a reader needs to trust or doubt it.

    ``statement`` is the only prose field in the layer and it is written once, at
    construction, from the engine value it embeds. The answer composer quotes statements
    verbatim; it never paraphrases, because a paraphrase is where a re-statement stops
    matching its source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(description="Deterministic identifier derived from learner and kind.")
    learner_id: str
    kind: EvidenceKind
    evidence_class: EvidenceClass
    reliability: ReliabilityGrade
    statement: str
    provenance: str = Field(description="The engine's own Provenance value, copied verbatim.")
    data_origin: str = Field(description="The engine's own DataOrigin value, copied verbatim.")
    source: str = Field(description="Where the value came from, as a module path or field name.")
    version: str | None = Field(
        default=None, description="Engine version under which the value was produced."
    )
    timestamp: datetime | None = Field(
        default=None, description="The instant the value refers to, when it has one."
    )
    subject: str | None = Field(
        default=None,
        description="Optional subject-scoped label. None when the value is not subject-specific.",
    )
    note: str | None = Field(default=None, description="Anything the statement must not overclaim.")


class GapRecord(BaseModel):
    """One stated shortfall, with a priority and the remedy that would close it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gap_id: str
    learner_id: str
    gap_type: GapType
    description: str
    priority: int = Field(ge=1, description="1 is the most important; lower wins.")
    subject: str | None = Field(default=None)
    remedy: str | None = Field(
        default=None, description="What would have to change for this gap to close."
    )


class StudentAnswer(BaseModel):
    """The full answer contract: the answer, everything behind it, and its limits.

    ``evidence``, ``data_gaps``, ``limitations``, and ``sources`` are *required* fields of
    the answer, never optional extras. An answer that cannot fill them is a refusal, which
    is represented by :attr:`refused` and carries no ``answer_type``. A refusal is not an
    empty answer: the space where an evidence list would go is how a reader tells a refusal
    from an omission.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    student: str
    question: str
    intent: QuestionIntent
    answer_type: AnswerType | None = Field(
        default=None,
        description="None exactly when refused. A class of claim is only assigned to an "
        "answer that was actually issued.",
    )
    answer: str
    refused: bool
    refusal_reason: str | None = None
    evidence: tuple[EvidenceRecord, ...]
    evidence_quality: ReliabilityGrade | None = None
    confidence: ConfidenceLevel | None = None
    data_gaps: tuple[GapRecord, ...]
    limitations: tuple[str, ...]
    generated_at: datetime
    sources: tuple[str, ...]
    engine_versions: dict[str, str]
    notice: SyntheticDataNotice


class StudentIntelligenceProfile(BaseModel):
    """The canonical profile for one learner: header plus evidence and gaps.

    A profile is a *collection of evidence*, not a narrative. It carries no summary
    sentence, because a summary is the layer's explanation output and belongs in a
    :class:`StudentAnswer` where the evidence it was built from travels beside it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    learner_id: str
    scenario_id: str
    title: str
    archetype: str = Field(
        description="The simulator archetype that generated the events. A property of the "
        "generator, never asserted about a learner.",
    )
    evidence: tuple[EvidenceRecord, ...]
    gaps: tuple[GapRecord, ...]
    engine_versions: dict[str, str]
    model_summary: dict[str, Any]
    notice: SyntheticDataNotice
    built_at: datetime

    def evidence_of(self, kind: EvidenceKind) -> tuple[EvidenceRecord, ...]:
        """Return the records of one kind, in stored order.

        Args:
            kind: The kind to select.

        Returns:
            The matching records.
        """
        return tuple(record for record in self.evidence if record.kind is kind)
