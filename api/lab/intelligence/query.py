"""The query layer: intent, sufficiency, and grounded answer composition.

This is the entire "AI" of the Student Intelligence layer and it is deliberately ordinary.
There is no hosted language model behind it — none is packaged with this repository — so the
answer composer works the way a supply chain must: it may only *re-state* evidence records
built from the engine's own output. The rules that keep it honest are stated here rather
than trusted to an LLM prompt:

* **Intent is classified by keywords, not by prompt.** A rule-based classifier cannot be
  steered. An unrecognised question is refused as ``UNKNOWN``, never best-guessed.
* **Sufficiency is checked against the evidence that survived scope filtering.** The same
  question answers differently for a teacher and for the learner, and the difference is
  enforced here, not in the UI.
* **A refusal distinguishes a data gap from a permission gap.** The needed evidence is
  withheld by role exactly when it exists in the profile, and each is refused with its own
  reason and surfaced as the matching gap type.
* **Every answer line is either a verbatim evidence statement or a fixed template string.**
  A test asserts exactly that, so a future edit cannot smuggle in an invented fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Final

from api.lab.intelligence.models import (
    AnswerType,
    EvidenceKind,
    EvidenceRecord,
    GapRecord,
    QuestionIntent,
    ReliabilityGrade,
    StudentAnswer,
    StudentIntelligenceProfile,
)
from api.lab.intelligence.rbac import AccessScope, restricted_gaps, visible_evidence
from focus_engine.schemas.primitives import ConfidenceLevel

__all__ = ["classify_intent", "answer_question"]

#: The synthetic-notice and provenance caveats travel with every answer. They are fixed
#: template strings, so a test can prove the answer invented nothing outside them.
_SYNTHETIC_LIMITATION: Final[str] = (
    "All data is synthetic. These answers describe the simulator, not a real learner."
)
_PREDICTED_LIMITATION: Final[str] = "PREDICTED values are model outputs, never measurements."
_CAUSATION_LIMITATION: Final[str] = (
    "Correlation is not causation. The engine reports observed behaviours and derived "
    "states, not causes."
)
_RESTRICTED_LIMITATION: Final[str] = (
    "PRIVILEGE_RESTRICTED evidence was withheld under the requested role and surfaced as "
    "a gap rather than a reading."
)
_AUTHORITY_LIMITATION: Final[str] = (
    "Authority statements are system-generated envelopes describing what is permitted, "
    "not predictions about the consequences of an action."
)

#: Fixed template sentences the composer may emit, per intent. Everything else in an answer
#: body must be a verbatim evidence statement or gap description.
_FIXED_LINES: Final[tuple[str, ...]] = (
    "Evidence:",
    "Data gaps relevant to this question:",
    "Limitations:",
    "Scope: the envelope's own permitted actions apply, subject to its recorded restrictions.",
    "The envelope does not meet the autonomy threshold, so autonomous action is not "
    "permitted for this learner.",
    "The authority envelope for this learner:",
    "The engine's stated chain, top to bottom: temporal state, prediction, uncertainty, "
    "then policy.",
    "No known data gaps for this learner.",
    "Subject-scoped evidence:",
    "The agreement between subject signals:",
)

_EXPLAIN_KINDS: Final = frozenset(
    {
        EvidenceKind.TEMPORAL,
        EvidenceKind.PREDICTION,
        EvidenceKind.UNCERTAINTY,
        EvidenceKind.POLICY,
    }
)
_EXPLANATION_KINDS: Final = frozenset(
    {
        EvidenceKind.TEMPORAL,
        EvidenceKind.CHANGE,
        EvidenceKind.PREDICTION,
        EvidenceKind.UNCERTAINTY,
        EvidenceKind.POLICY,
    }
)
_MISSING_KINDS: Final = frozenset(
    {EvidenceKind.EVENT, EvidenceKind.BASELINE, EvidenceKind.PROVENANCE}
)
_AUTONOMY_KINDS: Final = frozenset(
    {EvidenceKind.AUTHORITY, EvidenceKind.POLICY, EvidenceKind.INTERVENTION}
)

#: The three subject names phase-15 evidence can be scoped to. Used to decide whether a
#: SUBJECT question names one of the subjects a profile actually describes.
_SUBJECT_NAMES: Final[tuple[str, ...]] = (
    "algebra",
    "geometry",
    "reading",
    "math",
    "mathematics",
)

_PROFILE_WORDS: Final[tuple[str, ...]] = (
    "tell me about",
    "overview",
    "profile",
    "describe",
    "summari",
    "who is",
    "about this student",
    "student",
)
_EXPLAIN_WORDS: Final[tuple[str, ...]] = ("why", "explain", "reason", "cause")
_MISSING_WORDS: Final[tuple[str, ...]] = (
    "missing",
    "don't know",
    "do we know",
    "unknown",
    "insufficient",
    "unavailable",
    "not know",
    "lack",
)
_AUTONOMY_WORDS: Final[tuple[str, ...]] = (
    "autonom",
    "can the ai",
    "act",
    "allowed",
    "authority",
    "how far",
)
_AUTHORITY_WORDS: Final[tuple[str, ...]] = (
    "permitted",
    "authority to",
    "allowed to",
    "what can",
    "what may",
    "permission",
)
_OUTCOME_WORDS: Final[tuple[str, ...]] = (
    "what happened after",
    "has an intervention",
    "intervention been delivered",
    "did it work",
    "was it effective",
    "outcome",
)
_EXPLANATION_WORDS: Final[tuple[str, ...]] = (
    "why has",
    "why did",
    "why have",
    "why the",
    "why this",
    "reason for",
    "cause of",
)
_INTERVENTION_WORDS: Final[tuple[str, ...]] = (
    "intervention",
    "recap",
    "prompt was",
    "prompts were",
    "delivered",
)
_GAPS_WORDS: Final[tuple[str, ...]] = (
    "gaps",
    "evidence is missing",
    "evidence missing",
    "missing evidence",
    "which evidence",
    "lack evidence",
    "what evidence is lacking",
)
_CHANGE_WORDS: Final[tuple[str, ...]] = (
    "changed",
    "change",
    "trend",
    "trajectory",
    "over time",
    "progress",
    "recover",
)
_CURRENT_STATE_WORDS: Final[tuple[str, ...]] = (
    "current state",
    "current status",
    "current situation",
    "state right now",
    "latest state",
    "how is the student doing",
    "what is the state",
)
_COMPARISON_WORDS: Final[tuple[str, ...]] = (
    "compare",
    "agree",
    "disagree",
    "contradict",
    "signals",
    "consisten",
    "align",
    "both subjects",
)
_SUBJECT_WORDS: Final[tuple[str, ...]] = (
    "algebra",
    "geometry",
    "reading",
    "math",
    "mathematics",
    "performance in",
    "performing in",
    "subject performance",
)
_TIMELINE_WORDS: Final[tuple[str, ...]] = (
    "timeline",
    "chronolog",
    "sequence",
    "history of events",
    "day by day",
)
_EVIDENCE_WORDS: Final[tuple[str, ...]] = (
    "evidence",
    "sources",
    "provenance",
    "what do we know",
    "data points",
    "what is known",
)


class _Sufficiency(Enum):
    OK = "ok"
    MISSING_DATA = "missing_data"
    PRIVILEGE_RESTRICTED = "privilege_restricted"
    UNSUPPORTED = "unsupported"


def classify_intent(question: str) -> QuestionIntent:
    """Classify a question's intent by keyword.

    The order is load-bearing: the generic vocabularies (``profile``'s ``student``,
    ``explain``'s ``why``) sit last so that the specific vocabularies they would otherwise
    swallow — a subject name, a mention of an intervention delivery — are matched first.

    Args:
        question: The author's question, lowercased before matching.

    Returns:
        The best-supported intent, or ``UNKNOWN`` when no keyword set matched.
    """
    lowered = question.lower()
    for words, intent in (
        (_AUTONOMY_WORDS, QuestionIntent.AUTONOMY),
        (_AUTHORITY_WORDS, QuestionIntent.AUTHORITY),
        (_OUTCOME_WORDS, QuestionIntent.OUTCOME),
        (_EXPLANATION_WORDS, QuestionIntent.EXPLANATION),
        (_INTERVENTION_WORDS, QuestionIntent.INTERVENTION),
        (_EXPLAIN_WORDS, QuestionIntent.EXPLAIN),
        (_GAPS_WORDS, QuestionIntent.GAPS),
        (_MISSING_WORDS, QuestionIntent.MISSING),
        (_CHANGE_WORDS, QuestionIntent.CHANGE),
        (_CURRENT_STATE_WORDS, QuestionIntent.CURRENT_STATE),
        (_COMPARISON_WORDS, QuestionIntent.COMPARISON),
        (_SUBJECT_WORDS, QuestionIntent.SUBJECT),
        (_TIMELINE_WORDS, QuestionIntent.TIMELINE),
        (_EVIDENCE_WORDS, QuestionIntent.EVIDENCE),
        (_PROFILE_WORDS, QuestionIntent.PROFILE),
    ):
        if any(word in lowered for word in words):
            return intent
    return QuestionIntent.UNKNOWN


def _sufficiency_for(
    intent: QuestionIntent,
    profile: StudentIntelligenceProfile,
    scopes: frozenset[AccessScope],
) -> tuple[_Sufficiency, tuple[EvidenceKind, ...]]:
    """State whether an intent can be answered from the visible evidence alone.

    Args:
        intent: The classified intent.
        profile: The canonical profile.
        scopes: The granted scope set.

    Returns:
        The sufficiency and the intent's required kinds. The requirements are the only
        load-bearing set here: profile needs at least one observed fact, explain needs the
        state/confidence chain, autonomy needs the authority envelope.
    """
    if intent is QuestionIntent.UNKNOWN:
        return _Sufficiency.UNSUPPORTED, ()
    if intent in {QuestionIntent.PROFILE, QuestionIntent.EVIDENCE, QuestionIntent.TIMELINE}:
        optical = _visible_kinds(profile, scopes)
        required: tuple[EvidenceKind, ...] = (EvidenceKind.EVENT,)
        if not ({EvidenceKind.EVENT, EvidenceKind.CONTEXT} & optical):
            if profile.evidence_of(EvidenceKind.EVENT) or profile.evidence_of(EvidenceKind.CONTEXT):
                return _Sufficiency.PRIVILEGE_RESTRICTED, required
            return _Sufficiency.MISSING_DATA, required
        return _Sufficiency.OK, ()
    if intent in {QuestionIntent.GAPS, QuestionIntent.MISSING}:
        return _Sufficiency.OK, ()
    required = _required_kinds(intent)
    if any(not profile.evidence_of(kind) for kind in required):
        return _Sufficiency.MISSING_DATA, required
    if not set(required).issubset(_visible_kinds(profile, scopes)):
        return _Sufficiency.PRIVILEGE_RESTRICTED, required
    return _Sufficiency.OK, required


def _required_kinds(intent: QuestionIntent) -> tuple[EvidenceKind, ...]:
    """The kinds an intent cannot be answered without."""
    if intent is QuestionIntent.EXPLAIN:
        return (EvidenceKind.TEMPORAL, EvidenceKind.UNCERTAINTY)
    if intent is QuestionIntent.EXPLANATION:
        return (EvidenceKind.TEMPORAL, EvidenceKind.CHANGE, EvidenceKind.UNCERTAINTY)
    if intent in {QuestionIntent.AUTONOMY, QuestionIntent.AUTHORITY}:
        return (EvidenceKind.AUTHORITY,)
    if intent is QuestionIntent.CURRENT_STATE:
        return (EvidenceKind.TEMPORAL,)
    if intent is QuestionIntent.CHANGE:
        return (EvidenceKind.CHANGE,)
    if intent is QuestionIntent.COMPARISON:
        return (EvidenceKind.FUSION, EvidenceKind.SUBJECT)
    if intent is QuestionIntent.SUBJECT:
        return (EvidenceKind.SUBJECT,)
    if intent is QuestionIntent.INTERVENTION:
        return (EvidenceKind.INTERVENTION,)
    if intent is QuestionIntent.OUTCOME:
        return (EvidenceKind.OUTCOME, EvidenceKind.INTERVENTION)
    return ()


def _visible_kinds(
    profile: StudentIntelligenceProfile, scopes: frozenset[AccessScope]
) -> set[EvidenceKind]:
    """The kinds with at least one visible record."""
    return {record.kind for record in visible_evidence(profile, scopes)}


def _named_subjects(question: str) -> set[str]:
    """The subject names the question explicitly asks about, if any."""
    lowered = question.lower()
    return {name for name in _SUBJECT_NAMES if name in lowered}


def _select_evidence(
    intent: QuestionIntent,
    profile: StudentIntelligenceProfile,
    scopes: frozenset[AccessScope],
    question: str,
) -> tuple[EvidenceRecord, ...]:
    """Choose the evidence an intent's answer will rest on."""
    visible = visible_evidence(profile, scopes)
    if intent is QuestionIntent.PROFILE:
        return visible
    if intent is QuestionIntent.EXPLAIN:
        return tuple(
            record
            for record in visible
            if record.kind in (_EXPLAIN_KINDS | {EvidenceKind.PROVENANCE})
        )
    if intent is QuestionIntent.EXPLANATION:
        return tuple(
            record
            for record in visible
            if record.kind in (_EXPLANATION_KINDS | {EvidenceKind.PROVENANCE})
        )
    if intent in {QuestionIntent.MISSING, QuestionIntent.GAPS}:
        missing_kinds: frozenset[EvidenceKind] = (
            _MISSING_KINDS if intent is QuestionIntent.MISSING else frozenset()
        )
        return tuple(record for record in visible if record.kind in missing_kinds)
    if intent in {QuestionIntent.AUTONOMY, QuestionIntent.AUTHORITY}:
        return tuple(
            record
            for record in visible
            if record.kind in (_AUTONOMY_KINDS | {EvidenceKind.PROVENANCE})
        )
    if intent is QuestionIntent.CURRENT_STATE:
        return tuple(record for record in visible if record.kind is EvidenceKind.TEMPORAL)
    if intent is QuestionIntent.CHANGE:
        return tuple(
            record
            for record in visible
            if record.kind in {EvidenceKind.CHANGE, EvidenceKind.FUSION}
        )
    if intent is QuestionIntent.COMPARISON:
        return tuple(
            record
            for record in visible
            if record.kind in {EvidenceKind.FUSION, EvidenceKind.SUBJECT}
        )
    if intent is QuestionIntent.SUBJECT:
        subjects = profile.evidence_of(EvidenceKind.SUBJECT)
        named = _named_subjects(question)
        if named:
            subjects = tuple(record for record in subjects if record.subject in named)
        return tuple(
            record
            for record in visible
            if record.kind is EvidenceKind.SUBJECT and (record in subjects if named else True)
        )
    if intent is QuestionIntent.INTERVENTION:
        return tuple(record for record in visible if record.kind is EvidenceKind.INTERVENTION)
    if intent is QuestionIntent.OUTCOME:
        return tuple(
            record
            for record in visible
            if record.kind in {EvidenceKind.OUTCOME, EvidenceKind.INTERVENTION}
        )
    if intent is QuestionIntent.TIMELINE:
        return tuple(
            record
            for record in visible
            if record.kind in {EvidenceKind.EVENT, EvidenceKind.CONTEXT}
        )
    if intent is QuestionIntent.EVIDENCE:
        return tuple(
            record
            for record in visible
            if record.kind in {EvidenceKind.EVENT, EvidenceKind.CONTEXT, EvidenceKind.PROVENANCE}
        )
    return ()
    return ()


def _bullet(record: EvidenceRecord) -> str:
    """One answer line: the verbatim statement plus its standing."""
    return (
        f"- {record.statement} "
        f"[{record.evidence_class.value}/{record.provenance}, "
        f"reliability {record.reliability.value}]"
    )


def _gap_bullet(gap: GapRecord) -> str:
    """One gap line: the verbatim description plus its standing."""
    return f"- {gap.description} ({gap.gap_type.value}, priority {gap.priority})"


def _data_gaps(
    profile: StudentIntelligenceProfile, scopes: frozenset[AccessScope]
) -> tuple[GapRecord, ...]:
    """Combined canonical-plus-restricted gaps, prioritised."""
    return tuple(
        sorted(
            profile.gaps + restricted_gaps(profile, scopes),
            key=lambda gap: (gap.priority, gap.gap_id),
        )
    )


def _quality(selected: tuple[EvidenceRecord, ...]) -> tuple[ReliabilityGrade, ConfidenceLevel]:
    """Aggregate the reliability of the selected evidence into an answer's standing.

    A single model output drops confidence to ``low`` regardless of its own grading,
    because a prediction is never a measurement and the answer must not present it as one.

    Args:
        selected: The evidence the answer rests on.

    Returns:
        The evidence-quality grade and the confidence level.
    """
    if not selected:
        return ReliabilityGrade.LOW, ConfidenceLevel.UNKNOWN
    grades = {record.reliability for record in selected}
    classes = {record.evidence_class.value for record in selected}
    worst = (
        ReliabilityGrade.HIGH
        if grades == {ReliabilityGrade.HIGH}
        else ReliabilityGrade.LOW
        if ReliabilityGrade.LOW in grades
        else ReliabilityGrade.MEDIUM
    )
    if "predicted" in classes or "unavailable" in classes:
        return worst, ConfidenceLevel.LOW
    if worst is ReliabilityGrade.HIGH:
        return worst, ConfidenceLevel.HIGH
    if worst is ReliabilityGrade.MEDIUM:
        return worst, ConfidenceLevel.MEDIUM
    return worst, ConfidenceLevel.LOW


def _limitations(
    intent: QuestionIntent, selected: tuple[EvidenceRecord, ...], gaps: tuple[GapRecord, ...]
) -> tuple[str, ...]:
    out: list[str] = [_SYNTHETIC_LIMITATION]
    if any(record.evidence_class.value == "predicted" for record in selected):
        out.append(_PREDICTED_LIMITATION)
    if intent in {QuestionIntent.EXPLAIN, QuestionIntent.EXPLANATION}:
        out.append(_CAUSATION_LIMITATION)
    if intent is QuestionIntent.COMPARISON:
        out.append(_CAUSATION_LIMITATION)
    if intent in {QuestionIntent.AUTONOMY, QuestionIntent.AUTHORITY}:
        out.append(_AUTHORITY_LIMITATION)
    if any(gap.gap_type.value == "privilege_restricted" for gap in gaps):
        out.append(_RESTRICTED_LIMITATION)
    return tuple(out)


def answer_question(
    profile: StudentIntelligenceProfile,
    question: str,
    scopes: frozenset[AccessScope],
) -> StudentAnswer:
    """Answer one question against one learner's profile under a scope set.

    Args:
        profile: The canonical profile (already fully populated; has not been retrieved
            under the scope set — retrieval happens here from the caller-provided store).
        question: The author's question.
        scopes: The scopes granted to the requesting role.

    Returns:
        A completed :class:`StudentAnswer`. Refusals carry ``refused=True``, no answer
        type, and the constraints that produced them.
    """
    intent = classify_intent(question)
    sufficiency, required = _sufficiency_for(intent, profile, scopes)
    gaps = _data_gaps(profile, scopes)
    engine_versions = dict(profile.engine_versions)
    notice = profile.notice

    head = [
        f"Intent: {intent.value}; learner {profile.learner_id}.",
        profile.title,
        "",
    ]

    if sufficiency is _Sufficiency.OK:
        selected = _select_evidence(intent, profile, scopes, question)
        if intent is QuestionIntent.SUBJECT and not selected:
            return _refused(
                profile,
                question,
                intent,
                gaps,
                engine_versions,
                head,
                (
                    "Insufficient data: no subject-scoped evidence matches the named "
                    "subject in this question, so a subject answer cannot be grounded here."
                ),
            )
        quality, confidence = _quality(selected)
        limitations = _limitations(intent, selected, gaps)
        body = _compose(intent, selected, gaps, limitations, head)
        answer_type = _answer_type(intent, selected)
        return StudentAnswer(
            student=profile.learner_id,
            question=question,
            intent=intent,
            answer_type=answer_type,
            answer=body,
            refused=False,
            evidence=selected,
            evidence_quality=quality,
            confidence=confidence,
            data_gaps=gaps,
            limitations=limitations,
            generated_at=profile.built_at,
            sources=tuple(sorted({record.source for record in selected})),
            engine_versions=engine_versions,
            notice=notice,
        )

    reason = _refusal_reason(intent, sufficiency, required)
    return _refused(profile, question, intent, gaps, engine_versions, head, reason)


def _refused(
    profile: StudentIntelligenceProfile,
    question: str,
    intent: QuestionIntent,
    gaps: tuple[GapRecord, ...],
    engine_versions: dict[str, str],
    head: list[str],
    reason: str,
) -> StudentAnswer:
    """A refusal: same contract as an answer, with the claim space emptied."""
    body = "\n".join(
        [
            *head,
            f"Refused: {reason}",
            "",
            _FIXED_LINES[1],
            "\n".join(_gap_bullet(gap) for gap in gaps),
        ]
    )
    return StudentAnswer(
        student=profile.learner_id,
        question=question,
        intent=intent,
        answer_type=None,
        answer=body,
        refused=True,
        refusal_reason=reason,
        evidence=(),
        evidence_quality=None,
        confidence=None,
        data_gaps=gaps,
        limitations=(_SYNTHETIC_LIMITATION,),
        generated_at=profile.built_at,
        sources=(),
        engine_versions=engine_versions,
        notice=profile.notice,
    )


def _refusal_reason(
    intent: QuestionIntent, sufficiency: _Sufficiency, required: tuple[EvidenceKind, ...]
) -> str:
    if sufficiency is _Sufficiency.UNSUPPORTED:
        return (
            "This question does not map to a supported intent (profile, current_state, "
            "explain, explanation, missing, gaps, change, current completion, subject, "
            "comparison, intervention, outcome, evidence, authority, autonomy, timeline). "
            "The layer refuses rather than guesses."
        )
    if sufficiency is _Sufficiency.MISSING_DATA:
        names = ", ".join(kind.value for kind in required) or "evidence"
        return (
            f"Insufficient data: no {names} evidence exists for this learner, so a "
            f"{intent.value} answer cannot be grounded."
        )
    if sufficiency is _Sufficiency.PRIVILEGE_RESTRICTED:
        names = ", ".join(kind.value for kind in required)
        return (
            f"Required {names} evidence is restricted under the requested role. Re-query "
            "with a role entitled to the relevant scope."
        )
    return "Refused."  # pragma: no cover - the enum is exhaustive above


def _answer_type(intent: QuestionIntent, selected: tuple[EvidenceRecord, ...]) -> AnswerType:
    if intent is QuestionIntent.PROFILE:
        if any(record.evidence_class.value == "predicted" for record in selected):
            return AnswerType.PREDICTION
        if any(record.evidence_class.value != "observed" for record in selected):
            return AnswerType.DERIVED_FACT
        return AnswerType.FACT
    if intent in {QuestionIntent.EXPLAIN, QuestionIntent.EXPLANATION}:
        return AnswerType.INTERPRETATION
    if intent in {QuestionIntent.AUTONOMY, QuestionIntent.AUTHORITY}:
        return AnswerType.RECOMMENDATION
    if intent is QuestionIntent.MISSING or intent is QuestionIntent.GAPS:
        return AnswerType.FACT
    if intent is QuestionIntent.INTERVENTION or intent is QuestionIntent.TIMELINE:
        return AnswerType.FACT
    if intent is QuestionIntent.EVIDENCE:
        return AnswerType.FACT
    return AnswerType.DERIVED_FACT


def _compose(
    intent: QuestionIntent,
    selected: tuple[EvidenceRecord, ...],
    gaps: tuple[GapRecord, ...],
    limitations: tuple[str, ...],
    head: list[str],
) -> str:
    """Assemble the answer body from verbatim statements and fixed template lines only."""
    lines: list[str] = [*head]
    gaps_emitted = False

    if intent in {QuestionIntent.EXPLAIN, QuestionIntent.EXPLANATION}:
        lines.append(_FIXED_LINES[6])
        lines.extend(_bullet(record) for record in selected)
    elif intent in {QuestionIntent.AUTONOMY, QuestionIntent.AUTHORITY}:
        authority = [record for record in selected if record.kind is EvidenceKind.AUTHORITY]
        lines.append(_FIXED_LINES[5])
        lines.extend(_bullet(record) for record in authority)
        restricted = any(gap.gap_type.value == "privilege_restricted" for gap in gaps)
        if authority and not restricted:
            lines.append(_FIXED_LINES[3] if _meets_autonomy(authority) else _FIXED_LINES[4])
    elif intent in {QuestionIntent.MISSING, QuestionIntent.GAPS}:
        lines.extend(_bullet(record) for record in selected)
        if gaps:
            lines.append(_FIXED_LINES[1])
            lines.extend(_gap_bullet(gap) for gap in gaps)
            gaps_emitted = True
    elif intent is QuestionIntent.COMPARISON:
        lines.append(_FIXED_LINES[9])
        lines.extend(_bullet(record) for record in selected)
    elif intent is QuestionIntent.SUBJECT:
        lines.append(_FIXED_LINES[8])
        lines.extend(_bullet(record) for record in selected)
    else:
        lines.append(_FIXED_LINES[0])
        lines.extend(_bullet(record) for record in selected)

    if not gaps_emitted:
        if gaps:
            lines.append(_FIXED_LINES[1])
            lines.extend(_gap_bullet(gap) for gap in gaps)
        elif intent is not QuestionIntent.AUTONOMY and intent is not QuestionIntent.AUTHORITY:
            lines.append(_FIXED_LINES[7])

    lines.append(_FIXED_LINES[2])
    lines.extend(f"- {limitation}" for limitation in limitations)
    return "\n".join(lines)


def _meets_autonomy(selected: Sequence[EvidenceRecord]) -> bool:
    """Whether the selected authority statement says the threshold is met.

    A true/false reading of the always-fixed statement wording. If the authority record is
    absent for any reason the answer defaults to *not* permitted, which is the conservative
    reading and the one the envelope itself enforces when no envelope exists.
    """
    for record in selected:
        if (
            record.kind is EvidenceKind.AUTHORITY
            and "meets the autonomy threshold" in record.statement
        ):
            return True
    return False
