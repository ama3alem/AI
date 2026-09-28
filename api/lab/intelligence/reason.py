"""Deep-evidence reasoning for the Student Intelligence layer.

Everything in this module is a *deterministic reconstruction of the learner's own event
stream*. It invents nothing: subject roll-ups aggregate real ``question_started`` /
``question_answered`` events, change state is classified from real per-session accuracy,
and fusion compares subject signals to the global reading. The one honest escape hatch is
:data:`ChangeState.INSUFFICIENT_EVIDENCE` and :data:`EvidenceAgreement.INSUFFICIENT`: when
there is too little history to classify, the layer says so instead of guessing.

The engine never encodes a topic, so subject metadata only exists for the Phase 15 demo
streams (:mod:`api.lab.intelligence.demo`), where the simulator's output is deterministically
re-tagged. Canonical profiles carry no topic metadata and keep the ``subject-na`` gap; the
helpers here are only wired into the demo store.

The output of this module travels as evidence records, not as narrative: the builder embeds
the derived values in :class:`~api.lab.intelligence.models.EvidenceRecord` statements and the
query layer quotes those statements verbatim.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Final

from api.lab.intelligence.models import (
    GAP_PRIORITY,
    ChangeState,
    EvidenceAgreement,
    EvidenceClass,
    EvidenceKind,
    EvidenceRecord,
    GapRecord,
    GapType,
    ReliabilityGrade,
)
from focus_engine.events.types import (
    EventEnvelope,
    EventType,
    QuestionAnsweredPayload,
    QuestionStartedPayload,
)
from focus_engine.schemas.primitives import DataOrigin, Provenance

__all__ = [
    "MIN_SUBJECT_EVIDENCE",
    "MIN_SUBJECT_SESSIONS",
    "SIGNAL_DELTA",
    "STALE_AFTER_DAYS",
    "SUBJECT_TOKENS",
    "ScopeResolution",
    "SessionReading",
    "SubjectRollup",
    "build_change_record",
    "build_conflict_gap",
    "build_fusion_record",
    "build_subject_gaps",
    "build_subject_records",
    "classify_change",
    "read_sessions",
    "resolve_subject",
    "subject_rollups",
]

#: Fewest answered questions before a subject roll-up is allowed to claim anything.
MIN_SUBJECT_EVIDENCE: Final[int] = 8

#: Fewest sessions a subject must appear in before the roll-up is usable.
MIN_SUBJECT_SESSIONS: Final[int] = 3

#: Accuracy movement, in percentage points, treated as a real directional change.
SIGNAL_DELTA: Final[float] = 0.12

#: A subject with no question in this many days is stale.
STALE_AFTER_DAYS: Final[float] = 14.0

#: The subject vocabulary the query layer can resolve from a question.
SUBJECT_TOKENS: Final[tuple[str, ...]] = (
    "algebra",
    "geometry",
    "reading",
    "maths",
    "math",
    "science",
)


# --------------------------------------------------------------------------------------
# Resolution models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeResolution:
    """Result of resolving a question's subject scope against the known vocabulary."""

    requested: str
    """The lowercased question text."""

    resolved: str | None
    """The single resolved subject, or ``None`` when ambiguous or absent."""

    ambiguous: bool
    """True when more than one subject token matched the question."""

    reason: str
    """A statement of how the scope was resolved."""


@dataclass(frozen=True)
class SessionReading:
    """One session, reduced to the numbers the reasoning layer needs."""

    session_id: str
    started: datetime
    answered: int
    correct: int
    response_mean: float
    difficulty_mean: float


@dataclass(frozen=True)
class SubjectRollup:
    """A subject's evidence across the whole stream, with a directional signal."""

    subject: str
    answered: int
    correct: int
    accuracy: float
    recent_count: int
    recent_accuracy: float
    mean_difficulty: float
    last_seen: datetime | None
    sessions_represented: int
    sufficient: bool
    stale: bool
    signal: ChangeState


# --------------------------------------------------------------------------------------
# Stream reduction
# --------------------------------------------------------------------------------------


def _group_sessions(
    events: Sequence[EventEnvelope],
) -> tuple[tuple[str, tuple[EventEnvelope, ...]], ...]:
    """Split a flat stream into sessions ordered by first event (neutral grouping)."""
    buckets: dict[str, list[EventEnvelope]] = {}
    for event in events:
        buckets.setdefault(event.session_id, []).append(event)
    return tuple(
        sorted(
            ((session_id, tuple(group)) for session_id, group in buckets.items()),
            key=lambda item: item[1][0].timestamp,
        )
    )


def read_sessions(events: Sequence[EventEnvelope]) -> tuple[SessionReading, ...]:
    """Reduce a flat event stream into ordered per-session readings.

    Args:
        events: The flat, already-ordered stream.

    Returns:
        One reading per session, in first-event order.
    """
    readings: list[SessionReading] = []
    for session_id, session in _group_sessions(events):
        difficulties: dict[str, float] = {}
        answers: list[tuple[bool, float, str]] = []
        for event in session:
            payload = event.payload
            if event.event_type is EventType.QUESTION_STARTED and isinstance(
                payload, QuestionStartedPayload
            ):
                difficulties[payload.question_id] = (
                    payload.difficulty if payload.difficulty is not None else 0.5
                )
            elif event.event_type is EventType.QUESTION_ANSWERED and isinstance(
                payload, QuestionAnsweredPayload
            ):
                answers.append((payload.correct, payload.response_seconds, payload.question_id))
        if not answers:
            continue
        readings.append(
            SessionReading(
                session_id=session_id,
                started=session[0].timestamp,
                answered=len(answers),
                correct=sum(1 for correct, _, _ in answers),
                response_mean=mean(response for _, response, _ in answers),
                difficulty_mean=mean(
                    difficulties.get(question_id, 0.5) for _, _, question_id in answers
                ),
            )
        )
    return tuple(readings)


def subject_rollups(events: Sequence[EventEnvelope], *, now: datetime) -> tuple[SubjectRollup, ...]:
    """Aggregate every subject's questions across the whole stream.

    Args:
        events: The flat, already-ordered stream.
        now: The instant used to judge staleness.

    Returns:
        One roll-up per subject that appears at least once, ordered alphabetically.
    """
    tagged: dict[str, tuple[str, float]] = {}
    for event in events:
        payload = event.payload
        if event.event_type is EventType.QUESTION_STARTED and isinstance(
            payload, QuestionStartedPayload
        ):
            tagged[payload.question_id] = (
                payload.topic if payload.topic else "unknown",
                payload.difficulty if payload.difficulty is not None else 0.5,
            )

    buckets: dict[str, list[tuple[bool, datetime, str]]] = {}
    sessions_seen: dict[str, set[str]] = {}
    for event in events:
        if event.event_type is not EventType.QUESTION_ANSWERED:
            continue
        payload = event.payload
        if not isinstance(payload, QuestionAnsweredPayload):
            continue
        subject, _ = tagged.get(payload.question_id, ("unknown", 0.5))
        buckets.setdefault(subject, []).append(
            (payload.correct, event.timestamp, payload.question_id)
        )
        sessions_seen.setdefault(subject, set()).add(event.session_id)

    rollups: list[SubjectRollup] = []
    for subject, answers in sorted(buckets.items()):
        answered = len(answers)
        correct = sum(1 for ok, _, _ in answers)
        accuracy = correct / answered
        recent = answers[max(0, answered - max(1, answered // 3)) :]
        recent_count = len(recent)
        recent_accuracy = sum(1 for ok, _, _ in recent) / recent_count if recent_count else 0.0
        session_count = len(sessions_seen.get(subject, set()))
        difficulties = [
            tagged.get(question_id, ("unknown", 0.5))[1] for _, _, question_id in answers
        ]
        last_seen = max((when for _, when, _ in answers), default=None)
        rollups.append(
            SubjectRollup(
                subject=subject,
                answered=answered,
                correct=correct,
                accuracy=accuracy,
                recent_count=recent_count,
                recent_accuracy=recent_accuracy,
                mean_difficulty=mean(difficulties),
                last_seen=last_seen,
                sessions_represented=session_count,
                sufficient=answered >= MIN_SUBJECT_EVIDENCE
                and session_count >= MIN_SUBJECT_SESSIONS,
                stale=last_seen is not None
                and (now - last_seen).total_seconds() / 86400.0 > STALE_AFTER_DAYS,
                signal=_subject_signal(accuracy, recent_accuracy),
            )
        )
    return tuple(rollups)


def _subject_signal(overall: float, recent: float) -> ChangeState:
    delta = recent - overall
    if delta <= -SIGNAL_DELTA:
        return ChangeState.DECLINING
    if delta >= SIGNAL_DELTA:
        return ChangeState.IMPROVING
    return ChangeState.STABLE


def classify_change(events: Sequence[EventEnvelope]) -> tuple[ChangeState, dict[str, float]]:
    """Classify a learner's overall change from the true per-session accuracy vector.

    Args:
        events: The flat, already-ordered stream.

    Returns:
        The change state and the numbers that support it (early/late accuracy, response,
        and difficulty means).
    """
    readings = read_sessions(events)
    accuracies = [r.correct / r.answered for r in readings if r.answered]
    counts = len(accuracies)
    if counts < 4:
        return ChangeState.INSUFFICIENT_EVIDENCE, {}

    q = [mean(accuracies[i * counts // 4 : (i + 1) * counts // 4]) for i in range(4)]
    detail = {
        "sessions": counts,
        "accuracy_q1": q[0],
        "accuracy_q4": q[3],
        "response_early": _mean_of((r.response_mean for r in readings[: counts // 2]), 0.0),
        "response_late": _mean_of((r.response_mean for r in readings[counts // 2 :]), 0.0),
        "difficulty_early": _mean_of((r.difficulty_mean for r in readings[: counts // 2]), 0.0),
        "difficulty_late": _mean_of((r.difficulty_mean for r in readings[counts // 2 :]), 0.0),
    }

    if q[0] - q[1] >= SIGNAL_DELTA and q[2] - q[3] <= -SIGNAL_DELTA:
        return ChangeState.RECOVERING, detail
    if q[3] <= q[0] - SIGNAL_DELTA:
        return ChangeState.DECLINING, detail
    if q[3] >= q[0] + SIGNAL_DELTA:
        return ChangeState.IMPROVING, detail
    if pstdev(accuracies) > 0.25:
        return ChangeState.VOLATILE, detail
    return ChangeState.STABLE, detail


def _mean_of(values: Iterable[float], default: float) -> float:
    values = tuple(values)
    return mean(values) if values else default


# --------------------------------------------------------------------------------------
# Record builders (invoked from the demo profile path)
# --------------------------------------------------------------------------------------


def build_subject_records(
    learner_id: str, rollups: tuple[SubjectRollup, ...]
) -> tuple[EvidenceRecord, ...]:
    """One EVIDENCE-scoped record per subject with enough questions to describe."""
    records: list[EvidenceRecord] = []
    for rollup in rollups:
        if rollup.subject == "unknown" or rollup.answered < MIN_SUBJECT_EVIDENCE:
            continue
        records.append(
            EvidenceRecord(
                evidence_id=f"ev-{learner_id}-subject-{rollup.subject}",
                learner_id=learner_id,
                kind=EvidenceKind.SUBJECT,
                evidence_class=EvidenceClass.DERIVED,
                reliability=(
                    ReliabilityGrade.HIGH if rollup.sufficient else ReliabilityGrade.MEDIUM
                ),
                statement=(
                    f"Subject '{rollup.subject}': {rollup.answered} answered questions "
                    f"across {rollup.sessions_represented} sessions, accuracy "
                    f"{rollup.accuracy:.3f} overall vs {rollup.recent_accuracy:.3f} recent, "
                    f"mean difficulty {rollup.mean_difficulty:.3f}."
                ),
                provenance=Provenance.OBSERVED.value,
                data_origin=DataOrigin.SYNTHETIC.value,
                source="api.lab.intelligence.reason.build_subject_records",
                subject=rollup.subject,
                note=(
                    "A roll-up of behavioural events tagged with this subject's topic; it "
                    "is a reconstruction, not an assessment."
                ),
            )
        )
    return tuple(records)


def build_change_record(events: Sequence[EventEnvelope], learner_id: str) -> EvidenceRecord:
    """One CHANGE record describing how observed behaviour moved across time."""
    state, detail = classify_change(events)
    if not detail:
        reading = f"Change state {state.value}."
    else:
        reading = (
            f"Change state {state.value} over {detail['sessions']} sessions: accuracy moved "
            f"from {detail['accuracy_q1']:.3f} to {detail['accuracy_q4']:.3f}, response "
            f"time from {detail['response_early']:.1f}s to {detail['response_late']:.1f}s, "
            f"content difficulty from {detail['difficulty_early']:.2f} to "
            f"{detail['difficulty_late']:.2f}."
        )
    if state is ChangeState.INSUFFICIENT_EVIDENCE:
        reading += " Too little history to classify; no direction is claimed."
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-change",
        learner_id=learner_id,
        kind=EvidenceKind.CHANGE,
        evidence_class=EvidenceClass.DERIVED,
        reliability=(
            ReliabilityGrade.HIGH
            if state is not ChangeState.INSUFFICIENT_EVIDENCE
            else ReliabilityGrade.MEDIUM
        ),
        statement=reading,
        provenance=Provenance.OBSERVED.value,
        data_origin=DataOrigin.SYNTHETIC.value,
        source="api.lab.intelligence.reason.build_change_record",
        note="A reading of the learner's own event stream, not a cause and not an assessment.",
    )


def build_fusion_record(learner_id: str, rollups: tuple[SubjectRollup, ...]) -> EvidenceRecord:
    """One FUSION record reporting how subject signals line up with a global change."""
    usable = [r for r in rollups if r.sufficient]
    if not usable:
        return EvidenceRecord(
            evidence_id=f"ev-{learner_id}-fusion",
            learner_id=learner_id,
            kind=EvidenceKind.FUSION,
            evidence_class=EvidenceClass.UNAVAILABLE,
            reliability=ReliabilityGrade.MEDIUM,
            statement=(
                "Subject fusion insufficient: no subject has enough evidence to compare, so "
                "no agreement or conflict is claimed."
            ),
            provenance=Provenance.OBSERVED.value,
            data_origin=DataOrigin.SYNTHETIC.value,
            source="api.lab.intelligence.reason.build_fusion_record",
        )
    declining = [r.subject for r in usable if r.signal is ChangeState.DECLINING]
    improving = [r.subject for r in usable if r.signal is ChangeState.IMPROVING]
    if declining and improving:
        agreement = EvidenceAgreement.CONTRADICTORY
    elif declining:
        agreement = EvidenceAgreement.SUPPORTING
    else:
        agreement = EvidenceAgreement.NEUTRAL
    summary = "; ".join(
        f"{r.subject}: {r.signal.value}" for r in sorted(usable, key=lambda r: r.subject)
    )
    return EvidenceRecord(
        evidence_id=f"ev-{learner_id}-fusion",
        learner_id=learner_id,
        kind=EvidenceKind.FUSION,
        evidence_class=EvidenceClass.DERIVED,
        reliability=ReliabilityGrade.MEDIUM,
        statement=(
            f"Subject fusion agreement {agreement.value}: {summary}. "
            + (
                "The layer reports the disagreement rather than averaging it away."
                if agreement is EvidenceAgreement.CONTRADICTORY
                else "Subject signals do not contradict the wider reading."
            )
        ),
        provenance=Provenance.OBSERVED.value,
        data_origin=DataOrigin.SYNTHETIC.value,
        source="api.lab.intelligence.reason.build_fusion_record",
    )


def build_subject_gaps(
    learner_id: str, rollups: tuple[SubjectRollup, ...]
) -> tuple[GapRecord, ...]:
    """Gaps for subjects that cannot support a claim or that went silent."""
    gaps: list[GapRecord] = []
    for rollup in rollups:
        if rollup.subject == "unknown":
            continue
        if not rollup.sufficient:
            gaps.append(
                GapRecord(
                    gap_id=f"gap-{learner_id}-subject-{rollup.subject}-coverage",
                    learner_id=learner_id,
                    gap_type=GapType.LOW_COVERAGE,
                    description=(
                        f"Subject '{rollup.subject}': only {rollup.answered} answered "
                        f"questions across {rollup.sessions_represented} sessions; below "
                        f"the {MIN_SUBJECT_EVIDENCE}-question minimum, so no subject-level "
                        "claim is made."
                    ),
                    priority=GAP_PRIORITY[GapType.LOW_COVERAGE],
                    subject=rollup.subject,
                    remedy="More questions tagged with this subject.",
                )
            )
        elif rollup.stale:
            gaps.append(
                GapRecord(
                    gap_id=f"gap-{learner_id}-subject-{rollup.subject}-stale",
                    learner_id=learner_id,
                    gap_type=GapType.STALE,
                    description=(
                        f"Subject '{rollup.subject}' evidence is stale: no question for "
                        "over 14 days, so recent accuracy describes the last known run, "
                        "not now."
                    ),
                    priority=GAP_PRIORITY[GapType.STALE],
                    subject=rollup.subject,
                    remedy="A fresh session tagged with this subject.",
                )
            )
    return tuple(sorted(gaps, key=lambda gap: gap.priority))


def build_conflict_gap(learner_id: str, fusion: EvidenceRecord | None) -> GapRecord | None:
    """A CONFLICTING gap when subject signals contradict one another."""
    if fusion is None or "contradictory" not in fusion.statement:
        return None
    return GapRecord(
        gap_id=f"gap-{learner_id}-subject-conflict",
        learner_id=learner_id,
        gap_type=GapType.CONFLICTING,
        description=(
            "Subject signals contradict one another: distinct subjects moved in opposite "
            "directions, so a single overall story cannot be told without saying which "
            "subject it belongs to."
        ),
        priority=GAP_PRIORITY[GapType.CONFLICTING],
        remedy="Until the signals converge, answer subject-by-subject.",
    )


def resolve_subject(question: str) -> ScopeResolution:
    """Resolve which subject a question asks about.

    Args:
        question: The author's question, lowercased before matching.

    Returns:
        The resolution. ``resolved`` is ``None`` when no subject is named or more than one
        is named; both cases settle on "all subjects", for different reasons.
    """
    lowered = question.lower()
    matched = [token for token in SUBJECT_TOKENS if token in lowered]
    if not matched:
        return ScopeResolution(
            requested=question,
            resolved=None,
            ambiguous=False,
            reason="No subject was named in the question, so all subjects apply.",
        )
    if len(matched) > 1:
        return ScopeResolution(
            requested=question,
            resolved=None,
            ambiguous=True,
            reason="More than one subject was named, so all subjects apply.",
        )
    return ScopeResolution(
        requested=question,
        resolved=matched[0],
        ambiguous=False,
        reason=f"Subject scope resolved to '{matched[0]}'.",
    )
