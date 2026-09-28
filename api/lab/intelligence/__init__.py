"""The Student Intelligence layer: inspect-first, grounded, provenance-preserving answers.

Everything in this package reads from the finished Focus Intelligence Engine pipeline. It
adds no new behavioural understanding of its own: evidence records are re-stated values from
the engine's own typed results and stage log, never a second opinion. The boundary enforces
the rules that matter here --

* Every fact an answer cites is an :class:`EvidenceRecord` with a class, a provenance, and a
  source. The explanation layer can only restate those records.
* A PREDICTED value is never presented as an observation, and a refusal is never padded into
  a number.
* Role-based access is enforced *before* retrieval, and restricted evidence is surfaced as a
  ``PRIVILEGE_RESTRICTED`` gap rather than silently dropped.
* Missing data is a first-class answer: sufficiency is checked against the evidence that
  survived scope filtering, and an answer that cannot be supported is refused outright.
"""

from api.lab.intelligence.builder import (
    get_profile,
    get_store,
    known_learner_ids,
)
from api.lab.intelligence.models import (
    AnswerType,
    EvidenceClass,
    EvidenceKind,
    EvidenceRecord,
    GapRecord,
    GapType,
    QuestionIntent,
    ReliabilityGrade,
    StudentAnswer,
    StudentIntelligenceProfile,
)
from api.lab.intelligence.query import answer_question
from api.lab.intelligence.rbac import (
    ROLE_SCOPES,
    AccessScope,
    PermissionDeniedError,
    Role,
    enforce_access,
    role_from_header,
    scopes_for,
    visible_evidence,
)

__all__ = [
    "AccessScope",
    "AnswerType",
    "EvidenceClass",
    "EvidenceKind",
    "EvidenceRecord",
    "GapRecord",
    "GapType",
    "PermissionDeniedError",
    "QuestionIntent",
    "ReliabilityGrade",
    "Role",
    "ROLE_SCOPES",
    "StudentAnswer",
    "StudentIntelligenceProfile",
    "answer_question",
    "enforce_access",
    "get_profile",
    "get_store",
    "known_learner_ids",
    "role_from_header",
    "scopes_for",
    "visible_evidence",
]
