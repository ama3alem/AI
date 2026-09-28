"""Role-based access for the Student Intelligence layer, enforced before retrieval.

The load-bearing rule of this module is the order of operations: *the access check runs
before any evidence is read*. A request whose role is not entitled to a learner is refused
before the store is even consulted, so nothing about the contents of a restricted profile
leaks through timing, error text, or a partially built response.

**A learner may only ever speak for themselves.** A ``student`` role is refused anything
except the pseudonymous learner it names; there is no student that can read its classmates.
The requester's own identity travels in a header and must match exactly, and because ids are
pseudonymous this is a matching check, never a name lookup.

**Restriction is reported, not hidden.** Evidence a role is not entitled to see is *not*
silently dropped. It is surfaced as a ``PRIVILEGE_RESTRICTED`` gap, so a reader who hits the
limit can tell happened-by-role from happened-by-data: one responds to changing permissions
and the other to collecting more evidence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from api.lab.intelligence.models import (
    EvidenceKind,
    EvidenceRecord,
    GapRecord,
    GapType,
    StudentIntelligenceProfile,
)

__all__ = [
    "ALL_SCOPES",
    "AccessScope",
    "KIND_SCOPE",
    "PermissionDeniedError",
    "Role",
    "ROLE_SCOPES",
    "enforce_access",
    "role_from_header",
    "scopes_for",
    "restricted_gaps",
    "visible_evidence",
]


class Role(StrEnum):
    """Who is asking, and therefore what they may see. A header value, not a claim."""

    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"


class AccessScope(StrEnum):
    """One thing a role may see. Scopes are granted in sets, never graded."""

    PROFILE = "profile"
    EVIDENCE = "evidence"
    FOCUS_ENGINE = "focus_engine"
    INTERVENTIONS = "interventions"
    OUTCOMES = "outcomes"
    AUTHORITY = "authority"
    MISSING_DATA = "missing_data"
    SOURCES = "sources"


ALL_SCOPES: Final[frozenset[AccessScope]] = frozenset(AccessScope)

#: Which evidence kind belongs to which scope. One kind maps to exactly one scope, so a
#: role's visibility of the profile is fully determined by ROLE_SCOPES plus this table.
KIND_SCOPE: Final[dict[EvidenceKind, AccessScope]] = {
    EvidenceKind.PROVENANCE: AccessScope.SOURCES,
    EvidenceKind.EVENT: AccessScope.EVIDENCE,
    EvidenceKind.CONTEXT: AccessScope.EVIDENCE,
    EvidenceKind.FEATURE: AccessScope.FOCUS_ENGINE,
    EvidenceKind.BASELINE: AccessScope.FOCUS_ENGINE,
    EvidenceKind.TEMPORAL: AccessScope.FOCUS_ENGINE,
    EvidenceKind.PREDICTION: AccessScope.FOCUS_ENGINE,
    EvidenceKind.UNCERTAINTY: AccessScope.FOCUS_ENGINE,
    EvidenceKind.POLICY: AccessScope.FOCUS_ENGINE,
    EvidenceKind.AUTHORITY: AccessScope.AUTHORITY,
    EvidenceKind.INTERVENTION: AccessScope.INTERVENTIONS,
    EvidenceKind.OUTCOME: AccessScope.OUTCOMES,
    EvidenceKind.EVALUATION: AccessScope.OUTCOMES,
    EvidenceKind.SUBJECT: AccessScope.EVIDENCE,
    EvidenceKind.CHANGE: AccessScope.FOCUS_ENGINE,
    EvidenceKind.FUSION: AccessScope.FOCUS_ENGINE,
}

#: A learner sees their own evidence and sources and their missing data. Everything that
#: concerns the engine's internals -- the fitted model, the milestones, the delivery and its
#: outcome, the authority envelope -- is withheld and surfaced as restricted gaps instead.
ROLE_SCOPES: Final[dict[Role, frozenset[AccessScope]]] = {
    Role.STUDENT: frozenset(
        {
            AccessScope.PROFILE,
            AccessScope.EVIDENCE,
            AccessScope.MISSING_DATA,
            AccessScope.SOURCES,
        }
    ),
    Role.TEACHER: ALL_SCOPES,
    Role.ADMIN: ALL_SCOPES,
}


class PermissionDeniedError(Exception):
    """A role asked for a learner or an evidence scope it is not entitled to.

    Carries nothing about the denied content; the message names the role and the refusal
    rule, never what was behind the barrier.
    """


def role_from_header(value: str | None) -> Role:
    """Map the ``X-Student-Role`` header value to a role.

    Args:
        value: The raw header value, or ``None`` when the header was absent.

    Returns:
        The role.

    Raises:
        PermissionDeniedError: When the header is absent or names an unknown role. An
            absent role is refused rather than defaulted, because this repository never
            grants the most permissive option on a missing input.
    """
    if value is None:
        raise PermissionDeniedError("Missing X-Student-Role header; a role must be stated.")
    try:
        return Role(value.strip().lower())
    except ValueError:
        raise PermissionDeniedError(
            f"Unknown role {value!r}. A role must be one of "
            f"{', '.join(role.value for role in Role)}."
        ) from None


def scopes_for(role: Role) -> frozenset[AccessScope]:
    """Return the scopes a role is entitled to.

    Args:
        role: The role.

    Returns:
        Its granted scope set.
    """
    return ROLE_SCOPES[role]


def enforce_access(role: Role, requested_learner_id: str, requester_learner_id: str | None) -> None:
    """Enforce the role-to-learner rule *before* retrieval.

    Args:
        role: The requesting role.
        requested_learner_id: The learner the request asks about.
        requester_learner_id: The learner the requester claims to be, from the identity
            header. Required for and only for a ``student`` role.

    Raises:
        PermissionDeniedError: When a student asks about any learner other than itself
            (or fails to state an identity), or when the requested learner id is not a
            learner this layer knows -- refused without naming the known set, so that an
            id probe cannot learn what ids exist.
    """
    if role is Role.STUDENT:
        if requester_learner_id is None:
            raise PermissionDeniedError(
                "A student request must carry X-Student-Id naming the requesting learner. "
                "A student may only read their own record."
            )
        if requester_learner_id != requested_learner_id:
            raise PermissionDeniedError(
                f"A student may only query their own learner_id. Refused: the request asked "
                f"about {requested_learner_id!r} while identifying as {requester_learner_id!r}."
            )


def visible_evidence(
    profile: StudentIntelligenceProfile, scopes: frozenset[AccessScope]
) -> tuple[EvidenceRecord, ...]:
    """Filter a profile's evidence to what the scopes permit, preserving stored order.

    Args:
        profile: The canonical profile.
        scopes: The granted scope set.

    Returns:
        The permitted records, in stored order.
    """
    return tuple(record for record in profile.evidence if KIND_SCOPE[record.kind] in scopes)


def restricted_gaps(
    profile: StudentIntelligenceProfile, scopes: frozenset[AccessScope]
) -> tuple[GapRecord, ...]:
    """Report the evidence kinds withheld by ``scopes`` as restricted gaps.

    A restricted gap is only emitted for a kind whose evidence actually exists in the
    profile: a kind that never produced evidence is a data gap, not a permission gap, and
    the two must not be confusable.

    Args:
        profile: The canonical profile.
        scopes: The granted scope set.

    Returns:
        One ``PRIVILEGE_RESTRICTED`` gap per withheld kind that holds evidence.
    """
    gaps: list[GapRecord] = []
    for kind in (k for k in EvidenceKind if KIND_SCOPE[k] not in scopes):
        if profile.evidence_of(kind):
            scope_value = KIND_SCOPE[kind].value
            gaps.append(
                GapRecord(
                    gap_id=f"gap-{profile.learner_id}-privilege-{kind.value}",
                    learner_id=profile.learner_id,
                    gap_type=GapType.PRIVILEGE_RESTRICTED,
                    description=(
                        f"{kind.value} evidence for this learner exists but is restricted "
                        f"under the requested role (access scope {scope_value!r})."
                    ),
                    priority=8,
                    remedy="A role granted the required scope can see this evidence.",
                )
            )
    return tuple(gaps)
