"""Student Intelligence layer tests: grounded answers, RBAC-before-retrieval, honesty.

The layer being tested adds no understanding of its own — it re-states the engine's typed
outputs under a role scope. So the tests that matter here are anti-slop invariants rather
than behavioural checks:

* every line of every composed answer is either a verbatim engine statement, a verbatim
  gap description, or a line from the layer's own fixed template vocabulary — nothing else
  may appear (``test_no_answer_invents_words``);
* a ``predicted`` record is never dressed up as an observed fact, and an answer that leans
  on one declares itself ``prediction`` type with low confidence;
* access is enforced before retrieval, a learner can only ever see themselves, restricted
  evidence surfaces as a ``PRIVILEGE_RESTRICTED`` gap instead of disappearing, and an
  unknown id is refused without naming the set of known ids;
* missing data is first-class: what cannot be supported is refused with the difference
  between permission and data made explicit;
* the whole store rebuilds byte-for-byte under the fixed clock.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from api.lab.app import app
from api.lab.intelligence.builder import (
    build_store,
    get_profile,
    known_learner_ids,
)
from api.lab.intelligence.models import (
    EvidenceClass,
    EvidenceKind,
    GapType,
    QuestionIntent,
)
from api.lab.intelligence.query import (
    _FIXED_LINES,
    answer_question,
    classify_intent,
)
from api.lab.intelligence.rbac import (
    AccessScope,
    PermissionDeniedError,
    Role,
    enforce_access,
    restricted_gaps,
    role_from_header,
    scopes_for,
    visible_evidence,
)
from fastapi.testclient import TestClient

ALL_IDS = (
    "lab-stable-0001",
    "lab-decline-0001",
    "lab-rapid-0001",
    "lab-recovery-0001",
    "lab-thin-0001",
)

QUESTIONS = (
    "Tell me about this student",
    "Why is the behavioural state what it is",
    "What data is missing for this learner",
    "Can the AI intervene autonomously",
    "qgjqw lgj",
)


@pytest.fixture(scope="module", name="store")
def _store():
    return build_store()


def _expected_restricted_kinds(profile, scopes):
    return {
        kind
        for kind in EvidenceKind
        if kind not in {r.kind for r in visible_evidence(profile, scopes)}
        and profile.evidence_of(kind)
    }


# ---------------------------------------------------------------------------
# The store itself
# ---------------------------------------------------------------------------


def test_store_covers_every_scenario_learner(store):
    assert tuple(store) == known_learner_ids()
    assert len(store) == 5


def test_deterministic_rebuild():
    first = build_store()
    second = build_store()
    assert {k: v.model_dump() for k, v in first.items()} == {
        k: v.model_dump() for k, v in second.items()
    }


def test_all_profiles_carry_evidence_and_synthetic_notice(store):
    for learner_id, profile in store.items():
        assert profile.learner_id == learner_id
        assert profile.evidence
        assert profile.notice.warning
        assert "synthetic" in profile.notice.warning.lower()


def test_every_record_has_its_standing_fields(store):
    for profile in store.values():
        for record in profile.evidence:
            assert record.evidence_class
            assert record.kind
            assert record.reliability
            assert record.provenance
            assert record.source
            assert record.statement
            assert record.statement.strip()


def test_evidence_of_filters_by_kind(store):
    profile = store["lab-stable-0001"]
    assert all(r.kind is EvidenceKind.EVENT for r in profile.evidence_of(EvidenceKind.EVENT))
    assert profile.evidence_of(EvidenceKind.EVENT)


def test_gaps_are_prioritised_in_contract_order(store):
    for profile in store.values():
        priorities = [gap.priority for gap in profile.gaps]
        assert priorities == sorted(priorities)
        assert priorities[0] == 1
        assert {gap.gap_type for gap in profile.gaps}.isdisjoint({GapType.PRIVILEGE_RESTRICTED})


def test_thin_learner_gets_data_gaps_not_fabrication(store):
    profile = store["lab-thin-0001"]
    kinds = {gap.gap_type for gap in profile.gaps}
    assert GapType.MISSING in kinds
    assert GapType.UNVALIDATED in kinds
    temporal = profile.evidence_of(EvidenceKind.TEMPORAL)[0]
    assert "insufficient_data" in temporal.statement
    assert "no score" not in temporal.statement.lower()


def test_eduflow_native_attributes_never_invented(store):
    for profile in store.values():
        for gap in profile.gaps:
            if "grade" in gap.description or "class" in gap.description:
                assert "UNAVAILABLE" in gap.description
        gaps_text = "\n".join(gap.description for gap in profile.gaps).lower()
        assert "eduflow-native" in gaps_text


def test_predicted_never_tagged_as_observed(store):
    for profile in store.values():
        admin = answer_question(profile, "Tell me about this student", scopes_for(Role.ADMIN))
        for record in admin.evidence:
            if record.evidence_class is EvidenceClass.PREDICTED:
                assert record.evidence_class.value == "predicted"
                assert "[predicted/" in admin.answer


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Tell me about this student", QuestionIntent.PROFILE),
        ("why is this happening", QuestionIntent.EXPLAIN),
        ("what data is missing here", QuestionIntent.MISSING),
        ("can the AI act on its own", QuestionIntent.AUTONOMY),
        ("banana phone", QuestionIntent.UNKNOWN),
    ],
)
def test_intent_classification(question, intent):
    assert classify_intent(question) is intent


# ---------------------------------------------------------------------------
# Answering: the contracts
# ---------------------------------------------------------------------------


def test_answer_contract_is_complete(store):
    profile = store["lab-stable-0001"]
    answer = answer_question(profile, "Tell me about this student", scopes_for(Role.ADMIN))
    assert answer.student == profile.learner_id
    assert answer.question == "Tell me about this student"
    assert answer.intent is QuestionIntent.PROFILE
    assert answer.answer_type is not None
    assert answer.confidence is not None
    assert answer.evidence_quality is not None
    assert answer.answer
    assert answer.data_gaps
    assert answer.limitations
    assert answer.sources
    assert answer.engine_versions == dict(profile.engine_versions)
    assert answer.notice is profile.notice
    assert answer.generated_at == profile.built_at
    assert answer.generated_at.tzinfo == UTC


def test_all_answers_are_grounded_in_profile(store):
    for learner_id, profile in store.items():
        for question in QUESTIONS:
            for scopes in (scopes_for(Role.ADMIN), scopes_for(Role.STUDENT)):
                answer = answer_question(profile, question, scopes)
                if answer.refused:
                    continue
                for record in answer.evidence:
                    assert record.learner_id == learner_id


def test_answer_contains_no_invented_words(store):
    for profile in store.values():
        for question in QUESTIONS:
            for scopes in (scopes_for(Role.ADMIN), scopes_for(Role.STUDENT)):
                answer = answer_question(profile, question, scopes)
                if answer.refused:
                    continue
                allowed_bullets = {
                    f"- {r.statement} [{r.evidence_class.value}/{r.provenance}, "
                    f"reliability {r.reliability.value}]"
                    for r in answer.evidence
                }
                allowed_bullets |= {
                    f"- {g.description} ({g.gap_type.value}, priority {g.priority})"
                    for g in answer.data_gaps
                }
                allowed_bullets |= {f"- {limitation}" for limitation in answer.limitations}
                for line in answer.answer.splitlines():
                    if not line.strip():
                        continue
                    assert (
                        line in _FIXED_LINES
                        or line == profile.title
                        or line.startswith("Intent: ")
                        or line in allowed_bullets
                    ), f"invented line in {question!r} for {profile.learner_id}: {line!r}"


def test_prediction_evidence_forces_prediction_answer_type(store):
    for profile in store.values():
        answer = answer_question(profile, "Tell me about this student", scopes_for(Role.ADMIN))
        if any(r.evidence_class is EvidenceClass.PREDICTED for r in answer.evidence):
            assert answer.answer_type.value == "prediction"
            assert answer.confidence.value == "low"
        else:
            assert answer.answer_type.value in {"fact", "derived_fact"}


def test_confirmed_privacy_prediction_caveat_carried(store):
    profile = store["lab-stable-0001"]
    admin = answer_question(profile, "Tell me about this student", scopes_for(Role.ADMIN))
    student = answer_question(profile, "Tell me about this student", scopes_for(Role.STUDENT))
    assert any(
        "PREDICTED values are model outputs, never measurements" in limitation
        for limitation in admin.limitations
    )
    assert not any("PREDICTED" in limitation for limitation in student.limitations)


def test_autonomy_answer_states_the_threshold(store):
    profile = store["lab-decline-0001"]
    answer = answer_question(profile, "Can the AI intervene autonomously", scopes_for(Role.ADMIN))
    assert answer.answer_type.value == "recommendation"
    assert (
        "does not meet the autonomy threshold" in answer.answer
        or "envelope's own permitted actions apply" in answer.answer
    )


def test_sources_name_the_module_that_produced_the_record(store):
    for profile in store.values():
        answer = answer_question(profile, "What data is missing here", scopes_for(Role.ADMIN))
        for source in answer.sources:
            assert source
            assert "api" in source or "focus_engine" in source


# ---------------------------------------------------------------------------
# Sufficiency and refusal
# ---------------------------------------------------------------------------


def test_unknown_intent_refused_outright(store):
    profile = store["lab-stable-0001"]
    answer = answer_question(profile, "qgjqw lgj", scopes_for(Role.ADMIN))
    assert answer.refused
    assert answer.answer_type is None
    assert answer.confidence is None
    assert "does not map to a supported intent" in answer.refusal_reason


def test_student_explain_refused_as_privilege_restricted(store):
    profile = store["lab-stable-0001"]
    answer = answer_question(
        profile, "Why is the behavioural state what it is", scopes_for(Role.STUDENT)
    )
    assert answer.refused
    assert "restricted under the requested role" in answer.refusal_reason
    assert any(gap.gap_type is GapType.PRIVILEGE_RESTRICTED for gap in answer.data_gaps)


def test_student_autonomy_refused_without_authority(store):
    profile = store["lab-stable-0001"]
    answer = answer_question(profile, "Can the AI intervene autonomously", scopes_for(Role.STUDENT))
    assert answer.refused
    assert "authority" in answer.refusal_reason


def test_sufficiency_distinguishes_permission_from_data(store):
    profile = store["lab-thin-0001"]
    explain = answer_question(
        profile, "Why is the behavioural state what it is", scopes_for(Role.STUDENT)
    )
    assert "restricted under the requested role" in explain.refusal_reason


# ---------------------------------------------------------------------------
# Role-based access control, before retrieval
# ---------------------------------------------------------------------------


def test_role_from_header_maps_and_refuses():
    assert role_from_header("student") is Role.STUDENT
    assert role_from_header(" TEACHER ") is Role.TEACHER
    with pytest.raises(PermissionDeniedError):
        role_from_header(None)
    with pytest.raises(PermissionDeniedError):
        role_from_header("wizard")


def test_student_needs_identity_header():
    with pytest.raises(PermissionDeniedError):
        enforce_access(Role.STUDENT, "lab-stable-0001", None)


def test_student_cannot_read_another_learner():
    with pytest.raises(PermissionDeniedError):
        enforce_access(Role.STUDENT, "lab-decline-0001", "lab-stable-0001")


def test_student_may_read_self():
    enforce_access(Role.STUDENT, "lab-stable-0001", "lab-stable-0001")


def test_staff_ignore_the_identity_header():
    enforce_access(Role.TEACHER, "lab-decline-0001", None)
    enforce_access(Role.ADMIN, "lab-decline-0001", "someone-else")


def test_student_scopes_exclude_engine_internals():
    student_scopes = scopes_for(Role.STUDENT)
    assert (
        student_scopes
        & {
            AccessScope.FOCUS_ENGINE,
            AccessScope.AUTHORITY,
            AccessScope.INTERVENTIONS,
            AccessScope.OUTCOMES,
        }
        == set()
    )


def test_visible_evidence_respects_scopes(store):
    profile = store["lab-stable-0001"]
    student_scopes = scopes_for(Role.STUDENT)
    for record in visible_evidence(profile, student_scopes):
        assert record.kind not in {
            EvidenceKind.PREDICTION,
            EvidenceKind.AUTHORITY,
            EvidenceKind.POLICY,
            EvidenceKind.INTERVENTION,
            EvidenceKind.OUTCOME,
            EvidenceKind.EVALUATION,
        }


def test_restricted_gaps_exactly_the_withheld_kinds(store):
    for profile in store.values():
        student_scopes = scopes_for(Role.STUDENT)
        gaps = restricted_gaps(profile, student_scopes)
        expected = _expected_restricted_kinds(profile, student_scopes)
        assert {gap.description.split(" evidence ", 1)[0] for gap in gaps} == {
            kind.value for kind in expected
        }
        assert all(
            gap.gap_type is GapType.PRIVILEGE_RESTRICTED and gap.priority == 8 for gap in gaps
        )


def test_restricted_gaps_never_for_hollow_kinds(store):
    admin_scopes = scopes_for(Role.ADMIN)
    for profile in store.values():
        assert restricted_gaps(profile, admin_scopes) == ()


def test_unknown_learner_refusal_does_not_leak_known_ids():
    with pytest.raises(ValueError) as exc_info:
        get_profile("lab-no-such-learner")
    message = str(exc_info.value)
    assert "lab-no-such-learner" in message
    for known in ALL_IDS:
        assert known not in message


# ---------------------------------------------------------------------------
# HTTP boundary
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_http_missing_role_is_400(client):
    response = client.get("/api/students")
    assert response.status_code == 400


def test_http_unknown_role_is_400(client):
    response = client.get("/api/students", headers={"X-Student-Role": "wizard"})
    assert response.status_code == 400


def test_http_admin_lists_all_students(client):
    response = client.get("/api/students", headers={"X-Student-Role": "admin"})
    assert response.status_code == 200
    assert [row["learner_id"] for row in response.json()] == list(ALL_IDS)


def test_http_student_sees_only_self(client):
    response = client.get(
        "/api/students",
        headers={"X-Student-Role": "student", "X-Student-Id": "lab-stable-0001"},
    )
    assert response.status_code == 200
    assert [row["learner_id"] for row in response.json()] == ["lab-stable-0001"]


def test_http_student_other_learner_is_403(client):
    response = client.get(
        "/api/students/lab-decline-0001/intelligence",
        headers={"X-Student-Role": "student", "X-Student-Id": "lab-stable-0001"},
    )
    assert response.status_code == 403


def test_http_student_self_intelligence_is_200_and_scoped(client):
    response = client.get(
        "/api/students/lab-stable-0001/intelligence",
        headers={"X-Student-Role": "student", "X-Student-Id": "lab-stable-0001"},
    )
    assert response.status_code == 200
    payload = response.json()
    kinds = {record["kind"] for record in payload["evidence"]}
    assert "prediction" not in kinds
    assert "authority" not in kinds


def test_http_query_answer_is_grounded(client):
    response = client.post(
        "/api/students/lab-stable-0001/query",
        headers={"X-Student-Role": "admin"},
        json={"question": "What data is missing for this learner"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["refused"] is False
    assert payload["answer_type"] == "fact"
    assert payload["student"] == "lab-stable-0001"


def test_http_gaps_endpoint_orders_by_priority(client):
    response = client.get(
        "/api/students/lab-thin-0001/gaps",
        headers={"X-Student-Role": "admin"},
    )
    assert response.status_code == 200
    priorities = [gap["priority"] for gap in response.json()]
    assert priorities == sorted(priorities)


def test_http_unknown_learner_is_422(client):
    response = client.get(
        "/api/students/lab-no-such-learner/evidence",
        headers={"X-Student-Role": "admin"},
    )
    assert response.status_code == 422


def test_replayed_answers_are_identical(store):
    profile = store["lab-stable-0001"]
    first = answer_question(profile, "Tell me about this student", scopes_for(Role.ADMIN))
    second = answer_question(profile, "Tell me about this student", scopes_for(Role.ADMIN))
    assert first.model_dump() == second.model_dump()
    assert isinstance(first.generated_at, datetime)
    assert first.generated_at.tzinfo == UTC
