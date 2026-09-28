"""The lab's FastAPI server — thin, development-only, local-only.

This server exists solely to host the visual pipeline. It is not a production API, and the
``/api/`` prefix is a route convention rather than a versioning claim.

**Determinism.** Every request uses a fixed clock at 2026-06-15T00:00:00 UTC. The choice
of that timestamp is arbitrary; what matters is that it is chosen before any event is
examined, so it cannot leak information about the events it timestamps.

**Identity belongs to the events.** A pasted stream is analysed under the ``learner_id``
and ``session_id`` its own envelopes declare, and a caller whose stated identity
disagrees is rejected with a 422 naming both values. The engine refuses such a
contradiction independently, several layers down; this boundary rejects it first, because a
caller that has pasted one learner's events under another learner's id has a bug, and
telling them so is more useful than a server fault.

**Validation.** Nothing is repaired, reordered, or dropped silently. Every finding is
returned as a structured array for the UI to display, and a dataset that cannot proceed is
refused with its findings attached rather than analysed in a degraded form.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from api.lab.adapter import LabBundle, run_analysis
from api.lab.intelligence import (
    AccessScope,
    PermissionDeniedError,
    Role,
    answer_question,
    enforce_access,
    get_profile,
    get_store,
    role_from_header,
    scopes_for,
    visible_evidence,
)
from api.lab.intelligence.builder import build_demo_store, demo_learner_ids
from api.lab.intelligence.models import StudentIntelligenceProfile
from api.lab.intelligence.rbac import restricted_gaps
from api.lab.scenarios import (
    DEFAULT_SAMPLE_SEED,
    SCENARIO_IDS,
    canonical_example,
    describe,
    flatten,
    generate_sample_session,
    generate_scenario,
    get_scenario,
)
from api.lab.validation import coerce_events, validate_dataset
from focus_engine.events.types import EventEnvelope
from focus_engine.schemas.primitives import LearnerId, SessionId
from focus_engine.utils.clock import FixedClock

app = FastAPI(
    title="Focus Engine Lab",
    description="Development-only local visual pipeline for the Focus Engine.",
    docs_url="/docs",
)

_DETERMINISTIC_CLOCK = FixedClock(datetime(2026, 6, 15, tzinfo=UTC))

_STATIC_DIR = Path(__file__).parent / "static"


def _resolve_scenario(scenario_id: str) -> Any:
    """Look up a scenario, translating an unknown id into a 422 rather than a 500.

    Args:
        scenario_id: The identifier the caller supplied.

    Returns:
        The matching scenario definition.

    Raises:
        HTTPException: With status 422 when the id is not one of the shipped scenarios.
            The library raises :class:`ValueError`, which is correct for a library call
            and wrong for an HTTP boundary: an unrecognised id is a caller mistake, and
            surfacing it as a server fault would tell an operator the lab is broken when
            the only thing wrong was the request.
    """
    try:
        return get_scenario(scenario_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class ScenarioRequest(BaseModel):
    scenario_id: str = Field(
        ..., description="One of the pre-loaded scenario IDs.", pattern=r"^[a-z_]+$"
    )


class PasteRequest(BaseModel):
    """A pasted event stream.

    ``learner_id`` and ``session_id`` use the engine's own annotated identifier types
    rather than a looser string. The engine requires at least eight characters drawn from
    ``[A-Za-z0-9_-]``, and re-declaring a weaker rule at the API boundary would let a
    request through that the engine then refuses several layers deeper.
    """

    learner_id: LearnerId = Field(..., description="Pseudonymous learner identifier.")
    session_id: SessionId = Field(..., description="Pseudonymous session identifier.")
    raw_json: str = Field(
        ...,
        min_length=1,
        description=('A pasted JSON document: either a bare list of events or {"events": [...]}'),
    )


class QuestionRequest(BaseModel):
    """A natural question to answer from one learner's profile."""

    question: str = Field(..., min_length=1, description="The natural-language question.")


def _authorize(
    learner_id: str, role_header: str | None, requester_header: str | None
) -> tuple[Role, frozenset[AccessScope]]:
    """Resolve and enforce the Student Intelligence role *before* any retrieval.

    Args:
        learner_id: The learner the request asks about.
        role_header: The ``X-Student-Role`` header, absent or present.
        requester_header: The ``X-Student-Id`` header, absent or present.

    Returns:
        The resolved role and its scope set.

    Raises:
        HTTPException: With status 400 when the role header is missing or unknown, and
            with status 403 when the role is not permitted to read the learner. Both
            refusals happen before the store is consulted.
    """
    try:
        role = role_from_header(role_header)
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        enforce_access(role, learner_id, requester_header)
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return role, scopes_for(role)


def _resolve_profile(learner_id: str) -> StudentIntelligenceProfile:
    """Resolve a learner's profile, mapping an unknown id to a 422 without naming the set."""
    try:
        return get_profile(learner_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _generate(scenario_id: str) -> Any:
    """Generate a scenario, translating an unknown id into a 422 rather than a 500.

    Args:
        scenario_id: The identifier the caller supplied.

    Returns:
        The ``(spec, sessions)`` pair :func:`generate_scenario` produces.

    Raises:
        HTTPException: With status 422 when the id is not one of the shipped scenarios.
    """
    try:
        return generate_scenario(scenario_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _require_matching_identity(
    events: Sequence[EventEnvelope], learner_id: str, session_id: str
) -> None:
    """Refuse a call whose stated learner or session contradicts its own events.

    The events are authoritative about who they belong to. The engine enforces this
    independently and quite rightly - a baseline is never populated from another
    learner's behaviour - but it does so several layers down as a server fault. At the
    boundary the honest response is to reject the request as the caller error it is, and
    to say which field disagreed, because a caller who pasted one learner's events under
    another learner's id has a bug that only this message will reveal.

    Args:
        events: The accepted, validated events.
        learner_id: The learner the caller said the events belong to.
        session_id: The session the caller said the events belong to.

    Raises:
        HTTPException: With status 422 when the events name a different learner or the
            caller names a session that is not present in the stream.
    """
    observed_learners = {event.learner_id for event in events}
    if observed_learners != {learner_id}:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "learner_id does not match the events",
                "stated": learner_id,
                "observed": sorted(observed_learners),
                "message": (
                    "The events carry their own learner_id and the engine derives identity "
                    "from them. Resubmit with the learner_id the events declare, or fix the "
                    "events."
                ),
            },
        )
    if session_id not in {event.session_id for event in events}:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "session_id does not appear in the events",
                "stated": session_id,
                "observed": sorted({event.session_id for event in events}),
                "message": "The pipeline analysed a session that is not in the submitted stream.",
            },
        )


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    index = _STATIC_DIR / "index.html"
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.get("/api/canonical_example")
async def canonical_example_route() -> dict[str, Any]:
    """The smallest valid native session, validated by the engine before it is returned."""
    return canonical_example()


@app.get("/api/sample_session")
async def sample_session(seed: int = DEFAULT_SAMPLE_SEED, questions: int = 4) -> dict[str, Any]:
    """Emit a short, valid, native event session for the UI to fill the paste box with.

    The events come from the repository's own simulator, so they are valid by
    construction rather than by a template that has to be kept in step with the schemas.
    The seed is echoed back and is part of the response, so a reader can reproduce the exact
    document they were given.
    """
    try:
        return generate_sample_session(seed=seed, questions=questions)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"error": str(error)}) from error


@app.get("/api/scenarios")
async def list_scenarios() -> list[dict[str, Any]]:
    return [describe(sid) for sid in SCENARIO_IDS]


@app.post("/api/scenario_events")
async def scenario_events(req: ScenarioRequest) -> dict[str, Any]:
    spec, sessions = _generate(req.scenario_id)
    flat = flatten(sessions)
    return {
        "scenario_id": req.scenario_id,
        "description": describe(req.scenario_id),
        "events": [e.model_dump(mode="json") for e in flat],
        "session_count": len(sessions),
    }


@app.post("/api/validate")
async def validate_paste(req: PasteRequest) -> dict[str, Any]:
    parsed = coerce_events(req.raw_json)
    report = validate_dataset(parsed.rows, submitted=parsed.presented)
    return {
        "root_shape": parsed.root_shape.value,
        "parse_findings": [f.to_dict() for f in parsed.findings],
        "report": report.to_dict(),
    }


@app.post("/api/analyse/scenario")
async def analyse_scenario(req: ScenarioRequest) -> dict[str, Any]:
    spec, sessions = _generate(req.scenario_id)
    events = flatten(sessions)
    bundle: LabBundle = run_analysis(
        events,
        learner_id=spec.learner_id,
        session_id=sessions[-1][0].session_id,
        clock=_DETERMINISTIC_CLOCK,
    )
    return {
        "question_count": bundle.question_count,
        "model_summary": bundle.model_summary,
        "response": bundle.response.model_dump(mode="json"),
    }


@app.post("/api/analyse/paste")
async def analyse_paste(req: PasteRequest) -> dict[str, Any]:
    parsed = coerce_events(req.raw_json)
    report = validate_dataset(parsed.rows, submitted=parsed.presented)
    if not report.can_proceed:
        # The parse findings travel with the refusal. Dropping them reports "no events
        # reached validation" to a caller who in fact pasted a single event object, which
        # sends the reader hunting for an empty submission rather than at the root shape
        # they actually used. The empty_dataset finding is a *consequence* of the root
        # decision, not the cause, so the cause has to be in the response.
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Dataset cannot proceed",
                "root_shape": parsed.root_shape.value,
                "events_submitted": report.submitted,
                "events_accepted": report.accepted,
                "events_rejected": report.rejected,
                "parse_findings": [f.to_dict() for f in parsed.findings],
                "findings": [f.to_dict() for f in report.findings],
            },
        )
    _require_matching_identity(report.events, req.learner_id, req.session_id)
    bundle: LabBundle = run_analysis(
        report.events,
        learner_id=req.learner_id,
        session_id=req.session_id,
        clock=_DETERMINISTIC_CLOCK,
    )
    return {
        "question_count": bundle.question_count,
        "model_summary": bundle.model_summary,
        "root_shape": parsed.root_shape.value,
        "parse_findings": [f.to_dict() for f in parsed.findings],
        "validation": report.to_dict(),
        "response": bundle.response.model_dump(mode="json"),
    }


@app.post("/api/replay")
async def replay(req: ScenarioRequest) -> dict[str, Any]:
    """Run the same request twice and report whether the two responses are identical."""
    spec, sessions = _generate(req.scenario_id)
    events = flatten(sessions)
    session_id = sessions[-1][0].session_id
    first = run_analysis(
        events, learner_id=spec.learner_id, session_id=session_id, clock=_DETERMINISTIC_CLOCK
    )
    second = run_analysis(
        events, learner_id=spec.learner_id, session_id=session_id, clock=_DETERMINISTIC_CLOCK
    )
    one = first.response.model_dump(mode="json")
    two = second.response.model_dump(mode="json")
    return {
        "first": one,
        "second": two,
        "identical": one == two,
        "question_count": first.question_count,
    }


@app.get("/api/students")
async def list_students(
    role_header: str | None = Header(default=None, alias="X-Student-Role"),
    requester_header: str | None = Header(default=None, alias="X-Student-Id"),
) -> list[dict[str, str]]:
    """List the known learner ids, pruned to what the requester may see.

    A learner sees only the id they identify as - never the set of learners who exist -
    because an id listed is an id that exists, and this layer does not leak that set to a
    learner. Staff see the full pseudonymous roster, which is the point of a staff view.
    """
    try:
        role = role_from_header(role_header)
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store = get_store()
    ids: tuple[str, ...]
    if role is Role.STUDENT:
        if requester_header is not None and requester_header in store:
            ids = (requester_header,)
        else:
            ids = ()
    else:
        ids = tuple(store)
    return [{"learner_id": learner_id, "title": store[learner_id].title} for learner_id in ids]


@app.get("/api/students/{learner_id}/intelligence")
async def student_intelligence(
    learner_id: str,
    role_header: str | None = Header(default=None, alias="X-Student-Role"),
    requester_header: str | None = Header(default=None, alias="X-Student-Id"),
) -> dict[str, Any]:
    """The canonical profile under the requester's scopes, evidence and gaps attached."""
    _, scopes = _authorize(learner_id, role_header, requester_header)
    profile = _resolve_profile(learner_id)
    return {
        "learner_id": profile.learner_id,
        "scenario_id": profile.scenario_id,
        "title": profile.title,
        "archetype": profile.archetype,
        "built_at": profile.built_at.isoformat(),
        "notice": profile.notice.model_dump(mode="json"),
        "engine_versions": dict(profile.engine_versions),
        "model_summary": profile.model_summary,
        "evidence": [
            record.model_dump(mode="json") for record in visible_evidence(profile, scopes)
        ],
        "gaps": [
            gap.model_dump(mode="json")
            for gap in (*profile.gaps, *restricted_gaps(profile, scopes))
        ],
    }


@app.post("/api/students/{learner_id}/query")
async def student_query(
    learner_id: str,
    req: QuestionRequest,
    role_header: str | None = Header(default=None, alias="X-Student-Role"),
    requester_header: str | None = Header(default=None, alias="X-Student-Id"),
) -> dict[str, Any]:
    """Answer a natural question from the profile, under the requester's scopes."""
    _, scopes = _authorize(learner_id, role_header, requester_header)
    profile = _resolve_profile(learner_id)
    answer = answer_question(profile, req.question, scopes)
    return answer.model_dump(mode="json")


@app.get("/api/students/{learner_id}/evidence")
async def student_evidence(
    learner_id: str,
    role_header: str | None = Header(default=None, alias="X-Student-Role"),
    requester_header: str | None = Header(default=None, alias="X-Student-Id"),
) -> list[dict[str, Any]]:
    """The evidence records the requester's scopes permit, in stored order."""
    _, scopes = _authorize(learner_id, role_header, requester_header)
    profile = _resolve_profile(learner_id)
    return [record.model_dump(mode="json") for record in visible_evidence(profile, scopes)]


@app.get("/api/students/{learner_id}/gaps")
async def student_gaps(
    learner_id: str,
    role_header: str | None = Header(default=None, alias="X-Student-Role"),
    requester_header: str | None = Header(default=None, alias="X-Student-Id"),
) -> list[dict[str, Any]]:
    """Canonical data gaps plus the permissions that restricted evidence under the role."""
    _, scopes = _authorize(learner_id, role_header, requester_header)
    profile = _resolve_profile(learner_id)
    return [
        gap.model_dump(mode="json") for gap in (*profile.gaps, *restricted_gaps(profile, scopes))
    ]


# Intelligence demo endpoints (Phase 15)
@app.get("/api/intelligence-demo")
async def list_intelligence_demo() -> list[dict[str, str]]:
    """List the four Phase 15 demo learners."""
    store = build_demo_store()
    return [
        {"learner_id": learner_id, "title": store[learner_id].title}
        for learner_id in demo_learner_ids()
    ]


@app.get("/api/intelligence-demo/{learner_id}")
async def intelligence_demo_profile(
    learner_id: str,
    role_header: str | None = Header(default=None, alias="X-Student-Role"),
    requester_header: str | None = Header(default=None, alias="X-Student-Id"),
) -> dict[str, Any]:
    """The demo learner profile under the requester's scopes, evidence and gaps attached."""
    _, scopes = _authorize(learner_id, role_header, requester_header)
    store = build_demo_store()
    if learner_id not in store:
        raise HTTPException(status_code=422, detail={"error": "Learner not found in demo store"})
    profile = store[learner_id]
    return {
        "learner_id": profile.learner_id,
        "scenario_id": profile.scenario_id,
        "title": profile.title,
        "archetype": profile.archetype,
        "built_at": profile.built_at.isoformat(),
        "notice": profile.notice.model_dump(mode="json"),
        "engine_versions": dict(profile.engine_versions),
        "model_summary": profile.model_summary,
        "evidence": [
            record.model_dump(mode="json") for record in visible_evidence(profile, scopes)
        ],
        "gaps": [
            gap.model_dump(mode="json")
            for gap in (*profile.gaps, *restricted_gaps(profile, scopes))
        ],
    }


@app.post("/api/intelligence-demo/{learner_id}/query")
async def intelligence_demo_query(
    learner_id: str,
    req: QuestionRequest,
    role_header: str | None = Header(default=None, alias="X-Student-Role"),
    requester_header: str | None = Header(default=None, alias="X-Student-Id"),
) -> dict[str, Any]:
    """Answer a natural question from the demo learner profile, under the requester's scopes."""
    _, scopes = _authorize(learner_id, role_header, requester_header)
    store = build_demo_store()
    if learner_id not in store:
        raise HTTPException(status_code=422, detail={"error": "Learner not found in demo store"})
    profile = store[learner_id]
    answer = answer_question(profile, req.question, scopes)
    return answer.model_dump(mode="json")
