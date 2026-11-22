"""Career Copilot tool registry.

Every tool is a thin adapter over an *existing* endpoint or service, so the
agent sees exactly the numbers, guards and skill-gap logic the UI shows —
the LLM decides *which* tool to call and how to phrase the answer, never the
scores:

* ``get_recommendations`` runs the same pipeline as the Recommendations page
  (M0 retrieval + cross-encoder, M1 skill graph, M2 DCN ranker, M3 calibrated
  match % and tiers, M4 bandit) but through ``build_recommendation_feed`` —
  the read-only half — so chat views are never logged as SERVED
  impressions and never touch the bandit's online-learning signal.
* ``search_jobs`` is the Jobs page search (platform + live Adzuna, fresher
  detection) with the per-job skill gap attached.
* Write tools (``save_job``, ``update_career_direction``, ``remember``,
  ``forget``) are flagged ``writes=True``; the system prompt only allows them
  on an explicit user request.  Saving a job records the same 'save'
  interaction as the Save button — a genuine user action.

Each tool returns ``ToolResult``: ``data`` (compact JSON for the model) and
``cards`` (structured payload the chat UI renders).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.enums import ApplicationStatus, EmploymentType, ExperienceLevel, SkillSource
from app.models.profile import Profile
from app.models.skill import ProfileSkill
from app.models.user import User
from app.services.ingest import detect_fresher

log = logging.getLogger(__name__)

MAX_JOBS = 8


@dataclass
class AgentContext:
    db: Session
    user: User
    # every job card shown in this conversation, keyed by id — lets the model
    # refer back ("save the second one") and grounds explain_match for
    # transient live listings that have no DB row
    seen_jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    # the user's current message (deterministic backstops read it)
    user_message: str = ""
    # ordered ids of the most recent list shown (1-based references)
    last_list: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    data: dict[str, Any]
    cards: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = True


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., ToolResult]
    writes: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _profile(ctx: AgentContext) -> Profile | None:
    return ctx.db.scalar(
        select(Profile)
        .where(Profile.user_id == ctx.user.id)
        .options(
            selectinload(Profile.skills).selectinload(ProfileSkill.skill),
            selectinload(Profile.target_family),
        )
    )


def _fmt_salary(lo: int | None, hi: int | None, cur: str | None) -> str | None:
    if lo is None and hi is None:
        return None
    cur = cur or ""
    if lo is not None and hi is not None and lo != hi:
        return f"{cur} {lo:,}–{hi:,}".strip()
    return f"{cur} {(lo or hi):,}".strip()


def _job_card(job: Any, *, match_score: float | None = None, match_tier: str | None = None,
              reasons: list[str] | None = None) -> dict[str, Any]:
    """UI card from a JobDetailResponse (platform or live)."""
    gap = getattr(job, "skill_gap", None)
    skill_pct = None
    if gap is not None and gap.required_total:
        skill_pct = round(100.0 * gap.required_covered / gap.required_total, 1)
    level = getattr(job, "experience_level", None)
    return {
        "type": "job",
        "id": str(job.id),
        "title": job.title,
        "company": getattr(getattr(job, "company", None), "name", None),
        "location": job.location,
        "is_remote": bool(getattr(job, "is_remote", False)),
        "salary": _fmt_salary(job.salary_min, job.salary_max, getattr(job, "salary_currency", None)),
        "source": getattr(job, "source", None) or "native",
        "url": getattr(job, "external_url", None),
        "experience_level": getattr(level, "value", level),
        # calibrated M3 output (recommendations only) — never estimated here
        "match_score": round(match_score, 1) if match_score is not None else None,
        "match_tier": match_tier,
        "skill_match": skill_pct,
        "skill_gap": gap.model_dump() if gap is not None else None,
        "reasons": list(reasons or [])[:3],
    }


def _model_view(card: dict[str, Any], index: int) -> dict[str, Any]:
    """What the model sees for one job: compact and reference-able."""
    gap = card.get("skill_gap") or {}
    view = {
        "ref": index,
        "id": card["id"],
        "title": card["title"],
        "company": card["company"],
        "location": (card["location"] or "") + (" (remote)" if card["is_remote"] else ""),
        "salary": card["salary"],
        "level": card.get("experience_level"),
        "match_score": card["match_score"],
        "match_tier": card.get("match_tier"),
        "skill_match_pct": card["skill_match"],
        "have_skills": (gap.get("matched") or [])[:4],
        "learning_skills": (gap.get("learning") or [])[:3],
        "transferable_skills": (gap.get("transferable") or [])[:3],
        "missing_skills": (gap.get("missing_required") or [])[:4],
        "live_listing": card["source"] == "adzuna",
    }
    if card["reasons"]:
        view["why"] = card["reasons"][:2]
    return {k: v for k, v in view.items() if v not in (None, [], "", False)}


def _remember_list(ctx: AgentContext, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ctx.last_list = []
    views = []
    for i, card in enumerate(cards, start=1):
        ctx.seen_jobs[card["id"]] = card
        ctx.last_list.append(card["id"])
        views.append(_model_view(card, i))
    return views


def _resolve_job_ref(ctx: AgentContext, job_ref: Any) -> str | None:
    """Accept a job id, or a 1-based position in the last list shown."""
    ref = str(job_ref or "").strip().lstrip("#")
    if not ref:
        return None
    if ref.isdigit() and ctx.last_list:
        idx = int(ref) - 1
        if 0 <= idx < len(ctx.last_list):
            return ctx.last_list[idx]
    return ref


def _uuid(ref: str) -> UUID | None:
    try:
        return UUID(ref)
    except (ValueError, TypeError):
        return None


def _http_error(exc: HTTPException) -> ToolResult:
    return ToolResult(data={"error": str(exc.detail)}, ok=False)


def _enum(enum_cls: Any, value: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return enum_cls(str(value).lower())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# read tools
# ---------------------------------------------------------------------------

def get_profile(ctx: AgentContext) -> ToolResult:
    profile = _profile(ctx)
    if profile is None:
        return ToolResult(data={"error": "No profile yet — ask the user to complete their profile."}, ok=False)
    skills = [ps.skill.name for ps in profile.skills or [] if ps.skill is not None]
    data = {
        "name": f"{profile.first_name or ''} {profile.last_name or ''}".strip() or None,
        "headline": profile.headline,
        "location": profile.location,
        "years_of_experience": profile.years_of_experience,
        "skills": skills[:40],
        "target_roles": list(profile.target_roles or []),
        "interests": list(profile.career_interests or []),
        "learning_skills": list(profile.learning_skills or []),
        "target_family": profile.target_family.name if profile.target_family else None,
        "preferred_locations": list(profile.preferred_locations or []),
        "preferred_work_mode": profile.preferred_work_mode.value if profile.preferred_work_mode else None,
        "expected_salary": _fmt_salary(
            profile.expected_salary_min, profile.expected_salary_max, profile.expected_salary_currency
        ),
        "has_resume": bool(profile.resume_url or profile.raw_resume_text),
    }
    return ToolResult(data={k: v for k, v in data.items() if v not in (None, [], "")})


def search_jobs(
    ctx: AgentContext,
    query: str,
    location: str | None = None,
    remote_only: bool | None = None,
    experience_level: str | None = None,
    employment_type: str | None = None,
    min_salary: int | None = None,
    limit: int = 5,
) -> ToolResult:
    from app.api.jobs import list_jobs

    limit = max(1, min(int(limit or 5), MAX_JOBS))
    level = _enum(ExperienceLevel, experience_level)
    # Backstop: "fresher"/"0-1 years"/"entry level" in the query or the user's
    # own message means entry level even when the model forgot the filter.
    if level is None and (detect_fresher(query) or detect_fresher(ctx.user_message)):
        level = ExperienceLevel.ENTRY
    try:
        resp = list_jobs(
            db=ctx.db,
            viewer=ctx.user,
            q=(query or "").strip() or None,
            search=None,
            sort_by="relevance",
            location=(location or "").strip() or None,
            is_remote=True if remote_only else None,
            work_mode=None,
            employment_type=_enum(EmploymentType, employment_type),
            experience_level=level,
            skill=None,
            company_id=None,
            company=None,
            industry=None,
            family=None,
            skills=None,
            min_salary=int(min_salary) if min_salary else None,
            page=1,
            page_size=max(limit, 10),
        )
    except HTTPException as exc:
        return _http_error(exc)

    cards = [_job_card(item) for item in resp.items[:limit]]
    data: dict[str, Any] = {"total_matches": resp.total, "jobs": _remember_list(ctx, cards)}
    if level is not None:
        data["experience_level_applied"] = level.value
    if not cards:
        data["hint"] = "No matches. Suggest broadening the query or dropping a filter."
    return ToolResult(data=data, cards=list(cards))


def get_recommendations(ctx: AgentContext, limit: int = 5) -> ToolResult:
    from app.api.recommendations import build_recommendation_feed

    limit = max(1, min(int(limit or 5), MAX_JOBS))
    try:
        feed, _profile_row = build_recommendation_feed(ctx.db, ctx.user, limit)
    except HTTPException as exc:
        return _http_error(exc)

    cards = []
    for item in feed.items[:limit]:
        reasons: list[str] = []
        if item.explanation is not None:
            reasons = list(item.explanation.strengths[:2]) + list(item.explanation.concerns[:1])
        elif item.match_reasons:
            reasons = list(item.match_reasons[:2])
        cards.append(
            _job_card(item.job, match_score=item.match_score, match_tier=item.match_tier, reasons=reasons)
        )
    data: dict[str, Any] = {
        "feed_type": feed.recommendation_type,
        "profile_completeness": feed.profile_completeness_score,
        "ranking_model": feed.model_version or feed.strategy,
        "jobs": _remember_list(ctx, cards),
    }
    if feed.personalization_note:
        data["personalization_note"] = feed.personalization_note
    if feed.recommendation_type == "cold_start":
        data["hint"] = "Profile has no skills yet: suggest uploading a resume or adding skills."
    return ToolResult(data=data, cards=list(cards))


def explain_match(ctx: AgentContext, job_ref: str) -> ToolResult:
    from app.api.gap import get_job_gap

    ref = _resolve_job_ref(ctx, job_ref)
    if ref is None:
        return ToolResult(data={"error": "job_ref is required"}, ok=False)
    card = ctx.seen_jobs.get(ref)
    job_id = _uuid(ref)
    if job_id is None:
        # live (Adzuna) listing: no DB row — answer from the card we showed
        if card is None:
            return ToolResult(data={"error": f"Unknown job '{ref}'. Search or recommend first."}, ok=False)
        gap = card.get("skill_gap") or {}
        data = {
            "title": card["title"],
            "company": card["company"],
            "match_score": card.get("match_score"),
            "match_tier": card.get("match_tier"),
            "have_skills": gap.get("matched", []),
            "learning_skills": gap.get("learning", []),
            "transferable_skills": gap.get("transferable", []),
            "missing_required_skills": gap.get("missing_required", []),
            "nice_to_have": gap.get("missing_preferred", []),
            "note": "Live external listing — skill gap from the posting snippet only.",
        }
        data = {k: v for k, v in data.items() if v not in (None, [])}
        return ToolResult(data=data, cards=[{"type": "gap", "job_id": ref, **data}])
    try:
        gap = get_job_gap(job_id=job_id, current_user=ctx.user, db=ctx.db)
    except HTTPException as exc:
        return _http_error(exc)
    data: dict[str, Any] = {
        "title": gap.job_title,
        "skill_gap_score": gap.gap_score,
        "counts": gap.counts,
        "missing_required_skills": list(gap.missing_required)[:8],
        "weakest_link": gap.weakest_link,
        "skills": [{k: c.get(k) for k in ("skill", "status") if k in c} for c in (gap.classes or [])[:12]],
        "reasons": list(gap.reasons or [])[:5],
    }
    if card and card.get("match_score") is not None:
        # the calibrated ML score this job was shown with — quote that one
        data["match_score"] = card["match_score"]
        data["match_tier"] = card.get("match_tier")
    else:
        data["formula_match_score"] = gap.match_score
    return ToolResult(data=data, cards=[{"type": "gap", "job_id": str(job_id), **data}])


def career_direction_gap(ctx: AgentContext, include_live: bool = True) -> ToolResult:
    from app.services.direction_gap import direction_gaps

    profile = _profile(ctx)
    if profile is None:
        return ToolResult(data={"error": "No profile yet."}, ok=False)
    targets = direction_gaps(ctx.db, profile, include_live=bool(include_live))
    if not targets:
        return ToolResult(
            data={
                "message": "No target roles set yet.",
                "hint": "Offer to set target roles via update_career_direction, or the Career goals form on the Profile page.",
            }
        )
    data = {
        "targets": [
            {
                "target": t.target,
                "readiness_pct": t.readiness,
                "jobs_analyzed": t.jobs_analyzed,
                "top_missing": [f"{s.skill} ({round(s.share * 100)}% of postings)" for s in t.top_missing[:5]],
                "already_have": [s.skill for s in t.skills if s.status == "have"][:6],
                "learning": [s.skill for s in t.skills if s.status == "learning"][:4],
                "transferable": [s.skill for s in t.skills if s.status == "transferable"][:4],
            }
            for t in targets
        ]
    }
    cards = [{"type": "direction_gap", **t.as_dict()} for t in targets]
    return ToolResult(data=data, cards=cards)


def list_my_applications(ctx: AgentContext, status: str | None = None) -> ToolResult:
    from app.api.applications import get_my_applications

    try:
        rows = get_my_applications(
            current_user=ctx.user, db=ctx.db, page=1, limit=20, status_filter=_enum(ApplicationStatus, status)
        )
    except HTTPException as exc:
        return _http_error(exc)
    items = [
        {
            "job_id": str(r.job_id),
            "title": r.job.title if r.job else None,
            "company": r.job.company.name if r.job and r.job.company else None,
            "status": r.status.value,
            "updated": r.updated_at.date().isoformat() if r.updated_at else None,
        }
        for r in rows
    ]
    by_status: dict[str, int] = {}
    for it in items:
        by_status[it["status"]] = by_status.get(it["status"], 0) + 1
    return ToolResult(
        data={"count": len(items), "by_status": by_status, "applications": items[:12]},
        cards=[{"type": "applications", "items": items}],
    )


def tailor_resume(ctx: AgentContext, job_ref: str) -> ToolResult:
    from app.api.resume_optimize import optimize_resume

    ref = _resolve_job_ref(ctx, job_ref)
    job_id = _uuid(ref or "")
    if job_id is None:
        return ToolResult(
            data={"error": "Resume tailoring needs a platform job (live listings are not supported)."}, ok=False
        )
    try:
        out = optimize_resume(job_id=job_id, current_user=ctx.user, db=ctx.db)
    except HTTPException as exc:
        return _http_error(exc)
    ats = dict(out.ats)
    data = {
        "job_title": out.job_title,
        "ats": {k: ats[k] for k in list(ats)[:8]},
        "suggestions": [dict(s) for s in out.suggestions][:5],
        "llm_used": out.llm_used,
        "disclaimer": out.disclaimer,
    }
    return ToolResult(data=data, cards=[{"type": "resume", "job_id": str(job_id), **data}])


# ---------------------------------------------------------------------------
# write tools (explicit user request only)
# ---------------------------------------------------------------------------

def save_job(ctx: AgentContext, job_ref: str) -> ToolResult:
    from app.api.applications import save_job_for_later

    ref = _resolve_job_ref(ctx, job_ref)
    job_id = _uuid(ref or "")
    if job_id is None:
        card = ctx.seen_jobs.get(ref or "")
        hint = f" Open the posting instead: {card['url']}" if card and card.get("url") else ""
        return ToolResult(data={"error": "Live listings can't be saved to the tracker." + hint}, ok=False)
    try:
        out = save_job_for_later(job_id=job_id, current_user=ctx.user, db=ctx.db)
    except HTTPException as exc:
        return _http_error(exc)
    title = out.job.title if out.job else ctx.seen_jobs.get(str(job_id), {}).get("title")
    return ToolResult(data={"saved": True, "job_id": str(job_id), "title": title})


def update_career_direction(
    ctx: AgentContext,
    add_target_roles: list[str] | None = None,
    remove_target_roles: list[str] | None = None,
    add_learning_skills: list[str] | None = None,
    add_skills: list[str] | None = None,
) -> ToolResult:
    """Same rules as the Career goals form (PUT /profiles/me): roles capped at
    5 and mapped to a job family, learning skills canonicalised."""
    from app.api.profiles import update_my_profile
    from app.schemas.profile import ProfileUpdateRequest
    from app.services.skill_normalizer import resolve_skill

    profile = _profile(ctx)
    if profile is None:
        return ToolResult(data={"error": "No profile yet."}, ok=False)

    roles = [str(r).strip() for r in (profile.target_roles or []) if str(r).strip()]
    removed = {str(r).strip().lower() for r in (remove_target_roles or []) if str(r).strip()}
    roles = [r for r in roles if r.lower() not in removed]
    for raw in add_target_roles or []:
        role = str(raw).strip()[:100]
        if role and role.lower() not in {r.lower() for r in roles}:
            roles.append(role)
    learning = list(profile.learning_skills or []) + [str(s).strip() for s in (add_learning_skills or [])]

    changes: dict[str, Any] = {}
    if add_target_roles or remove_target_roles:
        changes["target_roles"] = roles[:5]
    if add_learning_skills:
        changes["learning_skills"] = [s for s in learning if s][:25]
    if changes:
        update_my_profile(payload=ProfileUpdateRequest(**changes), current_user=ctx.user, db=ctx.db)

    added_skills: list[str] = []
    if add_skills:
        profile = _profile(ctx)
        have = {ps.skill.name.lower() for ps in profile.skills or [] if ps.skill is not None}
        for raw in add_skills[:10]:
            name = str(raw).strip()[:80]
            if not name:
                continue
            skill = resolve_skill(ctx.db, name)
            if skill.name.lower() in have:
                continue
            have.add(skill.name.lower())
            profile.skills.append(ProfileSkill(profile_id=profile.id, skill_id=skill.id, source=SkillSource.MANUAL))
            added_skills.append(skill.name)
        ctx.db.commit()

    profile = _profile(ctx)
    return ToolResult(
        data={
            "target_roles": list(profile.target_roles or []),
            "target_family": profile.target_family.name if profile.target_family else None,
            "learning_skills": list(profile.learning_skills or []),
            "added_skills": added_skills,
            "note": "Recommendations, live job results and skill gaps now follow these goals.",
        }
    )


def remember(ctx: AgentContext, content: str, kind: str = "preference") -> ToolResult:
    from app.services.agent.memory import upsert_memory

    mem, created = upsert_memory(ctx.db, ctx.user.id, content, kind=kind, source="tool")
    if mem is None:
        return ToolResult(data={"error": "Nothing to remember."}, ok=False)
    return ToolResult(data={"remembered": mem.content, "kind": mem.kind, "new": created, "id": str(mem.id)})


def forget(ctx: AgentContext, memory_id: str) -> ToolResult:
    from app.services.agent.memory import delete_memory

    ok = delete_memory(ctx.db, ctx.user.id, memory_id)
    return ToolResult(data={"forgotten": ok}, ok=ok)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_JOB_REF = {
    "type": "string",
    "description": "Job id from a previous tool result, or its 1-based position ('2') in the last list shown.",
}
_STR_LIST = {"type": "array", "items": {"type": "string"}}

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "get_profile",
            "Read the user's profile: skills, experience, target roles, skills being learned, location and salary preferences.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            get_profile,
        ),
        Tool(
            "search_jobs",
            "Search live (Adzuna, India) + platform job listings, annotated with the user's skill gap (have / learning / transferable / missing). Use for any explicit search ('remote python jobs in Pune').",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Role/keywords/skills only, e.g. 'backend engineer python' - put seniority in experience_level, not here"},
                    "location": {"type": "string", "description": "City or country, if the user gave one"},
                    "remote_only": {"type": "boolean"},
                    "experience_level": {
                        "type": "string",
                        "enum": ["entry", "mid", "senior", "lead", "executive"],
                        "description": "Set 'entry' for fresher / graduate / trainee / junior / 0-1 years / campus / batch requests",
                    },
                    "employment_type": {
                        "type": "string",
                        "enum": ["full_time", "part_time", "contract", "internship", "temporary"],
                    },
                    "min_salary": {"type": "integer", "description": "Minimum annual salary in the listing currency"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_JOBS, "default": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            search_jobs,
        ),
        Tool(
            "get_recommendations",
            "Personalised job matches ranked by the trained ML pipeline, with a calibrated match % and tier (top/strong/possible), reasons and skill gaps. Use when the user asks what fits them / for their matches.",
            {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": MAX_JOBS, "default": 5}},
                "additionalProperties": False,
            },
            get_recommendations,
        ),
        Tool(
            "explain_match",
            "Detailed fit for ONE job already shown: skills the user has / is learning / can transfer / is missing, weakest link.",
            {"type": "object", "properties": {"job_ref": _JOB_REF}, "required": ["job_ref"], "additionalProperties": False},
            explain_match,
        ),
        Tool(
            "career_direction_gap",
            "For the user's target roles (career/domain switchers): which skills those roles demand across real postings, which the user has, is learning, can transfer, or lacks, plus a readiness %.",
            {
                "type": "object",
                "properties": {"include_live": {"type": "boolean", "default": True}},
                "additionalProperties": False,
            },
            career_direction_gap,
        ),
        Tool(
            "list_my_applications",
            "The user's saved jobs and applications with their pipeline status.",
            {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["saved", "applied", "viewed", "shortlisted", "interview", "offer", "hired", "rejected", "withdrawn"],
                    }
                },
                "additionalProperties": False,
            },
            list_my_applications,
        ),
        Tool(
            "tailor_resume",
            "ATS check + grounded resume improvement suggestions for one platform job (never invents experience).",
            {"type": "object", "properties": {"job_ref": _JOB_REF}, "required": ["job_ref"], "additionalProperties": False},
            tailor_resume,
        ),
        Tool(
            "save_job",
            "Save a platform job to the user's tracker. ONLY when the user explicitly asks to save it.",
            {"type": "object", "properties": {"job_ref": _JOB_REF}, "required": ["job_ref"], "additionalProperties": False},
            save_job,
            writes=True,
        ),
        Tool(
            "update_career_direction",
            "Change the user's career goals: add/remove target roles, add skills they are learning, or add skills they already have. ONLY when the user explicitly asks to change them.",
            {
                "type": "object",
                "properties": {
                    "add_target_roles": _STR_LIST,
                    "remove_target_roles": _STR_LIST,
                    "add_learning_skills": _STR_LIST,
                    "add_skills": _STR_LIST,
                },
                "additionalProperties": False,
            },
            update_career_direction,
            writes=True,
        ),
        Tool(
            "remember",
            "Store a durable fact about the user (preference, dealbreaker, goal) for future conversations, when they state one.",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Short third-person fact, e.g. 'Will not relocate outside Pune'"},
                    "kind": {"type": "string", "enum": ["preference", "dealbreaker", "goal", "fact"]},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            remember,
            writes=True,
        ),
        Tool(
            "forget",
            "Delete a stored memory by id when the user says it is wrong or outdated.",
            {"type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"], "additionalProperties": False},
            forget,
            writes=True,
        ),
    ]
}


def tool_schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS.values()]


def execute(ctx: AgentContext, name: str, arguments: dict[str, Any]) -> ToolResult:
    """Run one tool call defensively: unknown tools / bad args become errors
    the model can read and recover from, never exceptions."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(data={"error": f"Unknown tool '{name}'. Available: {sorted(TOOLS)}"}, ok=False)
    allowed = set((tool.parameters.get("properties") or {}).keys())
    args = {k: v for k, v in (arguments or {}).items() if k in allowed}
    missing = [k for k in tool.parameters.get("required", []) if args.get(k) in (None, "")]
    if missing:
        return ToolResult(data={"error": f"Missing required argument(s): {missing}"}, ok=False)
    try:
        return tool.fn(ctx, **args)
    except HTTPException as exc:
        return _http_error(exc)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, logged here
        log.exception("copilot tool %s failed", name)
        try:
            ctx.db.rollback()
        except Exception:  # pragma: no cover
            pass
        return ToolResult(data={"error": f"{name} failed: {type(exc).__name__}"}, ok=False)
