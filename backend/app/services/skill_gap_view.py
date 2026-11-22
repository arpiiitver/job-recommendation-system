"""Per-job skill-gap summary for the logged-in seeker (search + recommendations).

One shared comparison so every job the user sees - keyword search, AI search,
recommendations, live Adzuna listings included - shows the same breakdown:

    matched            job skills the seeker has
    learning           job skills the seeker marked as "currently learning"
    transferable       missing skills the seeker can transfer into (M1 skill
                       graph: curated edges + calibrated O*NET embeddings)
    missing_required   required job skills with no evidence
    missing_preferred  nice-to-have job skills with no evidence

Pure + cheap: one profile load per request, then set/graph look-ups per job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.models.profile import Profile
from app.models.skill import ProfileSkill
from app.schemas.job import JobDetailResponse, SkillGapSummary
from app.services.skill_graph import SkillGraph, load_seed_graph, transfer_similarity


@dataclass
class SeekerSkills:
    have: set[str] = field(default_factory=set)        # lowercase canonical names
    learning: set[str] = field(default_factory=set)
    graph: SkillGraph | None = None
    threshold: float = 0.65


def load_seeker_skills(db: Session, user_id) -> SeekerSkills | None:
    profile = db.scalar(
        select(Profile)
        .where(Profile.user_id == user_id)
        .options(selectinload(Profile.skills).selectinload(ProfileSkill.skill))
    )
    if profile is None:
        return None
    have = {
        ps.skill.name.lower().strip()
        for ps in (profile.skills or [])
        if ps.skill is not None and ps.skill.name
    }
    learning = {
        str(s).lower().strip() for s in (getattr(profile, "learning_skills", None) or []) if str(s).strip()
    } - have
    if not have and not learning:
        return None
    return SeekerSkills(
        have=have,
        learning=learning,
        graph=load_seed_graph(),
        threshold=float(get_settings().transfer_similarity_threshold),
    )


def summarize(seeker: SeekerSkills, job_skills: list[tuple[str, bool]]) -> SkillGapSummary:
    """job_skills: [(display name, is_required)]."""
    matched, learning, transferable, miss_req, miss_pref = [], [], [], [], []
    seen: set[str] = set()
    req_total = req_have = 0
    for name, required in job_skills:
        key = (name or "").lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if required:
            req_total += 1
        if key in seeker.have:
            matched.append(name)
            req_have += int(required)
            continue
        if key in seeker.learning:
            learning.append(name)
            continue
        sim = transfer_similarity(seeker.graph, seeker.have, name) if seeker.graph is not None else None
        if sim is not None and sim >= seeker.threshold:
            transferable.append(name)
            continue
        (miss_req if required else miss_pref).append(name)
    return SkillGapSummary(
        matched=matched,
        learning=learning,
        transferable=transferable,
        missing_required=miss_req,
        missing_preferred=miss_pref,
        required_total=req_total,
        required_covered=req_have,
    )


def annotate(seeker: SeekerSkills | None, jobs: list[JobDetailResponse]) -> None:
    """Attach `skill_gap` to each job in place (no-op when the user has no skills)."""
    if seeker is None:
        return
    for job in jobs:
        job.skill_gap = summarize(seeker, [(s.skill_name, bool(s.is_required)) for s in (job.skills or [])])
