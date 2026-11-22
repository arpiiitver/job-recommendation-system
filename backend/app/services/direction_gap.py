"""Career-direction gap: what do the roles I'm *targeting* demand?

For each target role on the profile (Career goals form, or set from the
Copilot chat) we pull real postings — live Adzuna results plus matching
platform jobs — count how often each skill is asked for, and classify every
in-demand skill against the seeker with the same rules as the per-job skill
gap (``skill_gap_view``): have / learning / transferable (M1 skill graph) /
missing.

``readiness_pct`` is demand-weighted coverage of the top skills:
have = 1, transferable = 0.5, learning = 0.25, missing = 0.  It is a
descriptive statistic over postings, not a model prediction.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.enums import JobStatus
from app.models.job import Job
from app.models.profile import Profile
from app.models.skill import JobSkill
from app.services.skill_gap_view import SeekerSkills, load_seeker_skills
from app.services.skill_graph import transfer_similarity

log = logging.getLogger(__name__)

MAX_TARGETS = 3
TOP_SKILLS = 12
STATUS_WEIGHT = {"have": 1.0, "transferable": 0.5, "learning": 0.25, "missing": 0.0}


@dataclass
class DemandSkill:
    skill: str
    share: float          # fraction of analysed postings that ask for it
    status: str           # have | learning | transferable | missing


@dataclass
class TargetGap:
    target: str
    jobs_analyzed: int
    readiness: float | None
    skills: list[DemandSkill] = field(default_factory=list)

    @property
    def top_missing(self) -> list[DemandSkill]:
        return [s for s in self.skills if s.status == "missing"]

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "jobs_analyzed": self.jobs_analyzed,
            "readiness": self.readiness,
            "skills": [s.__dict__ for s in self.skills],
        }


def _targets(profile: Profile) -> list[str]:
    roles = [str(r).strip() for r in (profile.target_roles or []) if str(r).strip()]
    if not roles:
        roles = [str(r).strip() for r in (profile.career_interests or []) if str(r).strip()]
    return roles[:MAX_TARGETS]


def _platform_skill_sets(db: Session, role: str, limit: int = 40) -> list[set[str]]:
    jobs = db.scalars(
        select(Job)
        .options(selectinload(Job.skills).selectinload(JobSkill.skill))
        .where(Job.status == JobStatus.PUBLISHED, Job.title.ilike(f"%{role}%"))
        .order_by(Job.created_at.desc())
        .limit(limit)
    ).all()
    out = []
    for job in jobs:
        names = {js.skill.name for js in job.skills or [] if js.skill is not None}
        if names:
            out.append(names)
    return out


def _live_skill_sets(role: str) -> list[set[str]]:
    from app.services.live_jobs import search_postings

    postings, _total = search_postings(role, page_size=40)
    return [{name for name, _req in p["skills"]} for p in postings if p.get("skills")]


def _status(seeker: SeekerSkills | None, name: str) -> str:
    if seeker is None:
        return "missing"
    key = name.lower().strip()
    if key in seeker.have:
        return "have"
    if key in seeker.learning:
        return "learning"
    if seeker.graph is not None:
        sim = transfer_similarity(seeker.graph, seeker.have, name)
        if sim is not None and sim >= seeker.threshold:
            return "transferable"
    return "missing"


def direction_gaps(db: Session, profile: Profile, *, include_live: bool = True) -> list[TargetGap]:
    seeker = load_seeker_skills(db, profile.user_id)
    out: list[TargetGap] = []
    for role in _targets(profile):
        skill_sets = _platform_skill_sets(db, role)
        if include_live:
            try:
                skill_sets += _live_skill_sets(role)
            except Exception:  # noqa: BLE001 - live leg is best effort
                log.warning("live postings unavailable for direction gap %r", role)
        if not skill_sets:
            out.append(TargetGap(target=role, jobs_analyzed=0, readiness=None))
            continue
        counts: Counter[str] = Counter()
        display: dict[str, str] = {}
        for names in skill_sets:
            for n in names:
                key = n.lower().strip()
                counts[key] += 1
                display.setdefault(key, n)
        total = len(skill_sets)
        skills = [
            DemandSkill(skill=display[k], share=round(c / total, 3), status=_status(seeker, display[k]))
            for k, c in counts.most_common(TOP_SKILLS)
        ]
        demand = sum(s.share for s in skills)
        readiness = (
            round(100 * sum(STATUS_WEIGHT[s.status] * s.share for s in skills) / demand, 1) if demand else None
        )
        out.append(TargetGap(target=role, jobs_analyzed=total, readiness=readiness, skills=skills))
    return out
