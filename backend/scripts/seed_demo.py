#!/usr/bin/env python
"""Demo + test fixture data for a fresh database (idempotent).

Creates the accounts the README's quick-login buttons and the integration
tests rely on, then runs the existing seeders in dependency order:

    1. demo accounts   aanya.seeker@example.com  (job seeker, with profile + skills)
                       recruiter@technova.io     (employer, TechNova Labs)
                       admin.qa@jobrec.io        (admin console)
       password for all: Password123!
    2. base skill catalog          (app.core.seed_skills)
    3. golden job postings         (scripts.seed_jobs — needs the employer)
    4. families, aliases, edges    (scripts.seed_canonical — assigns job families)

Usage (after ``alembic upgrade head``):
    cd backend
    python -m scripts.seed_demo

CI runs this before the test suite; a fresh database otherwise has no users
or jobs and every corpus-dependent test fails. Never run it against a
production database: the demo password is public.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from sqlalchemy import func, select  # noqa: E402

from app.core.db import SessionLocal  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.models.company import Company  # noqa: E402
from app.models.enums import ProficiencyLevel, SkillSource, UserRole  # noqa: E402
from app.models.profile import Profile  # noqa: E402
from app.models.skill import ProfileSkill  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.skill_normalizer import resolve_skill  # noqa: E402

DEMO_PASSWORD = "Password123!"
SEEKER_EMAIL = "aanya.seeker@example.com"
EMPLOYER_EMAIL = "recruiter@technova.io"
ADMIN_EMAIL = "admin.qa@jobrec.io"

# A full-stack fresher-to-junior profile; deliberately WITHOUT Docker (the
# simulator tests add it as the counterfactual skill).
SEEKER_SKILLS: list[tuple[str, ProficiencyLevel, float]] = [
    ("Python", ProficiencyLevel.ADVANCED, 3.0),
    ("SQL", ProficiencyLevel.INTERMEDIATE, 2.0),
    ("FastAPI", ProficiencyLevel.INTERMEDIATE, 1.5),
    ("React", ProficiencyLevel.INTERMEDIATE, 1.5),
    ("TypeScript", ProficiencyLevel.INTERMEDIATE, 1.0),
    ("PostgreSQL", ProficiencyLevel.INTERMEDIATE, 1.5),
    ("Git", ProficiencyLevel.ADVANCED, 3.0),
    ("Machine Learning", ProficiencyLevel.BEGINNER, 1.0),
]


def _user(db, email: str, role: UserRole) -> tuple[User, bool]:
    user = db.scalar(select(User).where(func.lower(User.email) == email))
    if user is not None:
        return user, False
    user = User(
        email=email,
        hashed_password=hash_password(DEMO_PASSWORD),
        role=role,
        is_active=True,
        is_verified=True,
    )
    db.add(user)
    db.flush()
    return user, True


def seed_accounts() -> None:
    with SessionLocal() as db:
        seeker, created = _user(db, SEEKER_EMAIL, UserRole.JOB_SEEKER)
        profile = db.scalar(select(Profile).where(Profile.user_id == seeker.id))
        if profile is None:
            profile = Profile(user_id=seeker.id, first_name="Aanya", last_name="Jain")
            db.add(profile)
            db.flush()
        if not profile.headline:
            profile.headline = "Full-stack developer (Python, FastAPI, React)"
            profile.location = "Pune, India"
            profile.years_of_experience = 1
            profile.bio = (
                "Computer engineering graduate building full-stack web apps with FastAPI, "
                "PostgreSQL and React; interested in machine learning."
            )
        have = {ps.skill_id for ps in profile.skills}
        for name, level, years in SEEKER_SKILLS:
            skill = resolve_skill(db, name)
            if skill.id in have:
                continue
            profile.skills.append(
                ProfileSkill(
                    profile_id=profile.id,
                    skill_id=skill.id,
                    proficiency_level=level,
                    years_of_experience=years,
                    source=SkillSource.MANUAL,
                )
            )
            have.add(skill.id)

        employer, _ = _user(db, EMPLOYER_EMAIL, UserRole.EMPLOYER)
        company = db.scalar(select(Company).where(Company.created_by_user_id == employer.id).limit(1))
        if company is None:
            db.add(
                Company(
                    name="TechNova Labs",
                    slug=f"technova-{str(employer.id)[:8]}",
                    created_by_user_id=employer.id,
                )
            )
        _user(db, ADMIN_EMAIL, UserRole.ADMIN)
        db.commit()
        print(f"Demo accounts ready ({SEEKER_EMAIL}, {EMPLOYER_EMAIL}, {ADMIN_EMAIL}){' — created' if created else ''}.")


def main() -> int:
    if os.environ.get("ENVIRONMENT", "").lower() in ("production", "prod"):
        print("Refusing to seed demo accounts with a public password in production.")
        return 1
    from app.core.seed_skills import seed_skills
    from scripts import seed_canonical, seed_jobs

    seed_accounts()
    with SessionLocal() as db:
        print(f"Base skills added: {seed_skills(db)}")
    seed_jobs.seed_jobs()
    seed_canonical.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
