"""User reports on job listings and the admin actions that resolve them.

Flow
----
1. A logged-in user reports a platform listing (``report_job``). The report
   is a ``ModerationFlag`` with ``origin="user_report"``; one OPEN report per
   user per job (enforced in the DB), so counts mean *distinct people*.
2. Severity escalates with consensus: scam/payment reasons are HIGH on their
   own, and every open report on a job becomes HIGH once
   ``MULTI_REPORT_THRESHOLD`` different users have reported it.
3. An admin works the "Reported by users" queue (``reported_jobs``) and either
   removes the listing (``remove_job``: archived + moderation lock + all open
   reports resolved) or dismisses the reports (``dismiss_reports``).
   ``restore_job`` lifts a lock that was applied by mistake.

Nothing here hides a listing automatically: reports are triage, the admin is
the decision maker (competitors could otherwise mass-report real postings).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.enums import FlagSeverity, FlagStatus, FlagTargetType, JobStatus
from app.models.job import Job
from app.models.moderation import ModerationFlag
from app.models.user import User

USER_REPORT = "user_report"
# "Flagged by multiple users": from this many distinct reporters a listing is
# highlighted for admins and its reports escalate to HIGH severity.
MULTI_REPORT_THRESHOLD = 2

REPORT_REASONS: dict[str, tuple[str, FlagSeverity]] = {
    "fake_or_scam": ("Fake job or scam", FlagSeverity.HIGH),
    "asks_for_payment": ("Asks candidates to pay a fee", FlagSeverity.HIGH),
    "misleading": ("Misleading pay, role or company details", FlagSeverity.MEDIUM),
    "already_filled": ("Position no longer available", FlagSeverity.LOW),
    "duplicate_or_spam": ("Duplicate or spam posting", FlagSeverity.MEDIUM),
    "offensive": ("Offensive or discriminatory content", FlagSeverity.HIGH),
    "other": ("Other", FlagSeverity.MEDIUM),
}


class ReportError(ValueError):
    """Business-rule rejection (mapped to HTTP 4xx by the API layer)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ReportOutcome:
    created: bool
    flag: ModerationFlag
    distinct_reporters: int


def _open_reports_q(job_id: UUID):
    return select(ModerationFlag).where(
        ModerationFlag.target_type == FlagTargetType.JOB,
        ModerationFlag.target_id == job_id,
        ModerationFlag.origin == USER_REPORT,
        ModerationFlag.status == FlagStatus.OPEN,
    )


def distinct_reporters(db: Session, job_id: UUID) -> int:
    return int(
        db.scalar(
            select(func.count(func.distinct(ModerationFlag.created_by_user_id))).where(
                ModerationFlag.target_type == FlagTargetType.JOB,
                ModerationFlag.target_id == job_id,
                ModerationFlag.origin == USER_REPORT,
                ModerationFlag.status == FlagStatus.OPEN,
            )
        )
        or 0
    )


def open_report_counts(db: Session, job_ids: list[UUID]) -> dict[UUID, int]:
    """Distinct open reporters per job (for employer/admin listings)."""
    if not job_ids:
        return {}
    return {
        jid: int(n)
        for jid, n in db.execute(
            select(
                ModerationFlag.target_id,
                func.count(func.distinct(ModerationFlag.created_by_user_id)),
            )
            .where(
                ModerationFlag.target_type == FlagTargetType.JOB,
                ModerationFlag.target_id.in_(job_ids),
                ModerationFlag.origin == USER_REPORT,
                ModerationFlag.status == FlagStatus.OPEN,
            )
            .group_by(ModerationFlag.target_id)
        ).all()
    }


def my_open_report(db: Session, user: User, job_id: UUID) -> ModerationFlag | None:
    return db.scalar(_open_reports_q(job_id).where(ModerationFlag.created_by_user_id == user.id))


def report_job(db: Session, user: User, job_id: UUID, reason: str, details: str | None) -> ReportOutcome:
    if reason not in REPORT_REASONS:
        raise ReportError(f"Unknown reason '{reason}'.", 422)
    job = db.get(Job, job_id)
    if job is None:
        raise ReportError("Job not found.", 404)
    if job.posted_by_user_id == user.id:
        raise ReportError("You cannot report your own posting.", 400)

    existing = my_open_report(db, user, job_id)
    if existing is not None:
        return ReportOutcome(created=False, flag=existing, distinct_reporters=distinct_reporters(db, job_id))

    label, severity = REPORT_REASONS[reason]
    flag = ModerationFlag(
        target_type=FlagTargetType.JOB,
        target_id=job.id,
        target_label=job.title[:255],
        reason=label,
        details=(details or "").strip()[:2000] or None,
        severity=severity,
        status=FlagStatus.OPEN,
        created_by_user_id=user.id,
        origin=USER_REPORT,
    )
    db.add(flag)
    try:
        db.flush()
    except IntegrityError:  # concurrent duplicate from the same user
        db.rollback()
        existing = my_open_report(db, user, job_id)
        if existing is None:
            raise
        return ReportOutcome(created=False, flag=existing, distinct_reporters=distinct_reporters(db, job_id))

    count = distinct_reporters(db, job_id)
    if count >= MULTI_REPORT_THRESHOLD:
        for f in db.scalars(_open_reports_q(job_id)).all():
            f.severity = FlagSeverity.HIGH
    db.commit()
    db.refresh(flag)
    return ReportOutcome(created=True, flag=flag, distinct_reporters=count)


def _close_reports(db: Session, admin: User, job_id: UUID, status: FlagStatus, note: str | None) -> int:
    now = datetime.now(timezone.utc)
    flags = db.scalars(
        select(ModerationFlag).where(
            ModerationFlag.target_type == FlagTargetType.JOB,
            ModerationFlag.target_id == job_id,
            ModerationFlag.status == FlagStatus.OPEN,
        )
    ).all()
    for f in flags:
        f.status = status
        f.resolved_by_user_id = admin.id
        f.resolved_at = now
        f.resolution_note = note
    return len(flags)


def remove_job(db: Session, admin: User, job_id: UUID, note: str | None) -> tuple[Job, int]:
    """Take a listing down: archived, locked against republishing, reports resolved."""
    job = db.get(Job, job_id)
    if job is None:
        raise ReportError("Job not found.", 404)
    note = (note or "").strip() or "Removed after review of user reports."
    job.status = JobStatus.ARCHIVED
    job.moderation_removed_at = datetime.now(timezone.utc)
    job.moderation_removed_by_user_id = admin.id
    job.moderation_note = note[:2000]
    job.updated_at = datetime.now(timezone.utc)
    closed = _close_reports(db, admin, job_id, FlagStatus.RESOLVED, note)
    db.commit()
    db.refresh(job)
    return job, closed


def dismiss_reports(db: Session, admin: User, job_id: UUID, note: str | None) -> int:
    if db.get(Job, job_id) is None:
        raise ReportError("Job not found.", 404)
    closed = _close_reports(
        db, admin, job_id, FlagStatus.DISMISSED, (note or "").strip() or "Reports reviewed — listing is legitimate."
    )
    db.commit()
    return closed


def restore_job(db: Session, admin: User, job_id: UUID, publish: bool = True) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise ReportError("Job not found.", 404)
    job.moderation_removed_at = None
    job.moderation_removed_by_user_id = None
    job.moderation_note = None
    if publish:
        job.status = JobStatus.PUBLISHED
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    return job


def lock_on_admin_status_change(job: Job, admin: User, new_status: JobStatus) -> None:
    """Admin moved a posting via the generic status endpoint: taking it down
    locks it like ``remove_job``; publishing lifts the lock."""
    if new_status in (JobStatus.CLOSED, JobStatus.ARCHIVED):
        if job.moderation_removed_at is None:
            job.moderation_removed_at = datetime.now(timezone.utc)
            job.moderation_removed_by_user_id = admin.id
            job.moderation_note = job.moderation_note or "Taken down by an administrator."
    elif new_status == JobStatus.PUBLISHED:
        job.moderation_removed_at = None
        job.moderation_removed_by_user_id = None
        job.moderation_note = None
