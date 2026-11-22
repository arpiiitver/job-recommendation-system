"""Report a job listing (any logged-in user) — feeds the admin moderation queue.

POST /jobs/{job_id}/report      create (idempotent per user while open)
GET  /jobs/{job_id}/report/me   has the caller already reported it?
GET  /job-reports/reasons       reason vocabulary for the UI
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.db import get_db
from app.core.rate_limit import rate_limited
from app.models.user import User
from app.services import job_reports as svc

router = APIRouter(tags=["job-reports"])

ReasonKey = Literal[
    "fake_or_scam", "asks_for_payment", "misleading", "already_filled", "duplicate_or_spam", "offensive", "other"
]


class ReportIn(BaseModel):
    reason: ReasonKey
    details: str | None = Field(default=None, max_length=2000)


class ReportOut(BaseModel):
    reported: bool
    created: bool = False
    reason: str | None = None


def _parse_job_id(job_id: str) -> UUID:
    try:
        return UUID(job_id)
    except ValueError:
        # live Adzuna listings ("adzuna-…") have no row on this platform
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Live listings from external job boards can't be reported here — "
            "please report them on the original site.",
        ) from None


@router.get("/job-reports/reasons")
def report_reasons() -> list[dict[str, str]]:
    return [{"key": k, "label": label} for k, (label, _sev) in svc.REPORT_REASONS.items()]


@router.post(
    "/jobs/{job_id}/report",
    response_model=ReportOut,
    dependencies=[Depends(rate_limited("job_report", per_minute=10))],
)
def report_job(
    job_id: str,
    payload: ReportIn,
    response: Response,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ReportOut:
    try:
        outcome = svc.report_job(db, current_user, _parse_job_id(job_id), payload.reason, payload.details)
    except svc.ReportError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    response.status_code = status.HTTP_201_CREATED if outcome.created else status.HTTP_200_OK
    # The reporter learns only that their report is recorded — not how many
    # others reported it (no signal for coordinated abuse).
    return ReportOut(reported=True, created=outcome.created, reason=outcome.flag.reason)


@router.get("/jobs/{job_id}/report/me", response_model=ReportOut)
def my_report(
    job_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ReportOut:
    try:
        jid = UUID(job_id)
    except ValueError:
        return ReportOut(reported=False)
    flag = svc.my_open_report(db, current_user, jid)
    return ReportOut(reported=flag is not None, reason=flag.reason if flag else None)
