"""User reports on job listings -> admin queue -> remove (locked) / dismiss / restore."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.core.db import SessionLocal
from app.main import app
from app.models.application import Application
from app.models.company import Company
from app.models.enums import FlagTargetType, UserRole
from app.models.job import Job
from app.models.moderation import ModerationFlag
from app.models.user import User

client = TestClient(app, raise_server_exceptions=False)
PW = "Password123!"


def _register(role: str = "job_seeker", **extra) -> dict:
    email = f"report_{uuid4().hex[:10]}@example.com"
    r = client.post("/api/v1/auth/register", json={"email": email, "password": PW, "role": role, **extra})
    assert r.status_code == 201, r.text
    body = r.json()
    return {"headers": {"Authorization": f"Bearer {body['access_token']}"}, "id": body["user"]["id"], "email": email}


@pytest.fixture
def world():
    """Employer with one published job, three seekers and an admin."""
    employer = _register("employer", company_name=f"ReportCo {uuid4().hex[:6]}")
    job = client.post(
        "/api/v1/jobs",
        headers=employer["headers"],
        json={
            "title": "Data entry from home",
            "description": "Pay a registration fee to start earning from home today.",
            "location": "Pune, India",
        },
    )
    assert job.status_code == 201, job.text
    job_id = job.json()["id"]
    r = client.patch(f"/api/v1/employer/postings/{job_id}/status", headers=employer["headers"], json={"status": "published"})
    assert r.status_code == 200, r.text
    seekers = [_register() for _ in range(3)]
    admin = _register()
    with SessionLocal() as db:
        db.get(User, UUID(admin["id"])).role = UserRole.ADMIN
        db.commit()
    yield {"employer": employer, "job_id": job_id, "seekers": seekers, "admin": admin}

    accounts = [employer, admin, *seekers]
    with SessionLocal() as db:
        jid = UUID(job_id)
        db.execute(delete(ModerationFlag).where(ModerationFlag.target_type == FlagTargetType.JOB, ModerationFlag.target_id == jid))
        db.execute(delete(Application).where(Application.job_id == jid))
        job_row = db.get(Job, jid)
        company_id = job_row.company_id if job_row else None
        if job_row is not None:
            db.delete(job_row)
        db.commit()
        if company_id is not None:
            company = db.get(Company, company_id)
            if company is not None and not db.scalar(select(Job.id).where(Job.company_id == company_id).limit(1)):
                db.delete(company)
                db.commit()
        for acct in accounts:
            user = db.get(User, UUID(acct["id"]))
            if user is not None:
                db.delete(user)
        db.commit()


def _report(acct, job_id, reason="fake_or_scam", **extra):
    return client.post(f"/api/v1/jobs/{job_id}/report", headers=acct["headers"], json={"reason": reason, **extra})


def test_reports_count_distinct_users_and_escalate(world):
    jid, s1, s2 = world["job_id"], world["seekers"][0], world["seekers"][1]
    r = _report(s1, jid, "misleading", details="salary is fake")
    assert r.status_code == 201 and r.json()["created"] is True
    again = _report(s1, jid, "fake_or_scam")
    assert again.status_code == 200 and again.json()["created"] is False  # one open report per user
    assert client.get(f"/api/v1/jobs/{jid}/report/me", headers=s1["headers"]).json()["reported"] is True
    assert client.get(f"/api/v1/jobs/{jid}/report/me", headers=s2["headers"]).json()["reported"] is False

    admin = world["admin"]["headers"]
    multi = client.get("/api/v1/admin/moderation/reported-jobs", headers=admin, params={"min_reporters": 2}).json()
    assert all(it["job_id"] != jid for it in multi["items"])  # only one reporter so far

    assert _report(s2, jid, "asks_for_payment").status_code == 201
    multi = client.get("/api/v1/admin/moderation/reported-jobs", headers=admin, params={"min_reporters": 2}).json()
    item = next(it for it in multi["items"] if it["job_id"] == jid)
    assert item["reporters"] == 2 and item["multi_reported"] is True
    assert {r["severity"] for r in item["reports"]} == {"high"}  # consensus escalates every open report
    overview = client.get("/api/v1/admin/overview", headers=admin).json()["moderation"]
    assert overview["multi_reported_jobs"] >= 1 and overview["user_reported_jobs"] >= 1


def test_report_guards(world):
    jid = world["job_id"]
    assert _report(world["employer"], jid).status_code == 400  # own posting
    assert _report(world["seekers"][0], jid, "not_a_reason").status_code == 422
    assert _report(world["seekers"][0], "adzuna-12345").status_code == 400  # live listing
    assert client.post(f"/api/v1/jobs/{jid}/report", json={"reason": "other"}).status_code in (401, 403)
    assert client.get("/api/v1/admin/moderation/reported-jobs", headers=world["seekers"][0]["headers"]).status_code == 403
    reasons = client.get("/api/v1/job-reports/reasons").json()
    assert {"fake_or_scam", "asks_for_payment"} <= {r["key"] for r in reasons}


def test_remove_locks_posting_and_restore_unlocks(world):
    jid, emp, admin = world["job_id"], world["employer"]["headers"], world["admin"]["headers"]
    for s in world["seekers"][:2]:
        _report(s, jid, "asks_for_payment")
    r = client.post(f"/api/v1/admin/moderation/reported-jobs/{jid}/remove", headers=admin, json={"note": "Charges a fee"})
    assert r.status_code == 200 and r.json()["reports_resolved"] == 2

    mine = next(j for j in client.get("/api/v1/employer/jobs", headers=emp).json() if j["id"] == jid)
    assert mine["status"] == "archived" and mine["moderation_note"] == "Charges a fee" and mine["open_reports"] == 0
    blocked = client.patch(f"/api/v1/employer/postings/{jid}/status", headers=emp, json={"status": "published"})
    assert blocked.status_code == 403
    assert client.put(f"/api/v1/jobs/{jid}", headers=emp, json={"status": "published"}).status_code == 403
    queue = client.get("/api/v1/admin/moderation/reported-jobs", headers=admin).json()
    assert all(it["job_id"] != jid for it in queue["items"])  # no open reports left

    r = client.post(f"/api/v1/admin/moderation/reported-jobs/{jid}/restore", headers=admin)
    assert r.status_code == 200 and r.json()["job_status"] == "published"
    ok = client.patch(f"/api/v1/employer/postings/{jid}/status", headers=emp, json={"status": "closed"})
    assert ok.status_code == 200


def test_dismiss_keeps_listing(world):
    jid, admin = world["job_id"], world["admin"]["headers"]
    _report(world["seekers"][0], jid, "misleading")
    r = client.post(f"/api/v1/admin/moderation/reported-jobs/{jid}/dismiss", headers=admin, json={})
    assert r.status_code == 200 and r.json()["reports_dismissed"] == 1
    with SessionLocal() as db:
        assert db.get(Job, UUID(jid)).moderation_removed_at is None
    # the same user may report again after a dismissal
    assert _report(world["seekers"][0], jid, "misleading").status_code == 201


def test_admin_takedown_via_status_locks_too(world):
    jid, emp, admin = world["job_id"], world["employer"]["headers"], world["admin"]["headers"]
    assert client.patch(f"/api/v1/admin/jobs/{jid}", headers=admin, json={"status": "closed"}).status_code == 200
    assert client.patch(f"/api/v1/employer/postings/{jid}/status", headers=emp, json={"status": "published"}).status_code == 403
    assert client.patch(f"/api/v1/admin/jobs/{jid}", headers=admin, json={"status": "published"}).status_code == 200
    with SessionLocal() as db:
        assert db.get(Job, UUID(jid)).moderation_removed_at is None


def test_employer_overview_funnel(world):
    jid, emp = world["job_id"], world["employer"]["headers"]
    apps = []
    for s in world["seekers"]:
        r = client.post(f"/api/v1/jobs/{jid}/apply", headers=s["headers"], json={})
        assert r.status_code == 201, r.text
        apps.append(r.json()["id"])
    assert client.patch(f"/api/v1/employer/applications/{apps[0]}/status", headers=emp, json={"status": "hired"}).status_code == 200
    ov = client.get("/api/v1/employer/overview", headers=emp).json()
    assert ov["total_applicants"] == 3 and ov["counts"]["hired"] == 1
    assert ov["waiting_review"] == 2 and len(ov["needs_review"]) == 2
    assert ov["conversion"]["hired"] == pytest.approx(33.3, abs=0.1)
    assert ov["recent_hires"][0]["job_id"] == jid
    # another employer never sees these applicants
    other = _register("employer", company_name=f"Other {uuid4().hex[:6]}")
    try:
        assert client.get("/api/v1/employer/overview", headers=other["headers"]).json()["total_applicants"] == 0
    finally:
        with SessionLocal() as db:
            user = db.get(User, UUID(other["id"]))
            company_ids = [c.id for c in db.scalars(select(Company).where(Company.created_by_user_id == user.id)).all()] if user else []
            for cid in company_ids:
                db.delete(db.get(Company, cid))
            if user is not None:
                db.delete(user)
            db.commit()
