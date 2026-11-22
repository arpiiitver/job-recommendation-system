"""Career Copilot: agent loop, tools, memory, fallback, ownership.

The LLM is a scripted fake that speaks the ``chat_tools`` contract, so the
loop (tool dispatch, error recovery, persistence, trace) is tested without
network or quota.  With no provider, the deterministic router must serve.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import app.services.llm.tool_provider as provider_mod
from app.core.db import SessionLocal
from app.main import app
from app.models.copilot import CopilotMemory
from app.models.user import User
from app.services.agent.tools import AgentContext, _resolve_job_ref, execute
from app.services.llm.provider import LlmUsage
from app.services.llm.tool_provider import ToolCall, ToolTurn

client = TestClient(app, raise_server_exceptions=False)


def _register(role: str = "job_seeker") -> dict:
    uid = uuid4().hex[:8]
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": f"copilot_{uid}@example.com", "password": "Password123!", "role": role},
    )
    assert reg.status_code == 201, reg.text
    body = reg.json()
    return {"headers": {"Authorization": f"Bearer {body['access_token']}"}, "id": body["user"]["id"]}


def _delete(user_id) -> None:
    with SessionLocal() as db:
        row = db.get(User, UUID(str(user_id)))
        if row is not None:
            db.delete(row)
            db.commit()


@pytest.fixture
def seeker():
    acct = _register()
    yield acct
    _delete(acct["id"])


class ScriptedProvider:
    """Fake tool-calling provider: replays a list of ToolTurns, records inputs."""

    name = "fake"
    model = "fake-agent"
    supports_tools = True

    def __init__(self, turns: list[ToolTurn] | None = None, fail: bool = False) -> None:
        self.turns = list(turns or [])
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def chat_tools(self, messages, tools, **kwargs) -> ToolTurn:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": [t["function"]["name"] for t in tools]})
        if self.fail:
            raise RuntimeError("provider down")
        if self.turns:
            return self.turns.pop(0)
        return ToolTurn(content="(no more script)", tool_calls=[], usage=LlmUsage())

    def complete_json(self, schema, prompt, **kwargs):  # memory extraction
        return schema(), LlmUsage()


def _call(name: str, args: dict | None = None, cid: str = "c1") -> ToolTurn:
    return ToolTurn(content="", tool_calls=[ToolCall(id=cid, name=name, arguments=args or {})], usage=LlmUsage(prompt_tokens=10, completion_tokens=5))


def _final(text: str) -> ToolTurn:
    return ToolTurn(content=text, tool_calls=[], usage=LlmUsage(prompt_tokens=10, completion_tokens=5))


def _use(monkeypatch, fake: ScriptedProvider) -> None:
    monkeypatch.setattr(provider_mod, "get_tool_provider", lambda: fake)


# --------------------------------------------------------------------------- #
# agent loop
# --------------------------------------------------------------------------- #

def test_agent_calls_tool_then_answers(seeker, monkeypatch):
    fake = ScriptedProvider([_call("get_profile"), _final("Your profile is empty — upload a resume.")])
    _use(monkeypatch, fake)

    r = client.post("/api/v1/copilot/chat", headers=seeker["headers"], json={"message": "what do you know about me?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "llm"
    assert body["reply"].startswith("Your profile is empty")
    assert [t["tool"] for t in body["trace"]] == ["get_profile"]
    assert body["model"] == "fake-agent"

    # the tool result was handed back to the model as a `tool` message
    second = fake.calls[1]["messages"]
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"
    assert second[0]["role"] == "system" and "JobRec Copilot" in second[0]["content"]
    assert "search_jobs" in fake.calls[0]["tools"]

    # persisted + listed
    conv = client.get(f"/api/v1/copilot/conversations/{body['conversation_id']}", headers=seeker["headers"]).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][1]["meta"]["trace"][0]["tool"] == "get_profile"
    assert conv["title"].startswith("what do you know")


def test_unknown_tool_error_is_recoverable(seeker, monkeypatch):
    fake = ScriptedProvider([_call("delete_database"), _final("Sorry, I can't do that.")])
    _use(monkeypatch, fake)
    body = client.post("/api/v1/copilot/chat", headers=seeker["headers"], json={"message": "hi"}).json()
    assert body["trace"][0]["ok"] is False
    assert "Unknown tool" in fake.calls[1]["messages"][-1]["content"]
    assert body["reply"] == "Sorry, I can't do that."


def test_step_budget_forces_final_answer(seeker, monkeypatch):
    from app.core.config import get_settings

    steps = get_settings().agent_max_steps
    turns = [_call("get_profile", cid=f"c{i}") for i in range(steps)] + [_final("done")]
    fake = ScriptedProvider(turns)
    _use(monkeypatch, fake)
    body = client.post("/api/v1/copilot/chat", headers=seeker["headers"], json={"message": "loop"}).json()
    assert body["reply"] == "done"
    assert len(body["trace"]) == steps
    assert fake.calls[-1]["tools"] == []  # last round has tools disabled


def test_provider_failure_degrades_to_router(seeker, monkeypatch):
    _use(monkeypatch, ScriptedProvider(fail=True))
    body = client.post("/api/v1/copilot/chat", headers=seeker["headers"], json={"message": "recommend me jobs"}).json()
    assert body["mode"] == "fallback"
    assert "AI is unavailable" in body["reply"]


def test_history_replayed_across_turns(seeker, monkeypatch):
    fake = ScriptedProvider([_final("first answer"), _final("second answer")])
    _use(monkeypatch, fake)
    first = client.post("/api/v1/copilot/chat", headers=seeker["headers"], json={"message": "first question"}).json()
    client.post(
        "/api/v1/copilot/chat",
        headers=seeker["headers"],
        json={"message": "follow up", "conversation_id": first["conversation_id"]},
    )
    replay = [m for m in fake.calls[1]["messages"] if m["role"] != "system"]
    assert [m["content"] for m in replay] == ["first question", "first answer", "follow up"]


# --------------------------------------------------------------------------- #
# deterministic router (no provider — conftest default)
# --------------------------------------------------------------------------- #

def test_fallback_remember_and_recommend(seeker):
    r = client.post(
        "/api/v1/copilot/chat",
        headers=seeker["headers"],
        json={"message": "remember that I will not relocate outside Pune"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "fallback"
    mems = client.get("/api/v1/copilot/memories", headers=seeker["headers"]).json()
    assert any("relocate outside Pune" in m["content"] for m in mems)

    rec = client.post("/api/v1/copilot/chat", headers=seeker["headers"], json={"message": "recommend jobs for me"}).json()
    assert rec["trace"][0]["tool"] == "get_recommendations"


def test_fallback_direction_gap_without_targets(seeker):
    body = client.post(
        "/api/v1/copilot/chat", headers=seeker["headers"], json={"message": "I want to switch careers"}
    ).json()
    assert body["trace"][0]["tool"] == "career_direction_gap"
    assert "target roles" in body["reply"].lower()


def test_status_reports_fallback(seeker):
    s = client.get("/api/v1/copilot/status", headers=seeker["headers"]).json()
    assert s["mode"] == "fallback" and "search_jobs" in s["tools"]


def test_digest_endpoint(seeker):
    r = client.post("/api/v1/copilot/digest", headers=seeker["headers"])
    assert r.status_code == 200, r.text
    convs = client.get("/api/v1/copilot/conversations", headers=seeker["headers"]).json()
    assert any(c["title"] == "Daily matches" for c in convs)


# --------------------------------------------------------------------------- #
# memory + ownership + tools
# --------------------------------------------------------------------------- #

def test_memory_dedupe_and_delete(seeker):
    h = seeker["headers"]
    a = client.post("/api/v1/copilot/memories", headers=h, json={"content": "Prefers remote roles", "kind": "preference"})
    assert a.status_code == 201
    client.post("/api/v1/copilot/memories", headers=h, json={"content": "prefers remote roles", "kind": "preference"})
    mems = client.get("/api/v1/copilot/memories", headers=h).json()
    assert len(mems) == 1
    assert client.delete(f"/api/v1/copilot/memories/{mems[0]['id']}", headers=h).status_code == 204
    assert client.get("/api/v1/copilot/memories", headers=h).json() == []


def test_conversations_are_private(seeker):
    body = client.post("/api/v1/copilot/chat", headers=seeker["headers"], json={"message": "hello"}).json()
    other = _register()
    try:
        r = client.get(f"/api/v1/copilot/conversations/{body['conversation_id']}", headers=other["headers"])
        assert r.status_code == 404
        r = client.post(
            "/api/v1/copilot/chat",
            headers=other["headers"],
            json={"message": "hi", "conversation_id": body["conversation_id"]},
        )
        assert r.status_code == 404
        with SessionLocal() as db:
            assert db.query(CopilotMemory).filter(CopilotMemory.user_id == UUID(other["id"])).count() == 0
    finally:
        _delete(other["id"])


def test_employers_cannot_use_copilot():
    emp = _register("employer")
    try:
        r = client.post("/api/v1/copilot/chat", headers=emp["headers"], json={"message": "hi"})
        assert r.status_code == 403
    finally:
        _delete(emp["id"])


def test_job_ref_resolution_and_arg_validation():
    ctx = AgentContext(db=None, user=None)  # type: ignore[arg-type]
    ctx.last_list = ["aaa", "bbb"]
    assert _resolve_job_ref(ctx, "2") == "bbb"
    assert _resolve_job_ref(ctx, "9") == "9"
    assert _resolve_job_ref(ctx, "some-id") == "some-id"
    res = execute(ctx, "explain_match", {})
    assert not res.ok and "Missing required" in res.data["error"]
    res = execute(ctx, "search_jobs", {"query": ""})
    assert not res.ok


def test_fallback_ordinals_resolve_against_last_list(monkeypatch):
    from app.services.agent import fallback
    from app.services.agent.tools import ToolResult

    calls = []
    monkeypatch.setattr(fallback, "execute", lambda ctx, name, args: (calls.append((name, args)), ToolResult(data={"title": "T", "error": "x"}, ok=False))[1])
    ctx = AgentContext(db=None, user=None)  # type: ignore[arg-type]
    ctx.last_list = ["a", "b", "c"]
    fallback.route(ctx, "Why is the second one a good fit?")
    fallback.route(ctx, "save the last one")
    fallback.route(ctx, "tailor my resume for the first job")
    assert calls == [("explain_match", {"job_ref": "2"}), ("save_job", {"job_ref": "3"}), ("tailor_resume", {"job_ref": "1"})]


# --------------------------------------------------------------------------- #
# harmony with the ML pipeline: chat must never log impressions / bandit rows
# --------------------------------------------------------------------------- #

def test_chat_never_writes_recommendation_logs(seeker, monkeypatch):
    """Copilot uses the read-only half of the feed: no Recommendation snapshots
    or SERVED events (the bandit's online rewards and offline eval read those)."""
    from app.api import recommendations as rec_api
    from app.models.recommendation import Recommendation, RecommendationEvent

    calls: list[int] = []
    real_build = rec_api.build_recommendation_feed

    def _spy(db, user, limit):
        calls.append(limit)
        return real_build(db, user, limit)

    monkeypatch.setattr(rec_api, "build_recommendation_feed", _spy)
    persisted: list[int] = []
    monkeypatch.setattr(rec_api, "_persist_served", lambda *a, **k: persisted.append(1))

    client.post("/api/v1/copilot/chat", headers=seeker["headers"], json={"message": "recommend jobs for me"})
    client.post("/api/v1/copilot/digest", headers=seeker["headers"])

    assert len(calls) == 2, "copilot must use the shared ranking pipeline"
    assert persisted == [], "copilot must not log SERVED impressions"
    with SessionLocal() as db:
        from app.models.profile import Profile

        prof = db.query(Profile).filter(Profile.user_id == UUID(seeker["id"])).one_or_none()
        if prof is not None:
            assert db.query(Recommendation).filter(Recommendation.profile_id == prof.id).count() == 0
            assert db.query(RecommendationEvent).filter(RecommendationEvent.profile_id == prof.id).count() == 0


def test_tool_message_echoes_provider_fields():
    from app.services.llm.tool_provider import assistant_tool_message

    turn = ToolTurn(
        content="",
        tool_calls=[
            ToolCall(
                id="c9",
                name="search_jobs",
                arguments={"query": "python"},
                raw_arguments='{"query":"python"}',
                extra_content={"google": {"thought_signature": "sig"}},
            )
        ],
        usage=LlmUsage(),
    )
    msg = assistant_tool_message(turn)
    call = msg["tool_calls"][0]
    assert call["function"]["arguments"] == '{"query":"python"}'
    assert call["extra_content"]["google"]["thought_signature"] == "sig"


def test_rate_limit_window():
    from app.core import rate_limit

    rate_limit.reset()
    assert rate_limit.check("t", 2, now=0.0) is None
    assert rate_limit.check("t", 2, now=1.0) is None
    assert rate_limit.check("t", 2, now=2.0) is not None
    assert rate_limit.check("t", 2, now=61.0) is None
    rate_limit.reset()
