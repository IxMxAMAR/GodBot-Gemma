"""Tests for sub-project 33 — /api/sessions/search."""
from __future__ import annotations

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.session import Session
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    app = build_app(sessions_root=sessions_root)
    return TestClient(app), sessions_root


def test_search_requires_q(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/search")
    assert r.status_code == 422  # FastAPI: missing required query param


def test_search_empty_q_string_400(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/search?q=")
    assert r.status_code == 400


def test_search_finds_in_user_message(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("how do I refactor the auth module")
    r = c.get("/api/sessions/search?q=auth")
    body = r.json()
    assert len(body["matches"]) == 1
    m = body["matches"][0]
    assert m["id"] == s.id
    assert m["role"] == "user"
    assert "auth" in m["snippet"].lower()


def test_search_finds_in_assistant_message(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("question")
    s.append_assistant_final("the answer is FOOBAR_TOKEN")
    r = c.get("/api/sessions/search?q=foobar_token")
    body = r.json()
    assert len(body["matches"]) == 1
    assert body["matches"][0]["role"] == "assistant"


def test_search_finds_in_tool_result(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.append_tool_result("c1", "found PATTERN_X in line 42")
    r = c.get("/api/sessions/search?q=pattern_x")
    body = r.json()
    assert len(body["matches"]) == 1
    assert body["matches"][0]["role"] == "tool"


def test_search_filters_by_workspace(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    ws_a = tmp_path / "ws_a"
    ws_a.mkdir()
    ws_b = tmp_path / "ws_b"
    ws_b.mkdir()
    s_a = Session.create(root=sroot, model="m", workspace_root=str(ws_a))
    s_a.append_user("auth flow question")
    s_b = Session.create(root=sroot, model="m", workspace_root=str(ws_b))
    s_b.append_user("auth flow question")

    r = c.get(f"/api/sessions/search?q=auth&workspace={ws_a.resolve()}")
    body = r.json()
    assert len(body["matches"]) == 1
    assert body["matches"][0]["id"] == s_a.id


def test_search_respects_limit(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    for _ in range(5):
        s = Session.create(root=sroot, model="m")
        s.append_user("query has a marker_word in it")
    r = c.get("/api/sessions/search?q=marker_word&limit=2")
    body = r.json()
    assert len(body["matches"]) == 2


def test_search_no_matches_returns_empty(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("hello world")
    r = c.get("/api/sessions/search?q=nothing-like-this")
    assert r.json()["matches"] == []


def test_search_one_match_per_session(tmp_path, monkeypatch):
    """Each session contributes at most one match (first hit wins)."""
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("first auth mention")
    s.append_assistant_final("second auth mention")
    r = c.get("/api/sessions/search?q=auth")
    body = r.json()
    assert len(body["matches"]) == 1
    # First hit is the user message.
    assert body["matches"][0]["role"] == "user"
