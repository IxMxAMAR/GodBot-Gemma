"""Tests for sub-project 72 — /api/sessions/{sid}/messages."""
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


def test_messages_404_unknown(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/messages")
    assert r.status_code == 404


def test_messages_basic_pagination(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    for i in range(10):
        s.append_user(f"q{i}")
        s.append_assistant_final(f"a{i}")
    r = c.get(f"/api/sessions/{s.id}/messages?limit=5")
    body = r.json()
    assert len(body["messages"]) == 5
    assert body["total"] == 20
    assert body["has_more"] is True


def test_messages_offset_pagination(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    for i in range(5):
        s.append_user(f"q{i}")
    r = c.get(f"/api/sessions/{s.id}/messages?offset=2&limit=2")
    body = r.json()
    assert [m["content"] for m in body["messages"]] == ["q2", "q3"]
    assert body["has_more"] is True


def test_messages_index_present(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("hi")
    s.append_assistant_final("hello")
    body = c.get(f"/api/sessions/{s.id}/messages").json()
    assert body["messages"][0]["index"] == 0
    assert body["messages"][1]["index"] == 1


def test_messages_role_filter(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("hi")
    s.append_assistant_final("hello")
    s.append_user("again")
    s.append_assistant_final("yes")
    body = c.get(f"/api/sessions/{s.id}/messages?role=user").json()
    assert len(body["messages"]) == 2
    assert all(m["role"] == "user" for m in body["messages"])


def test_messages_invalid_role_400(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.get(f"/api/sessions/{s.id}/messages?role=tool")
    assert r.status_code == 400


def test_messages_no_more_when_exhausted(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("only")
    body = c.get(f"/api/sessions/{s.id}/messages").json()
    assert body["has_more"] is False
    assert body["total"] == 1
