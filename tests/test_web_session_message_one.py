"""Tests for sub-project 90 — /api/sessions/{sid}/messages/{idx}."""
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


def test_message_404_unknown_session(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/messages/0")
    assert r.status_code == 404


def test_message_404_index_out_of_range(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("hi")
    r = c.get(f"/api/sessions/{s.id}/messages/99")
    assert r.status_code == 404


def test_message_returns_correct_index(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("first")
    s.append_assistant_final("reply 1")
    s.append_user("second")
    s.append_assistant_final("reply 2")

    body = c.get(f"/api/sessions/{s.id}/messages/2").json()
    assert body["role"] == "user"
    assert body["content"] == "second"
    assert body["index"] == 2


def test_message_first_index(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("opening")
    body = c.get(f"/api/sessions/{s.id}/messages/0").json()
    assert body["index"] == 0
    assert body["role"] == "user"


def test_message_last_index(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("q")
    s.append_assistant_final("a")
    body = c.get(f"/api/sessions/{s.id}/messages/1").json()
    assert body["index"] == 1
    assert body["role"] == "assistant"


def test_message_empty_session_404(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.get(f"/api/sessions/{s.id}/messages/0")
    assert r.status_code == 404
