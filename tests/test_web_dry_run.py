"""Tests for sub-project 49 — /api/agent/dry_run prompt debugger."""
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


def test_dry_run_400_missing_session_id(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/dry_run", json={"message": "hi"})
    assert r.status_code == 400


def test_dry_run_404_unknown_session(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/dry_run", json={"session_id": "no-such", "message": "hi"})
    assert r.status_code == 404


def test_dry_run_default_mode_renders_prompt(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.post("/api/agent/dry_run", json={"session_id": s.id, "message": "hi there"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "default"
    assert "Available tools:" in body["system_prompt"]
    assert isinstance(body["tool_catalog"], list)
    assert any(t["name"] == "read_file" for t in body["tool_catalog"])


def test_dry_run_plan_mode_detected(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.post("/api/agent/dry_run", json={"session_id": s.id, "message": "/plan rewrite README"})
    body = r.json()
    assert body["mode"] == "plan"
    assert "PLAN MODE" in body["system_prompt"]


def test_dry_run_custom_command_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    cmd_dir = tmp_path / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "doc.toml").write_text(
        'description = "doc"\nsystem_prompt_suffix = "DOC-MODE"\n', encoding="utf-8"
    )

    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    app = build_app(sessions_root=sessions_root)
    c = TestClient(app)
    s = Session.create(root=sessions_root, model="m")
    r = c.post("/api/agent/dry_run", json={"session_id": s.id, "message": "/doc add docstrings"})
    body = r.json()
    assert body["mode"] == "custom_command"
    assert body["command_name"] == "doc"
    assert "DOC-MODE" in body["system_prompt"]


def test_dry_run_falls_back_to_last_user_message(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("/plan from history")
    # No `message` field — should pick the last user event from the session.
    r = c.post("/api/agent/dry_run", json={"session_id": s.id})
    body = r.json()
    assert body["mode"] == "plan"


def test_dry_run_does_not_mutate_session(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("existing message")
    before = list(s._events())
    c.post("/api/agent/dry_run", json={"session_id": s.id, "message": "extra"})
    after = list(s._events())
    assert before == after  # no events appended
