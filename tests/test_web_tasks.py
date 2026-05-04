"""Integration tests for /api/tasks (sub-project 13)."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


def test_start_task_requires_goal(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/tasks", json={})
    assert r.status_code == 400


def test_tool_overrides_must_be_list(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/tasks", json={"goal": "x", "tool_overrides": "read_file"})
    assert r.status_code == 400


def test_get_unknown_task_404(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/tasks/t-nopenope")
    assert r.status_code == 404


def test_cancel_unknown_task_404(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/tasks/t-nopenope/cancel")
    assert r.status_code == 404


def test_start_task_returns_pending_record(tmp_path, monkeypatch):
    """Patch run_turn so the task completes quickly without an LLM call."""
    async def fake_run_turn(*, session, emit, **_):
        session.append_assistant_final("ok")
        from godbot.core.events import DoneEvent
        await emit(DoneEvent(step_count=1))

    import godbot.core.agent as agent_mod
    import godbot.core.tasks as tasks_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", tasks_mod.TaskRunner())

    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/tasks", json={
        "goal": "investigate the auth flow",
        "workspace": str(tmp_path),
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"].startswith("t-")
    assert body["goal"] == "investigate the auth flow"
    assert body["status"] in ("pending", "running", "done")


def test_list_tasks_returns_array(tmp_path, monkeypatch):
    async def fake_run_turn(*, session, emit, **_):
        session.append_assistant_final("ok")
        from godbot.core.events import DoneEvent
        await emit(DoneEvent(step_count=1))

    import godbot.core.agent as agent_mod
    import godbot.core.tasks as tasks_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", tasks_mod.TaskRunner())

    c = _client(tmp_path, monkeypatch)
    c.post("/api/tasks", json={"goal": "g1", "workspace": str(tmp_path)})
    c.post("/api/tasks", json={"goal": "g2", "workspace": str(tmp_path)})
    r = c.get("/api/tasks")
    assert r.status_code == 200
    body = r.json()
    assert "tasks" in body
    assert len(body["tasks"]) == 2
