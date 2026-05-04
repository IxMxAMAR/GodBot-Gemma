"""Tests for sub-project 93 — /api/agent/quickrun."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


def test_quickrun_400_no_goal(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/quickrun", json={})
    assert r.status_code == 400


def test_quickrun_400_invalid_max_wait(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/quickrun", json={"goal": "x", "max_wait_seconds": "not-a-number"})
    assert r.status_code == 400


def test_quickrun_completes_synchronously(tmp_path, monkeypatch):
    """With a stubbed run_turn the task finishes fast; quickrun returns
    the final answer in one round trip."""
    async def fake_run_turn(*, session, emit, **_):
        from godbot.core.events import DoneEvent
        session.append_assistant_final("the answer")
        await emit(DoneEvent(step_count=1))

    import godbot.core.agent as agent_mod
    import godbot.core.tasks as tasks_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", tasks_mod.TaskRunner())

    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/quickrun", json={
        "goal": "what is 2+2?",
        "workspace": str(tmp_path),
        "max_wait_seconds": 5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert body["result"] == "the answer"
    assert body["task_id"].startswith("t-")
    assert body["session_id"] is not None
    assert body["step_count"] == 1
    assert body["elapsed_ms"] >= 0


def test_quickrun_returns_running_on_timeout(tmp_path, monkeypatch):
    """When the runner doesn't finish before max_wait, status='running'."""
    import asyncio

    async def stuck_run_turn(*, session, cancel, emit, **_):
        for _ in range(50):
            if cancel.is_set():
                from godbot.core.events import ErrorEvent
                await emit(ErrorEvent(message="cancelled", recoverable=False))
                return
            await asyncio.sleep(0.1)

    import godbot.core.agent as agent_mod
    import godbot.core.tasks as tasks_mod
    monkeypatch.setattr(agent_mod, "run_turn", stuck_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", tasks_mod.TaskRunner())

    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/quickrun", json={
        "goal": "stuck",
        "workspace": str(tmp_path),
        "max_wait_seconds": 2,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("running", "pending")
    # Cancel cleanup so the test doesn't leak the stuck task.
    c.post(f"/api/tasks/{body['task_id']}/cancel")


def test_quickrun_max_wait_clamped(tmp_path, monkeypatch):
    """max_wait_seconds clamps to [2, 600]."""
    async def fake_run_turn(*, session, emit, **_):
        from godbot.core.events import DoneEvent
        session.append_assistant_final("ok")
        await emit(DoneEvent(step_count=1))

    import godbot.core.agent as agent_mod
    import godbot.core.tasks as tasks_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", tasks_mod.TaskRunner())

    c = _client(tmp_path, monkeypatch)
    started = time.monotonic()
    r = c.post("/api/agent/quickrun", json={
        "goal": "x",
        "workspace": str(tmp_path),
        "max_wait_seconds": 0.01,  # below floor of 2s
    })
    elapsed = time.monotonic() - started
    assert r.status_code == 200
    # Should have completed quickly because the task finishes under the floor.
    assert elapsed < 5.0
