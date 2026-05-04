"""Tests for sub-project 45 — /api/tasks/{tid}/wait long-poll."""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


def test_wait_404_unknown_task(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/tasks/t-nopenope/wait")
    assert r.status_code == 404


def test_wait_returns_immediately_for_terminal_task(tmp_path, monkeypatch):
    """Patch run_turn so the task finishes synchronously, then wait."""
    async def fake_run_turn(*, session, emit, **_):
        from godbot.core.events import DoneEvent
        session.append_assistant_final("ok")
        await emit(DoneEvent(step_count=1))

    import godbot.core.agent as agent_mod
    import godbot.core.tasks as tasks_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", tasks_mod.TaskRunner())

    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/tasks", json={"goal": "x", "workspace": str(tmp_path)})
    tid = r.json()["id"]
    # Give the runner a moment to land in terminal status.
    time.sleep(0.2)
    started = time.monotonic()
    r = c.get(f"/api/tasks/{tid}/wait?timeout=10")
    elapsed = time.monotonic() - started
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    # Should be near-instant since the task was already terminal.
    assert elapsed < 1.0


def test_wait_blocks_then_unblocks_when_status_flips(tmp_path, monkeypatch):
    """Drive the TaskRunner directly (avoiding TestClient cross-request loop
    quirks) and verify the wait endpoint sees the status flip."""
    import godbot.core.tasks as tasks_mod

    runner = tasks_mod.TaskRunner()
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", runner)
    # Inject a record in pending state, then flip to done from another thread.
    rec = tasks_mod.TaskRecord(
        id="t-flipme", goal="x", workspace=None, session_id=None,
        status="pending", started_at=time.time(),
    )
    runner._records["t-flipme"] = rec

    import threading
    def flip_after_delay():
        time.sleep(0.5)
        rec.status = "done"
        rec.result = "ok"
        rec.ended_at = time.time()
    threading.Thread(target=flip_after_delay, daemon=True).start()

    c = _client(tmp_path, monkeypatch)
    started = time.monotonic()
    r = c.get("/api/tasks/t-flipme/wait?timeout=5")
    elapsed = time.monotonic() - started
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert elapsed >= 0.4  # waited for the flip
    assert elapsed < 4.5   # didn't approach timeout


def test_wait_returns_running_record_on_timeout(tmp_path, monkeypatch):
    """If the task doesn't finish before timeout, return the in-progress record."""
    started_evt = asyncio.Event()

    async def stuck_run_turn(*, session, cancel, emit, **_):
        started_evt.set()
        # Stay running long enough that the wait timeout fires first.
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
    r = c.post("/api/tasks", json={"goal": "x", "workspace": str(tmp_path)})
    tid = r.json()["id"]
    # Wait with a tight timeout; should return the running record.
    r = c.get(f"/api/tasks/{tid}/wait?timeout=1")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("pending", "running")
    # Cleanup: cancel the stuck task.
    c.post(f"/api/tasks/{tid}/cancel")
    time.sleep(0.5)


def test_wait_clamps_huge_timeout(tmp_path, monkeypatch):
    """Sending a giant timeout should not crash the daemon — internally
    clamped to 300s. We just verify the endpoint accepts it without 4xx
    and returns an immediate response for a terminal task."""
    async def fake_run_turn(*, session, emit, **_):
        from godbot.core.events import DoneEvent
        session.append_assistant_final("ok")
        await emit(DoneEvent(step_count=1))

    import godbot.core.agent as agent_mod
    import godbot.core.tasks as tasks_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", tasks_mod.TaskRunner())

    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/tasks", json={"goal": "x", "workspace": str(tmp_path)})
    tid = r.json()["id"]
    time.sleep(0.2)
    r = c.get(f"/api/tasks/{tid}/wait?timeout=99999")
    assert r.status_code == 200
