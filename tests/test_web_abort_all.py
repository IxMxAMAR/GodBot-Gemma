"""Tests for sub-project 98 — /api/agent/abort_all."""
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


def test_abort_all_no_active_returns_zeros(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/abort_all")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["sessions_signaled"] == 0
    assert body["tasks_signaled"] == 0


def test_abort_all_signals_cancel_events(tmp_path, monkeypatch):
    """Manually plant a cancel event in the web module and confirm
    abort_all sets it."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    from godbot.interfaces import web as webmod
    ev = asyncio.Event()
    webmod._cancels["fake-sid"] = ev
    try:
        assert ev.is_set() is False
        r = c.post("/api/agent/abort_all")
        body = r.json()
        assert body["sessions_signaled"] == 1
        assert ev.is_set() is True
    finally:
        webmod._cancels.pop("fake-sid", None)


def test_abort_all_signals_non_terminal_task(tmp_path, monkeypatch):
    """Drive the TaskRunner directly (avoiding TestClient cross-request loop
    quirks) to verify abort_all signals a pending task."""
    import godbot.core.tasks as tasks_mod

    runner = tasks_mod.TaskRunner()
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", runner)
    rec = tasks_mod.TaskRecord(
        id="t-aborttest", goal="x", workspace=None, session_id=None,
        status="running", started_at=0.0,
    )
    runner._records["t-aborttest"] = rec
    runner._cancels["t-aborttest"] = asyncio.Event()

    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/abort_all")
    body = r.json()
    assert body["tasks_signaled"] == 1
    assert runner._cancels["t-aborttest"].is_set() is True


def test_abort_all_skips_already_terminal(tmp_path, monkeypatch):
    """A finished task should NOT be counted as signaled."""
    async def fake_run_turn(*, session, emit, **_):
        from godbot.core.events import DoneEvent
        session.append_assistant_final("ok")
        await emit(DoneEvent(step_count=1))

    import godbot.core.agent as agent_mod
    import godbot.core.tasks as tasks_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", tasks_mod.TaskRunner())

    c = _client(tmp_path, monkeypatch)
    c.post("/api/tasks", json={"goal": "x", "workspace": str(tmp_path)})
    import time as _time
    _time.sleep(0.4)  # task should be done
    r = c.post("/api/agent/abort_all")
    body = r.json()
    assert body["tasks_signaled"] == 0


def test_abort_all_idempotent(tmp_path, monkeypatch):
    """Calling abort_all twice in a row is safe (already-set events skip)."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    from godbot.interfaces import web as webmod
    webmod._cancels["sid-a"] = asyncio.Event()
    try:
        first = c.post("/api/agent/abort_all").json()
        assert first["sessions_signaled"] == 1
        second = c.post("/api/agent/abort_all").json()
        assert second["sessions_signaled"] == 0  # already set
    finally:
        webmod._cancels.pop("sid-a", None)
