"""Tests for sub-project 25 — SSE stream endpoint for background tasks."""
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


def test_task_stream_404_on_unknown(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/tasks/t-nopenope/stream")
    assert r.status_code == 404


def test_task_stream_yields_status_and_terminal(tmp_path, monkeypatch):
    """Schedule a task with a stubbed run_turn so it finishes immediately,
    then attach the SSE stream and read events to terminal."""
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
    assert r.status_code == 200
    tid = r.json()["id"]

    # Give the runner a moment to produce its events.
    time.sleep(0.2)

    events = []
    ids = []
    with c.stream("GET", f"/api/tasks/{tid}/stream", timeout=5.0) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("id:"):
                ids.append(line.split(":", 1)[1].strip())
            # Both `done` and a terminal status close the stream.
            if "done" in events:
                break
    # We must see the pending status, the running status, and at minimum a done.
    assert "status" in events
    assert "done" in events
    # IDs are monotonically increasing seq numbers.
    int_ids = [int(x) for x in ids]
    assert int_ids == sorted(int_ids)


def test_task_stream_shows_tool_calls(tmp_path, monkeypatch):
    async def fake_run_turn(*, session, emit, **_):
        from godbot.core.events import DoneEvent, ToolCallEvent, ToolResultEvent
        await emit(ToolCallEvent(id="c1", name="read_file", args={"path": "x"}))
        await emit(ToolResultEvent(id="c1", preview="hello", blob=None, duration_ms=42))
        session.append_assistant_final("done")
        await emit(DoneEvent(step_count=2))

    import godbot.core.agent as agent_mod
    import godbot.core.tasks as tasks_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", tasks_mod.TaskRunner())

    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/tasks", json={"goal": "x", "workspace": str(tmp_path)})
    tid = r.json()["id"]
    time.sleep(0.2)

    events = []
    with c.stream("GET", f"/api/tasks/{tid}/stream", timeout=5.0) as resp:
        for line in resp.iter_lines():
            if line.startswith("event:"):
                ev_name = line.split(":", 1)[1].strip()
                events.append(ev_name)
            if "done" in events:
                break
    assert "tool_call" in events
    assert "tool_result" in events
    assert "done" in events


def test_task_stream_replays_for_late_subscriber(tmp_path, monkeypatch):
    """Attach AFTER the runner finishes — the event log buffer should
    still replay all events including the terminal status."""
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

    # Wait for the task to fully finish.
    deadline = time.time() + 3
    while time.time() < deadline:
        rec = c.get(f"/api/tasks/{tid}").json()
        if rec["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)

    # Late subscriber: should still see the buffered events via replay.
    events = []
    with c.stream("GET", f"/api/tasks/{tid}/stream", timeout=5.0) as resp:
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if "done" in events or any(s in events for s in ("status",)) and "done" not in events and len(events) >= 4:
                # Either we saw the explicit terminal status row, or we've
                # consumed enough events that the stream should be ending.
                pass
            if "done" in events:
                break
    assert "done" in events


def test_task_stream_static_response_when_log_missing(tmp_path, monkeypatch):
    """If a task record exists but its EventLog has been GC'd (e.g.
    loaded from persistence after a restart), the endpoint emits a
    one-shot status event with the current record and exits."""
    import godbot.core.tasks as tasks_mod

    runner = tasks_mod.TaskRunner()
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", runner)
    # Inject a fake terminal record without an event log.
    rec = tasks_mod.TaskRecord(
        id="t-imported",
        goal="x", workspace=None, session_id=None,
        status="interrupted", started_at=time.time(),
        ended_at=time.time(), result="", error="restart",
    )
    runner._records["t-imported"] = rec

    c = _client(tmp_path, monkeypatch)
    events = []
    datas = []
    with c.stream("GET", "/api/tasks/t-imported/stream", timeout=5.0) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                datas.append(line.split(":", 1)[1].strip())
    assert "status" in events
    # The one-shot data should mention the interrupted status.
    assert any("interrupted" in d for d in datas)
