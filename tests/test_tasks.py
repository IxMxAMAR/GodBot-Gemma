"""Tests for godbot.core.tasks (sub-project 13 — background agent tasks).

Drives :class:`TaskRunner` directly with a stubbed agent loop so the tests
don't need a live LLM. The web layer is covered separately.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from godbot.core.tasks import TaskRunner, _new_task_id


def test_task_id_format():
    tid = _new_task_id()
    assert tid.startswith("t-")
    assert len(tid) == 10  # "t-" + 8 hex
    assert all(c in "0123456789abcdef" for c in tid[2:])


@pytest.mark.asyncio
async def test_start_task_returns_pending_then_runs(tmp_path, monkeypatch):
    """Patch run_turn so we don't actually call an LLM, verify state transitions."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))

    captured: dict = {}

    async def fake_run_turn(*, session, registry, emit, cancel, max_steps, max_context,
                            system_prompt, system_prompt_builder=None, **kwargs):
        # Mark that we ran, then write a fake assistant final answer.
        captured["ran"] = True
        captured["session_id"] = session.id
        captured["tool_overrides"] = list(session.tool_overrides or [])
        session.append_assistant_final("the answer")
        from godbot.core.events import DoneEvent
        await emit(DoneEvent(step_count=1))

    import godbot.core.tasks as tasks_mod
    import godbot.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", TaskRunner())

    runner = tasks_mod.DEFAULT_RUNNER
    rec = runner.start_task(
        goal="research the auth flow",
        sessions_root=tmp_path / "sessions",
        workspace=str(tmp_path),
    )
    assert rec.status == "pending"
    # Yield to let the runner task start.
    for _ in range(20):
        await asyncio.sleep(0.05)
        if rec.status == "done":
            break

    assert rec.status == "done"
    assert rec.result == "the answer"
    assert rec.session_id == captured["session_id"]
    assert rec.step_count == 1
    # Safe-only filter active by default → tool_overrides excludes dangerous tools.
    # The fake session got a non-empty override list whose entries are all
    # non-dangerous in DEFAULT.
    from godbot.core.registry import DEFAULT
    danger = {t.name for t in DEFAULT.all() if t.dangerous}
    assert all(name not in danger for name in captured["tool_overrides"])


@pytest.mark.asyncio
async def test_explicit_tool_overrides_pass_through(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))

    captured: dict = {}

    async def fake_run_turn(*, session, **_):
        captured["overrides"] = list(session.tool_overrides or [])
        session.append_assistant_final("ok")
        from godbot.core.events import DoneEvent
        await _["emit"](DoneEvent(step_count=1))

    import godbot.core.tasks as tasks_mod
    import godbot.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", TaskRunner())

    runner = tasks_mod.DEFAULT_RUNNER
    rec = runner.start_task(
        goal="x",
        sessions_root=tmp_path / "sessions",
        workspace=str(tmp_path),
        tool_overrides=["read_file"],
    )
    for _ in range(20):
        await asyncio.sleep(0.05)
        if rec.status in ("done", "error"):
            break
    assert rec.status == "done"
    assert captured["overrides"] == ["read_file"]


@pytest.mark.asyncio
async def test_task_records_tool_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))

    async def fake_run_turn(*, session, emit, **_):
        from godbot.core.events import DoneEvent, ToolCallEvent, ToolResultEvent
        await emit(ToolCallEvent(id="c1", name="read_file", args={"path": "x"}))
        await emit(ToolResultEvent(id="c1", preview="hello", blob=None, duration_ms=42))
        session.append_assistant_final("done")
        await emit(DoneEvent(step_count=2))

    import godbot.core.tasks as tasks_mod
    import godbot.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", TaskRunner())

    runner = tasks_mod.DEFAULT_RUNNER
    rec = runner.start_task(
        goal="x", sessions_root=tmp_path / "sessions", workspace=str(tmp_path),
    )
    for _ in range(20):
        await asyncio.sleep(0.05)
        if rec.status == "done":
            break
    assert rec.status == "done"
    assert len(rec.tool_calls) == 1
    tc = rec.tool_calls[0]
    assert tc["name"] == "read_file"
    assert tc["result_preview"] == "hello"
    assert tc["duration_ms"] == 42


@pytest.mark.asyncio
async def test_cancel_task_marks_cancelled(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))

    started = asyncio.Event()

    async def fake_run_turn(*, session, cancel, emit, **_):
        started.set()
        # Wait for cancel to be set.
        for _ in range(50):
            if cancel.is_set():
                from godbot.core.events import ErrorEvent
                await emit(ErrorEvent(message="cancelled", recoverable=False))
                return
            await asyncio.sleep(0.05)

    import godbot.core.tasks as tasks_mod
    import godbot.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", TaskRunner())

    runner = tasks_mod.DEFAULT_RUNNER
    rec = runner.start_task(
        goal="x", sessions_root=tmp_path / "sessions", workspace=str(tmp_path),
    )
    await started.wait()
    sent = runner.cancel_task(rec.id)
    assert sent is True
    for _ in range(50):
        await asyncio.sleep(0.05)
        if rec.status in ("done", "error", "cancelled"):
            break
    assert rec.status == "cancelled"


def test_cancel_unknown_task_returns_false(tmp_path):
    runner = TaskRunner()
    assert runner.cancel_task("t-deadbeef") is False


def test_get_task_unknown_returns_none(tmp_path):
    runner = TaskRunner()
    assert runner.get_task("t-deadbeef") is None


@pytest.mark.asyncio
async def test_list_tasks_newest_first_with_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))

    async def fake_run_turn(*, session, emit, **_):
        session.append_assistant_final("ok")
        from godbot.core.events import DoneEvent
        await emit(DoneEvent(step_count=1))

    import godbot.core.tasks as tasks_mod
    import godbot.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", TaskRunner())

    runner = tasks_mod.DEFAULT_RUNNER
    recs = []
    for i in range(3):
        recs.append(runner.start_task(
            goal=f"goal-{i}", sessions_root=tmp_path / "sessions", workspace=str(tmp_path),
        ))
        await asyncio.sleep(0.01)
    for _ in range(50):
        await asyncio.sleep(0.05)
        if all(r.status == "done" for r in recs):
            break
    listed = runner.list_tasks()
    assert len(listed) == 3
    # Newest first.
    assert listed[0].id == recs[-1].id
    short = runner.list_tasks(limit=2)
    assert len(short) == 2


@pytest.mark.asyncio
async def test_clear_finished_only_drops_terminal(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))

    async def fake_run_turn(*, session, emit, **_):
        session.append_assistant_final("ok")
        from godbot.core.events import DoneEvent
        await emit(DoneEvent(step_count=1))

    import godbot.core.tasks as tasks_mod
    import godbot.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(tasks_mod, "DEFAULT_RUNNER", TaskRunner())

    runner = tasks_mod.DEFAULT_RUNNER
    r1 = runner.start_task(goal="x", sessions_root=tmp_path / "sessions", workspace=str(tmp_path))
    for _ in range(50):
        await asyncio.sleep(0.05)
        if r1.status == "done":
            break
    n = runner.clear_finished()
    assert n == 1
    assert runner.get_task(r1.id) is None
