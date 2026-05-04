"""Tests for sub-project 17 — TaskRunner disk persistence.

Verifies that task records survive a TaskRunner re-instantiation, and
that records left in a non-terminal state get flipped to ``interrupted``
on reload.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from godbot.core.tasks import TaskRecord, TaskRunner, _new_task_id


def _make_record(**overrides):
    base = dict(
        id=_new_task_id(),
        goal="x",
        workspace=None,
        session_id=None,
        status="running",
        started_at=time.time(),
    )
    base.update(overrides)
    return TaskRecord(**base)


def test_attach_persistence_creates_dir(tmp_path):
    runner = TaskRunner()
    n = runner.attach_persistence(tmp_path / "tasks")
    assert n == 0
    assert (tmp_path / "tasks").exists()


def test_attach_persistence_loads_existing(tmp_path):
    pdir = tmp_path / "tasks"
    pdir.mkdir()
    rec = _make_record(status="done", result="answer")
    (pdir / f"{rec.id}.json").write_text(
        json.dumps({
            "id": rec.id, "goal": rec.goal, "workspace": rec.workspace,
            "session_id": rec.session_id, "status": "done",
            "started_at": rec.started_at, "ended_at": time.time(),
            "result": "answer", "error": "", "step_count": 1,
            "tool_calls": [], "model": "", "provider": "",
        }),
        encoding="utf-8",
    )
    runner = TaskRunner()
    n = runner.attach_persistence(pdir)
    assert n == 1
    loaded = runner.get_task(rec.id)
    assert loaded is not None
    assert loaded.status == "done"
    assert loaded.result == "answer"


def test_attach_persistence_flips_non_terminal_to_interrupted(tmp_path):
    pdir = tmp_path / "tasks"
    pdir.mkdir()
    rec = _make_record(status="running")
    (pdir / f"{rec.id}.json").write_text(
        json.dumps({
            "id": rec.id, "goal": "x", "workspace": None,
            "session_id": None, "status": "running",
            "started_at": rec.started_at, "ended_at": None,
            "result": "", "error": "", "step_count": 0,
            "tool_calls": [], "model": "", "provider": "",
        }),
        encoding="utf-8",
    )
    runner = TaskRunner()
    runner.attach_persistence(pdir)
    loaded = runner.get_task(rec.id)
    assert loaded is not None
    assert loaded.status == "interrupted"
    assert "restart" in loaded.error.lower()
    assert loaded.ended_at is not None
    # And the on-disk file is rewritten with the flipped status.
    on_disk = json.loads((pdir / f"{rec.id}.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "interrupted"


def test_attach_persistence_skips_corrupt_files(tmp_path):
    pdir = tmp_path / "tasks"
    pdir.mkdir()
    (pdir / "t-bad.json").write_text("{not json", encoding="utf-8")
    runner = TaskRunner()
    n = runner.attach_persistence(pdir)
    assert n == 0


@pytest.mark.asyncio
async def test_start_task_writes_to_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))

    async def fake_run_turn(*, session, emit, **_):
        session.append_assistant_final("done")
        from godbot.core.events import DoneEvent
        await emit(DoneEvent(step_count=1))

    import godbot.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)

    pdir = tmp_path / "tasks"
    runner = TaskRunner()
    runner.attach_persistence(pdir)
    rec = runner.start_task(
        goal="hello",
        sessions_root=tmp_path / "sessions",
        workspace=str(tmp_path),
    )
    # Record should be on disk immediately (status pending or running).
    p = pdir / f"{rec.id}.json"
    assert p.exists()
    for _ in range(40):
        await asyncio.sleep(0.05)
        if rec.status == "done":
            break
    # And the final state was persisted.
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["status"] == "done"


@pytest.mark.asyncio
async def test_clear_finished_deletes_disk_files(tmp_path, monkeypatch):
    async def fake_run_turn(*, session, emit, **_):
        session.append_assistant_final("ok")
        from godbot.core.events import DoneEvent
        await emit(DoneEvent(step_count=1))

    import godbot.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", fake_run_turn)
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))

    pdir = tmp_path / "tasks"
    runner = TaskRunner()
    runner.attach_persistence(pdir)
    rec = runner.start_task(
        goal="x", sessions_root=tmp_path / "sessions", workspace=str(tmp_path),
    )
    for _ in range(40):
        await asyncio.sleep(0.05)
        if rec.status == "done":
            break
    p = pdir / f"{rec.id}.json"
    assert p.exists()
    runner.clear_finished()
    assert not p.exists()
