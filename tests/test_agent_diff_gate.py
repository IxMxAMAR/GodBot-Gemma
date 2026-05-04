"""Tests for fs_diff payload + args_override path on dangerous-tool gates."""
from __future__ import annotations
import asyncio
import json

import pytest

from godbot.core.agent import run_turn
from godbot.core.events import GateEvent, ToolResultEvent
from godbot.core.registry import Registry
from godbot.core.session import Session
from tests._mock_llm import MockLLM


@pytest.mark.asyncio
async def test_write_file_gate_includes_fs_diff(tmp_path):
    """Gating a write_file call emits a GateEvent with fs_diff populated."""
    reg = Registry()

    @reg.tool(dangerous=True)
    def write_file(path: str, content: str) -> str:
        """write."""
        from pathlib import Path
        Path(path).write_text(content, encoding="utf-8")
        return f"wrote {path}"

    target = tmp_path / "hello.py"
    target.write_text("old\n", encoding="utf-8")

    s = Session.create(root=tmp_path / "sessions", model="m")
    s.append_user("rewrite hello.py")

    llm = MockLLM([
        json.dumps({
            "thought": "rewrite",
            "action": "write_file",
            "args": {"path": str(target), "content": "new\n"},
        }),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    events: list = []

    async def emit(ev):
        events.append(ev)
        if isinstance(ev, GateEvent):
            asyncio.create_task(_resolve(ev.id, "allow"))

    async def _resolve(call_id, decision):
        await asyncio.sleep(0.01)
        s.resolve_gate(call_id, decision)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )

    gates = [e for e in events if isinstance(e, GateEvent)]
    assert len(gates) == 1
    assert gates[0].fs_diff is not None
    assert gates[0].fs_diff["path"] == str(target)
    assert gates[0].fs_diff["before"] == "old\n"
    assert gates[0].fs_diff["after"] == "new\n"
    # Tool actually ran with original args.
    assert target.read_text(encoding="utf-8") == "new\n"


@pytest.mark.asyncio
async def test_write_file_gate_creating_new_file_has_null_before(tmp_path):
    """When the target doesn't exist, fs_diff.before is None (null in JSON)."""
    reg = Registry()

    @reg.tool(dangerous=True)
    def write_file(path: str, content: str) -> str:
        """write."""
        from pathlib import Path
        Path(path).write_text(content, encoding="utf-8")
        return "ok"

    target = tmp_path / "new.txt"  # doesn't exist
    s = Session.create(root=tmp_path / "sessions", model="m")
    s.append_user("create file")

    llm = MockLLM([
        json.dumps({
            "thought": "create",
            "action": "write_file",
            "args": {"path": str(target), "content": "fresh"},
        }),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    gates: list = []

    async def emit(ev):
        if isinstance(ev, GateEvent):
            gates.append(ev)
            asyncio.create_task(_resolve(ev.id, "allow"))

    async def _resolve(call_id, decision):
        await asyncio.sleep(0.01)
        s.resolve_gate(call_id, decision)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )

    assert len(gates) == 1
    assert gates[0].fs_diff is not None
    assert gates[0].fs_diff["before"] is None
    assert gates[0].fs_diff["after"] == "fresh"


@pytest.mark.asyncio
async def test_edit_file_gate_simulates_replacement(tmp_path):
    """edit_file fs_diff.after is the simulated find/replace result."""
    reg = Registry()

    @reg.tool(dangerous=True)
    def edit_file(path: str, old: str, new: str) -> str:
        """edit."""
        from pathlib import Path
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return "ok"

    target = tmp_path / "f.py"
    target.write_text("alpha beta gamma\n", encoding="utf-8")

    s = Session.create(root=tmp_path / "sessions", model="m")
    s.append_user("edit")

    llm = MockLLM([
        json.dumps({
            "thought": "edit",
            "action": "edit_file",
            "args": {"path": str(target), "old": "beta", "new": "BETA"},
        }),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    gates: list = []

    async def emit(ev):
        if isinstance(ev, GateEvent):
            gates.append(ev)
            asyncio.create_task(_resolve(ev.id, "allow"))

    async def _resolve(call_id, decision):
        await asyncio.sleep(0.01)
        s.resolve_gate(call_id, decision)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )

    assert len(gates) == 1
    assert gates[0].fs_diff is not None
    assert gates[0].fs_diff["before"] == "alpha beta gamma\n"
    assert gates[0].fs_diff["after"] == "alpha BETA gamma\n"


@pytest.mark.asyncio
async def test_args_override_applied(tmp_path):
    """resolve_gate(args_override={...}) overrides the agent's args before execute."""
    reg = Registry()
    captured: dict = {}

    @reg.tool(dangerous=True)
    def write_file(path: str, content: str) -> str:
        """write."""
        captured["content"] = content
        captured["path"] = path
        return "ok"

    target = tmp_path / "x.txt"
    s = Session.create(root=tmp_path / "sessions", model="m")
    s.append_user("write")

    llm = MockLLM([
        json.dumps({
            "thought": "write",
            "action": "write_file",
            "args": {"path": str(target), "content": "AGENT"},
        }),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])

    async def emit(ev):
        if isinstance(ev, GateEvent):
            asyncio.create_task(_resolve(ev.id))

    async def _resolve(call_id):
        await asyncio.sleep(0.01)
        s.resolve_gate(call_id, "allow", args_override={"content": "USER_EDITED"})

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )

    assert captured["content"] == "USER_EDITED"
    assert captured["path"] == str(target)


@pytest.mark.asyncio
async def test_args_override_bad_type_rejected(tmp_path):
    """A malicious override that breaks the tool's schema is rejected before execute."""
    reg = Registry()
    ran = {"called": False}

    @reg.tool(dangerous=True)
    def write_file(path: str, content: str) -> str:
        """write."""
        ran["called"] = True
        return "should not run"

    target = tmp_path / "x.txt"
    s = Session.create(root=tmp_path / "sessions", model="m")
    s.append_user("write")

    llm = MockLLM([
        json.dumps({
            "thought": "write",
            "action": "write_file",
            "args": {"path": str(target), "content": "AGENT"},
        }),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    results: list = []

    async def emit(ev):
        if isinstance(ev, ToolResultEvent):
            results.append(ev)
        if isinstance(ev, GateEvent):
            asyncio.create_task(_resolve(ev.id))

    async def _resolve(call_id):
        await asyncio.sleep(0.01)
        # content must be a string per tool schema; a list violates it.
        s.resolve_gate(call_id, "allow", args_override={"content": ["not", "a", "string"]})

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )

    assert ran["called"] is False
    assert results and "args invalid after override" in results[0].preview


@pytest.mark.asyncio
async def test_non_fs_dangerous_tool_has_null_fs_diff(tmp_path):
    """Non-FS-write dangerous tools (e.g. shells) still gate with fs_diff=None."""
    reg = Registry()

    @reg.tool(dangerous=True)
    def run_shell(cmd: str) -> str:
        """run."""
        return "ok"

    s = Session.create(root=tmp_path / "sessions", model="m")
    s.append_user("run")

    llm = MockLLM([
        json.dumps({"thought": "run", "action": "run_shell", "args": {"cmd": "ls"}}),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    gates: list = []

    async def emit(ev):
        if isinstance(ev, GateEvent):
            gates.append(ev)
            asyncio.create_task(_resolve(ev.id))

    async def _resolve(call_id):
        await asyncio.sleep(0.01)
        s.resolve_gate(call_id, "allow")

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )

    assert len(gates) == 1
    assert gates[0].fs_diff is None
