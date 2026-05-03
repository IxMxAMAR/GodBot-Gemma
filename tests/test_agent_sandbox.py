import asyncio
import json
import pytest
from godbot.core.agent import run_turn
from godbot.core.events import GateEvent, ToolResultEvent, DoneEvent
from godbot.core.registry import Registry
from godbot.core.session import Session
from tests._mock_llm import MockLLM


@pytest.mark.asyncio
async def test_workspace_active_auto_approves_write_file(tmp_path):
    reg = Registry()

    @reg.tool(dangerous=True)
    def write_file(path: str, content: str) -> str:
        """write."""
        return f"wrote {path}"

    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    s = Session.create(
        root=tmp_path / "sessions", model="m",
        workspace_root=str(ws_root), auto_approve_in_sandbox=True,
    )
    s.append_user("hi")

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "write_file",
                    "args": {"path": "hello.txt", "content": "y"}}),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    events = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )
    # No gate should have been emitted.
    assert not any(isinstance(e, GateEvent) for e in events)
    assert any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_workspace_active_still_gates_run_powershell(tmp_path):
    reg = Registry()

    @reg.tool(dangerous=True)
    def run_powershell(cmd: str) -> str:
        """ps."""
        return "ran"

    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    s = Session.create(
        root=tmp_path / "sessions", model="m",
        workspace_root=str(ws_root), auto_approve_in_sandbox=True,
    )
    s.append_user("hi")

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "run_powershell", "args": {"cmd": "Get-ChildItem"}}),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    events = []

    async def emit(ev):
        events.append(ev)
        if isinstance(ev, GateEvent):
            asyncio.create_task(_resolve(s, ev.id))

    async def _resolve(sess, call_id):
        await asyncio.sleep(0.01)
        sess.resolve_gate(call_id, "allow")

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )
    # Gate STILL fires for shell tools even in sandbox mode.
    assert any(isinstance(e, GateEvent) for e in events)


@pytest.mark.asyncio
async def test_sandbox_revokes_prior_always_for_shell(tmp_path):
    """Even if the user clicked 'always' on run_powershell pre-workspace,
    sandbox YOLO mode re-enables the gate for non-FS-safe dangerous tools."""
    reg = Registry()

    @reg.tool(dangerous=True)
    def run_powershell(cmd: str) -> str:
        """ps."""
        return "ran"

    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    s = Session.create(
        root=tmp_path / "sessions", model="m",
        workspace_root=str(ws_root), auto_approve_in_sandbox=True,
    )
    s.mark_auto_approved("run_powershell")  # pre-existing blanket approval
    s.append_user("hi")

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "run_powershell", "args": {"cmd": "Get-ChildItem"}}),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    events = []

    async def emit(ev):
        events.append(ev)
        if isinstance(ev, GateEvent):
            asyncio.create_task(_resolve(s, ev.id))

    async def _resolve(sess, call_id):
        await asyncio.sleep(0.01)
        sess.resolve_gate(call_id, "allow")

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )
    # Gate STILL fires even though run_powershell is auto_approved.
    assert any(isinstance(e, GateEvent) for e in events)


@pytest.mark.asyncio
async def test_no_workspace_means_normal_gating(tmp_path):
    reg = Registry()

    @reg.tool(dangerous=True)
    def write_file(path: str, content: str) -> str:
        """write."""
        return "wrote"

    s = Session.create(root=tmp_path / "sessions", model="m")
    s.append_user("hi")

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "write_file",
                    "args": {"path": "hello.txt", "content": "y"}}),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    events = []

    async def emit(ev):
        events.append(ev)
        if isinstance(ev, GateEvent):
            asyncio.create_task(_resolve(s, ev.id))

    async def _resolve(sess, call_id):
        await asyncio.sleep(0.01)
        sess.resolve_gate(call_id, "allow")

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )
    assert any(isinstance(e, GateEvent) for e in events)
