import asyncio
import json
import pytest
from godbot.core.agent import run_turn
from godbot.core.events import GateEvent, ToolResultEvent, DoneEvent
from godbot.core.registry import Registry
from godbot.core.session import Session
from tests._mock_llm import MockLLM


@pytest.mark.asyncio
async def test_gate_allow_runs_tool(tmp_path):
    reg = Registry()

    @reg.tool(dangerous=True)
    def boom(target: str) -> str:
        """Boom."""
        return f"boomed {target}"

    s = Session.create(root=tmp_path, model="m")
    s.append_user("boom")

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "boom", "args": {"target": "x"}}),
        json.dumps({"thought": "done", "final_answer": "ok"}),
    ])
    events = []

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
    trs = [e for e in events if isinstance(e, ToolResultEvent)]
    assert trs and "boomed x" in trs[0].preview


@pytest.mark.asyncio
async def test_gate_deny_returns_denied_message(tmp_path):
    reg = Registry()

    @reg.tool(dangerous=True)
    def boom(target: str) -> str:
        """Boom."""
        return "should not run"

    s = Session.create(root=tmp_path, model="m")
    s.append_user("boom")

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "boom", "args": {"target": "y"}}),
        json.dumps({"thought": "done", "final_answer": "ok"}),
    ])
    events = []

    async def emit(ev):
        events.append(ev)
        if isinstance(ev, GateEvent):
            asyncio.create_task(_resolve(ev.id, "deny"))

    async def _resolve(call_id, decision):
        await asyncio.sleep(0.01)
        s.resolve_gate(call_id, decision)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )
    trs = [e for e in events if isinstance(e, ToolResultEvent)]
    assert trs and "denied" in trs[0].preview.lower()


@pytest.mark.asyncio
async def test_gate_always_then_skips_next_time(tmp_path):
    reg = Registry()

    @reg.tool(dangerous=True)
    def boom(target: str) -> str:
        """Boom."""
        return "ran"

    s = Session.create(root=tmp_path, model="m")
    s.append_user("boom twice")

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "boom", "args": {"target": "1"}}),
        json.dumps({"thought": "x", "action": "boom", "args": {"target": "2"}}),
        json.dumps({"thought": "done", "final_answer": "ok"}),
    ])
    gate_events = []

    async def emit(ev):
        if isinstance(ev, GateEvent):
            gate_events.append(ev)
            asyncio.create_task(_resolve(ev.id, "always"))

    async def _resolve(call_id, decision):
        await asyncio.sleep(0.01)
        s.resolve_gate(call_id, decision)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )
    assert len(gate_events) == 1
    assert s.is_auto_approved("boom")
