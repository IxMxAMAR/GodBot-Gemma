import asyncio
import json
import pytest
from godbot.client.embedded import EmbeddedRunner
from godbot.core.events import DoneEvent, TokenEvent
from godbot.core.registry import Registry
from tests._mock_llm import MockLLM


@pytest.mark.asyncio
async def test_embedded_runner_yields_events_to_done(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    reg = Registry()
    llm = MockLLM([json.dumps({"thought": "easy", "final_answer": "ok"})])
    runner = EmbeddedRunner.create(
        sessions_root=tmp_path / "sessions",
        registry=reg,
        llm=llm,
    )
    events = []
    async for ev in runner.run("hi"):
        events.append(ev)
    types = [type(e).__name__ for e in events]
    assert "DoneEvent" in types
    # MockLLM streams character by character — many TokenEvents.
    assert any(isinstance(e, TokenEvent) for e in events)


@pytest.mark.asyncio
async def test_embedded_runner_resolve_gate_routes_to_session(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    reg = Registry()

    @reg.tool(dangerous=True)
    def boom(target: str) -> str:
        """Boom."""
        return "ran"

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "boom", "args": {"target": "1"}}),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])
    runner = EmbeddedRunner.create(
        sessions_root=tmp_path / "sessions",
        registry=reg,
        llm=llm,
    )

    async def _resolve_after(runner, call_id, delay):
        await asyncio.sleep(delay)
        await runner.resolve_gate(call_id, "allow")

    async def consume():
        events = []
        async for ev in runner.run("boom"):
            events.append(ev)
            if ev.__class__.__name__ == "GateEvent":
                # Schedule resolve in a separate task so the consumer continues.
                asyncio.create_task(_resolve_after(runner, ev.id, 0.01))
        return events

    events = await consume()
    types = [type(e).__name__ for e in events]
    assert "ToolResultEvent" in types
    assert "DoneEvent" in types


@pytest.mark.asyncio
async def test_embedded_runner_stop_cancels_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    reg = Registry()
    # Script never ends with final_answer; loop runs until cancel.
    llm = MockLLM([
        json.dumps({"thought": "loop", "final_answer": "x"}),
    ])
    runner = EmbeddedRunner.create(
        sessions_root=tmp_path / "sessions",
        registry=reg,
        llm=llm,
    )
    await runner.stop()  # cancel before we even start
    events = [ev async for ev in runner.run("hi")]
    types = [type(e).__name__ for e in events]
    assert "ErrorEvent" in types
