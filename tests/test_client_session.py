import asyncio
import json
import pytest
import respx
import httpx
from godbot.client import Session
from godbot.core.events import DoneEvent
from godbot.core.registry import Registry
from tests._mock_llm import MockLLM


@pytest.mark.asyncio
async def test_session_falls_through_to_embedded_when_no_daemon(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    reg = Registry()
    llm = MockLLM([json.dumps({"thought": "ok", "final_answer": "hi"})])

    def factory():
        from godbot.client.embedded import EmbeddedRunner
        return EmbeddedRunner.create(
            sessions_root=tmp_path / "sessions", registry=reg, llm=llm,
        )

    async with Session(
        base_url="http://127.0.0.1:9999",  # nothing listening
        embedded_factory=factory,
    ) as s:
        assert s.mode == "embedded"
        events = [ev async for ev in s.run("hi")]
        assert any(isinstance(e, DoneEvent) for e in events)


@respx.mock
@pytest.mark.asyncio
async def test_session_uses_daemon_when_health_returns_200(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    respx.get("http://127.0.0.1:7878/api/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.post("http://127.0.0.1:7878/api/sessions/new").mock(
        return_value=httpx.Response(200, json={"session_id": "s1"})
    )
    respx.post("http://127.0.0.1:7878/api/chat").mock(
        return_value=httpx.Response(200, json={"session_id": "s1"})
    )
    body = "event: token\ndata: {\"type\":\"token\",\"text\":\"hi\"}\n\nevent: done\ndata: {\"type\":\"done\",\"step_count\":1}\n\n"
    respx.get("http://127.0.0.1:7878/api/chat/stream").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )
    )
    async with Session() as s:
        assert s.mode == "daemon"
        events = [ev async for ev in s.run("hi")]
    types = [type(e).__name__ for e in events]
    assert "DoneEvent" in types


@pytest.mark.asyncio
async def test_session_resolve_gate_in_embedded_mode_calls_runner(tmp_path, monkeypatch):
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

    def factory():
        from godbot.client.embedded import EmbeddedRunner
        return EmbeddedRunner.create(
            sessions_root=tmp_path / "sessions", registry=reg, llm=llm,
        )

    async with Session(base_url="http://127.0.0.1:9999", embedded_factory=factory) as s:
        async def _resolve(sess, call_id):
            await asyncio.sleep(0.01)
            await sess.resolve_gate(call_id, "allow")

        async def consume():
            async for ev in s.run("boom"):
                if ev.__class__.__name__ == "GateEvent":
                    asyncio.create_task(_resolve(s, ev.id))
        await consume()
