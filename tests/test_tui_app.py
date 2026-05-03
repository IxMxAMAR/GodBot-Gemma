import asyncio
import json
from pathlib import Path

import pytest

from godbot.interfaces.tui import TuiApp
from godbot.client.embedded import EmbeddedRunner
from godbot.core.registry import Registry
from tests._mock_llm import MockLLM


def _factory(tmp_path, llm, reg):
    def make():
        return EmbeddedRunner.create(
            sessions_root=tmp_path / "sessions",
            registry=reg,
            llm=llm,
        )
    return make


@pytest.mark.asyncio
async def test_send_message_renders_assistant_bubble(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    reg = Registry()
    llm = MockLLM([json.dumps({"thought": "ok", "final_answer": "hi back"})])

    app = TuiApp(
        base_url="http://127.0.0.1:9999",  # nothing listening
        auto_launch=False,
        sessions_root=tmp_path / "sessions",
    )
    app._embedded_factory = _factory(tmp_path, llm, reg)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Send message
        composer = app.query_one("MessageInput")
        composer._area.text = "hi"
        composer._submit()
        # Wait for the stream to complete.
        for _ in range(40):
            await pilot.pause(0.05)
            if "hi back" in app.conversation.model.last_assistant_text():
                break
        assert "hi back" in app.conversation.model.last_assistant_text()


@pytest.mark.asyncio
async def test_gate_button_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    reg = Registry()

    @reg.tool(dangerous=True)
    def boom(target: str) -> str:
        """boom."""
        return "boomed"

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "boom", "args": {"target": "x"}}),
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])

    app = TuiApp(
        base_url="http://127.0.0.1:9999",
        auto_launch=False,
        sessions_root=tmp_path / "sessions",
    )
    app._embedded_factory = _factory(tmp_path, llm, reg)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        composer = app.query_one("MessageInput")
        composer._area.text = "boom"
        composer._submit()
        # Wait for gate.
        for _ in range(40):
            await pilot.pause(0.05)
            if app.conversation.model.pending_gates():
                break
        gates = app.conversation.model.pending_gates()
        assert len(gates) == 1
        # Click Allow on the gate widget
        from godbot.interfaces.tui_widgets.conversation import GateDecided
        app.post_message(GateDecided(call_id=gates[0].id, decision="allow"))
        for _ in range(40):
            await pilot.pause(0.05)
            if any(c.result == "boomed" for c in app.conversation.model._tool_cards.values()):
                break
        assert any(c.result == "boomed" for c in app.conversation.model._tool_cards.values())
