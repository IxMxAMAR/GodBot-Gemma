"""Live smoke test — only runs when GODBOT_LIVE=1 and LM Studio has a Gemma loaded.

Run: `GODBOT_LIVE=1 pytest tests/test_lmstudio_smoke.py -v`
"""
from __future__ import annotations
import asyncio
import os
import pytest

from godbot.core.agent import run_turn
from godbot.core.events import DoneEvent
from godbot.core.llm import LLMClient
from godbot.core.registry import Registry
from godbot.core.session import Session
from godbot.prompts import build_system_prompt


pytestmark = pytest.mark.skipif(os.environ.get("GODBOT_LIVE") != "1", reason="set GODBOT_LIVE=1")


@pytest.mark.asyncio
async def test_live_final_answer(tmp_path):
    reg = Registry()

    @reg.tool()
    def get_secret() -> str:
        """Returns a constant secret string."""
        return "PURPLE_OWL"

    s = Session.create(root=tmp_path, model="auto")
    s.append_user("Use the get_secret tool, then tell me the secret in your final_answer.")
    llm = LLMClient(base_url="http://localhost:1234/v1", model="auto")
    info = llm.probe()
    print(f"Live model: {info.id} ctx={info.context_length}")

    events = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=10, max_context=info.context_length or 28000,
        system_prompt=build_system_prompt(reg.all()),
    )
    final = next((e for e in events if isinstance(e, DoneEvent)), None)
    assert final is not None, [type(e).__name__ for e in events]
    msgs = s.messages_for_llm()
    answer = msgs[-1]["content"].lower()
    assert "purple" in answer or "owl" in answer
