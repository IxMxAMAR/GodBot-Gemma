import asyncio
import json
import pytest
from godbot.core.agent import run_turn
from godbot.core.events import ErrorEvent, DoneEvent
from godbot.core.registry import Registry
from godbot.core.session import Session
from tests._mock_llm import MockLLM


@pytest.mark.asyncio
async def test_max_steps_emits_error(tmp_path):
    reg = Registry()

    @reg.tool()
    def loop(text: str) -> str:
        """Always echo."""
        return text

    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")

    scripts = [
        json.dumps({"thought": str(i), "action": "loop", "args": {"text": "x"}})
        for i in range(20)
    ]
    llm = MockLLM(scripts)
    events = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=3, max_context=10000, system_prompt="SYS",
    )
    errs = [e for e in events if isinstance(e, ErrorEvent)]
    assert errs and "max_steps" in errs[-1].message


@pytest.mark.asyncio
async def test_cancellation_pre_step(tmp_path):
    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    llm = MockLLM([json.dumps({"thought": "x", "final_answer": "y"})])
    cancel = asyncio.Event()
    cancel.set()
    events = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=cancel, max_steps=3, max_context=10000, system_prompt="SYS",
    )
    assert any(isinstance(e, ErrorEvent) and e.message == "cancelled" for e in events)
