import asyncio
import json
import pytest
from godbot.core.agent import run_turn
from godbot.core.events import TokenEvent, DoneEvent
from godbot.core.registry import Registry
from godbot.core.session import Session
from tests._mock_llm import MockLLM


@pytest.mark.asyncio
async def test_final_answer_in_one_turn(tmp_path):
    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    llm = MockLLM([json.dumps({"thought": "easy", "final_answer": "hello!"})])
    events = []

    async def emit(ev):
        events.append(ev)

    cancel = asyncio.Event()
    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit, cancel=cancel,
        max_steps=5, max_context=10000, system_prompt="SYS",
    )
    assert any(isinstance(e, DoneEvent) for e in events)
    msgs = s.messages_for_llm()
    assert msgs[-1] == {"role": "assistant", "content": "hello!"}
