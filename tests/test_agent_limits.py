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
    # The new error message should include actionable guidance about how
    # to bump or disable the brake — discovered while debugging the
    # rootkit_demo loop where "max_steps exceeded" alone was unhelpful.
    assert "config.toml" in errs[-1].message or "0 to disable" in errs[-1].message


@pytest.mark.asyncio
async def test_max_steps_zero_disables_brake(tmp_path):
    """max_steps=0 means unlimited. The agent runs until it emits a
    final_answer (or the user cancels). Verify with a script that
    eventually returns final_answer after many tool calls."""
    reg = Registry()

    @reg.tool()
    def noop(text: str = "") -> str:
        """No-op."""
        return "ok"

    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")

    # 50 tool turns then a final_answer — would exceed the old default
    # of 25 and the new default of 100, but with max_steps=0 it runs.
    scripts = [
        json.dumps({"thought": str(i), "action": "noop", "args": {"text": "x"}})
        for i in range(50)
    ]
    scripts.append(json.dumps({"thought": "done", "final_answer": "all done"}))

    llm = MockLLM(scripts)
    events = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=0, max_context=10000,
        system_prompt="SYS",
    )
    # We should NOT see a max_steps error.
    assert not any(
        isinstance(e, ErrorEvent) and "max_steps" in e.message
        for e in events
    )
    # We SHOULD see DoneEvent with step_count > 25 (proving the brake
    # didn't fire at the old default).
    dones = [e for e in events if isinstance(e, DoneEvent)]
    assert dones
    assert dones[-1].step_count > 25


@pytest.mark.asyncio
async def test_max_steps_negative_also_disables(tmp_path):
    """Sentinel: max_steps=-1 (or any negative) disables the brake too,
    since the test 'max_steps <= 0' covers it."""
    reg = Registry()

    @reg.tool()
    def noop(text: str = "") -> str:
        """No-op."""
        return "ok"

    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    scripts = [
        json.dumps({"thought": str(i), "action": "noop", "args": {"text": "x"}})
        for i in range(30)
    ]
    scripts.append(json.dumps({"thought": "done", "final_answer": "ok"}))
    llm = MockLLM(scripts)
    events = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=-1, max_context=10000,
        system_prompt="SYS",
    )
    assert not any(
        isinstance(e, ErrorEvent) and "max_steps" in e.message
        for e in events
    )


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
