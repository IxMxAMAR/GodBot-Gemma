import asyncio
import json
import pytest
from godbot.core.agent import run_turn
from godbot.core.events import ToolCallEvent, ToolResultEvent, DoneEvent
from godbot.core.registry import Registry
from godbot.core.session import Session
from tests._mock_llm import MockLLM


@pytest.mark.asyncio
async def test_tool_call_then_final(tmp_path):
    reg = Registry()

    @reg.tool()
    def echo(text: str) -> str:
        """Echo text."""
        return f"echo:{text}"

    s = Session.create(root=tmp_path, model="m")
    s.append_user("say hi")

    llm = MockLLM([
        json.dumps({"thought": "use echo", "action": "echo", "args": {"text": "hi"}}),
        json.dumps({"thought": "got it", "final_answer": "echo:hi"}),
    ])
    events = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )
    tcs = [e for e in events if isinstance(e, ToolCallEvent)]
    trs = [e for e in events if isinstance(e, ToolResultEvent)]
    dones = [e for e in events if isinstance(e, DoneEvent)]
    assert len(tcs) == 1 and tcs[0].name == "echo"
    assert len(trs) == 1
    assert len(dones) == 1


@pytest.mark.asyncio
async def test_args_validation_failure_recovers(tmp_path):
    reg = Registry()

    @reg.tool()
    def needs_path(path: str) -> str:
        """Needs path."""
        return path

    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "needs_path", "args": {}}),
        json.dumps({"thought": "give up", "final_answer": "ok"}),
    ])
    events = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )
    trs = [e for e in events if e.__class__.__name__ == "ToolResultEvent"]
    assert any("invalid" in (e.preview or "").lower() or "required" in (e.preview or "").lower() for e in trs)


@pytest.mark.asyncio
async def test_tool_subset_filters_schema(tmp_path):
    reg = Registry()

    @reg.tool()
    def echo(text: str) -> str:
        """Echo."""
        return text

    @reg.tool()
    def banned(text: str) -> str:
        """Should not appear."""
        return text

    s = Session.create(root=tmp_path, model="m")
    s.set_tool_overrides(["echo"])
    s.append_user("call banned")

    llm = MockLLM([
        json.dumps({"thought": "x", "action": "banned", "args": {"text": "y"}}),  # schema rejects this
        json.dumps({"thought": "ok", "final_answer": "ok"}),
    ])
    events = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000, system_prompt="SYS",
    )
    # The first turn should be rejected by the schema (action enum doesn't include 'banned'),
    # synthetic_tool_result appended, then the agent recovers with final_answer.
    assert any(isinstance(e, DoneEvent) for e in events)
    # 'banned' should never be invoked.
    tcs = [e for e in events if e.__class__.__name__ == "ToolCallEvent"]
    assert all(e.name != "banned" for e in tcs)


@pytest.mark.asyncio
async def test_system_prompt_builder_filters_by_overrides(tmp_path):
    """The system prompt sent to the LLM must reflect session.tool_overrides,
    not the full registry — otherwise the model sees toggled-off tools in the
    catalog and wastes a turn calling them only to be rejected by the schema."""
    from godbot.prompts import build_system_prompt

    reg = Registry()

    @reg.tool()
    def echo(text: str) -> str:
        """Echo text."""
        return text

    @reg.tool()
    def banned(text: str) -> str:
        """Should not appear in the system prompt."""
        return text

    s = Session.create(root=tmp_path, model="m")
    s.set_tool_overrides(["echo"])
    s.append_user("hi")

    llm = MockLLM([
        json.dumps({"thought": "ok", "final_answer": "done"}),
    ])

    async def emit(ev):
        pass

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000,
        system_prompt="SHOULD_NOT_BE_USED",
        system_prompt_builder=build_system_prompt,
    )

    sys_msg = llm.calls[0]["messages"][0]
    assert sys_msg["role"] == "system"
    content = sys_msg["content"]
    assert "echo" in content
    assert "banned" not in content
    # The static fallback prompt must not be used when builder is supplied.
    assert "SHOULD_NOT_BE_USED" not in content
