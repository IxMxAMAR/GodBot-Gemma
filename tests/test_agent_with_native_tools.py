"""End-to-end agent loop tests with the new Provider abstraction.

Uses a fake in-process Provider (no HTTP) to assert the agent loop:
  - drives a NATIVE_TOOLS turn → executes the tool → completes on the
    second turn with a final answer
  - drives a REACT_JSON turn through a Provider unchanged from the legacy
    behaviour
  - still accepts the legacy ``llm=`` shim (MockLLM) without changes
"""
from __future__ import annotations

import asyncio
import json

import pytest

from godbot.core.agent import run_turn
from godbot.core.events import DoneEvent, ToolCallEvent, ToolResultEvent
from godbot.core.providers import (
    NATIVE_TOOLS,
    REACT_JSON,
    ModelInfo,
    ParsedToolCall,
    Provider,
    ProviderConfig,
    TurnResult,
)
from godbot.core.registry import Registry
from godbot.core.session import Session
from tests._mock_llm import MockLLM


class _ScriptedProvider(Provider):
    """A scripted Provider for agent-loop tests.

    ``responses`` is a list where each entry is one of:
      - ``("react", text)`` — return a ReAct envelope as raw_text
      - ``("native_call", tool_name, args)`` — emit a native tool_use
      - ``("native_final", text)`` — emit a final assistant message
    """

    def __init__(self, config: ProviderConfig, responses, protocol: str = NATIVE_TOOLS):
        super().__init__(config)
        self._responses = list(responses)
        self._protocol = protocol
        self.calls: list[dict] = []

    async def list_models(self):
        return [ModelInfo(id="fake", supports_native_tools=True, provider=self.config.name)]

    async def select_model(self, requested):
        return ModelInfo(
            id=requested or "fake",
            supports_native_tools=(self._protocol == NATIVE_TOOLS),
            provider=self.config.name,
        )

    def preferred_protocol(self, model):
        return self._protocol

    async def complete_streaming(
        self, model, messages, on_delta, protocol,
        tool_schemas=None, react_schema=None, cancel=None,
        temperature=0.7, max_tokens=4096,
    ):
        self.calls.append({
            "messages": messages, "protocol": protocol,
            "tool_schemas": tool_schemas, "react_schema": react_schema,
        })
        if not self._responses:
            raise AssertionError("_ScriptedProvider: no more scripted responses")
        kind, *rest = self._responses.pop(0)
        if kind == "react":
            text = rest[0]
            for ch in text:
                if cancel is not None and cancel.is_set():
                    return TurnResult(finish_reason="cancelled")
                r = on_delta(ch)
                if asyncio.iscoroutine(r):
                    await r
            return TurnResult(raw_text=text, finish_reason="stop")
        if kind == "native_final":
            text = rest[0]
            for ch in text:
                r = on_delta(ch)
                if asyncio.iscoroutine(r):
                    await r
            return TurnResult(final_answer=text, raw_text=text, finish_reason="stop")
        if kind == "native_call":
            name, args = rest
            return TurnResult(
                tool_calls=[ParsedToolCall(id=f"call_{name}", name=name, args=args)],
                raw_text="",
                finish_reason="tool_calls",
            )
        raise AssertionError(f"unknown scripted entry {kind!r}")


# ---- ReAct path through a Provider --------------------------------------


@pytest.mark.asyncio
async def test_react_path_through_provider_completes(tmp_path):
    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    p = _ScriptedProvider(
        ProviderConfig(name="fake"),
        responses=[("react", json.dumps({"thought": "easy", "final_answer": "hi back"}))],
        protocol=REACT_JSON,
    )
    events: list = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        provider=p, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=3, max_context=10000,
        system_prompt="SYS",
    )
    assert any(isinstance(e, DoneEvent) for e in events)
    msgs = s.messages_for_llm()
    assert msgs[-1] == {"role": "assistant", "content": "hi back"}


# ---- Native-tools end-to-end --------------------------------------------


@pytest.mark.asyncio
async def test_native_tools_call_then_final(tmp_path):
    reg = Registry()

    @reg.tool()
    def echo(text: str = "") -> str:
        """Echo the input back."""
        return f"echoed: {text}"

    s = Session.create(root=tmp_path, model="m")
    s.append_user("please echo")
    p = _ScriptedProvider(
        ProviderConfig(name="fake"),
        responses=[
            ("native_call", "echo", {"text": "hi"}),
            ("native_final", "Done!"),
        ],
        protocol=NATIVE_TOOLS,
    )
    events: list = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        provider=p, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=5, max_context=10000,
        system_prompt="SYS",
    )

    # Tool call + result + done all surfaced.
    assert any(isinstance(e, ToolCallEvent) and e.name == "echo" for e in events)
    assert any(isinstance(e, ToolResultEvent) for e in events)
    assert any(isinstance(e, DoneEvent) for e in events)

    # Final assistant message recorded.
    msgs = s.messages_for_llm()
    assert msgs[-1] == {"role": "assistant", "content": "Done!"}

    # Provider was given the OpenAI tool schema in turn 1.
    assert p.calls[0]["protocol"] == NATIVE_TOOLS
    schemas = p.calls[0]["tool_schemas"]
    assert schemas and schemas[0]["function"]["name"] == "echo"


# ---- Backward compat: legacy llm= shim still works ----------------------


@pytest.mark.asyncio
async def test_legacy_llm_shim_still_works(tmp_path):
    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    llm = MockLLM([json.dumps({"thought": "easy", "final_answer": "hello"})])
    events: list = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=3, max_context=10000,
        system_prompt="SYS",
    )
    assert any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_run_turn_requires_exactly_one_of_llm_or_provider(tmp_path):
    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    with pytest.raises(ValueError):
        await run_turn(
            session=s, registry=reg, emit=lambda e: None,
            cancel=asyncio.Event(), max_steps=1, max_context=10000,
            system_prompt="x",
        )


# ---- session.protocol override is honoured ------------------------------


@pytest.mark.asyncio
async def test_session_protocol_override_forces_react(tmp_path):
    reg = Registry()
    s = Session.create(
        root=tmp_path, model="m",
        provider="fake", model_name="fake", protocol=REACT_JSON,
    )
    s.append_user("hi")
    # Provider would prefer NATIVE_TOOLS, but session pins REACT_JSON.
    p = _ScriptedProvider(
        ProviderConfig(name="fake"),
        responses=[("react", json.dumps({"thought": "k", "final_answer": "ok"}))],
        protocol=NATIVE_TOOLS,
    )
    events: list = []

    async def emit(ev):
        events.append(ev)

    await run_turn(
        provider=p, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=3, max_context=10000,
        system_prompt="SYS",
    )
    assert any(isinstance(e, DoneEvent) for e in events)
    # The provider was called with REACT_JSON, not NATIVE_TOOLS.
    assert p.calls[0]["protocol"] == REACT_JSON
    assert p.calls[0]["react_schema"] is not None
