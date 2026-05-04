"""AnthropicProvider tests — all wire calls mocked via respx."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from godbot.core.providers import NATIVE_TOOLS, REACT_JSON, ProviderConfig
from godbot.core.providers.anthropic import (
    AnthropicProvider,
    _openai_tools_to_anthropic,
    _translate_messages,
)
from godbot.core.providers.base import ModelInfo


def _sse(events: list[tuple[str, dict]]) -> str:
    """Build an Anthropic SSE event-stream body.

    Anthropic prefixes each event with both ``event:`` and ``data:`` lines;
    our parser only consumes the ``data:`` payload, so we emit just that
    (the ``type`` field inside the JSON drives our state machine)."""
    body = ""
    for _name, payload in events:
        body += "data: " + json.dumps(payload) + "\n\n"
    return body


# ---- helpers ------------------------------------------------------------


def test_translate_messages_splits_system():
    msgs = [
        {"role": "system", "content": "you are a helper"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    sys_text, anth = _translate_messages(msgs)
    assert sys_text == "you are a helper"
    assert anth == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_translate_messages_collapses_same_role():
    """Anthropic rejects consecutive same-role turns (e.g. user-after-user
    when a synthetic_tool_result is appended). Collapsed into one."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "<system>retry please</system>"},
    ]
    _, anth = _translate_messages(msgs)
    assert len(anth) == 1
    assert anth[0]["role"] == "user"
    assert "hi" in anth[0]["content"]
    assert "retry please" in anth[0]["content"]


def test_openai_tools_to_anthropic_shape():
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List a directory.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
    ]
    out = _openai_tools_to_anthropic(schemas)
    assert out == [
        {
            "name": "list_dir",
            "description": "List a directory.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


# ---- list_models / select_model -----------------------------------------


@pytest.mark.asyncio
async def test_list_models_returns_curated_set():
    p = AnthropicProvider(ProviderConfig(name="anthropic"))
    models = await p.list_models()
    ids = {m.id for m in models}
    assert "claude-3-5-sonnet-latest" in ids
    assert all(m.supports_native_tools for m in models)


@pytest.mark.asyncio
async def test_select_model_auto_uses_default_or_sonnet():
    p = AnthropicProvider(ProviderConfig(name="anthropic"))
    info = await p.select_model("auto")
    assert info.id == "claude-3-5-sonnet-latest"

    p2 = AnthropicProvider(
        ProviderConfig(name="anthropic", default_model="claude-haiku-4-5")
    )
    info2 = await p2.select_model("auto")
    assert info2.id == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_select_model_unknown_synthesised():
    p = AnthropicProvider(ProviderConfig(name="anthropic"))
    info = await p.select_model("claude-future-model")
    assert info.id == "claude-future-model"
    assert info.supports_native_tools is True


def test_preferred_protocol_default_native():
    p = AnthropicProvider(ProviderConfig(name="anthropic"))
    assert p.preferred_protocol(ModelInfo(id="claude-3-5-sonnet-latest")) == NATIVE_TOOLS


def test_preferred_protocol_pin_overrides():
    p = AnthropicProvider(
        ProviderConfig(name="anthropic", protocol=REACT_JSON)
    )
    assert p.preferred_protocol(ModelInfo(id="claude-3-5-sonnet-latest")) == REACT_JSON


# ---- complete_streaming -------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_streaming_text_only_final_answer():
    body = _sse([
        ("message_start", {
            "type": "message_start",
            "message": {"id": "m1", "usage": {"input_tokens": 10, "output_tokens": 0}},
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi "},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "there"},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5},
        }),
        ("message_stop", {"type": "message_stop"}),
    ])
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = AnthropicProvider(ProviderConfig(name="anthropic", api_key="sk-fake"))
    seen: list[str] = []
    res = await p.complete_streaming(
        ModelInfo(id="claude-3-5-sonnet-latest", supports_native_tools=True),
        [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
        on_delta=lambda t: seen.append(t),
        protocol=NATIVE_TOOLS,
        tool_schemas=[],
    )
    assert "".join(seen) == "Hi there"
    assert res.final_answer == "Hi there"
    assert res.tool_calls == []
    assert res.finish_reason == "end_turn"
    assert res.usage["input_tokens"] == 10
    assert res.usage["output_tokens"] == 5


@respx.mock
@pytest.mark.asyncio
async def test_streaming_tool_use_block_assembled():
    body = _sse([
        ("message_start", {
            "type": "message_start",
            "message": {"id": "m1", "usage": {"input_tokens": 12, "output_tokens": 0}},
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_abc", "name": "list_dir", "input": {}},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"pa'},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'th": "."}'},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 4},
        }),
        ("message_stop", {"type": "message_stop"}),
    ])
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = AnthropicProvider(ProviderConfig(name="anthropic", api_key="sk-fake"))
    res = await p.complete_streaming(
        ModelInfo(id="claude-3-5-sonnet-latest", supports_native_tools=True),
        [{"role": "user", "content": "list ."}],
        on_delta=lambda t: None,
        protocol=NATIVE_TOOLS,
        tool_schemas=[{"type": "function", "function": {"name": "list_dir", "parameters": {}}}],
    )
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc.id == "tu_abc"
    assert tc.name == "list_dir"
    assert tc.args == {"path": "."}
    assert res.final_answer is None
    assert res.finish_reason == "tool_use"


@respx.mock
@pytest.mark.asyncio
async def test_streaming_request_body_translated_correctly():
    captured: dict = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_sse([("message_stop", {"type": "message_stop"})]),
        )

    respx.post("https://api.anthropic.com/v1/messages").mock(side_effect=handler)
    p = AnthropicProvider(ProviderConfig(name="anthropic", api_key="sk-fake"))
    await p.complete_streaming(
        ModelInfo(id="claude-3-5-sonnet-latest"),
        [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
        on_delta=lambda t: None,
        protocol=NATIVE_TOOLS,
        tool_schemas=[
            {"type": "function", "function": {"name": "echo", "description": "d",
                                                 "parameters": {"type": "object"}}}
        ],
    )
    body = captured["body"]
    assert body["model"] == "claude-3-5-sonnet-latest"
    assert body["system"] == "S"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["tools"] == [{
        "name": "echo", "description": "d",
        "input_schema": {"type": "object"},
    }]
    assert captured["headers"].get("x-api-key") == "sk-fake"
    assert captured["headers"].get("anthropic-version") == "2023-06-01"


@respx.mock
@pytest.mark.asyncio
async def test_streaming_cancellation():
    p = AnthropicProvider(ProviderConfig(name="anthropic", api_key="sk-fake"))
    cancel = asyncio.Event()
    cancel.set()
    res = await p.complete_streaming(
        ModelInfo(id="claude-3-5-sonnet-latest"),
        [{"role": "user", "content": "hi"}],
        on_delta=lambda t: None,
        protocol=NATIVE_TOOLS,
        cancel=cancel,
    )
    assert res.finish_reason == "cancelled"
