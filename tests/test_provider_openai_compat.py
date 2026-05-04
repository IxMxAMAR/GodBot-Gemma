"""Tests for GenericOpenAICompatProvider.

All HTTP is mocked via respx; nothing here ever touches a real endpoint.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from godbot.core.providers import (
    NATIVE_TOOLS,
    REACT_JSON,
    ProviderConfig,
)
from godbot.core.providers.openai_compat import GenericOpenAICompatProvider


# ---- helpers ------------------------------------------------------------


def _sse(chunks: list[str]) -> str:
    body = ""
    for c in chunks:
        body += "data: " + c + "\n\n"
    body += "data: [DONE]\n\n"
    return body


def _models_payload(*ids_and_ctx: tuple[str, int | None]) -> dict:
    data = []
    for entry in ids_and_ctx:
        mid, ctx = entry
        item: dict = {"id": mid}
        if ctx is not None:
            item["loaded_context_length"] = ctx
        data.append(item)
    return {"data": data}


# ---- list_models / select_model -----------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_list_models_parses_data():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(
            200,
            json=_models_payload(("gemma-3", 32768), ("qwen-7b", 4096)),
        )
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="lmstudio", base_url="http://localhost:1234/v1")
    )
    models = await p.list_models()
    assert [m.id for m in models] == ["gemma-3", "qwen-7b"]
    assert models[0].context_length == 32768
    # Gemma identified as ReAct-only (small/legacy heuristic).
    assert models[0].supports_native_tools is False
    # Qwen-7b → native tools eligible.
    assert models[1].supports_native_tools is True


@respx.mock
@pytest.mark.asyncio
async def test_select_model_lmstudio_auto_picks_gemma():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(
            200,
            json=_models_payload(("nomic-embed", 2048), ("gemma-3n-e4b-it", 32768)),
        )
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="lmstudio", base_url="http://localhost:1234/v1")
    )
    info = await p.select_model("auto")
    assert info.id == "gemma-3n-e4b-it"
    assert info.context_length == 32768


@respx.mock
@pytest.mark.asyncio
async def test_select_model_lmstudio_auto_no_gemma_raises():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json=_models_payload(("qwen-7b", 4096)))
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="lmstudio", base_url="http://localhost:1234/v1")
    )
    with pytest.raises(RuntimeError, match="gemma"):
        await p.select_model("auto")


@respx.mock
@pytest.mark.asyncio
async def test_select_model_pinned_match():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(
            200,
            json=_models_payload(("qwen", 4096), ("gemma-3n", 32768)),
        )
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="lmstudio", base_url="http://localhost:1234/v1")
    )
    info = await p.select_model("qwen")
    assert info.id == "qwen"


@respx.mock
@pytest.mark.asyncio
async def test_select_model_unknown_synthesised_for_cloud():
    """For non-lmstudio providers, an unknown model id returns a synthetic
    ModelInfo so cloud APIs that don't expose /models still work."""
    respx.get("http://api.example.com/v1/models").mock(
        return_value=httpx.Response(200, json=_models_payload(("gpt-4o", None)))
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="openai", base_url="http://api.example.com/v1")
    )
    info = await p.select_model("gpt-5-future")
    assert info.id == "gpt-5-future"
    assert info.supports_native_tools is True


@respx.mock
@pytest.mark.asyncio
async def test_select_model_cloud_auto_uses_default_model():
    respx.get("http://api.example.com/v1/models").mock(
        return_value=httpx.Response(200, json=_models_payload(("gpt-4o", None)))
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(
            name="openai",
            base_url="http://api.example.com/v1",
            default_model="gpt-4o",
        )
    )
    info = await p.select_model("auto")
    assert info.id == "gpt-4o"


# ---- preferred_protocol -------------------------------------------------


def test_preferred_protocol_native_for_capable_model():
    p = GenericOpenAICompatProvider(ProviderConfig(name="openai"))
    from godbot.core.providers.base import ModelInfo
    assert p.preferred_protocol(ModelInfo(id="gpt-4o", supports_native_tools=True)) == NATIVE_TOOLS


def test_preferred_protocol_react_for_gemma():
    p = GenericOpenAICompatProvider(ProviderConfig(name="lmstudio"))
    from godbot.core.providers.base import ModelInfo
    assert p.preferred_protocol(ModelInfo(id="gemma-3", supports_native_tools=False)) == REACT_JSON


def test_preferred_protocol_config_pin_overrides_heuristic():
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="openai", protocol=REACT_JSON)
    )
    from godbot.core.providers.base import ModelInfo
    # Even a tools-capable model is forced to ReAct when pinned.
    assert p.preferred_protocol(ModelInfo(id="gpt-4o", supports_native_tools=True)) == REACT_JSON


# ---- complete_streaming: REACT_JSON path --------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_streaming_react_accumulates_and_parses():
    envelope = json.dumps({"thought": "easy", "final_answer": "hi!"})
    body = _sse([
        json.dumps({"choices": [{"delta": {"content": envelope[:5]}}]}),
        json.dumps({"choices": [{"delta": {"content": envelope[5:]}, "finish_reason": "stop"}]}),
    ])
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="lmstudio", base_url="http://localhost:1234/v1")
    )
    from godbot.core.providers.base import ModelInfo
    seen: list[str] = []
    res = await p.complete_streaming(
        ModelInfo(id="gemma-3", provider="lmstudio"),
        [{"role": "user", "content": "hi"}],
        on_delta=lambda t: seen.append(t),
        protocol=REACT_JSON,
        react_schema={"oneOf": [{"type": "object"}]},
    )
    assert "".join(seen) == envelope
    assert res.final_answer == "hi!"
    assert res.thought == "easy"
    assert res.raw_text == envelope


@respx.mock
@pytest.mark.asyncio
async def test_streaming_react_invalid_json_returns_raw():
    body = _sse([
        json.dumps({"choices": [{"delta": {"content": "not-json"}}]}),
    ])
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="lmstudio", base_url="http://localhost:1234/v1")
    )
    from godbot.core.providers.base import ModelInfo
    res = await p.complete_streaming(
        ModelInfo(id="gemma-3"), [], lambda t: None, REACT_JSON,
        react_schema={"oneOf": [{"type": "object"}]},
    )
    assert res.final_answer is None
    assert res.tool_calls == []
    assert res.raw_text == "not-json"


@respx.mock
@pytest.mark.asyncio
async def test_streaming_react_response_format_passed_through():
    captured: dict = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse([json.dumps({"choices": [{"delta": {"content": "x"}}]})]),
        )

    respx.post("http://localhost:1234/v1/chat/completions").mock(side_effect=handler)
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="lmstudio", base_url="http://localhost:1234/v1")
    )
    from godbot.core.providers.base import ModelInfo
    schema = {"oneOf": [{"type": "object"}]}
    await p.complete_streaming(
        ModelInfo(id="gemma-3"), [], lambda t: None, REACT_JSON, react_schema=schema,
    )
    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert captured["body"]["response_format"]["json_schema"]["name"] == "react"


# ---- complete_streaming: NATIVE_TOOLS path ------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_streaming_native_tool_calls_assembled():
    """Mock OpenAI's streamed tool_call deltas across two chunks; verify the
    provider reassembles {id, name, args_json} into a single ParsedToolCall."""
    body = _sse([
        json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_xyz", "function": {"name": "list_dir", "arguments": '{"pa'}}
        ]}}]}),
        json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'th": "."}'}}
        ]}, "finish_reason": "tool_calls"}]}),
    ])
    respx.post("http://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="openai", base_url="http://api.example.com/v1")
    )
    from godbot.core.providers.base import ModelInfo
    res = await p.complete_streaming(
        ModelInfo(id="gpt-4o", supports_native_tools=True, provider="openai"),
        [{"role": "user", "content": "list it"}],
        lambda t: None,
        NATIVE_TOOLS,
        tool_schemas=[{"type": "function", "function": {"name": "list_dir"}}],
    )
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc.id == "call_xyz"
    assert tc.name == "list_dir"
    assert tc.args == {"path": "."}
    assert res.final_answer is None
    assert res.finish_reason == "tool_calls"


@respx.mock
@pytest.mark.asyncio
async def test_streaming_native_no_tool_calls_returns_final_answer():
    body = _sse([
        json.dumps({"choices": [{"delta": {"content": "Hello "}}]}),
        json.dumps({"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]}),
    ])
    respx.post("http://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="openai", base_url="http://api.example.com/v1")
    )
    from godbot.core.providers.base import ModelInfo
    seen: list[str] = []
    res = await p.complete_streaming(
        ModelInfo(id="gpt-4o", supports_native_tools=True, provider="openai"),
        [], lambda t: seen.append(t), NATIVE_TOOLS, tool_schemas=[],
    )
    assert "".join(seen) == "Hello world"
    assert res.final_answer == "Hello world"
    assert res.tool_calls == []
    assert res.finish_reason == "stop"


@respx.mock
@pytest.mark.asyncio
async def test_streaming_native_tool_schemas_forwarded():
    captured: dict = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse([json.dumps({"choices": [{"delta": {"content": "x"}}]})]),
        )

    respx.post("http://api.example.com/v1/chat/completions").mock(side_effect=handler)
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="openai", base_url="http://api.example.com/v1")
    )
    from godbot.core.providers.base import ModelInfo
    schemas = [{"type": "function", "function": {"name": "list_dir", "parameters": {}}}]
    await p.complete_streaming(
        ModelInfo(id="gpt-4o", supports_native_tools=True),
        [], lambda t: None, NATIVE_TOOLS, tool_schemas=schemas,
    )
    assert captured["body"]["tools"] == schemas
    assert captured["body"]["tool_choice"] == "auto"


@respx.mock
@pytest.mark.asyncio
async def test_streaming_cancellation():
    body = _sse([
        json.dumps({"choices": [{"delta": {"content": "a"}}]}),
        json.dumps({"choices": [{"delta": {"content": "b"}}]}),
    ])
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="lmstudio", base_url="http://localhost:1234/v1")
    )
    from godbot.core.providers.base import ModelInfo
    cancel = asyncio.Event()
    cancel.set()
    res = await p.complete_streaming(
        ModelInfo(id="gemma-3"), [], lambda t: None, REACT_JSON,
        react_schema={"oneOf": [{"type": "object"}]}, cancel=cancel,
    )
    # Cancelled before any byte streamed.
    assert res.finish_reason == "cancelled"


@respx.mock
@pytest.mark.asyncio
async def test_streaming_native_usage_normalised():
    body = _sse([
        json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}),
    ])
    respx.post("http://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = GenericOpenAICompatProvider(
        ProviderConfig(name="openai", base_url="http://api.example.com/v1")
    )
    from godbot.core.providers.base import ModelInfo
    res = await p.complete_streaming(
        ModelInfo(id="gpt-4o", supports_native_tools=True),
        [], lambda t: None, NATIVE_TOOLS, tool_schemas=[],
    )
    assert res.usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
