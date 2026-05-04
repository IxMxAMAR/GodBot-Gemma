"""GeminiProvider tests — all wire calls mocked via respx."""
from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx

from godbot.core.providers import NATIVE_TOOLS, REACT_JSON, ProviderConfig
from godbot.core.providers.base import ModelInfo
from godbot.core.providers.gemini import (
    GeminiProvider,
    _gemini_usage,
    _openai_tools_to_gemini,
    _translate_messages_to_contents,
)


def _sse(events: list[dict]) -> str:
    body = ""
    for payload in events:
        body += "data: " + json.dumps(payload) + "\n\n"
    return body


# ---- helpers ------------------------------------------------------------


def test_translate_messages_splits_system_and_renames_role():
    msgs = [
        {"role": "system", "content": "you are X"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    sys_text, contents = _translate_messages_to_contents(msgs)
    assert sys_text == "you are X"
    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
    ]


def test_translate_messages_collapses_same_role():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    _, contents = _translate_messages_to_contents(msgs)
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "a"}, {"text": "b"}]


def test_openai_tools_to_gemini():
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "list_dir", "description": "List dir.",
                "parameters": {"type": "object", "properties": {"p": {"type": "string"}}},
            },
        }
    ]
    out = _openai_tools_to_gemini(schemas)
    assert out == [
        {
            "name": "list_dir",
            "description": "List dir.",
            "parameters": {"type": "object", "properties": {"p": {"type": "string"}}},
        }
    ]


def test_gemini_usage_normalisation():
    out = _gemini_usage({"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15})
    assert out == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


# ---- list_models / select_model -----------------------------------------


@pytest.mark.asyncio
async def test_list_models_returns_known_set():
    p = GeminiProvider(ProviderConfig(name="gemini"))
    models = await p.list_models()
    ids = {m.id for m in models}
    assert "gemini-2.0-flash" in ids
    assert all(m.supports_native_tools for m in models)


@pytest.mark.asyncio
async def test_select_model_auto_uses_default():
    p = GeminiProvider(ProviderConfig(name="gemini"))
    info = await p.select_model("auto")
    assert info.id == "gemini-2.0-flash"


@pytest.mark.asyncio
async def test_select_model_unknown_synthesised():
    p = GeminiProvider(ProviderConfig(name="gemini"))
    info = await p.select_model("gemini-future")
    assert info.id == "gemini-future"
    assert info.supports_native_tools is True


def test_preferred_protocol_default_native():
    p = GeminiProvider(ProviderConfig(name="gemini"))
    assert p.preferred_protocol(ModelInfo(id="gemini-2.0-flash")) == NATIVE_TOOLS


# ---- complete_streaming -------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_streaming_text_only_final_answer():
    body = _sse([
        {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello "}]},
            }],
        },
        {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "world"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3, "totalTokenCount": 11},
        },
    ])
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent"
    ).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = GeminiProvider(ProviderConfig(name="gemini", api_key="test-key"))
    seen: list[str] = []
    res = await p.complete_streaming(
        ModelInfo(id="gemini-2.0-flash", supports_native_tools=True),
        [{"role": "user", "content": "hi"}],
        on_delta=lambda t: seen.append(t),
        protocol=NATIVE_TOOLS,
        tool_schemas=[],
    )
    assert "".join(seen) == "Hello world"
    assert res.final_answer == "Hello world"
    assert res.tool_calls == []
    assert res.finish_reason == "STOP"
    assert res.usage == {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}


@respx.mock
@pytest.mark.asyncio
async def test_streaming_function_call():
    body = _sse([
        {
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"functionCall": {"name": "list_dir", "args": {"path": "."}}},
                ]},
                "finishReason": "STOP",
            }],
        },
    ])
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent"
    ).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    p = GeminiProvider(ProviderConfig(name="gemini", api_key="test-key"))
    res = await p.complete_streaming(
        ModelInfo(id="gemini-2.0-flash", supports_native_tools=True),
        [{"role": "user", "content": "list ."}],
        on_delta=lambda t: None,
        protocol=NATIVE_TOOLS,
        tool_schemas=[{"type": "function", "function": {"name": "list_dir", "parameters": {}}}],
    )
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc.name == "list_dir"
    assert tc.args == {"path": "."}
    assert res.final_answer is None


@respx.mock
@pytest.mark.asyncio
async def test_streaming_request_body_translated_correctly():
    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_sse([
                {"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"}]}
            ]),
        )

    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent"
    ).mock(side_effect=handler)
    p = GeminiProvider(ProviderConfig(name="gemini", api_key="test-key"))
    await p.complete_streaming(
        ModelInfo(id="gemini-2.0-flash"),
        [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
        on_delta=lambda t: None,
        protocol=NATIVE_TOOLS,
        tool_schemas=[
            {"type": "function", "function": {"name": "echo", "description": "d", "parameters": {"type": "object"}}}
        ],
    )
    body = captured["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "S"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["tools"] == [{
        "functionDeclarations": [{
            "name": "echo", "description": "d", "parameters": {"type": "object"},
        }]
    }]
    # API key is appended as a query parameter.
    parsed = urlparse(captured["url"])
    qs = parse_qs(parsed.query)
    assert qs.get("key") == ["test-key"]
    assert qs.get("alt") == ["sse"]


@respx.mock
@pytest.mark.asyncio
async def test_streaming_cancellation():
    p = GeminiProvider(ProviderConfig(name="gemini", api_key="test-key"))
    cancel = asyncio.Event()
    cancel.set()
    res = await p.complete_streaming(
        ModelInfo(id="gemini-2.0-flash"),
        [{"role": "user", "content": "hi"}],
        on_delta=lambda t: None,
        protocol=NATIVE_TOOLS,
        cancel=cancel,
    )
    assert res.finish_reason == "cancelled"
