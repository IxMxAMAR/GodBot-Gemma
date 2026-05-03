import asyncio
import json
import respx
import httpx
import pytest
from godbot.core.llm import LLMClient


def _sse(chunks):
    out = ""
    for c in chunks:
        out += "data: " + c + "\n\n"
    out += "data: [DONE]\n\n"
    return out


@respx.mock
@pytest.mark.asyncio
async def test_streaming_accumulates_tokens():
    body = _sse([
        '{"choices":[{"delta":{"content":"hel"}}]}',
        '{"choices":[{"delta":{"content":"lo"}}]}',
    ])
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "g", "loaded_context_length": 32768}]})
    )
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    client = LLMClient(base_url="http://localhost:1234/v1", model="g")
    seen: list[str] = []

    async def on_delta(t: str):
        seen.append(t)

    full = await client.complete_streaming(
        messages=[{"role": "user", "content": "hi"}],
        on_delta=on_delta,
        response_format=None,
    )
    assert full == "hello"
    assert seen == ["hel", "lo"]


@respx.mock
@pytest.mark.asyncio
async def test_streaming_cancellation():
    body = _sse([
        '{"choices":[{"delta":{"content":"a"}}]}',
        '{"choices":[{"delta":{"content":"b"}}]}',
        '{"choices":[{"delta":{"content":"c"}}]}',
    ])
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "g"}]})
    )
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    client = LLMClient(base_url="http://localhost:1234/v1", model="g")
    cancel = asyncio.Event()
    cancel.set()
    full = await client.complete_streaming(
        messages=[{"role": "user", "content": "hi"}],
        on_delta=lambda t: None,
        response_format=None,
        cancel=cancel,
    )
    assert isinstance(full, str)


@respx.mock
@pytest.mark.asyncio
async def test_response_format_passed_through():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(['{"choices":[{"delta":{"content":"x"}}]}']),
        )

    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "g"}]})
    )
    respx.post("http://localhost:1234/v1/chat/completions").mock(side_effect=handler)
    client = LLMClient(base_url="http://localhost:1234/v1", model="g")
    schema = {"oneOf": [{"type": "object"}]}
    await client.complete_streaming(
        messages=[{"role": "user", "content": "hi"}],
        on_delta=lambda t: None,
        response_format={"type": "json_schema", "json_schema": {"name": "react", "schema": schema}},
    )
    assert captured["body"]["response_format"]["type"] == "json_schema"
