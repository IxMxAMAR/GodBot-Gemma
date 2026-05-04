"""Tests for sub-project 29 — streaming inline completions."""
from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.completion import stream_completion
from godbot.interfaces.web import build_app


def _sse_chunk_payload(text: str) -> str:
    """Build an OpenAI-style streaming chunk for one delta."""
    import json as _json
    return _json.dumps({
        "choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}],
    })


def _stream_body(chunks: list[str]) -> str:
    body = ""
    for c in chunks:
        body += "data: " + _sse_chunk_payload(c) + "\n\n"
    body += "data: [DONE]\n\n"
    return body


# --- core stream_completion ----


@respx.mock
@pytest.mark.asyncio
async def test_stream_completion_yields_chunks():
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=_stream_body(["Hel", "lo, ", "wor", "ld"]),
            headers={"content-type": "text/event-stream"},
        )
    )
    out = []
    async for chunk in stream_completion(
        prefix="say ", provider_name="lmstudio",
        base_url="http://localhost:1234/v1", api_key="", model="m",
    ):
        out.append(chunk)
    assert "".join(out) == "Hello, world"


@pytest.mark.asyncio
async def test_stream_completion_empty_input_yields_nothing():
    out = []
    async for chunk in stream_completion(
        prefix="", suffix="", provider_name="lmstudio",
        base_url="http://localhost:1234/v1", api_key="", model="m",
    ):
        out.append(chunk)
    assert out == []


@pytest.mark.asyncio
async def test_stream_completion_unsupported_provider_raises():
    with pytest.raises(ValueError):
        async for _ in stream_completion(
            prefix="x", provider_name="anthropic",
            base_url="https://api.anthropic.com", api_key="k", model="claude",
        ):
            pass


@respx.mock
@pytest.mark.asyncio
async def test_stream_completion_skips_done_marker_and_malformed_lines():
    """`[DONE]` ends the stream; malformed JSON lines are silently skipped."""
    raw = (
        "data: " + _sse_chunk_payload("ok") + "\n\n"
        "data: not_valid_json\n\n"
        "data: [DONE]\n\n"
    )
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, text=raw, headers={"content-type": "text/event-stream"},
        )
    )
    out = []
    async for chunk in stream_completion(
        prefix="x", provider_name="lmstudio",
        base_url="http://localhost:1234/v1", api_key="", model="m",
    ):
        out.append(chunk)
    assert out == ["ok"]


# --- endpoint ----


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        '[providers]\n'
        'default = "lmstudio"\n'
        '[providers.lmstudio]\n'
        'base_url = "http://localhost:1234/v1"\n'
        'default_model = "test-model"\n',
        encoding="utf-8",
    )
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


@respx.mock
def test_stream_endpoint_emits_chunk_then_done(tmp_path, monkeypatch):
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=_stream_body(["foo", "bar"]),
            headers={"content-type": "text/event-stream"},
        )
    )
    c = _client(tmp_path, monkeypatch)
    events = []
    datas = []
    with c.stream("POST", "/api/complete/stream", json={"prefix": "x"}) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                datas.append(line.split(":", 1)[1].strip())
    # We see two chunk events plus one done.
    assert events.count("chunk") == 2
    assert "done" in events
    # Joined chunk texts should equal "foobar".
    import json as _json
    chunk_texts = [
        _json.loads(d)["text"] for ev, d in zip(events, datas)
        if ev == "chunk"
    ]
    assert "".join(chunk_texts) == "foobar"


def test_stream_endpoint_unsupported_provider_returns_501(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        '[providers]\n'
        'default = "anthropic"\n'
        '[providers.anthropic]\n'
        'default_model = "claude-haiku"\n',
        encoding="utf-8",
    )
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.post("/api/complete/stream", json={"prefix": "x"})
    assert r.status_code == 501


def test_stream_endpoint_unknown_provider_400(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/complete/stream", json={"prefix": "x", "provider": "no-such"})
    assert r.status_code == 400
