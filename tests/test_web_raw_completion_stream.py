"""Tests for sub-project 89 — POST /api/agent/raw_completion/stream."""
from __future__ import annotations

import json

import httpx
import respx
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.interfaces.web import build_app


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


def _sse_chunk(text: str) -> str:
    return json.dumps({
        "choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}],
    })


def _stream_body(chunks: list[str]) -> str:
    body = ""
    for c in chunks:
        body += "data: " + _sse_chunk(c) + "\n\n"
    body += "data: [DONE]\n\n"
    return body


def test_raw_stream_400_on_no_messages(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/raw_completion/stream", json={})
    assert r.status_code == 400


def test_raw_stream_400_on_bad_message_shape(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/raw_completion/stream", json={
        "messages": [{"role": "user"}],  # missing content
    })
    assert r.status_code == 400


def test_raw_stream_501_on_anthropic(tmp_path, monkeypatch):
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
    r = c.post("/api/agent/raw_completion/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 501


def test_raw_stream_400_unknown_provider(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/raw_completion/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
        "provider": "no-such",
    })
    assert r.status_code == 400


@respx.mock
def test_raw_stream_emits_chunks_then_done(tmp_path, monkeypatch):
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=_stream_body(["foo", "bar", "baz"]),
            headers={"content-type": "text/event-stream"},
        )
    )
    c = _client(tmp_path, monkeypatch)
    events: list[str] = []
    payloads: list[str] = []
    with c.stream("POST", "/api/agent/raw_completion/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
    }) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                payloads.append(line.split(":", 1)[1].strip())
    assert events.count("chunk") == 3
    assert "done" in events
    chunk_texts = [
        json.loads(p)["text"]
        for ev, p in zip(events, payloads) if ev == "chunk"
    ]
    assert "".join(chunk_texts) == "foobarbaz"


@respx.mock
def test_raw_stream_passes_payload(tmp_path, monkeypatch):
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, text=_stream_body(["x"]),
            headers={"content-type": "text/event-stream"},
        )
    )
    c = _client(tmp_path, monkeypatch)
    with c.stream("POST", "/api/agent/raw_completion/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 32,
    }) as resp:
        for _ in resp.iter_lines():
            pass
    upstream = json.loads(route.calls.last.request.content)
    assert upstream["temperature"] == 0.0
    assert upstream["max_tokens"] == 32
    assert upstream["stream"] is True
