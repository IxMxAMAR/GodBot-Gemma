"""Tests for sub-project 73 — /api/agent/raw_completion."""
from __future__ import annotations

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


def test_raw_completion_400_no_messages(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/raw_completion", json={})
    assert r.status_code == 400


def test_raw_completion_400_bad_message_shape(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/raw_completion", json={
        "messages": [{"role": "user"}],  # missing content
    })
    assert r.status_code == 400


def test_raw_completion_400_unknown_provider(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/raw_completion", json={
        "messages": [{"role": "user", "content": "hi"}],
        "provider": "no-such",
    })
    assert r.status_code == 400


def test_raw_completion_501_anthropic(tmp_path, monkeypatch):
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
    r = c.post("/api/agent/raw_completion", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 501


@respx.mock
def test_raw_completion_happy_path(tmp_path, monkeypatch):
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "hello back"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "model": "test-model",
            },
        )
    )
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/raw_completion", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == "hello back"
    assert body["finish_reason"] == "stop"
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["output_tokens"] == 5
    assert body["usage"]["total_tokens"] == 15
    assert body["model"] == "test-model"


@respx.mock
def test_raw_completion_passes_temperature_and_max_tokens(tmp_path, monkeypatch):
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
        )
    )
    c = _client(tmp_path, monkeypatch)
    c.post("/api/agent/raw_completion", json={
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 64,
    })
    import json as _json
    upstream = _json.loads(route.calls.last.request.content)
    assert upstream["temperature"] == 0.0
    assert upstream["max_tokens"] == 64
    assert upstream["stream"] is False


@respx.mock
def test_raw_completion_502_on_upstream_failure(tmp_path, monkeypatch):
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/raw_completion", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 502


def test_raw_completion_400_on_invalid_temperature(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/agent/raw_completion", json={
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": "not-a-number",
    })
    assert r.status_code == 400
