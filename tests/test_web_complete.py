"""Integration tests for POST /api/complete (sub-project 12)."""
from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401 — tool autodiscovery
from godbot.interfaces.web import build_app


def _new_client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    # Provide a minimal config.toml so providers.default + providers.lmstudio
    # are populated.
    cfg_dir = tmp_path
    (cfg_dir / "config.toml").write_text(
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
def test_complete_endpoint_happy_path(tmp_path, monkeypatch):
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": " a + b"}}]},
        )
    )
    c = _new_client(tmp_path, monkeypatch)
    r = c.post("/api/complete", json={
        "prefix": "def add(a, b):\n    return",
        "language": "python",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "a + b" in body["completion"]
    assert body["model"] == "test-model"
    assert body["elapsed_ms"] >= 0


@respx.mock
def test_complete_endpoint_passes_max_tokens(tmp_path, monkeypatch):
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]}),
    )
    c = _new_client(tmp_path, monkeypatch)
    r = c.post("/api/complete", json={"prefix": "x", "max_tokens": 16})
    assert r.status_code == 200
    import json as _json
    upstream = _json.loads(route.calls.last.request.content)
    assert upstream["max_tokens"] == 16


def test_complete_endpoint_unknown_provider_400(tmp_path, monkeypatch):
    c = _new_client(tmp_path, monkeypatch)
    r = c.post("/api/complete", json={"prefix": "x", "provider": "no_such"})
    assert r.status_code == 400


def test_complete_endpoint_unsupported_provider_501(tmp_path, monkeypatch):
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
    r = c.post("/api/complete", json={"prefix": "x"})
    assert r.status_code == 501


@respx.mock
def test_complete_endpoint_resolves_auto_via_provider(tmp_path, monkeypatch):
    """When the requested model is 'auto', the daemon must resolve it via the
    provider's select_model. We seed /v1/models with one entry so it picks that."""
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "gemma-3", "loaded_context_length": 32768}]},
        ),
    )
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "yo"}}]},
        ),
    )
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        '[providers]\n'
        'default = "lmstudio"\n'
        '[providers.lmstudio]\n'
        'base_url = "http://localhost:1234/v1"\n'
        'default_model = "auto"\n',
        encoding="utf-8",
    )
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.post("/api/complete", json={"prefix": "x"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "gemma-3"
