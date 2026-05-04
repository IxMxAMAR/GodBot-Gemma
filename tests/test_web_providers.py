"""Tests for the provider-related web endpoints (Task 9).

  - GET  /api/providers
  - GET  /api/providers/{name}/models
  - POST /api/sessions/new with provider/model fields
  - GET  /api/sessions/{sid} surfaces provider/model_name/protocol
"""
from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401 — auto-discovery
from godbot.interfaces.web import build_app


def _build(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    return build_app(sessions_root=tmp_path / "sessions")


# ---- /api/providers -----------------------------------------------------


def test_providers_endpoint_lists_configured(tmp_path, monkeypatch):
    app = _build(tmp_path, monkeypatch)
    c = TestClient(app)
    r = c.get("/api/providers")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "lmstudio"
    names = {p["name"] for p in body["providers"]}
    # The default config writes lmstudio/ollama/openai/anthropic/gemini/groq/together/openrouter
    for required in ("lmstudio", "ollama", "openai", "anthropic", "gemini"):
        assert required in names


def test_providers_endpoint_reports_api_key_presence(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-from-env")
    app = _build(tmp_path, monkeypatch)
    c = TestClient(app)
    body = c.get("/api/providers").json()
    openai = next(p for p in body["providers"] if p["name"] == "openai")
    assert openai["has_api_key"] is True
    # Anthropic key not set in this test → reports false
    anthropic = next(p for p in body["providers"] if p["name"] == "anthropic")
    assert anthropic["has_api_key"] is False


# ---- /api/providers/{name}/models --------------------------------------


def test_provider_models_unknown_404(tmp_path, monkeypatch):
    app = _build(tmp_path, monkeypatch)
    c = TestClient(app)
    r = c.get("/api/providers/does-not-exist/models")
    assert r.status_code == 404


def test_provider_models_anthropic_curated(tmp_path, monkeypatch):
    """Anthropic returns a curated list with no live HTTP call."""
    app = _build(tmp_path, monkeypatch)
    c = TestClient(app)
    r = c.get("/api/providers/anthropic/models")
    assert r.status_code == 200
    body = r.json()
    ids = {m["id"] for m in body["models"]}
    assert "claude-3-5-sonnet-latest" in ids
    assert all(m["supports_native_tools"] for m in body["models"])


@respx.mock
def test_provider_models_openai_compat_hits_models_endpoint(tmp_path, monkeypatch):
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "gemma-3", "loaded_context_length": 32768}]},
        )
    )
    app = _build(tmp_path, monkeypatch)
    c = TestClient(app)
    r = c.get("/api/providers/lmstudio/models")
    assert r.status_code == 200
    body = r.json()
    assert body["models"] == [
        {"id": "gemma-3", "context_length": 32768, "supports_native_tools": False}
    ]


# ---- session creation extends the schema --------------------------------


def test_session_new_default_provider_is_lmstudio(tmp_path, monkeypatch):
    app = _build(tmp_path, monkeypatch)
    c = TestClient(app)
    body = c.post("/api/sessions/new").json()
    assert body["provider"] == "lmstudio"
    assert body["model_name"] == "auto"
    assert body["protocol"] is None


def test_session_new_with_provider_and_model(tmp_path, monkeypatch):
    app = _build(tmp_path, monkeypatch)
    c = TestClient(app)
    body = c.post("/api/sessions/new", json={
        "provider": "anthropic",
        "model_name": "claude-3-5-sonnet-latest",
        "protocol": "native",
    }).json()
    assert body["provider"] == "anthropic"
    assert body["model_name"] == "claude-3-5-sonnet-latest"
    assert body["protocol"] == "native"
    sid = body["session_id"]

    # GET /api/sessions/<sid> surfaces the same fields.
    r = c.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    g = r.json()
    assert g["provider"] == "anthropic"
    assert g["model_name"] == "claude-3-5-sonnet-latest"
    assert g["protocol"] == "native"
