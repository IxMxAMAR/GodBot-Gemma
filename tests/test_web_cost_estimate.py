"""Tests for sub-project 36 — POST /api/cost/estimate."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


def test_estimate_known_provider(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/cost/estimate", json={
        "provider": "openai", "model": "gpt-4o-mini",
        "input_tokens": 1_000_000, "output_tokens": 500_000,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["matched"] is True
    # gpt-4o-mini: 0.15/M in + 0.60/M out → 0.15 + 0.30 = 0.45
    assert body["usd"] == pytest.approx(0.45)


def test_estimate_local_provider_is_free(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/cost/estimate", json={
        "provider": "lmstudio", "model": "gemma",
        "input_tokens": 10**9, "output_tokens": 10**9,
    })
    body = r.json()
    assert body["usd"] == 0.0
    assert body["matched"] is True


def test_estimate_unknown_provider_unmatched(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/cost/estimate", json={
        "provider": "no-such-provider", "model": "no-such-model",
        "input_tokens": 1000, "output_tokens": 500,
    })
    body = r.json()
    assert body["matched"] is False
    assert body["usd"] == 0.0


def test_estimate_400_missing_provider(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/cost/estimate", json={"model": "x"})
    assert r.status_code == 400


def test_estimate_400_missing_model(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/cost/estimate", json={"provider": "x"})
    assert r.status_code == 400


def test_estimate_400_negative_tokens(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/cost/estimate", json={
        "provider": "openai", "model": "gpt-4o",
        "input_tokens": -1, "output_tokens": 0,
    })
    assert r.status_code == 400


def test_estimate_400_non_int_tokens(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/cost/estimate", json={
        "provider": "openai", "model": "gpt-4o",
        "input_tokens": "many", "output_tokens": 0,
    })
    assert r.status_code == 400


def test_estimate_zero_tokens_yields_zero(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/cost/estimate", json={
        "provider": "openai", "model": "gpt-4o-mini",
        "input_tokens": 0, "output_tokens": 0,
    })
    body = r.json()
    assert body["usd"] == 0.0
    assert body["matched"] is True
