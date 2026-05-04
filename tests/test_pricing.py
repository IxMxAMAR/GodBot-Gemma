"""Tests for godbot.core.pricing (sub-project 23)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.pricing import (
    ModelRate,
    _lookup_rate,
    compute_cost,
    load_pricing,
)
from godbot.core.session import Session
from godbot.interfaces.web import build_app


# --- table loading -----------------------------------------------------


def test_load_pricing_returns_builtin_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    table = load_pricing()
    # Sanity: openai is in builtins.
    assert "openai" in table
    assert "gpt-4o-mini" in table["openai"]
    assert table["openai"]["gpt-4o-mini"].input_per_1m > 0


def test_load_pricing_overrides_builtins(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "pricing.toml").write_text(
        '[openai."gpt-4o"]\n'
        'input_per_1m = 99.0\n'
        'output_per_1m = 199.0\n',
        encoding="utf-8",
    )
    table = load_pricing()
    assert table["openai"]["gpt-4o"].input_per_1m == 99.0
    assert table["openai"]["gpt-4o"].output_per_1m == 199.0


def test_load_pricing_adds_new_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "pricing.toml").write_text(
        '[brand_new_provider."some-model"]\n'
        'input_per_1m = 0.10\n'
        'output_per_1m = 0.40\n',
        encoding="utf-8",
    )
    table = load_pricing()
    assert "brand_new_provider" in table
    assert table["brand_new_provider"]["some-model"].input_per_1m == 0.10


def test_load_pricing_corrupt_toml_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "pricing.toml").write_text("not [valid toml [[[", encoding="utf-8")
    table = load_pricing()
    # Builtin defaults must still be present.
    assert "openai" in table


# --- lookup ------------------------------------------------------------


def test_lookup_exact_match():
    table = {"openai": {"gpt-4o": ModelRate(2.5, 10.0)}}
    rate = _lookup_rate(table, "openai", "gpt-4o")
    assert rate is not None
    assert rate.input_per_1m == 2.5


def test_lookup_lowercase_match():
    table = {"openai": {"gpt-4o": ModelRate(2.5, 10.0)}}
    rate = _lookup_rate(table, "openai", "GPT-4o")
    assert rate is not None


def test_lookup_strips_date_suffix():
    """Anthropic-style id 'claude-haiku-4-5-20251001' should resolve to
    'claude-haiku-4-5' if that exists in the table."""
    table = {"anthropic": {"claude-haiku-4-5": ModelRate(1.0, 5.0)}}
    rate = _lookup_rate(table, "anthropic", "claude-haiku-4-5-20251001")
    assert rate is not None
    assert rate.input_per_1m == 1.0


def test_lookup_unknown_returns_none():
    rate = _lookup_rate({"openai": {}}, "openai", "no-such-model")
    assert rate is None


# --- compute_cost ------------------------------------------------------


def test_compute_cost_known_provider_and_model():
    table = {"openai": {"gpt-4o": ModelRate(2.5, 10.0)}}
    out = compute_cost(
        provider="openai", model="gpt-4o",
        input_tokens=1_000_000, output_tokens=500_000,
        table=table,
    )
    assert out.matched is True
    assert out.input_usd == pytest.approx(2.5)
    assert out.output_usd == pytest.approx(5.0)
    assert out.usd == pytest.approx(7.5)


def test_compute_cost_local_provider_is_free():
    out = compute_cost(
        provider="lmstudio", model="any",
        input_tokens=10**9, output_tokens=10**9,
        table={"lmstudio": {}},
    )
    assert out.usd == 0.0
    assert out.matched is True
    assert out.rate is not None
    assert out.rate.input_per_1m == 0.0


def test_compute_cost_unknown_returns_zero_unmatched():
    out = compute_cost(
        provider="some-cloud", model="unknown-model",
        input_tokens=100, output_tokens=50,
        table={},
    )
    assert out.usd == 0.0
    assert out.matched is False
    assert out.rate is None


def test_compute_cost_to_dict_round_trip():
    table = {"openai": {"gpt-4o": ModelRate(2.5, 10.0)}}
    out = compute_cost(
        provider="openai", model="gpt-4o",
        input_tokens=1000, output_tokens=500, table=table,
    )
    d = out.to_dict()
    assert d["matched"] is True
    assert d["provider"] == "openai"
    assert d["model"] == "gpt-4o"
    assert d["rate"]["input_per_1m"] == 2.5
    assert "usd" in d


# --- endpoint ----------------------------------------------------------


def test_session_cost_endpoint_unknown_session_404(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/sessions/no-such/cost")
    assert r.status_code == 404


def test_session_cost_endpoint_known_pricing(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    s = Session.create(
        root=sessions_root, model="m",
        provider="openai", model_name="gpt-4o-mini",
    )
    s.add_usage({"input_tokens": 1_000_000, "output_tokens": 500_000, "total_tokens": 1_500_000})

    app = build_app(sessions_root=sessions_root)
    c = TestClient(app)
    r = c.get(f"/api/sessions/{s.id}/cost")
    assert r.status_code == 200
    body = r.json()
    assert body["matched"] is True
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    # gpt-4o-mini in builtins: 0.15 in / 0.60 out per 1M.
    assert body["input_usd"] == pytest.approx(0.15)
    assert body["output_usd"] == pytest.approx(0.30)
    assert body["usd"] == pytest.approx(0.45)
    assert body["usage"]["input_tokens"] == 1_000_000


def test_session_cost_endpoint_lmstudio_is_free(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    s = Session.create(
        root=sessions_root, model="gemma-3", provider="lmstudio", model_name="gemma-3",
    )
    s.add_usage({"input_tokens": 5_000_000, "output_tokens": 2_000_000, "total_tokens": 7_000_000})
    app = build_app(sessions_root=sessions_root)
    c = TestClient(app)
    r = c.get(f"/api/sessions/{s.id}/cost")
    body = r.json()
    assert body["usd"] == 0.0
    assert body["matched"] is True


def test_session_cost_endpoint_unknown_model_unmatched(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    s = Session.create(
        root=sessions_root, model="weird", provider="openai", model_name="not-a-real-model",
    )
    s.add_usage({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    app = build_app(sessions_root=sessions_root)
    c = TestClient(app)
    r = c.get(f"/api/sessions/{s.id}/cost")
    body = r.json()
    assert body["matched"] is False
    assert body["usd"] == 0.0
