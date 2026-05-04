"""Tests for sub-project 28 — per-session budget guardrail."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.session import Session
from godbot.interfaces.web import build_app


# --- Session-level budget API ----


def test_budget_starts_unset(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    b = s.budget
    assert b["max_total_tokens"] is None
    assert b["max_usd"] is None


def test_set_and_get_budget(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.set_budget(max_total_tokens=10000, max_usd=0.50)
    b = s.budget
    assert b["max_total_tokens"] == 10000
    assert b["max_usd"] == 0.50


def test_set_budget_clear_field(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.set_budget(max_total_tokens=10000)
    assert s.budget["max_total_tokens"] == 10000
    s.set_budget()  # both caps cleared
    assert s.budget["max_total_tokens"] is None
    assert s.budget["max_usd"] is None


def test_is_over_budget_no_caps(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.add_usage({"input_tokens": 999_999_999, "output_tokens": 0, "total_tokens": 999_999_999})
    over, _ = s.is_over_budget()
    assert over is False


def test_is_over_budget_token_cap(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.set_budget(max_total_tokens=1000)
    s.add_usage({"input_tokens": 600, "output_tokens": 0, "total_tokens": 600})
    over, _ = s.is_over_budget()
    assert over is False
    s.add_usage({"input_tokens": 500, "output_tokens": 0, "total_tokens": 500})
    over, reason = s.is_over_budget()
    assert over is True
    assert "token budget" in reason


def test_is_over_budget_usd_cap_lmstudio_never_fires(tmp_path):
    """Local providers cost $0; max_usd cap should never trip."""
    s = Session.create(root=tmp_path, model="m", provider="lmstudio", model_name="gemma")
    s.set_budget(max_usd=0.01)
    s.add_usage({"input_tokens": 10**8, "output_tokens": 10**8, "total_tokens": 2 * 10**8})
    over, _ = s.is_over_budget()
    assert over is False


def test_is_over_budget_usd_cap_known_provider_fires(tmp_path):
    s = Session.create(
        root=tmp_path, model="m",
        provider="openai", model_name="gpt-4o-mini",
    )
    # gpt-4o-mini: $0.15/M input. Set a tiny cap.
    s.set_budget(max_usd=0.001)
    s.add_usage({"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000})
    over, reason = s.is_over_budget()
    assert over is True
    assert "cost budget" in reason


def test_is_over_budget_unknown_provider_does_not_block(tmp_path):
    """When pricing is unmatched, max_usd cap doesn't fire (matched=False)."""
    s = Session.create(
        root=tmp_path, model="m",
        provider="random_provider", model_name="who-knows",
    )
    s.set_budget(max_usd=0.001)
    s.add_usage({"input_tokens": 999_999_999, "output_tokens": 0, "total_tokens": 999_999_999})
    over, _ = s.is_over_budget()
    assert over is False


# --- Agent loop integration ----


@pytest.mark.asyncio
async def test_agent_loop_aborts_when_over_token_budget(tmp_path):
    from godbot.core.agent import run_turn
    from godbot.core.events import DoneEvent, ErrorEvent
    from godbot.core.registry import Registry
    from tests._mock_llm import MockLLM

    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    s.set_budget(max_total_tokens=100)
    # Pre-load usage so step 0's budget check trips immediately.
    s.add_usage({"input_tokens": 200, "output_tokens": 0, "total_tokens": 200})
    s.append_user("hi")
    llm = MockLLM([json.dumps({"thought": "x", "final_answer": "ok"})])

    events = []
    async def emit(ev):
        events.append(ev)

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit, cancel=asyncio.Event(),
        max_steps=3, max_context=10000, system_prompt="SYS",
    )
    err_events = [e for e in events if isinstance(e, ErrorEvent)]
    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(err_events) >= 1
    assert "token budget" in err_events[0].message
    assert len(done_events) == 0


# --- Web endpoints ----


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    app = build_app(sessions_root=sessions_root)
    return TestClient(app), sessions_root


def test_get_budget_endpoint(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.set_budget(max_total_tokens=2000)
    r = c.get(f"/api/sessions/{s.id}/budget")
    assert r.status_code == 200
    body = r.json()
    assert body["max_total_tokens"] == 2000


def test_post_budget_endpoint_sets_caps(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.post(
        f"/api/sessions/{s.id}/budget",
        json={"max_total_tokens": 5000, "max_usd": 0.25},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["max_total_tokens"] == 5000
    assert body["max_usd"] == 0.25


def test_post_budget_endpoint_clears_via_null(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.set_budget(max_total_tokens=1000, max_usd=0.10)
    r = c.post(
        f"/api/sessions/{s.id}/budget",
        json={"max_total_tokens": None},
    )
    body = r.json()
    assert body["max_total_tokens"] is None
    # Other cap untouched.
    assert body["max_usd"] == 0.10


def test_get_budget_404_unknown_session(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/budget")
    assert r.status_code == 404
