"""Tests for sub-project 20 — per-session token usage tracking."""
from __future__ import annotations

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.session import Session
from godbot.interfaces.web import build_app


def test_usage_property_zero_default(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    u = s.usage
    assert u == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "turns": 0}


def test_add_usage_accumulates(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.add_usage({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    s.add_usage({"input_tokens": 200, "output_tokens": 75, "total_tokens": 275})
    u = s.usage
    assert u["input_tokens"] == 300
    assert u["output_tokens"] == 125
    assert u["total_tokens"] == 425
    assert u["turns"] == 2


def test_add_usage_skips_empty(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.add_usage({})
    s.add_usage({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    u = s.usage
    assert u == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "turns": 0}


def test_add_usage_handles_partial_dicts(tmp_path):
    """Some providers omit total_tokens; only input/output present."""
    s = Session.create(root=tmp_path, model="m")
    s.add_usage({"input_tokens": 100, "output_tokens": 30})
    u = s.usage
    assert u["input_tokens"] == 100
    assert u["output_tokens"] == 30
    assert u["total_tokens"] == 0  # not provided, stays 0
    assert u["turns"] == 1


def test_add_usage_persists_across_load(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    sid = s.id
    s.add_usage({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    s2 = Session.load(tmp_path, sid)
    u = s2.usage
    assert u["input_tokens"] == 100
    assert u["turns"] == 1


def test_add_usage_no_op_on_garbage(tmp_path):
    """Non-int values should not crash add_usage; the session stays at zero."""
    s = Session.create(root=tmp_path, model="m")
    s.add_usage({"input_tokens": "not_a_number"})
    assert s.usage["turns"] == 0


def test_session_get_endpoint_includes_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    s = Session.create(root=sessions_root, model="m")
    s.add_usage({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    app = build_app(sessions_root=sessions_root)
    c = TestClient(app)
    r = c.get(f"/api/sessions/{s.id}")
    assert r.status_code == 200
    body = r.json()
    assert "usage" in body
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["output_tokens"] == 5
    assert body["usage"]["turns"] == 1


def test_usage_endpoint_returns_normalized_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    s = Session.create(root=sessions_root, model="m")
    s.add_usage({"input_tokens": 200, "output_tokens": 50, "total_tokens": 250})
    s.add_usage({"input_tokens": 100, "output_tokens": 25, "total_tokens": 125})

    app = build_app(sessions_root=sessions_root)
    c = TestClient(app)
    r = c.get(f"/api/sessions/{s.id}/usage")
    assert r.status_code == 200
    body = r.json()
    assert body["input_tokens"] == 300
    assert body["output_tokens"] == 75
    assert body["total_tokens"] == 375
    assert body["turns"] == 2


def test_usage_endpoint_404_on_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/sessions/no-such-session/usage")
    assert r.status_code == 404


def test_agent_loop_does_not_inflate_usage_on_legacy_llm(tmp_path):
    """Legacy LLMClient returns empty usage; turn count must stay zero."""
    import asyncio
    import json
    from godbot.core.agent import run_turn
    from godbot.core.events import DoneEvent
    from godbot.core.registry import Registry
    from tests._mock_llm import MockLLM

    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    llm = MockLLM([json.dumps({"thought": "ok", "final_answer": "hi back"})])

    events = []
    async def emit(ev):
        events.append(ev)

    asyncio.run(run_turn(
        llm=llm, session=s, registry=reg, emit=emit, cancel=asyncio.Event(),
        max_steps=3, max_context=10000, system_prompt="SYS",
    ))
    assert any(isinstance(e, DoneEvent) for e in events)
    # MockLLM doesn't carry usage; session.usage stays zero.
    assert s.usage["turns"] == 0
