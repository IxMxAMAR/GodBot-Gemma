"""Tests for sub-project 111 — /api/sessions/{sid}/context_usage."""
from __future__ import annotations

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.session import Session
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    app = build_app(sessions_root=sessions_root)
    return TestClient(app), sessions_root


def test_context_usage_404_unknown(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/context_usage")
    assert r.status_code == 404


def test_context_usage_basic_shape(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.get(f"/api/sessions/{s.id}/context_usage")
    assert r.status_code == 200
    body = r.json()
    for k in ("used_tokens", "max_context", "percent", "turns", "avg_turn_tokens"):
        assert k in body


def test_context_usage_grows_with_messages(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    initial = c.get(f"/api/sessions/{s.id}/context_usage").json()["used_tokens"]
    s.append_user("x" * 4000)  # ~1000 tokens
    s.append_assistant_final("y" * 4000)
    grown = c.get(f"/api/sessions/{s.id}/context_usage").json()["used_tokens"]
    assert grown > initial
    # Grew by roughly 8000/4 = 2000 tokens.
    assert grown - initial >= 1500


def test_context_usage_percent_is_capped_at_100(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    # Stuff the session way over the configured 28000-token max_context.
    huge = "z" * 200_000  # ~50000 tokens
    s.append_user(huge)
    body = c.get(f"/api/sessions/{s.id}/context_usage").json()
    assert body["percent"] <= 100.0


def test_context_usage_max_context_from_config(tmp_path, monkeypatch):
    """Custom max_context in config flows through to the response."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        '[llm]\nmax_context = 8000\n', encoding="utf-8",
    )
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    app = build_app(sessions_root=sessions_root)
    c = TestClient(app)
    s = Session.create(root=sessions_root, model="m")
    body = c.get(f"/api/sessions/{s.id}/context_usage").json()
    assert body["max_context"] == 8000


def test_context_usage_turns_reflects_usage(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.add_usage({"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})
    s.add_usage({"input_tokens": 500, "output_tokens": 100, "total_tokens": 600})
    body = c.get(f"/api/sessions/{s.id}/context_usage").json()
    assert body["turns"] == 2
    assert body["avg_turn_tokens"] == 750  # (1000 + 500) / 2


def test_context_usage_no_turns_avg_is_zero(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    body = c.get(f"/api/sessions/{s.id}/context_usage").json()
    assert body["turns"] == 0
    assert body["avg_turn_tokens"] == 0
