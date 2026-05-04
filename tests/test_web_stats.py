"""Tests for sub-project 32 — /api/stats aggregate metrics endpoint."""
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


def test_stats_empty_state(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["sessions"] == 0
    assert body["turns"] == 0
    assert body["tool_calls_total"] == 0
    assert body["top_tools"] == []
    assert body["tasks"]["total"] == 0
    assert body["usage"]["total_tokens"] == 0
    assert body["estimated_cost_usd"] == 0.0


def test_stats_counts_sessions_and_turns(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s1 = Session.create(root=sroot, model="m")
    s2 = Session.create(root=sroot, model="m")
    s1.add_usage({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    s1.add_usage({"input_tokens": 200, "output_tokens": 75, "total_tokens": 275})
    s2.add_usage({"input_tokens": 50, "output_tokens": 25, "total_tokens": 75})
    r = c.get("/api/stats")
    body = r.json()
    assert body["sessions"] == 2
    assert body["turns"] == 3
    assert body["usage"]["input_tokens"] == 350
    assert body["usage"]["output_tokens"] == 150
    assert body["usage"]["total_tokens"] == 500


def test_stats_aggregates_tool_calls_and_returns_top(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("hi")
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.append_tool_result("c1", "ok")
    s.append_assistant_tool_call("c2", "read_file", {"path": "y"}, raw="{}")
    s.append_tool_result("c2", "ok")
    s.append_assistant_tool_call("c3", "git_status", {}, raw="{}")
    s.append_tool_result("c3", "clean")
    r = c.get("/api/stats")
    body = r.json()
    assert body["tool_calls_total"] == 3
    by_name = {entry["name"]: entry["count"] for entry in body["top_tools"]}
    assert by_name["read_file"] == 2
    assert by_name["git_status"] == 1
    # The top entry is read_file.
    assert body["top_tools"][0]["name"] == "read_file"


def test_stats_includes_estimated_cost_for_known_provider(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(
        root=sroot, model="m",
        provider="openai", model_name="gpt-4o-mini",
    )
    # gpt-4o-mini: $0.15/M in + $0.60/M out.
    s.add_usage({"input_tokens": 1_000_000, "output_tokens": 500_000, "total_tokens": 1_500_000})
    r = c.get("/api/stats")
    body = r.json()
    # 0.15 + 0.30 = 0.45 USD
    assert body["estimated_cost_usd"] == 0.45


def test_stats_skips_unmatched_provider_in_cost(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(
        root=sroot, model="m",
        provider="totally_unknown", model_name="?",
    )
    s.add_usage({"input_tokens": 999_999, "output_tokens": 999_999, "total_tokens": 1_999_998})
    r = c.get("/api/stats")
    body = r.json()
    # Unmatched provider doesn't contribute cost.
    assert body["estimated_cost_usd"] == 0.0
    # But token counts still flow through.
    assert body["usage"]["total_tokens"] == 1_999_998


def test_stats_resilient_to_unreadable_session_dir(tmp_path, monkeypatch):
    """A directory in sessions_root that isn't a valid session is skipped."""
    c, sroot = _client(tmp_path, monkeypatch)
    bogus = sroot / "not-a-real-session"
    bogus.mkdir()
    # No meta.json — should be skipped.
    r = c.get("/api/stats")
    assert r.status_code == 200
    assert r.json()["sessions"] == 0
