"""Tests for sub-project 100 — /api/admin/snapshot diagnostic dump."""
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


def test_snapshot_basic_shape(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/admin/snapshot")
    assert r.status_code == 200
    body = r.json()
    for key in ("version", "health", "stats", "workspaces", "audit", "generated_at"):
        assert key in body


def test_snapshot_health_field(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    body = c.get("/api/admin/snapshot").json()
    health = body["health"]
    assert health["status"] == "ok"
    assert "tools" in health
    assert health["tools"]["total"] > 0
    assert "uptime_seconds" in health


def test_snapshot_aggregates_session_data(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(
        root=sroot, model="m", provider="openai", model_name="gpt-4o-mini",
        workspace_root=str(tmp_path),
    )
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.append_tool_result("c1", "ok")
    s.add_usage({"input_tokens": 1_000_000, "output_tokens": 500_000, "total_tokens": 1_500_000})

    body = c.get("/api/admin/snapshot").json()
    assert body["stats"]["sessions"] == 1
    assert body["stats"]["tool_calls_total"] == 1
    assert any(t["name"] == "read_file" for t in body["stats"]["top_tools"])
    assert body["stats"]["usage"]["total_tokens"] == 1_500_000
    assert body["stats"]["estimated_cost_usd"] > 0


def test_snapshot_includes_workspaces(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    ws_a = tmp_path / "ws_a"; ws_a.mkdir()
    Session.create(root=sroot, model="m", workspace_root=str(ws_a))
    body = c.get("/api/admin/snapshot").json()
    paths = [w["path"] for w in body["workspaces"]]
    assert str(ws_a.resolve()) in paths


def test_snapshot_audit_includes_dangerous_calls(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "write_file", {"path": "x", "content": "y"}, raw="{}")
    s.append_tool_result("c1", "ok: wrote")
    body = c.get("/api/admin/snapshot").json()
    assert len(body["audit"]) == 1
    assert body["audit"][0]["name"] == "write_file"


def test_snapshot_audit_skips_safe_calls(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.append_tool_result("c1", "content")
    body = c.get("/api/admin/snapshot").json()
    assert body["audit"] == []


def test_snapshot_generated_at_iso(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    import re
    body = c.get("/api/admin/snapshot").json()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", body["generated_at"])


def test_snapshot_auth_gated(tmp_path, monkeypatch):
    """When auth is on, /api/admin/snapshot requires the bearer token."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.setenv("GODBOT_API_TOKEN", "s3cr3t")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    assert c.get("/api/admin/snapshot").status_code == 401
    assert c.get("/api/admin/snapshot", headers={"Authorization": "Bearer s3cr3t"}).status_code == 200
