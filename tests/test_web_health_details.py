"""Tests for sub-project 43 — /api/health/details."""
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


def test_health_details_basic_shape(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/health/details")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert isinstance(body["uptime_seconds"], int)
    assert body["uptime_seconds"] >= 0
    assert "providers" in body
    assert "default" in body["providers"]
    assert "configured" in body["providers"]
    assert isinstance(body["mcp_servers"], list)
    assert "tools" in body
    assert body["tools"]["total"] > 0
    assert body["tools"]["dangerous"] >= 1  # write_file et al.
    assert "background_tasks" in body
    assert body["background_tasks"]["total"] >= 0


def test_health_details_session_count(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    Session.create(root=sroot, model="m")
    Session.create(root=sroot, model="m")
    r = c.get("/api/health/details")
    body = r.json()
    assert body["sessions"]["count"] == 2


def test_health_endpoint_unchanged(tmp_path, monkeypatch):
    """/api/health (unsuffixed) keeps its bare shape — back-compat."""
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok"}


def test_health_details_no_provider_probes(tmp_path, monkeypatch):
    """The endpoint must not hit upstream provider URLs (cheap to call
    repeatedly). We just verify the call returns quickly and doesn't
    require any TOML config. No HTTP mocking needed since no requests
    should fly."""
    c, _ = _client(tmp_path, monkeypatch)
    import time as _time
    started = _time.monotonic()
    for _ in range(5):
        r = c.get("/api/health/details")
        assert r.status_code == 200
    elapsed = _time.monotonic() - started
    # Five calls should comfortably finish in well under a second.
    assert elapsed < 5.0
