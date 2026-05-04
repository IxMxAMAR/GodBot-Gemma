"""Tests for sub-project 94 — /api/version."""
from __future__ import annotations

import re

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


def test_version_endpoint_returns_string(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/version")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body
    assert isinstance(body["version"], str)


def test_version_endpoint_semver_shape(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    body = c.get("/api/version").json()
    # Either real semver from importlib.metadata or "0.0.0" fallback.
    assert re.match(r"^\d+\.\d+", body["version"])


def test_version_endpoint_no_auth_required_when_disabled(tmp_path, monkeypatch):
    """When auth is off (the default), /api/version is open."""
    monkeypatch.delenv("GODBOT_API_TOKEN", raising=False)
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/version")
    assert r.status_code == 200


def test_version_endpoint_requires_auth_when_enabled(tmp_path, monkeypatch):
    """Unlike /api/health, /api/version IS gated by auth when enabled.
    (Liveness probes use /api/health.)"""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.setenv("GODBOT_API_TOKEN", "s3cr3t")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/version")
    # Auth applies to /api/* except /api/health.
    assert r.status_code == 401
