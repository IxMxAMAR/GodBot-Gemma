"""Tests for sub-project 88 — optional API token auth."""
from __future__ import annotations

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.interfaces.web import build_app


def test_auth_disabled_by_default(tmp_path, monkeypatch):
    """No token in config + no env var = open access (back-compat)."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.delenv("GODBOT_API_TOKEN", raising=False)
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    assert c.get("/api/tools").status_code == 200
    assert c.get("/api/health").status_code == 200


def test_auth_blocks_unauthenticated_when_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.setenv("GODBOT_API_TOKEN", "s3cr3t")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/tools")
    assert r.status_code == 401
    assert "Bearer" in r.headers.get("www-authenticate", "")


def test_auth_health_endpoint_stays_open(tmp_path, monkeypatch):
    """Liveness probe must NOT require auth even when auth is enabled."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.setenv("GODBOT_API_TOKEN", "s3cr3t")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_auth_correct_token_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.setenv("GODBOT_API_TOKEN", "s3cr3t")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/tools", headers={"Authorization": "Bearer s3cr3t"})
    assert r.status_code == 200


def test_auth_wrong_token_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.setenv("GODBOT_API_TOKEN", "s3cr3t")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/tools", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_auth_missing_bearer_prefix_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.setenv("GODBOT_API_TOKEN", "s3cr3t")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/tools", headers={"Authorization": "s3cr3t"})  # no "Bearer " prefix
    assert r.status_code == 401


def test_auth_env_overrides_config(tmp_path, monkeypatch):
    """GODBOT_API_TOKEN env var wins over [auth] token in config.toml."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        '[auth]\ntoken = "from-config"\n', encoding="utf-8"
    )
    monkeypatch.setenv("GODBOT_API_TOKEN", "from-env")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    # Config token should NOT work; env var should.
    assert c.get("/api/tools", headers={"Authorization": "Bearer from-config"}).status_code == 401
    assert c.get("/api/tools", headers={"Authorization": "Bearer from-env"}).status_code == 200


def test_auth_config_only(tmp_path, monkeypatch):
    """Token in [auth] section alone (no env var) also gates."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.delenv("GODBOT_API_TOKEN", raising=False)
    (tmp_path / "config.toml").write_text(
        '[auth]\ntoken = "cfg-token"\n', encoding="utf-8"
    )
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    assert c.get("/api/tools").status_code == 401
    assert c.get("/api/tools", headers={"Authorization": "Bearer cfg-token"}).status_code == 200


def test_auth_static_routes_open(tmp_path, monkeypatch):
    """Non-/api routes (static frontend) stay open even with auth on."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.setenv("GODBOT_API_TOKEN", "s3cr3t")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    # /static doesn't exist as a route in tests (no static dir mounted),
    # but the response must NOT be 401 — it'll be 404 instead.
    r = c.get("/some-nonexistent-static-file")
    assert r.status_code != 401
