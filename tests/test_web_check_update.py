"""Tests for sub-project 107 — /api/version/check_update."""
from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


@respx.mock
def test_check_update_returns_latest(tmp_path, monkeypatch):
    respx.get("https://api.github.com/repos/IxMxAMAR/GodBot-Gemma/releases/latest").mock(
        return_value=httpx.Response(
            200,
            json={
                "tag_name": "v9.9.9",
                "html_url": "https://github.com/IxMxAMAR/GodBot-Gemma/releases/tag/v9.9.9",
            },
        )
    )
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/version/check_update")
    assert r.status_code == 200
    body = r.json()
    assert body["latest"] == "9.9.9"
    assert body["update_available"] is True
    assert "9.9.9" in body["release_url"]
    assert body["error"] is None


@respx.mock
def test_check_update_strips_v_prefix(tmp_path, monkeypatch):
    respx.get("https://api.github.com/repos/IxMxAMAR/GodBot-Gemma/releases/latest").mock(
        return_value=httpx.Response(
            200, json={"tag_name": "v0.1.0", "html_url": "https://x.com"},
        )
    )
    c = _client(tmp_path, monkeypatch)
    body = c.get("/api/version/check_update").json()
    assert body["latest"] == "0.1.0"


@respx.mock
def test_check_update_404_no_releases(tmp_path, monkeypatch):
    """GitHub returns 404 when a repo has no published releases."""
    respx.get("https://api.github.com/repos/IxMxAMAR/GodBot-Gemma/releases/latest").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )
    c = _client(tmp_path, monkeypatch)
    body = c.get("/api/version/check_update").json()
    assert body["latest"] is None
    assert body["update_available"] is False
    assert "no published releases" in body["error"]


@respx.mock
def test_check_update_network_error_is_non_fatal(tmp_path, monkeypatch):
    respx.get("https://api.github.com/repos/IxMxAMAR/GodBot-Gemma/releases/latest").mock(
        side_effect=httpx.ConnectError("network down")
    )
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/version/check_update")
    assert r.status_code == 200
    body = r.json()
    assert body["latest"] is None
    assert body["update_available"] is False
    assert body["error"] is not None


@respx.mock
def test_check_update_no_update_when_versions_equal(tmp_path, monkeypatch):
    """We can't easily get the actual installed version in tests, but
    verify the response shape includes the comparison."""
    respx.get("https://api.github.com/repos/IxMxAMAR/GodBot-Gemma/releases/latest").mock(
        return_value=httpx.Response(
            200, json={"tag_name": "v0.0.0", "html_url": "https://x.com"},
        )
    )
    c = _client(tmp_path, monkeypatch)
    body = c.get("/api/version/check_update").json()
    # 0.0.0 is the fallback; if installed version is also 0.0.0 (editable
    # install without metadata), this comparison is False.
    assert body["update_available"] is False


def test_check_update_auth_gated_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    monkeypatch.setenv("GODBOT_API_TOKEN", "s3cr3t")
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/version/check_update")
    assert r.status_code == 401
