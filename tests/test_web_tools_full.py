"""Tests for sub-project 87 — /api/tools/full endpoint."""
from __future__ import annotations

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


def test_tools_full_returns_list(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/tools/full")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) > 0


def test_tools_full_includes_schema(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    body = c.get("/api/tools/full").json()
    by_name = {t["name"]: t for t in body}
    # read_file is a known tool with a 'path' arg.
    rf = by_name.get("read_file")
    assert rf is not None
    assert "schema" in rf
    schema = rf["schema"]
    assert schema.get("type") == "object"
    assert "properties" in schema
    assert "path" in schema["properties"]


def test_tools_full_includes_timeout_and_dangerous(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    body = c.get("/api/tools/full").json()
    for t in body:
        assert "timeout" in t
        assert isinstance(t["timeout"], int)
        assert "dangerous" in t


def test_tools_full_dangerous_flag_correct(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    body = c.get("/api/tools/full").json()
    by_name = {t["name"]: t for t in body}
    # write_file is dangerous, read_file is not.
    assert by_name["write_file"]["dangerous"] is True
    assert by_name["read_file"]["dangerous"] is False


def test_tools_full_count_matches_short_list(tmp_path, monkeypatch):
    """tools_full returns the same number of tools as /api/tools."""
    c = _client(tmp_path, monkeypatch)
    short = c.get("/api/tools").json()
    full = c.get("/api/tools/full").json()
    assert len(short) == len(full)
