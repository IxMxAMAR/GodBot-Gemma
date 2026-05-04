"""Tests for sub-project 92 — /api/sessions/{sid}/clear_overrides."""
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


def test_clear_overrides_404_unknown(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/sessions/no-such/clear_overrides")
    assert r.status_code == 404


def test_clear_overrides_resets_to_none(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.set_tool_overrides(["read_file"])
    assert s.tool_overrides == ["read_file"]

    r = c.post(f"/api/sessions/{s.id}/clear_overrides")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["tool_overrides"] is None

    # Reload from disk to confirm persistence.
    s2 = Session.load(sroot, s.id)
    assert s2.tool_overrides is None


def test_clear_overrides_idempotent(tmp_path, monkeypatch):
    """Calling clear_overrides on a session that already has no overrides
    is a no-op success."""
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    assert s.tool_overrides is None
    r = c.post(f"/api/sessions/{s.id}/clear_overrides")
    assert r.status_code == 200
    body = r.json()
    assert body["tool_overrides"] is None


def test_clear_overrides_does_not_affect_other_meta(tmp_path, monkeypatch):
    """Other session meta fields stay untouched."""
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m", workspace_root=str(tmp_path))
    s.set_tool_overrides(["read_file"])
    s.set_pinned(True)
    s.set_budget(max_total_tokens=5000)

    c.post(f"/api/sessions/{s.id}/clear_overrides")
    s2 = Session.load(sroot, s.id)
    # Other fields remain intact.
    assert s2.pinned is True
    assert s2.budget["max_total_tokens"] == 5000
    assert s2._meta.get("workspace_root") == str(__import__("pathlib").Path(tmp_path).resolve())
