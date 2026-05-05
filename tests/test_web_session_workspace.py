"""Tests for sub-project 105 — POST /api/sessions/{sid}/workspace."""
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


def test_workspace_404_unknown(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/sessions/no-such/workspace", json={"workspace": str(tmp_path)})
    assert r.status_code == 404


def test_workspace_set_to_path(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    ws = tmp_path / "newproj"
    ws.mkdir()
    r = c.post(f"/api/sessions/{s.id}/workspace", json={"workspace": str(ws)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["workspace_root"] == str(ws.resolve())

    s2 = Session.load(sroot, s.id)
    assert s2._meta.get("workspace_root") == str(ws.resolve())


def test_workspace_clear(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    ws = tmp_path / "ws"
    ws.mkdir()
    s = Session.create(root=sroot, model="m", workspace_root=str(ws))
    r = c.post(f"/api/sessions/{s.id}/workspace", json={"workspace": None})
    body = r.json()
    assert body["workspace_root"] is None


def test_workspace_invalid_path_400(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.post(
        f"/api/sessions/{s.id}/workspace",
        json={"workspace": str(tmp_path / "nonexistent")},
    )
    assert r.status_code == 400


def test_workspace_non_string_400(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.post(f"/api/sessions/{s.id}/workspace", json={"workspace": 12345})
    assert r.status_code == 400


def test_workspace_preserves_auto_approve_when_unspecified(tmp_path, monkeypatch):
    """Not specifying auto_approve_in_sandbox keeps the existing flag."""
    c, sroot = _client(tmp_path, monkeypatch)
    ws_a = tmp_path / "a"
    ws_a.mkdir()
    ws_b = tmp_path / "b"
    ws_b.mkdir()
    s = Session.create(
        root=sroot, model="m",
        workspace_root=str(ws_a), auto_approve_in_sandbox=True,
    )
    r = c.post(f"/api/sessions/{s.id}/workspace", json={"workspace": str(ws_b)})
    body = r.json()
    assert body["auto_approve_in_sandbox"] is True


def test_workspace_explicit_auto_approve_overrides(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    ws = tmp_path / "ws"
    ws.mkdir()
    s = Session.create(
        root=sroot, model="m", auto_approve_in_sandbox=True,
    )
    r = c.post(
        f"/api/sessions/{s.id}/workspace",
        json={"workspace": str(ws), "auto_approve_in_sandbox": False},
    )
    body = r.json()
    assert body["auto_approve_in_sandbox"] is False
