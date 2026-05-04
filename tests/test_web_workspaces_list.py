"""Tests for sub-project 44 — /api/workspaces."""
from __future__ import annotations

import time

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


def test_workspaces_empty_state(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/workspaces")
    assert r.status_code == 200
    assert r.json() == {"workspaces": []}


def test_workspaces_distinct_only(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    ws_a = tmp_path / "ws_a"; ws_a.mkdir()
    ws_b = tmp_path / "ws_b"; ws_b.mkdir()
    Session.create(root=sroot, model="m", workspace_root=str(ws_a))
    time.sleep(0.01)  # ensure distinct started_at strings
    Session.create(root=sroot, model="m", workspace_root=str(ws_a))
    Session.create(root=sroot, model="m", workspace_root=str(ws_b))
    r = c.get("/api/workspaces")
    body = r.json()
    paths = [w["path"] for w in body["workspaces"]]
    assert len(paths) == 2
    assert str(ws_a.resolve()) in paths
    assert str(ws_b.resolve()) in paths


def test_workspaces_session_count_aggregated(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    ws = tmp_path / "ws"; ws.mkdir()
    Session.create(root=sroot, model="m", workspace_root=str(ws))
    Session.create(root=sroot, model="m", workspace_root=str(ws))
    Session.create(root=sroot, model="m", workspace_root=str(ws))
    r = c.get("/api/workspaces")
    body = r.json()
    assert len(body["workspaces"]) == 1
    assert body["workspaces"][0]["sessions"] == 3


def test_workspaces_skips_sessions_without_workspace(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    Session.create(root=sroot, model="m")  # no workspace_root
    ws = tmp_path / "ws"; ws.mkdir()
    Session.create(root=sroot, model="m", workspace_root=str(ws))
    r = c.get("/api/workspaces")
    paths = [w["path"] for w in r.json()["workspaces"]]
    assert paths == [str(ws.resolve())]


def test_workspaces_ordered_by_last_activity_desc(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    ws_old = tmp_path / "old"; ws_old.mkdir()
    ws_new = tmp_path / "new"; ws_new.mkdir()
    Session.create(root=sroot, model="m", workspace_root=str(ws_old))
    time.sleep(1.05)  # started_at has 1-second resolution
    Session.create(root=sroot, model="m", workspace_root=str(ws_new))
    r = c.get("/api/workspaces")
    body = r.json()
    paths = [w["path"] for w in body["workspaces"]]
    assert paths[0] == str(ws_new.resolve())
    assert paths[1] == str(ws_old.resolve())
