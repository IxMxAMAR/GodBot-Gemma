"""Tests for sub-project 79 — workspace + pinned filters on /api/sessions."""
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


def test_sessions_list_returns_richer_shape(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m", workspace_root=str(tmp_path))
    body = c.get("/api/sessions").json()
    entry = next(e for e in body if e["id"] == s.id)
    assert "model" in entry
    assert "provider" in entry
    assert "workspace_root" in entry
    assert "started_at" in entry
    assert "pinned" in entry
    assert entry["pinned"] is False


def test_sessions_filter_by_workspace(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    ws_a = tmp_path / "ws_a"; ws_a.mkdir()
    ws_b = tmp_path / "ws_b"; ws_b.mkdir()
    s_a = Session.create(root=sroot, model="m", workspace_root=str(ws_a))
    Session.create(root=sroot, model="m", workspace_root=str(ws_b))
    body = c.get(f"/api/sessions?workspace={ws_a.resolve()}").json()
    assert [e["id"] for e in body] == [s_a.id]


def test_sessions_filter_pinned_only(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    pinned = Session.create(root=sroot, model="m")
    Session.create(root=sroot, model="m")
    pinned.set_pinned(True)
    body = c.get("/api/sessions?pinned_only=true").json()
    ids = [e["id"] for e in body]
    assert pinned.id in ids
    assert len(ids) == 1


def test_sessions_limit(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    for _ in range(5):
        Session.create(root=sroot, model="m")
    body = c.get("/api/sessions?limit=2").json()
    assert len(body) == 2


def test_sessions_newest_first(tmp_path, monkeypatch):
    """Ordering: most recent session first."""
    import time
    c, sroot = _client(tmp_path, monkeypatch)
    older = Session.create(root=sroot, model="m")
    time.sleep(1.05)  # started_at has 1s resolution
    newer = Session.create(root=sroot, model="m")
    body = c.get("/api/sessions").json()
    ids = [e["id"] for e in body]
    assert ids.index(newer.id) < ids.index(older.id)


def test_sessions_workspace_filter_no_match(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    Session.create(root=sroot, model="m")
    body = c.get(f"/api/sessions?workspace={tmp_path}/no-such").json()
    assert body == []
