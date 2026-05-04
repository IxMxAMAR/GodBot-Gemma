"""Tests for sub-project 52 — session pin/unpin and cleanup exemption."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

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


def test_session_starts_unpinned(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    assert s.pinned is False


def test_set_pinned_persists(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.set_pinned(True)
    assert s.pinned is True
    s2 = Session.load(tmp_path, s.id)
    assert s2.pinned is True


def test_pin_endpoint_sets(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.post(f"/api/sessions/{s.id}/pin", json={"pinned": True})
    assert r.status_code == 200
    body = r.json()
    assert body["pinned"] is True
    # Reload from disk to confirm persistence.
    s2 = Session.load(sroot, s.id)
    assert s2.pinned is True


def test_pin_endpoint_unpin(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.set_pinned(True)
    r = c.post(f"/api/sessions/{s.id}/pin", json={"pinned": False})
    assert r.status_code == 200
    assert r.json()["pinned"] is False


def test_pin_endpoint_400_missing_field(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.post(f"/api/sessions/{s.id}/pin", json={})
    assert r.status_code == 400


def test_pin_endpoint_404_unknown(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/sessions/no-such/pin", json={"pinned": True})
    assert r.status_code == 404


def _set_started_days_ago(meta_path, days):
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["started_at"] = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(data), encoding="utf-8")


def test_cleanup_exempts_pinned_sessions(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    pinned = Session.create(root=sroot, model="m")
    unpinned = Session.create(root=sroot, model="m")
    # Both look "old" by date; only the pinned one survives.
    _set_started_days_ago(sroot / pinned.id / "meta.json", 60)
    _set_started_days_ago(sroot / unpinned.id / "meta.json", 60)
    pinned.set_pinned(True)

    r = c.post("/api/sessions/cleanup", json={})
    body = r.json()
    assert pinned.id not in body["deleted"]
    assert unpinned.id in body["deleted"]
    assert (sroot / pinned.id).exists() is True
    assert (sroot / unpinned.id).exists() is False


def test_explicit_delete_still_works_on_pinned(tmp_path, monkeypatch):
    """DELETE /api/sessions/{sid} bypasses pinned (cleanup exemption ≠
    delete protection)."""
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.set_pinned(True)
    r = c.delete(f"/api/sessions/{s.id}")
    assert r.status_code == 200
    assert (sroot / s.id).exists() is False
