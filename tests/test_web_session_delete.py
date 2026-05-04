"""Tests for sub-project 34 — session deletion + bulk cleanup."""
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


# --- single delete ----


def test_delete_404_unknown(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.delete("/api/sessions/no-such")
    assert r.status_code == 404


def test_delete_removes_directory_and_record(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    sdir = sroot / s.id
    assert sdir.exists()
    r = c.delete(f"/api/sessions/{s.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["deleted"] == s.id
    assert not sdir.exists()
    # And the GET endpoint now 404s.
    assert c.get(f"/api/sessions/{s.id}").status_code == 404


def test_delete_idempotent_returns_404_on_second(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    c.delete(f"/api/sessions/{s.id}")
    r = c.delete(f"/api/sessions/{s.id}")
    assert r.status_code == 404


# --- bulk cleanup ----


def _set_started_at(meta_path, days_ago: int) -> None:
    """Rewrite a session's meta.json started_at to N days ago."""
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["started_at"] = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(data), encoding="utf-8")


def test_cleanup_default_30_day_cutoff(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    fresh = Session.create(root=sroot, model="m")
    old = Session.create(root=sroot, model="m")
    _set_started_at(sroot / old.id / "meta.json", days_ago=60)

    r = c.post("/api/sessions/cleanup", json={})
    body = r.json()
    assert old.id in body["deleted"]
    assert fresh.id not in body["deleted"]
    assert body["kept"] >= 1
    assert (sroot / old.id).exists() is False
    assert (sroot / fresh.id).exists() is True


def test_cleanup_dry_run_does_not_delete(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    _set_started_at(sroot / s.id / "meta.json", days_ago=60)

    r = c.post("/api/sessions/cleanup", json={"dry_run": True})
    body = r.json()
    assert body["dry_run"] is True
    assert s.id in body["deleted"]
    # Directory still on disk.
    assert (sroot / s.id).exists() is True


def test_cleanup_custom_cutoff(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    _set_started_at(sroot / s.id / "meta.json", days_ago=10)

    # 30-day default keeps it.
    r = c.post("/api/sessions/cleanup", json={"older_than_days": 30})
    assert s.id not in r.json()["deleted"]
    # 7-day cutoff deletes it.
    r = c.post("/api/sessions/cleanup", json={"older_than_days": 7})
    assert s.id in r.json()["deleted"]


def test_cleanup_rejects_zero_days(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/sessions/cleanup", json={"older_than_days": 0})
    assert r.status_code == 400


def test_cleanup_handles_corrupt_meta(tmp_path, monkeypatch):
    """A session with unreadable meta.json is counted as kept, not crashed."""
    c, sroot = _client(tmp_path, monkeypatch)
    bogus = sroot / "broken-2026-01-01"
    bogus.mkdir()
    (bogus / "meta.json").write_text("not json", encoding="utf-8")
    r = c.post("/api/sessions/cleanup", json={"older_than_days": 1})
    assert r.status_code == 200
    body = r.json()
    assert "broken-2026-01-01" not in body["deleted"]
    assert body["kept"] >= 1
