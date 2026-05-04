"""Tests for sub-project 35 — per-turn feedback."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.session import Session
from godbot.interfaces.web import build_app


# --- Session-level API ----


def test_append_feedback_records_event(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    s.append_assistant_final("hello")
    s.append_feedback(target_event_index=1, rating="up", comment="nice")
    summary = s.feedback_summary()
    assert summary["up"] == 1
    assert summary["down"] == 0
    assert summary["latest_by_index"][1]["rating"] == "up"
    assert summary["latest_by_index"][1]["comment"] == "nice"


def test_append_feedback_invalid_rating_raises(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    with pytest.raises(ValueError):
        s.append_feedback(target_event_index=0, rating="meh")


def test_append_feedback_latest_wins(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    s.append_assistant_final("hello")
    s.append_feedback(target_event_index=1, rating="up")
    s.append_feedback(target_event_index=1, rating="down", comment="changed mind")
    summary = s.feedback_summary()
    # Both events counted in totals.
    assert summary["up"] == 1
    assert summary["down"] == 1
    # latest_by_index reflects the most recent rating.
    assert summary["latest_by_index"][1]["rating"] == "down"
    assert summary["latest_by_index"][1]["comment"] == "changed mind"


def test_feedback_persists_across_load(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_assistant_final("answer")
    s.append_feedback(target_event_index=0, rating="down")
    s2 = Session.load(tmp_path, s.id)
    assert s2.feedback_summary()["down"] == 1


def test_comment_is_truncated_at_1000_chars(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("x")
    huge = "y" * 2000
    s.append_feedback(target_event_index=0, rating="up", comment=huge)
    summary = s.feedback_summary()
    assert len(summary["latest_by_index"][0]["comment"]) == 1000


# --- Endpoints ----


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    app = build_app(sessions_root=sessions_root)
    return TestClient(app), sessions_root


def test_post_feedback_endpoint(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("q")
    s.append_assistant_final("a")
    r = c.post(
        f"/api/sessions/{s.id}/feedback",
        json={"target_index": 1, "rating": "up", "comment": "great answer"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    fb = c.get(f"/api/sessions/{s.id}/feedback").json()
    assert fb["up"] == 1
    assert fb["latest_by_index"]["1"]["rating"] == "up"


def test_post_feedback_404_unknown_session(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/sessions/no-such/feedback", json={"target_index": 0, "rating": "up"})
    assert r.status_code == 404


def test_post_feedback_400_missing_index(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.post(f"/api/sessions/{s.id}/feedback", json={"rating": "up"})
    assert r.status_code == 400


def test_post_feedback_400_bad_rating(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.post(f"/api/sessions/{s.id}/feedback", json={"target_index": 0, "rating": "yes"})
    assert r.status_code == 400


def test_get_feedback_404_unknown(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/feedback")
    assert r.status_code == 404
