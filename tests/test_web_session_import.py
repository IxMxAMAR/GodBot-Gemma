"""Tests for sub-project 51 — /api/sessions/import (round-trip with /export)."""
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


def test_import_400_on_missing_events(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/sessions/import", json={})
    assert r.status_code == 400


def test_import_400_on_non_object_body(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    # FastAPI rejects non-dict JSON before our 400 fires; either response
    # is acceptable as long as it's an error.
    r = c.post("/api/sessions/import", json=["not", "a", "dict"])
    assert r.status_code in (400, 422)


def test_export_then_import_round_trip(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    src = Session.create(
        root=sroot, model="m", provider="openai", model_name="gpt-4o-mini",
    )
    src.append_user("hi")
    src.append_assistant_final("hello")
    src.add_usage({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    src.set_budget(max_total_tokens=5000)

    exported = c.get(f"/api/sessions/{src.id}/export").json()
    r = c.post("/api/sessions/import", json=exported)
    assert r.status_code == 200
    body = r.json()
    new_sid = body["session_id"]
    assert body["imported_from"] == src.id
    assert body["events_imported"] == 2

    new = Session.load(sroot, new_sid)
    msgs = new.messages_for_llm()
    assert len(msgs) == 2
    assert msgs[0]["content"] == "hi"
    assert msgs[1]["content"] == "hello"
    assert new.provider == "openai"
    assert new.model_name == "gpt-4o-mini"
    # Usage carried over.
    assert new.usage["turns"] == 1
    assert new.usage["total_tokens"] == 150
    # Budget carried over.
    assert new.budget["max_total_tokens"] == 5000


def test_import_assigns_fresh_id(tmp_path, monkeypatch):
    """The imported session's id must NOT match the source."""
    c, sroot = _client(tmp_path, monkeypatch)
    src = Session.create(root=sroot, model="m")
    src.append_user("x")
    exp = c.get(f"/api/sessions/{src.id}/export").json()
    r = c.post("/api/sessions/import", json=exp)
    new_sid = r.json()["session_id"]
    assert new_sid != src.id


def test_import_skips_malformed_events(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    payload = {
        "model": "m", "provider": "lmstudio", "model_name": "m",
        "events": [
            {"type": "user", "content": "hi"},
            "not a dict",  # should be skipped
            {"no_type": "field"},  # missing type, skipped
            {"type": "assistant_final", "content": "hello"},
        ],
    }
    r = c.post("/api/sessions/import", json=payload)
    assert r.status_code == 200
    assert r.json()["events_imported"] == 2


def test_import_works_without_optional_fields(tmp_path, monkeypatch):
    """Minimal valid import: just events + model."""
    c, sroot = _client(tmp_path, monkeypatch)
    payload = {
        "events": [{"type": "user", "content": "hi"}],
    }
    r = c.post("/api/sessions/import", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["events_imported"] == 1
    new = Session.load(sroot, body["session_id"])
    assert new.provider == "lmstudio"  # default
