"""Tests for sub-project 26 — tool result blob retrieval endpoint."""
from __future__ import annotations

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.session import BLOB_INLINE_LIMIT, Session
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    app = build_app(sessions_root=sessions_root)
    return TestClient(app), sessions_root


def test_blob_404_on_unknown_session(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/blobs/c0")
    assert r.status_code == 404


def test_blob_404_on_unknown_call_id(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    r = c.get(f"/api/sessions/{s.id}/blobs/c-nope")
    assert r.status_code == 404


def test_blob_returns_full_content_when_oversized(tmp_path, monkeypatch):
    """Tool results larger than BLOB_INLINE_LIMIT trigger blob storage;
    the endpoint must return the full text, not the truncated LLM view."""
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    big = "x" * (BLOB_INLINE_LIMIT + 100)
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.record_tool_result("c1", big)

    r = c.get(f"/api/sessions/{s.id}/blobs/c1")
    assert r.status_code == 200
    body = r.json()
    assert body["call_id"] == "c1"
    assert body["length"] == len(big)
    assert body["content"] == big


def test_blob_as_text_returns_plain_text(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    big = "y" * (BLOB_INLINE_LIMIT + 100)
    s.append_assistant_tool_call("c2", "x", {}, raw="{}")
    s.record_tool_result("c2", big)

    r = c.get(f"/api/sessions/{s.id}/blobs/c2?as_text=true")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == big


def test_blob_404_when_result_was_inline_only(tmp_path, monkeypatch):
    """Small results never write a blob — call_id has no on-disk file."""
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c3", "x", {}, raw="{}")
    # 100 chars is well under the inline limit, so no blob is persisted.
    s.record_tool_result("c3", "small result")
    r = c.get(f"/api/sessions/{s.id}/blobs/c3")
    assert r.status_code == 404
