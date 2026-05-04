"""Tests for sub-project 80 — /api/sessions/{sid}/transcript."""
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


def test_transcript_404_unknown(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/transcript")
    assert r.status_code == 404


def test_transcript_basic(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("hello")
    s.append_assistant_final("hi back")
    r = c.get(f"/api/sessions/{s.id}/transcript")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert "User: hello" in text
    assert "Assistant: hi back" in text


def test_transcript_skips_tools_by_default(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("read file")
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.append_tool_result("c1", "content")
    s.append_assistant_final("done")
    text = c.get(f"/api/sessions/{s.id}/transcript").text
    assert "Tool call:" not in text
    assert "Tool result:" not in text
    assert "User: read file" in text
    assert "Assistant: done" in text


def test_transcript_include_tools_true(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.append_tool_result("c1", "the file content")
    text = c.get(f"/api/sessions/{s.id}/transcript?include_tools=true").text
    assert "Tool call: read_file" in text
    assert "Tool result:" in text
    assert "the file content" in text


def test_transcript_empty_session(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    text = c.get(f"/api/sessions/{s.id}/transcript").text
    assert text == "\n"  # rstrip + trailing newline


def test_transcript_truncates_huge_tool_result(tmp_path, monkeypatch):
    """Tool results are capped at 500 chars when included."""
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "x", {}, raw="{}")
    s.append_tool_result("c1", "Z" * 1000)
    text = c.get(f"/api/sessions/{s.id}/transcript?include_tools=true").text
    # Tool result line max 500 chars after the "Tool result: " prefix.
    tr_line = next(ln for ln in text.splitlines() if ln.startswith("Tool result:"))
    payload = tr_line[len("Tool result: "):]
    assert len(payload) <= 500
