"""Tests for sub-project 61 — /api/sessions/{sid}/explain."""
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


def test_explain_404_unknown(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/explain")
    assert r.status_code == 404


def test_explain_basic(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("rewrite the README to mention the memory panel")
    s.append_assistant_final("done")
    r = c.get(f"/api/sessions/{s.id}/explain")
    body = r.json()
    assert body["label"].startswith("rewrite the README")
    assert body["messages"] == 2
    assert body["tool_calls"] == 0
    assert body["errors"] == 0


def test_explain_truncates_long_label(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    huge = "x" * 200
    s.append_user(huge)
    body = c.get(f"/api/sessions/{s.id}/explain").json()
    assert len(body["label"]) == 80


def test_explain_collapses_newlines(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("line one\nline two")
    body = c.get(f"/api/sessions/{s.id}/explain").json()
    assert "\n" not in body["label"]
    assert "line one line two" in body["label"]


def test_explain_counts_tool_calls(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("read x")
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.append_tool_result("c1", "ok")
    s.append_assistant_tool_call("c2", "git_status", {}, raw="{}")
    s.append_tool_result("c2", "clean")
    s.append_assistant_final("done")
    body = c.get(f"/api/sessions/{s.id}/explain").json()
    assert body["tool_calls"] == 2
    assert "read_file" in body["tools_used"]
    assert "git_status" in body["tools_used"]


def test_explain_no_user_message(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    body = c.get(f"/api/sessions/{s.id}/explain").json()
    assert body["label"] == "(no user message)"
    assert body["messages"] == 0


def test_explain_errors_counted(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("hi")
    s.append_synthetic_tool_result("schema validation failed")
    s.append_synthetic_tool_result("retry nudge")
    body = c.get(f"/api/sessions/{s.id}/explain").json()
    assert body["errors"] == 2


def test_explain_tools_used_top_10_only(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    # 12 distinct tool names; top_10 list should cap at 10.
    for i in range(12):
        s.append_assistant_tool_call(f"c{i}", f"tool_{i}", {}, raw="{}")
        s.append_tool_result(f"c{i}", "ok")
    body = c.get(f"/api/sessions/{s.id}/explain").json()
    assert len(body["tools_used"]) == 10
