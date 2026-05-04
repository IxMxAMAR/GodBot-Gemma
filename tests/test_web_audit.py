"""Tests for sub-project 40 — /api/audit dangerous-call log."""
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


def test_audit_empty_state(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/audit")
    assert r.status_code == 200
    assert r.json()["calls"] == []


def test_audit_includes_dangerous_call(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "write_file", {"path": "x", "content": "y"}, raw="{}")
    s.append_tool_result("c1", "ok: wrote 1 char to x")

    r = c.get("/api/audit")
    body = r.json()
    assert len(body["calls"]) == 1
    entry = body["calls"][0]
    assert entry["name"] == "write_file"
    assert entry["call_id"] == "c1"
    assert "ok: wrote" in entry["result_preview"]
    assert entry["gated"] is True


def test_audit_skips_safe_tools(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.append_tool_result("c1", "content")
    r = c.get("/api/audit")
    assert r.json()["calls"] == []


def test_audit_pairs_call_and_result(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "write_file", {"path": "x", "content": "y"}, raw="{}")
    s.append_tool_result("c1", "RESULT_X")
    s.append_assistant_tool_call("c2", "edit_file", {"path": "y", "old": "a", "new": "b"}, raw="{}")
    s.append_tool_result("c2", "RESULT_Y")
    r = c.get("/api/audit")
    body = r.json()
    by_id = {entry["call_id"]: entry for entry in body["calls"]}
    assert "RESULT_X" in by_id["c1"]["result_preview"]
    assert "RESULT_Y" in by_id["c2"]["result_preview"]


def test_audit_handles_call_without_result(tmp_path, monkeypatch):
    """A tool_call without a matching tool_result (e.g. cancelled) still
    surfaces in the audit, with a placeholder preview."""
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "run_powershell", {"cmd": "echo hi"}, raw="{}")
    # No tool_result event.
    r = c.get("/api/audit")
    body = r.json()
    assert len(body["calls"]) == 1
    assert "no result" in body["calls"][0]["result_preview"]


def test_audit_respects_limit(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    for i in range(5):
        s = Session.create(root=sroot, model="m")
        s.append_assistant_tool_call(f"c{i}", "write_file", {"path": f"f{i}", "content": "x"}, raw="{}")
        s.append_tool_result(f"c{i}", "ok")
    r = c.get("/api/audit?limit=2")
    body = r.json()
    assert len(body["calls"]) == 2


def test_audit_truncates_long_result_preview(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_assistant_tool_call("c1", "write_file", {"path": "x", "content": "y"}, raw="{}")
    big = "Q" * 1000
    s.append_tool_result("c1", big)
    r = c.get("/api/audit")
    body = r.json()
    # Preview is capped at 300 chars.
    assert len(body["calls"][0]["result_preview"]) <= 300
