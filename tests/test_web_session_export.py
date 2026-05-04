"""Tests for sub-project 31 — session export endpoint."""
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


def test_export_404_unknown_session(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/export")
    assert r.status_code == 404


def test_export_json_includes_meta_events_usage(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m", provider="openai", model_name="gpt-4o-mini")
    s.append_user("hi")
    s.append_assistant_final("hello")
    s.add_usage({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})

    r = c.get(f"/api/sessions/{s.id}/export")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == s.id
    assert body["provider"] == "openai"
    assert body["model_name"] == "gpt-4o-mini"
    assert body["usage"]["turns"] == 1
    # Events array contains both user + assistant entries.
    types = [ev.get("type") for ev in body["events"]]
    assert "user" in types
    assert "assistant_final" in types
    # Cost block is populated from pricing table.
    assert body["cost"]["matched"] is True
    assert body["cost"]["usd"] > 0


def test_export_json_includes_budget(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.set_budget(max_total_tokens=5000)
    r = c.get(f"/api/sessions/{s.id}/export")
    body = r.json()
    assert body["budget"]["max_total_tokens"] == 5000


def test_export_markdown_format(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m", provider="lmstudio", model_name="gemma")
    s.append_user("read main.py")
    s.append_assistant_tool_call("c1", "read_file", {"path": "main.py"}, raw="{}")
    s.append_tool_result("c1", "def x(): pass\n")
    s.append_assistant_final("It defines x()")
    s.add_usage({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    r = c.get(f"/api/sessions/{s.id}/export?format=markdown")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    text = r.text
    assert f"# Session `{s.id}`" in text
    assert "## User" in text
    assert "## Assistant" in text
    assert "### Tool call: `read_file`" in text
    assert "It defines x()" in text
    # Local provider so no cost line.
    assert "**Usage:**" in text


def test_export_markdown_truncates_huge_tool_result(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("dump file")
    s.append_assistant_tool_call("c1", "read_file", {"path": "huge"}, raw="{}")
    big_content = "X" * 5000
    s.append_tool_result("c1", big_content, blob="c1")
    r = c.get(f"/api/sessions/{s.id}/export?format=markdown")
    text = r.text
    assert "truncated" in text
    # Truncation happens in markdown rendering; raw events still hold full content.
    full = c.get(f"/api/sessions/{s.id}/export").json()
    full_evs = [ev for ev in full["events"] if ev.get("type") == "tool_result"]
    assert len(full_evs[0]["content"]) == 5000
