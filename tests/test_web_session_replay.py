"""Tests for sub-project 58 — /api/sessions/{sid}/replay."""
from __future__ import annotations

import time

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


def test_replay_404_unknown_session(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.get("/api/sessions/no-such/replay")
    assert r.status_code == 404


def test_replay_emits_each_event(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("hi")
    s.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    s.append_tool_result("c1", "content")
    s.append_assistant_final("done")

    events = []
    with c.stream("GET", f"/api/sessions/{s.id}/replay", timeout=5.0) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if "replay_done" in events:
                break

    assert "user" in events
    assert "assistant_tool_call" in events
    assert "tool_result" in events
    assert "assistant_final" in events
    assert "replay_done" in events


def test_replay_empty_session_just_emits_done(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    events = []
    with c.stream("GET", f"/api/sessions/{s.id}/replay", timeout=5.0) as resp:
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if "replay_done" in events:
                break
    assert events == ["replay_done"]


def test_replay_delay_throttles(tmp_path, monkeypatch):
    """A non-zero delay_ms should make the stream take roughly N * delay."""
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    for i in range(5):
        s.append_user(f"msg {i}")
    started = time.monotonic()
    with c.stream("GET", f"/api/sessions/{s.id}/replay?delay_ms=100", timeout=10.0) as resp:
        for line in resp.iter_lines():
            if "replay_done" in line:
                break
    elapsed = time.monotonic() - started
    # 5 events × 100ms = ~500ms minimum (allow slack for dispatch overhead).
    assert elapsed >= 0.4
    # And not absurdly long.
    assert elapsed < 5.0


def test_replay_data_payload_is_full_event(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    s = Session.create(root=sroot, model="m")
    s.append_user("hello world")

    import json as _json
    payloads = []
    with c.stream("GET", f"/api/sessions/{s.id}/replay", timeout=5.0) as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                payloads.append(line.split(":", 1)[1].strip())
    # First payload is the user event with content.
    parsed = _json.loads(payloads[0])
    assert parsed.get("type") == "user"
    assert parsed.get("content") == "hello world"
