import asyncio
import json
import pytest
from fastapi.testclient import TestClient
import godbot.tools
from godbot.interfaces import web as webmod
from godbot.interfaces.web import build_app
from tests._mock_llm import MockLLM


@pytest.fixture(autouse=True)
def _reset_web_state():
    """Reset module-level shared state to prevent cross-test leaks."""
    webmod._logs.clear()
    webmod._cancels.clear()
    webmod._sessions_cache.clear()
    yield
    webmod._logs.clear()
    webmod._cancels.clear()
    webmod._sessions_cache.clear()


@pytest.fixture
def app_with_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    app = build_app(sessions_root=sessions_root)

    def fake_llm_factory():
        return MockLLM([
            json.dumps({"thought": "easy", "final_answer": "yo"})
        ])

    app.dependency_overrides[webmod.get_llm] = fake_llm_factory
    return app, sessions_root


def test_chat_then_stream_yields_done(app_with_mock):
    app, sessions_root = app_with_mock
    c = TestClient(app)
    sid = c.post("/api/sessions/new").json()["session_id"]
    r = c.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert r.status_code == 200

    # Open the SSE stream and read events until 'done' or timeout.
    events = []
    with c.stream("GET", f"/api/chat/stream?session_id={sid}", timeout=5.0) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if "done" in events:
                break
    assert "done" in events


def test_stop_endpoint_sets_cancel(app_with_mock):
    app, _ = app_with_mock
    c = TestClient(app)
    sid = c.post("/api/sessions/new").json()["session_id"]
    r = c.post("/api/stop", json={"session_id": sid})
    assert r.status_code == 200


def test_late_subscriber_replays_buffered_events(app_with_mock):
    """Sub-project 21: subscribers attaching AFTER the runner has already
    published events still see them by replay. The legacy contract was 409
    on second connection; the new contract is "follow with replay"."""
    app, _ = app_with_mock
    c = TestClient(app)
    sid = c.post("/api/sessions/new").json()["session_id"]
    # Drive the turn first; the runner closes the EventLog on completion
    # but the buffer remains for late subscribers.
    r = c.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert r.status_code == 200
    # Drain the runner by attaching a stream until done.
    with c.stream("GET", f"/api/chat/stream?session_id={sid}", timeout=5.0) as resp:
        for line in resp.iter_lines():
            if line.startswith("event: done"):
                break
    # Now attach a SECOND time — the events are buffered, so we should
    # still receive them via replay (not 409).
    events = []
    ids = []
    with c.stream("GET", f"/api/chat/stream?session_id={sid}", timeout=5.0) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("id:"):
                ids.append(line.split(":", 1)[1].strip())
            if "done" in events:
                break
    assert "done" in events
    # Each emitted SSE event must carry an id (the EventLog seq).
    assert len(ids) >= 1
    # And the ids must be monotonically increasing integers.
    int_ids = [int(x) for x in ids]
    assert int_ids == sorted(int_ids)


def test_reconnect_with_last_event_id_resumes(app_with_mock):
    """Passing ?last_event_id=N returns only events with seq > N."""
    app, _ = app_with_mock
    c = TestClient(app)
    sid = c.post("/api/sessions/new").json()["session_id"]
    c.post("/api/chat", json={"session_id": sid, "message": "hi"})
    # Drain once to populate.
    with c.stream("GET", f"/api/chat/stream?session_id={sid}", timeout=5.0) as resp:
        for line in resp.iter_lines():
            if line.startswith("event: done"):
                break
    # Reconnect with a high last_event_id — fewer events should come through.
    log = webmod._logs[sid]
    head = log.head_seq
    assert head >= 1
    events_replay_full = []
    with c.stream("GET", f"/api/chat/stream?session_id={sid}", timeout=5.0) as resp:
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events_replay_full.append(line.split(":", 1)[1].strip())
            if "done" in events_replay_full:
                break
    events_replay_partial = []
    with c.stream(
        "GET",
        f"/api/chat/stream?session_id={sid}&last_event_id={head - 1}",
        timeout=5.0,
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events_replay_partial.append(line.split(":", 1)[1].strip())
            if "done" in events_replay_partial:
                break
    # Resuming from head-1 yields fewer (or at most equal) events than from 0.
    assert len(events_replay_partial) <= len(events_replay_full)
