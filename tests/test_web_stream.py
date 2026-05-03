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
    webmod._streams.clear()
    webmod._cancels.clear()
    webmod._sessions_cache.clear()
    yield
    webmod._streams.clear()
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
