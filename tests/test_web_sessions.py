from fastapi.testclient import TestClient
import godbot.tools  # discovery
from godbot.interfaces.web import build_app


def test_health_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_and_list_session(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.post("/api/sessions/new")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    r2 = c.get("/api/sessions")
    assert sid in [s["id"] for s in r2.json()]


def test_get_session_history(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    sid = c.post("/api/sessions/new").json()["session_id"]
    r = c.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == sid
    assert body["messages"] == []
