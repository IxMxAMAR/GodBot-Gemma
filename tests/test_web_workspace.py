from fastapi.testclient import TestClient
import godbot.tools
from godbot.interfaces.web import build_app


def test_new_session_with_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    ws = tmp_path / "ws"
    ws.mkdir()
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.post("/api/sessions/new", json={
        "workspace": str(ws),
        "auto_approve_in_sandbox": True,
    })
    sid = r.json()["session_id"]
    s = c.get(f"/api/sessions/{sid}").json()
    assert s.get("workspace_root") is not None
    assert s.get("auto_approve_in_sandbox") is True
