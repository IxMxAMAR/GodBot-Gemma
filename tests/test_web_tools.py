from fastapi.testclient import TestClient
import godbot.tools
from godbot.interfaces.web import build_app


def test_list_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/tools")
    body = r.json()
    names = [t["name"] for t in body]
    assert "read_file" in names


def test_toggle_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    sid = c.post("/api/sessions/new").json()["session_id"]
    r = c.post("/api/tools/toggle", json={"session_id": sid, "name": "run_powershell", "enabled": False})
    assert r.status_code == 200
    s = c.get(f"/api/sessions/{sid}").json()
    # tool_overrides updated
    assert "run_powershell" not in (s.get("tool_overrides") or [])
