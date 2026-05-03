from fastapi.testclient import TestClient
from godbot.interfaces.web import build_app


def test_root_serves_html(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    assert "html" in r.headers.get("content-type", "").lower()
    assert "GodBot" in r.text
