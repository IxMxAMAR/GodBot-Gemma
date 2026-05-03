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


def test_app_js_exists():
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "godbot" / "interfaces" / "static" / "app.js"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    # Spot-check the JS has the key entry points the plan calls for.
    for key in ("openStream", "appendToken", "appendToolCall", "appendGate", "send", "newSession", "init"):
        assert key in text, f"missing JS entry point: {key}"
