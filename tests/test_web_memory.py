"""Tests for sub-project 10.3 — memory web endpoints.

Covers GET /api/memory, POST /api/memory/pin, POST /api/memory/delete, and
POST /api/memory/auto_summarize. Uses FastAPI's TestClient with an isolated
GODBOT_HOME so notes don't leak between tests.
"""
from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401 — tool auto-discovery
from godbot.interfaces.web import build_app


def _new_app(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


def _save_note(client: TestClient, content: str, workspace: str, tags: list[str] | None = None):
    """Create a workspace-scoped note via the agent's save_note tool path.

    We bypass the daemon's chat loop because we only need notes in the store;
    use the underlying tool directly (which is the same code path).
    """
    from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
    from godbot.tools.memory import save_note

    ws = Workspace.of(workspace)
    token = set_workspace(ws)
    try:
        save_note(content, tags=tags or [])
    finally:
        _ws_current.reset(token)


def test_memory_list_returns_workspace_scoped_notes(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    ws_a = tmp_path / "ws_a"
    ws_a.mkdir()
    ws_b = tmp_path / "ws_b"
    ws_b.mkdir()
    _save_note(client, "alpha note", str(ws_a))
    _save_note(client, "beta note", str(ws_b))

    r = client.get(f"/api/memory?workspace={ws_a}")
    assert r.status_code == 200
    notes = r.json()["notes"]
    assert len(notes) == 1
    assert notes[0]["content"] == "alpha note"


def test_memory_list_substring_filter(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    ws = tmp_path / "ws"
    ws.mkdir()
    _save_note(client, "kubernetes deployment", str(ws))
    _save_note(client, "tooling polish", str(ws))

    r = client.get(f"/api/memory?workspace={ws}&q=kuber")
    assert r.status_code == 200
    notes = r.json()["notes"]
    assert len(notes) == 1
    assert "kubernetes" in notes[0]["content"]


def test_memory_list_unscoped_returns_everything(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    ws_a = tmp_path / "ws_a"
    ws_a.mkdir()
    ws_b = tmp_path / "ws_b"
    ws_b.mkdir()
    _save_note(client, "alpha note", str(ws_a))
    _save_note(client, "beta note", str(ws_b))

    r = client.get("/api/memory")
    assert r.status_code == 200
    notes = r.json()["notes"]
    assert len(notes) == 2


def test_pin_endpoint_adds_pin_tag(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    ws = tmp_path / "ws"
    ws.mkdir()
    _save_note(client, "important fact", str(ws))

    notes = client.get(f"/api/memory?workspace={ws}").json()["notes"]
    ts = notes[0]["timestamp"]

    r = client.post("/api/memory/pin", json={"timestamp": ts, "pin": True})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    notes = client.get(f"/api/memory?workspace={ws}").json()["notes"]
    assert "pinned" in (notes[0]["tags"] or [])


def test_pin_endpoint_unpin(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    ws = tmp_path / "ws"
    ws.mkdir()
    _save_note(client, "fact", str(ws), tags=["pinned"])

    notes = client.get(f"/api/memory?workspace={ws}").json()["notes"]
    ts = notes[0]["timestamp"]
    assert "pinned" in (notes[0]["tags"] or [])

    client.post("/api/memory/pin", json={"timestamp": ts, "pin": False})
    notes = client.get(f"/api/memory?workspace={ws}").json()["notes"]
    assert "pinned" not in (notes[0]["tags"] or [])


def test_pin_missing_timestamp_400(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    r = client.post("/api/memory/pin", json={"pin": True})
    assert r.status_code == 400


def test_pin_unknown_timestamp_404(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    r = client.post("/api/memory/pin", json={"timestamp": "9999-01-01_00-00-00", "pin": True})
    assert r.status_code == 404


def test_delete_endpoint_removes_note(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    ws = tmp_path / "ws"
    ws.mkdir()
    _save_note(client, "to be deleted", str(ws))

    notes = client.get(f"/api/memory?workspace={ws}").json()["notes"]
    ts = notes[0]["timestamp"]

    r = client.post("/api/memory/delete", json={"timestamp": ts})
    assert r.status_code == 200

    notes = client.get(f"/api/memory?workspace={ws}").json()["notes"]
    assert notes == []


def test_delete_unknown_timestamp_404(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    r = client.post("/api/memory/delete", json={"timestamp": "9999-01-01_00-00-00"})
    assert r.status_code == 404


def test_auto_summarize_creates_project_summary_note(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    ws = tmp_path / "ws_proj"
    ws.mkdir()
    (ws / "pyproject.toml").write_text(
        '[project]\nname = "demo"\ndependencies = ["fastapi"]\n', encoding="utf-8"
    )
    (ws / "README.md").write_text("# Demo\nA tiny test project.", encoding="utf-8")

    r = client.post("/api/memory/auto_summarize", json={"workspace": str(ws)})
    assert r.status_code == 200
    body = r.json()
    assert body["cached"] is False
    assert "demo" in body["summary"].lower()
    assert "fastapi" in body["summary"].lower()

    notes = client.get(f"/api/memory?workspace={ws}").json()["notes"]
    assert any("project_summary" in (n.get("tags") or []) for n in notes)


def test_auto_summarize_returns_cached_on_second_call(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    ws = tmp_path / "ws_proj"
    ws.mkdir()
    (ws / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")

    first = client.post("/api/memory/auto_summarize", json={"workspace": str(ws)}).json()
    second = client.post("/api/memory/auto_summarize", json={"workspace": str(ws)}).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert first["summary"] == second["summary"]


def test_auto_summarize_force_reruns(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    ws = tmp_path / "ws_proj"
    ws.mkdir()
    (ws / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")

    client.post("/api/memory/auto_summarize", json={"workspace": str(ws)})
    second = client.post(
        "/api/memory/auto_summarize", json={"workspace": str(ws), "force": True}
    ).json()
    assert second["cached"] is False


def test_auto_summarize_missing_workspace_400(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    r = client.post("/api/memory/auto_summarize", json={})
    assert r.status_code == 400


def test_auto_summarize_invalid_workspace_400(tmp_path, monkeypatch):
    client = _new_app(tmp_path, monkeypatch)
    r = client.post("/api/memory/auto_summarize", json={"workspace": str(tmp_path / "no-such")})
    assert r.status_code == 400
