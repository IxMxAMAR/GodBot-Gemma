from pathlib import Path
import json
from godbot.core.session import Session


def test_create_with_workspace(tmp_path):
    ws_root = tmp_path / "myws"
    ws_root.mkdir()
    s = Session.create(root=tmp_path / "sessions", model="m",
                       workspace_root=str(ws_root), auto_approve_in_sandbox=True)
    meta = json.loads((s.dir / "meta.json").read_text())
    assert meta["workspace_root"] == str(ws_root.resolve())
    assert meta["auto_approve_in_sandbox"] is True


def test_workspace_property_builds_workspace(tmp_path):
    ws_root = tmp_path / "myws"
    ws_root.mkdir()
    s = Session.create(root=tmp_path / "sessions", model="m",
                       workspace_root=str(ws_root), auto_approve_in_sandbox=True)
    ws = s.workspace
    assert ws is not None
    assert ws.root == ws_root.resolve()
    assert ws.auto_approve_in_sandbox is True


def test_workspace_property_is_none_when_unset(tmp_path):
    s = Session.create(root=tmp_path / "sessions", model="m")
    assert s.workspace is None


def test_legacy_meta_without_workspace_loads_clean(tmp_path):
    sdir = tmp_path / "sessions" / "old_session"
    sdir.mkdir(parents=True)
    (sdir / "blobs").mkdir()
    (sdir / "events.jsonl").touch()
    (sdir / "meta.json").write_text(json.dumps({
        "id": "old_session", "started_at": "x", "ended_at": None,
        "model": "m", "tool_overrides": None, "auto_approved_tools": [],
        "yolo": False, "rag_collection": None,
    }))
    s = Session.load(root=tmp_path / "sessions", session_id="old_session")
    assert s.workspace is None


def test_set_workspace_persists(tmp_path):
    ws_root = tmp_path / "myws"
    ws_root.mkdir()
    s = Session.create(root=tmp_path / "sessions", model="m")
    s.set_workspace(str(ws_root), auto_approve=True)
    loaded = Session.load(root=tmp_path / "sessions", session_id=s.id)
    assert loaded.workspace is not None
    assert loaded.workspace.auto_approve_in_sandbox is True
