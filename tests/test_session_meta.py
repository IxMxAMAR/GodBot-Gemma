import json
from godbot.core.session import Session


def test_create_new_session(tmp_path):
    s = Session.create(root=tmp_path, model="gemma-3n-e4b-it")
    assert s.dir.exists()
    assert (s.dir / "meta.json").exists()
    meta = json.loads((s.dir / "meta.json").read_text())
    assert meta["model"] == "gemma-3n-e4b-it"
    assert meta["auto_approved_tools"] == []
    assert meta["yolo"] is False
    assert meta["tool_overrides"] is None
    assert "id" in meta and "started_at" in meta


def test_load_existing_session(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    sid = s.id
    loaded = Session.load(root=tmp_path, session_id=sid)
    assert loaded.id == sid
    assert loaded.model == "m"


def test_set_yolo_persists(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.set_yolo(True)
    loaded = Session.load(root=tmp_path, session_id=s.id)
    assert loaded.yolo is True


def test_mark_auto_approved(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.mark_auto_approved("write_file")
    assert s.is_auto_approved("write_file")
    loaded = Session.load(root=tmp_path, session_id=s.id)
    assert loaded.is_auto_approved("write_file")
