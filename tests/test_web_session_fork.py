"""Tests for sub-project 38 — POST /api/sessions/{sid}/fork."""
from __future__ import annotations

from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.session import BLOB_INLINE_LIMIT, Session
from godbot.interfaces.web import build_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    app = build_app(sessions_root=sessions_root)
    return TestClient(app), sessions_root


def test_fork_404_unknown_source(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/sessions/no-such/fork", json={})
    assert r.status_code == 404


def test_fork_full_history_default(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    src = Session.create(root=sroot, model="m", provider="openai", model_name="gpt-4o-mini")
    src.append_user("first")
    src.append_assistant_final("reply 1")
    src.append_user("second")
    src.append_assistant_final("reply 2")
    r = c.post(f"/api/sessions/{src.id}/fork", json={})
    assert r.status_code == 200
    body = r.json()
    new_sid = body["session_id"]
    assert body["forked_from"] == src.id
    assert body["events_copied"] == 4

    new = Session.load(sroot, new_sid)
    msgs = new.messages_for_llm()
    # 2 user + 2 assistant turns carried over.
    assert len(msgs) == 4
    # Provider + model inherited.
    assert new.provider == "openai"
    assert new.model_name == "gpt-4o-mini"


def test_fork_up_to_index_truncates(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    src = Session.create(root=sroot, model="m")
    src.append_user("a")
    src.append_assistant_final("b")
    src.append_user("c")
    src.append_assistant_final("d")
    # Take the first 2 events only (a + b).
    r = c.post(f"/api/sessions/{src.id}/fork", json={"up_to_index": 2})
    body = r.json()
    assert body["events_copied"] == 2

    new = Session.load(sroot, body["session_id"])
    msgs = new.messages_for_llm()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "a"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "b"


def test_fork_up_to_zero_starts_empty(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    src = Session.create(root=sroot, model="m")
    src.append_user("never carried")
    r = c.post(f"/api/sessions/{src.id}/fork", json={"up_to_index": 0})
    body = r.json()
    assert body["events_copied"] == 0
    new = Session.load(sroot, body["session_id"])
    assert new.messages_for_llm() == []


def test_fork_400_negative_index(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    src = Session.create(root=sroot, model="m")
    r = c.post(f"/api/sessions/{src.id}/fork", json={"up_to_index": -1})
    assert r.status_code == 400


def test_fork_400_index_past_end(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    src = Session.create(root=sroot, model="m")
    src.append_user("only event")
    r = c.post(f"/api/sessions/{src.id}/fork", json={"up_to_index": 99})
    assert r.status_code == 400


def test_fork_does_not_carry_usage_or_budget(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    src = Session.create(root=sroot, model="m")
    src.append_user("hi")
    src.add_usage({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    src.set_budget(max_total_tokens=999)
    r = c.post(f"/api/sessions/{src.id}/fork", json={})
    new = Session.load(sroot, r.json()["session_id"])
    # Fresh start: no usage, no budget caps.
    assert new.usage["turns"] == 0
    assert new.budget["max_total_tokens"] is None


def test_fork_copies_referenced_blobs(tmp_path, monkeypatch):
    c, sroot = _client(tmp_path, monkeypatch)
    src = Session.create(root=sroot, model="m")
    src.append_assistant_tool_call("c1", "read_file", {"path": "x"}, raw="{}")
    big = "Z" * (BLOB_INLINE_LIMIT + 100)
    src.record_tool_result("c1", big)
    # The blob exists for src.
    src_blob = src.dir / "blobs" / "c1.txt"
    assert src_blob.exists()

    r = c.post(f"/api/sessions/{src.id}/fork", json={})
    new = Session.load(sroot, r.json()["session_id"])
    new_blob = new.dir / "blobs" / "c1.txt"
    assert new_blob.exists()
    assert new_blob.read_text(encoding="utf-8") == big
