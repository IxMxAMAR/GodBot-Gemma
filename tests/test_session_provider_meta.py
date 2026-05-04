"""Session provider/model_name/protocol metadata round-trip tests."""
from __future__ import annotations

import json
from pathlib import Path

from godbot.core.providers import NATIVE_TOOLS, REACT_JSON
from godbot.core.session import Session


def test_create_default_provider(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    assert s.provider == "lmstudio"
    assert s.model_name == "m"  # falls back to legacy ``model``
    assert s.protocol is None


def test_create_with_provider(tmp_path):
    s = Session.create(
        root=tmp_path, model="auto",
        provider="anthropic",
        model_name="claude-3-5-sonnet-latest",
        protocol=NATIVE_TOOLS,
    )
    assert s.provider == "anthropic"
    assert s.model_name == "claude-3-5-sonnet-latest"
    assert s.protocol == NATIVE_TOOLS


def test_provider_meta_roundtrips_through_disk(tmp_path):
    s = Session.create(
        root=tmp_path, model="auto",
        provider="openai", model_name="gpt-4o", protocol=NATIVE_TOOLS,
    )
    s2 = Session.load(tmp_path, s.id)
    assert s2.provider == "openai"
    assert s2.model_name == "gpt-4o"
    assert s2.protocol == NATIVE_TOOLS


def test_legacy_meta_without_provider_defaults_to_lmstudio(tmp_path):
    """A meta.json from before sub-project 7 lacks provider/model_name/
    protocol — load() must accept it and surface lmstudio defaults."""
    sid = "legacy_session"
    sdir = tmp_path / sid
    sdir.mkdir()
    (sdir / "blobs").mkdir()
    legacy_meta = {
        "id": sid,
        "started_at": "2024-01-01T00:00:00",
        "ended_at": None,
        "model": "gemma-3",
        "tool_overrides": None,
        "auto_approved_tools": [],
        "yolo": False,
        "rag_collection": None,
        "workspace_root": None,
        "auto_approve_in_sandbox": False,
    }
    (sdir / "meta.json").write_text(json.dumps(legacy_meta), encoding="utf-8")
    (sdir / "events.jsonl").touch()

    s = Session.load(tmp_path, sid)
    assert s.provider == "lmstudio"
    assert s.model_name == "gemma-3"
    assert s.protocol is None


def test_set_provider_persists(tmp_path):
    s = Session.create(root=tmp_path, model="auto")
    s.set_provider("groq", model_name="llama-3.3-70b-versatile", protocol=REACT_JSON)
    s2 = Session.load(tmp_path, s.id)
    assert s2.provider == "groq"
    assert s2.model_name == "llama-3.3-70b-versatile"
    assert s2.protocol == REACT_JSON


def test_set_provider_partial_update(tmp_path):
    s = Session.create(
        root=tmp_path, model="auto",
        provider="openai", model_name="gpt-4o", protocol=NATIVE_TOOLS,
    )
    # Only change provider; keep model/protocol.
    s.set_provider("anthropic")
    s2 = Session.load(tmp_path, s.id)
    assert s2.provider == "anthropic"
    assert s2.model_name == "gpt-4o"
    assert s2.protocol == NATIVE_TOOLS
