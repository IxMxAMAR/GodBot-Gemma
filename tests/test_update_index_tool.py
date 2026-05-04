"""Tests for sub-project 19 — update_workspace_index tool."""
from __future__ import annotations

import pytest

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.knowledge import update_workspace_index, workspace_collection_name


def test_no_workspace_message(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    out = update_workspace_index(path="anything.py")
    assert "no active workspace" in out.lower()


def test_outside_workspace_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    ws = tmp_path / "ws"; ws.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")

    workspace = Workspace.of(str(ws))
    token = set_workspace(workspace)
    try:
        out = update_workspace_index(path=str(outside))
        assert out.startswith("[error]")
        assert "outside the active workspace" in out
    finally:
        _ws_current.reset(token)


def test_unindexed_workspace_hints_user(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    ws = tmp_path / "ws"; ws.mkdir()
    f = ws / "main.py"
    f.write_text("def add(a,b): return a+b\n", encoding="utf-8")
    workspace = Workspace.of(str(ws))
    token = set_workspace(workspace)
    try:
        out = update_workspace_index(path="main.py")
        assert "not indexed" in out
        assert "/api/rag/index_workspace" in out
    finally:
        _ws_current.reset(token)


def test_nonexistent_file_errors(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"; ws.mkdir()
    # Seed the collection so the unindexed-hint doesn't fire first.
    (ws / "seed.py").write_text("y = 2\n", encoding="utf-8")
    from godbot.index import build_index
    coll = workspace_collection_name(str(ws))
    build_index(ws, collection=coll)
    workspace = Workspace.of(str(ws))
    token = set_workspace(workspace)
    try:
        out = update_workspace_index(path="ghost.py")
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_update_file_round_trip(tmp_path, monkeypatch):
    """End-to-end: build, edit, update, search returns updated content."""
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"; ws.mkdir()
    f = ws / "auth.py"
    f.write_text("def login(): return 'old-flow'\n", encoding="utf-8")

    from godbot.index import build_index
    from godbot.tools.knowledge import search_workspace
    coll = workspace_collection_name(str(ws))
    build_index(ws, collection=coll)

    workspace = Workspace.of(str(ws))
    token = set_workspace(workspace)
    try:
        # Edit the file: replace "old-flow" with "newflowmarker".
        f.write_text("def login(): return 'newflowmarker'\n", encoding="utf-8")
        out = update_workspace_index(path="auth.py")
        assert out.startswith("ok:")
        assert "auth.py" in out
        # Search now finds the updated content.
        hits = search_workspace(query="newflowmarker", top_k=3)
        assert "newflowmarker" in hits
    finally:
        _ws_current.reset(token)
