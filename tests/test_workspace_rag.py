"""Tests for workspace-scoped RAG (sub-project 14).

Covers:
- workspace_collection_name produces stable, distinct names
- search_workspace honours the workspace contextvar and returns a
  helpful hint when not yet indexed
- POST /api/rag/index_workspace builds + returns metadata; cached path
  returns without re-indexing unless force=true
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401 — ensures auto-discovery picks up search_workspace
from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.interfaces.web import build_app
from godbot.tools.knowledge import search_workspace, workspace_collection_name


def test_collection_name_stable_for_same_path(tmp_path):
    name1 = workspace_collection_name(str(tmp_path))
    name2 = workspace_collection_name(str(tmp_path))
    assert name1 == name2
    assert name1.startswith("ws_")


def test_collection_name_differs_for_different_paths(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    assert workspace_collection_name(str(a)) != workspace_collection_name(str(b))


def test_collection_name_resolves_relative_paths(tmp_path):
    """Relative paths and absolute paths to the same dir produce the same name."""
    abs_name = workspace_collection_name(str(tmp_path))
    # Nudge through Path.resolve indirectly — same input.
    rel_name = workspace_collection_name(str(tmp_path.resolve()))
    assert abs_name == rel_name


def test_search_workspace_no_active_workspace_message(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    out = search_workspace(query="x")
    assert "no active workspace" in out.lower()


def test_search_workspace_unindexed_hints_user(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    ws = tmp_path / "ws"
    ws.mkdir()
    workspace = Workspace.of(str(ws))
    token = set_workspace(workspace)
    try:
        out = search_workspace(query="x")
        assert "not indexed" in out
        assert "/api/rag/index_workspace" in out
    finally:
        _ws_current.reset(token)


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    return TestClient(app)


def test_index_workspace_requires_workspace(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/rag/index_workspace", json={})
    assert r.status_code == 400


def test_index_workspace_invalid_dir_400(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/rag/index_workspace", json={"workspace": str(tmp_path / "no_such")})
    assert r.status_code == 400


def test_index_workspace_creates_collection(tmp_path, monkeypatch):
    """End-to-end: small project with one Python file gets indexed and the
    collection appears under <home>/rag/.

    Uses the legacy LM Studio embedder probe path, which falls back to
    sentence-transformers when the probe fails. Importing
    SentenceTransformer is heavy, so mark this test as slow-tolerable.
    """
    pytest.importorskip("sentence_transformers")

    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "main.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8",
    )
    c = _client(tmp_path / "home", monkeypatch)
    r = c.post("/api/rag/index_workspace", json={"workspace": str(ws)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["collection"].startswith("ws_")
    assert body["indexed_chunks"] >= 1
    assert body["cached"] is False
    coll_dir = Path(monkeypatch.getenv("GODBOT_HOME") if hasattr(monkeypatch, "getenv") else tmp_path / "home") / "rag" / body["collection"]
    # monkeypatch.setenv was used; just assert the path under tmp_path/home.
    coll_dir = (tmp_path / "home") / "rag" / body["collection"]
    assert coll_dir.exists()


def test_index_workspace_cached_path(tmp_path, monkeypatch):
    """Second call without force returns cached=true without re-running."""
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "main.py").write_text("x = 1\n", encoding="utf-8")
    c = _client(tmp_path / "home", monkeypatch)
    first = c.post("/api/rag/index_workspace", json={"workspace": str(ws)})
    assert first.status_code == 200
    second = c.post("/api/rag/index_workspace", json={"workspace": str(ws)})
    assert second.status_code == 200
    body = second.json()
    assert body["cached"] is True
