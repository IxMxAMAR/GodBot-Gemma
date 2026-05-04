"""Tests for sub-project 59 — mkdir tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.fs import mkdir


def test_mkdir_creates_directory(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = mkdir(path="newdir")
        assert out.startswith("ok: created")
        assert (tmp_path / "newdir").is_dir()
    finally:
        _ws_current.reset(token)


def test_mkdir_creates_parents(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = mkdir(path="deep/nested/dir")
        assert out.startswith("ok:")
        assert (tmp_path / "deep" / "nested" / "dir").is_dir()
    finally:
        _ws_current.reset(token)


def test_mkdir_idempotent_when_exists(tmp_path):
    (tmp_path / "existing").mkdir()
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = mkdir(path="existing")
        assert out.startswith("ok: exists")
    finally:
        _ws_current.reset(token)


def test_mkdir_rejects_when_path_is_file(tmp_path):
    f = tmp_path / "blocker.txt"
    f.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = mkdir(path="blocker.txt")
        assert out.startswith("[error]")
        assert "not a directory" in out
    finally:
        _ws_current.reset(token)


def test_mkdir_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = mkdir(path=str(tmp_path / "outside_dir"))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_mkdir_parents_false_fails_when_intermediate_missing(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = mkdir(path="not_yet/here", parents=False)
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)
