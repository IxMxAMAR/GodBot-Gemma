"""Tests for sub-project 57 — delete_file, copy_file, move_file."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.fs import copy_file, delete_file, move_file


# --- delete_file ----


def test_delete_file_basic(tmp_path):
    f = tmp_path / "victim.txt"
    f.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = delete_file(path="victim.txt")
        assert out.startswith("ok: deleted")
        assert not f.exists()
    finally:
        _ws_current.reset(token)


def test_delete_file_missing(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = delete_file(path="ghost.txt")
        assert out.startswith("[error]")
        assert "not found" in out
    finally:
        _ws_current.reset(token)


def test_delete_file_refuses_directory(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = delete_file(path="subdir")
        assert out.startswith("[error]")
        assert "directory" in out
    finally:
        _ws_current.reset(token)


def test_delete_file_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = delete_file(path=str(outside))
        assert out.startswith("[error]")
        assert outside.exists()  # not deleted
    finally:
        _ws_current.reset(token)


# --- copy_file ----


def test_copy_file_basic(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = copy_file(src="src.txt", dst="dst.txt")
        assert out.startswith("ok:")
        assert (tmp_path / "dst.txt").read_text(encoding="utf-8") == "hello"
        assert (tmp_path / "src.txt").exists()  # source preserved
    finally:
        _ws_current.reset(token)


def test_copy_file_creates_parent_dirs(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = copy_file(src="src.txt", dst="nested/deep/copy.txt")
        assert out.startswith("ok:")
        assert (tmp_path / "nested" / "deep" / "copy.txt").exists()
    finally:
        _ws_current.reset(token)


def test_copy_file_refuses_overwrite(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("new", encoding="utf-8")
    dst = tmp_path / "dst.txt"
    dst.write_text("existing", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = copy_file(src="src.txt", dst="dst.txt")
        assert out.startswith("[error]")
        assert "destination exists" in out
        assert dst.read_text(encoding="utf-8") == "existing"  # untouched
    finally:
        _ws_current.reset(token)


def test_copy_file_missing_source(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = copy_file(src="ghost.txt", dst="dst.txt")
        assert out.startswith("[error]")
        assert "source not found" in out
    finally:
        _ws_current.reset(token)


# --- move_file ----


def test_move_file_basic(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = move_file(src="src.txt", dst="renamed.txt")
        assert out.startswith("ok: moved")
        assert not src.exists()
        assert (tmp_path / "renamed.txt").read_text(encoding="utf-8") == "hello"
    finally:
        _ws_current.reset(token)


def test_move_file_into_subdir(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = move_file(src="src.txt", dst="sub/dest.txt")
        assert out.startswith("ok:")
        assert not src.exists()
        assert (tmp_path / "sub" / "dest.txt").exists()
    finally:
        _ws_current.reset(token)


def test_move_file_refuses_overwrite(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("new", encoding="utf-8")
    dst = tmp_path / "dst.txt"
    dst.write_text("existing", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = move_file(src="src.txt", dst="dst.txt")
        assert out.startswith("[error]")
        # Both files still present, untouched.
        assert src.exists()
        assert dst.read_text(encoding="utf-8") == "existing"
    finally:
        _ws_current.reset(token)


def test_move_file_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    src = ws_dir / "src.txt"
    src.write_text("x", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = move_file(src="src.txt", dst=str(outside))
        assert out.startswith("[error]")
        assert src.exists()  # source preserved
        assert not outside.exists()  # not created
    finally:
        _ws_current.reset(token)
