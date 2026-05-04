"""Tests for sub-project 46 — compare_files tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import compare_files


def test_compare_files_identical(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello\nworld\n", encoding="utf-8")
    b.write_text("hello\nworld\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = compare_files(path_a="a.txt", path_b="b.txt")
        assert "identical" in out
    finally:
        _ws_current.reset(token)


def test_compare_files_unified_diff(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("apple\nbanana\ncherry\n", encoding="utf-8")
    b.write_text("apple\nBANANA\ncherry\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = compare_files(path_a="a.txt", path_b="b.txt")
        assert "---" in out
        assert "+++" in out
        assert "-banana" in out
        assert "+BANANA" in out
    finally:
        _ws_current.reset(token)


def test_compare_files_missing_arg_errors():
    out = compare_files(path_a="a.txt", path_b="")
    assert out.startswith("[error]")


def test_compare_files_missing_file_errors(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = compare_files(path_a="a.txt", path_b="ghost.txt")
        assert out.startswith("[error]")
        assert "not a file" in out
    finally:
        _ws_current.reset(token)


def test_compare_files_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    a = ws_dir / "a.txt"
    a.write_text("x", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("y", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = compare_files(path_a="a.txt", path_b=str(outside))
        assert out.startswith("[error]")
        assert "outside the active workspace" in out
    finally:
        _ws_current.reset(token)


def test_compare_files_truncates_huge_diff(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("\n".join(f"line {i}" for i in range(200)) + "\n", encoding="utf-8")
    b.write_text("\n".join(f"DIFF {i}" for i in range(200)) + "\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = compare_files(path_a="a.txt", path_b="b.txt", max_lines=20)
        assert "truncated at 20 lines" in out
    finally:
        _ws_current.reset(token)


def test_compare_files_supports_absolute_path_inside_workspace(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("v1\n", encoding="utf-8")
    b.write_text("v2\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = compare_files(path_a=str(a), path_b=str(b))
        assert "-v1" in out
        assert "+v2" in out
    finally:
        _ws_current.reset(token)
