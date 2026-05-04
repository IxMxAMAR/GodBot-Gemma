"""Tests for sub-project 97 — read_lines_around tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import read_lines_around


def _seed(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n", encoding="utf-8")
    return f


def test_basic_context(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = read_lines_around(path="f.txt", line=5)
        # Default context=3 → lines 2..8.
        assert " 2\tline 2" in out
        assert ">5\tline 5" in out
        assert " 8\tline 8" in out
        # Lines 1 and 9 NOT present.
        assert " 1\tline 1" not in out
        assert " 9\tline 9" not in out
    finally:
        _ws_current.reset(token)


def test_context_zero(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = read_lines_around(path="f.txt", line=5, context=0)
        # Just one line.
        assert out.count("\n") == 0
        assert ">5\tline 5" in out
    finally:
        _ws_current.reset(token)


def test_target_marker(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = read_lines_around(path="f.txt", line=5)
        # Only the target line is marked with `>`.
        marker_count = sum(1 for ln in out.splitlines() if ln.startswith(">"))
        assert marker_count == 1
    finally:
        _ws_current.reset(token)


def test_line_at_start(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = read_lines_around(path="f.txt", line=1, context=3)
        # Should start at line 1 (no negative lines).
        assert ">1\tline 1" in out
    finally:
        _ws_current.reset(token)


def test_line_at_end(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = read_lines_around(path="f.txt", line=10, context=3)
        assert ">10\tline 10" in out
    finally:
        _ws_current.reset(token)


def test_invalid_line(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = read_lines_around(path="f.txt", line=0)
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_line_past_end(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = read_lines_around(path="f.txt", line=999)
        assert out.startswith("[error]")
        assert "> total" in out
    finally:
        _ws_current.reset(token)


def test_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("a\nb\nc\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = read_lines_around(path=str(outside), line=2)
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_context_clamped(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        # context=99 clamped to 50; on a 10-line file, returns all 10.
        out = read_lines_around(path="f.txt", line=5, context=99)
        assert " 1\tline 1" in out
        assert " 10\tline 10" in out
    finally:
        _ws_current.reset(token)
