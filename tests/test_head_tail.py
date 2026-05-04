"""Tests for sub-project 55 — head + tail tools."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import head, tail


def _seed(tmp_path, n: int):
    f = tmp_path / "log.txt"
    f.write_text("\n".join(f"line {i}" for i in range(1, n + 1)) + "\n", encoding="utf-8")
    return f


def test_head_default_50_lines(tmp_path):
    _seed(tmp_path, 200)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = head(path="log.txt")
        line_count = len(out.splitlines())
        assert line_count == 50
        # First and last visible lines come from the head.
        assert "1\tline 1" in out
        assert "50\tline 50" in out
        assert "line 51" not in out
    finally:
        _ws_current.reset(token)


def test_head_custom_count(tmp_path):
    _seed(tmp_path, 100)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = head(path="log.txt", lines=5)
        assert len(out.splitlines()) == 5
        assert "5\tline 5" in out
        assert "line 6" not in out
    finally:
        _ws_current.reset(token)


def test_head_caps_at_5000(tmp_path):
    _seed(tmp_path, 100)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        # Asking for 99999 should not crash; clamps to 5000 (i.e. all 100 here).
        out = head(path="log.txt", lines=99999)
        assert len(out.splitlines()) == 100
    finally:
        _ws_current.reset(token)


def test_head_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = head(path="empty.txt")
        assert "(empty file)" in out
    finally:
        _ws_current.reset(token)


def test_head_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "log.txt"
    outside.write_text("x\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = head(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_tail_default(tmp_path):
    _seed(tmp_path, 200)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tail(path="log.txt")
        line_count = len(out.splitlines())
        assert line_count == 50
        # Last 50 lines should be 151..200.
        assert "200\tline 200" in out
        assert "151\tline 151" in out
        assert "150\tline 150" not in out
    finally:
        _ws_current.reset(token)


def test_tail_custom_count(tmp_path):
    _seed(tmp_path, 50)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tail(path="log.txt", lines=3)
        lines_visible = out.splitlines()
        assert len(lines_visible) == 3
        # Should be the last three: 48, 49, 50.
        assert "48\tline 48" in lines_visible[0]
        assert "50\tline 50" in lines_visible[-1]
    finally:
        _ws_current.reset(token)


def test_tail_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tail(path="empty.txt")
        assert "(empty file)" in out
    finally:
        _ws_current.reset(token)


def test_tail_smaller_than_count(tmp_path):
    """Asking for 50 lines from a 5-line file returns all 5."""
    _seed(tmp_path, 5)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tail(path="log.txt", lines=50)
        assert len(out.splitlines()) == 5
        assert "1\tline 1" in out
        assert "5\tline 5" in out
    finally:
        _ws_current.reset(token)


def test_head_missing_file_errors(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = head(path="ghost.log")
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)
