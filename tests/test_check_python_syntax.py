"""Tests for sub-project 50 — check_python_syntax tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import check_python_syntax


def test_check_inline_valid():
    out = check_python_syntax(code="def f(): return 1")
    assert out.startswith("ok:")
    assert "parses cleanly" in out


def test_check_inline_invalid_returns_position():
    out = check_python_syntax(code="def f(\n")
    assert out.startswith("[error]")
    assert "line" in out
    assert "col" in out


def test_check_file_valid(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = check_python_syntax(path="ok.py")
        assert out.startswith("ok:")
    finally:
        _ws_current.reset(token)


def test_check_file_invalid(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = check_python_syntax(path="bad.py")
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_check_no_args_errors():
    out = check_python_syntax()
    assert out.startswith("[error]")


def test_check_both_args_errors():
    out = check_python_syntax(path="x.py", code="def f(): pass")
    assert out.startswith("[error]")
    assert "only one" in out


def test_check_non_python_path_rejected(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# hello", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = check_python_syntax(path="notes.md")
        assert out.startswith("[error]")
        assert "not a Python file" in out
    finally:
        _ws_current.reset(token)


def test_check_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def f(): pass\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = check_python_syntax(path=str(outside))
        assert out.startswith("[error]")
        assert "outside the active workspace" in out
    finally:
        _ws_current.reset(token)
