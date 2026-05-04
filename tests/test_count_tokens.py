"""Tests for sub-project 39 — count_tokens tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import count_tokens


def test_count_tokens_inline_text():
    out = count_tokens(text="hello world")
    assert "approx" in out
    assert "tokens" in out
    # 11 chars / 4 = 2 tokens (integer division).
    assert "11 chars" in out
    assert "1 lines" in out


def test_count_tokens_empty_args_errors():
    out = count_tokens()
    assert out.startswith("[error]")


def test_count_tokens_both_args_errors():
    out = count_tokens(text="hi", path="x")
    assert out.startswith("[error]")
    assert "only one" in out


def test_count_tokens_path_workspace_relative(tmp_path):
    f = tmp_path / "main.py"
    f.write_text("def foo():\n    return 42\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_tokens(path="main.py")
        assert "approx" in out
        # 25 chars total in our content.
        assert "main.py" in out
    finally:
        _ws_current.reset(token)


def test_count_tokens_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = count_tokens(path=str(outside))
        assert out.startswith("[error]")
        assert "outside the active workspace" in out
    finally:
        _ws_current.reset(token)


def test_count_tokens_missing_file(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_tokens(path="ghost.py")
        assert out.startswith("[error]")
        assert "not a file" in out
    finally:
        _ws_current.reset(token)


def test_count_tokens_handles_multiline():
    multiline = "line1\nline2\nline3\n"
    out = count_tokens(text=multiline)
    assert "3 lines" in out
