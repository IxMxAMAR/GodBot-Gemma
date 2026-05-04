"""Tests for sub-project 65 — text_metrics tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import text_metrics


def test_text_metrics_inline_basic():
    out = text_metrics(text="hello world")
    assert "11 chars" in out
    assert "11 bytes" in out
    assert "2 words" in out
    assert "1 lines" in out
    assert "longest line 11" in out


def test_text_metrics_multiline():
    body = "line1\nlonger second line\nthird"
    out = text_metrics(text=body)
    assert "3 lines" in out
    assert "longest line 18" in out  # "longer second line" = 18 chars


def test_text_metrics_unicode_byte_count():
    """Unicode chars count as multiple bytes in UTF-8."""
    out = text_metrics(text="héllo")  # 5 chars, 6 bytes (é = 2)
    assert "5 chars" in out
    assert "6 bytes" in out


def test_text_metrics_path(tmp_path):
    f = tmp_path / "demo.txt"
    f.write_text("one two three\nfour five\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = text_metrics(path="demo.txt")
        assert "5 words" in out
        assert "2 lines" in out
    finally:
        _ws_current.reset(token)


def test_text_metrics_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = text_metrics(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_text_metrics_no_args_errors():
    out = text_metrics()
    assert out.startswith("[error]")


def test_text_metrics_both_args_errors():
    out = text_metrics(text="x", path="y.txt")
    assert out.startswith("[error]")


def test_text_metrics_empty_input():
    out = text_metrics(text="")
    assert out.startswith("[error]")  # empty falls into "no args" branch
