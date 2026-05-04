"""Tests for sub-project 60 — format_json + format_python."""
from __future__ import annotations

import json

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import format_json, format_python


# --- format_json ----


def test_format_json_default_indent():
    out = format_json(text='{"a":1,"b":[1,2,3]}')
    parsed = json.loads(out)
    assert parsed == {"a": 1, "b": [1, 2, 3]}
    assert "\n" in out
    assert "  " in out  # default 2-space indent


def test_format_json_custom_indent():
    out = format_json(text='{"a": 1}', indent=4)
    assert "    " in out


def test_format_json_indent_clamped():
    """Indents outside [1, 8] are clamped."""
    out = format_json(text='{"a": 1}', indent=999)
    # Doesn't crash, returns something parseable.
    assert json.loads(out) == {"a": 1}


def test_format_json_sort_keys():
    out = format_json(text='{"z":1,"a":2}', sort_keys=True)
    parsed_lines = out.splitlines()
    # First key after the opening brace should be "a", not "z".
    a_idx = next(i for i, ln in enumerate(parsed_lines) if '"a"' in ln)
    z_idx = next(i for i, ln in enumerate(parsed_lines) if '"z"' in ln)
    assert a_idx < z_idx


def test_format_json_invalid_input():
    out = format_json(text='not json')
    assert out.startswith("[error]")


def test_format_json_empty():
    out = format_json(text="")
    assert out.startswith("[error]")


# --- format_python ----


def test_format_python_inline():
    out = format_python(text="def x(  a , b   ): return a+b")
    assert "def x(a, b):" in out
    assert "return a + b" in out


def test_format_python_path(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("x   =1\ny=  2", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = format_python(path="f.py")
        assert "x = 1" in out
        assert "y = 2" in out
    finally:
        _ws_current.reset(token)


def test_format_python_syntax_error():
    out = format_python(text="def broken(\n")
    assert out.startswith("[error]")
    assert "syntax" in out or "line" in out


def test_format_python_no_args():
    out = format_python()
    assert out.startswith("[error]")


def test_format_python_both_args_errors():
    out = format_python(text="x=1", path="f.py")
    assert out.startswith("[error]")
    assert "only one" in out


def test_format_python_non_python_path(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# hi", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = format_python(path="notes.md")
        assert out.startswith("[error]")
        assert "not a Python file" in out
    finally:
        _ws_current.reset(token)
