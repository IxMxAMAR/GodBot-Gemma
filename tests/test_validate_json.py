"""Tests for sub-project 42 — validate_json tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import validate_json


def test_validate_json_object_inline():
    out = validate_json(text='{"a": 1, "b": 2}')
    assert out.startswith("ok:")
    assert "object with 2 key" in out


def test_validate_json_array_inline():
    out = validate_json(text='[1, 2, 3, 4]')
    assert out.startswith("ok:")
    assert "array of 4 item" in out


def test_validate_json_scalar_inline():
    out = validate_json(text='42')
    assert out.startswith("ok:")
    assert "int" in out


def test_validate_json_string_inline():
    out = validate_json(text='"hello"')
    assert out.startswith("ok:")
    assert "str" in out


def test_validate_json_invalid_returns_line_col():
    out = validate_json(text='{"a": 1, "b": }')
    assert out.startswith("[error]")
    assert "line" in out and "col" in out


def test_validate_json_empty_args_errors():
    out = validate_json()
    assert out.startswith("[error]")


def test_validate_json_both_args_errors():
    out = validate_json(text="{}", path="x")
    assert out.startswith("[error]")
    assert "only one" in out


def test_validate_json_path_workspace_relative(tmp_path):
    f = tmp_path / "config.json"
    f.write_text('{"port": 8080}', encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = validate_json(path="config.json")
        assert out.startswith("ok:")
        assert "object with 1 key" in out
    finally:
        _ws_current.reset(token)


def test_validate_json_path_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = validate_json(path=str(outside))
        assert out.startswith("[error]")
        assert "outside the active workspace" in out
    finally:
        _ws_current.reset(token)


def test_validate_json_missing_file(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = validate_json(path="nope.json")
        assert out.startswith("[error]")
        assert "not a file" in out
    finally:
        _ws_current.reset(token)


def test_validate_json_corrupt_file_reports_position(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text('{"a": 1\n"b": 2}\n', encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = validate_json(path="bad.json")
        assert out.startswith("[error]")
        assert "line 2" in out or "line 1" in out
    finally:
        _ws_current.reset(token)
