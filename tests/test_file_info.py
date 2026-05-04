"""Tests for sub-project 69 — file_info tool."""
from __future__ import annotations

import re

from godbot.core.registry import DEFAULT
from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import file_info


def test_file_info_basic_file(tmp_path):
    f = tmp_path / "thing.txt"
    f.write_text("hello", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = file_info(path="thing.txt")
        assert "size: 5 bytes" in out
        assert "type: file" in out
        assert re.search(r"modified: \d{4}-\d{2}-\d{2}T", out)
    finally:
        _ws_current.reset(token)


def test_file_info_includes_mime_when_known(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html></html>", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = file_info(path="page.html")
        assert "mime: text/html" in out
    finally:
        _ws_current.reset(token)


def test_file_info_directory_type(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = file_info(path="subdir")
        assert "type: directory" in out
    finally:
        _ws_current.reset(token)


def test_file_info_missing(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = file_info(path="ghost.txt")
        assert out.startswith("[error]")
        assert "not found" in out
    finally:
        _ws_current.reset(token)


def test_file_info_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = file_info(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_file_info_empty_path():
    out = file_info(path="")
    assert out.startswith("[error]")
    assert "path required" in out


def test_file_info_is_registered_non_dangerous():
    spec = DEFAULT.spec("file_info")
    assert spec is not None
    assert spec.dangerous is False
