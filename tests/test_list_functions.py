"""Tests for sub-project 54 — list_functions tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import list_functions


def test_list_top_level_functions(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text(
        "def add(a, b):\n    return a + b\n"
        "def subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = list_functions(path="demo.py")
        assert "def add" in out
        assert "def subtract" in out
        assert "L1:" in out
    finally:
        _ws_current.reset(token)


def test_list_class_with_methods(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text(
        "class Greeter:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "    def hello(self):\n"
        "        return self.name\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = list_functions(path="demo.py")
        assert "class Greeter" in out
        assert "def Greeter.__init__" in out
        assert "def Greeter.hello" in out
    finally:
        _ws_current.reset(token)


def test_list_async_functions(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text(
        "async def fetch(url):\n    return url\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = list_functions(path="demo.py")
        assert "async def fetch" in out
    finally:
        _ws_current.reset(token)


def test_list_empty_file(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("x = 1\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = list_functions(path="empty.py")
        assert "(no top-level definitions)" in out
    finally:
        _ws_current.reset(token)


def test_list_missing_path():
    out = list_functions(path="")
    assert out.startswith("[error]")
    assert "path required" in out


def test_list_non_python_rejected(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("[a]\nb = 1", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = list_functions(path="config.toml")
        assert out.startswith("[error]")
        assert "not a Python file" in out
    finally:
        _ws_current.reset(token)


def test_list_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def x(): pass\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = list_functions(path=str(outside))
        assert out.startswith("[error]")
        assert "outside the active workspace" in out
    finally:
        _ws_current.reset(token)


def test_list_syntax_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = list_functions(path="bad.py")
        assert out.startswith("[error]")
        assert "syntax" in out
    finally:
        _ws_current.reset(token)


def test_list_signature_capture(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text(
        "def fn(a: int, b: str = 'x', *args, **kwargs):\n    pass\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = list_functions(path="demo.py")
        # Some form of the args (likely via ast.unparse) should be in output.
        assert "def fn" in out
    finally:
        _ws_current.reset(token)
