"""Tests for sub-project 48 — extract_function tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import extract_function


def test_extract_basic_function(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a - b\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = extract_function(path="demo.py", name="add")
        assert "def add(a, b):" in out
        assert "return a + b" in out
        assert "subtract" not in out  # only the matching function
    finally:
        _ws_current.reset(token)


def test_extract_includes_decorators(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text(
        "from functools import lru_cache\n"
        "@lru_cache\n"
        "@staticmethod\n"
        "def cached_fn(x):\n"
        "    return x * 2\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = extract_function(path="demo.py", name="cached_fn")
        assert "@lru_cache" in out
        assert "@staticmethod" in out
        assert "def cached_fn(x):" in out
    finally:
        _ws_current.reset(token)


def test_extract_class_definition(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text(
        "class Greeter:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "    def hello(self):\n"
        "        return f'Hi, {self.name}'\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = extract_function(path="demo.py", name="Greeter")
        assert "class Greeter:" in out
        assert "def __init__" in out
        assert "def hello" in out
    finally:
        _ws_current.reset(token)


def test_extract_method_by_bare_name(tmp_path):
    """Methods nested inside classes are matched by bare name."""
    f = tmp_path / "demo.py"
    f.write_text(
        "class Thing:\n"
        "    def helper(self, x):\n"
        "        return x + 1\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = extract_function(path="demo.py", name="helper")
        assert "def helper(self, x):" in out
        assert "return x + 1" in out
    finally:
        _ws_current.reset(token)


def test_extract_async_function(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text(
        "async def fetch(url):\n"
        "    return await client.get(url)\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = extract_function(path="demo.py", name="fetch")
        assert "async def fetch" in out
    finally:
        _ws_current.reset(token)


def test_extract_not_found(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text("def x(): pass\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = extract_function(path="demo.py", name="missing")
        assert out.startswith("[error]")
        assert "not found" in out
    finally:
        _ws_current.reset(token)


def test_extract_missing_args():
    out = extract_function(path="", name="x")
    assert out.startswith("[error]")


def test_extract_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def x(): pass\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = extract_function(path=str(outside), name="x")
        assert out.startswith("[error]")
        assert "outside the active workspace" in out
    finally:
        _ws_current.reset(token)


def test_extract_syntax_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = extract_function(path="bad.py", name="broken")
        assert out.startswith("[error]")
        assert "syntax" in out
    finally:
        _ws_current.reset(token)
