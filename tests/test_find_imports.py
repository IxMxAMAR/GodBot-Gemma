"""Tests for sub-project 47 — find_imports tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import find_imports


def test_find_imports_basic(tmp_path):
    f = tmp_path / "demo.py"
    f.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from typing import Any, Optional\n"
        "import json as J\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_imports(path="demo.py")
        assert "import os" in out
        assert "import sys" in out
        assert "import json as J" in out
        assert "from pathlib import Path" in out
        assert "from typing import Any, Optional" in out
        assert "direct (3)" in out
        assert "from (2)" in out
    finally:
        _ws_current.reset(token)


def test_find_imports_relative_levels(tmp_path):
    f = tmp_path / "rel.py"
    f.write_text(
        "from . import utils\n"
        "from ..pkg import helper\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_imports(path="rel.py")
        assert "from . import utils" in out
        assert "from ..pkg import helper" in out
    finally:
        _ws_current.reset(token)


def test_find_imports_no_imports(tmp_path):
    f = tmp_path / "plain.py"
    f.write_text("x = 1\nprint(x)\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_imports(path="plain.py")
        assert "(no imports)" in out
    finally:
        _ws_current.reset(token)


def test_find_imports_syntax_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def x(\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_imports(path="bad.py")
        assert out.startswith("[error]")
        assert "syntax" in out
        assert "line" in out
    finally:
        _ws_current.reset(token)


def test_find_imports_non_python_rejected(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("[a]\nb = 1", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_imports(path="config.toml")
        assert out.startswith("[error]")
        assert "not a Python file" in out
    finally:
        _ws_current.reset(token)


def test_find_imports_missing_path():
    out = find_imports(path="")
    assert out.startswith("[error]")
    assert "path required" in out


def test_find_imports_missing_file(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_imports(path="ghost.py")
        assert out.startswith("[error]")
        assert "not a file" in out
    finally:
        _ws_current.reset(token)


def test_find_imports_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("import os\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = find_imports(path=str(outside))
        assert out.startswith("[error]")
        assert "outside the active workspace" in out
    finally:
        _ws_current.reset(token)
