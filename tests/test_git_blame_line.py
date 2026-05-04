"""Tests for sub-project 96 — git_blame_line tool."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.git_tools import git_blame_line


GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git binary not on PATH")


def _init_repo(path):
    subprocess.run([GIT, "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run([GIT, "config", "user.email", "test@example.com"], cwd=path, check=True, capture_output=True)
    subprocess.run([GIT, "config", "user.name", "Test User"], cwd=path, check=True, capture_output=True)
    subprocess.run([GIT, "config", "commit.gpgsign", "false"], cwd=path, check=True, capture_output=True)


@needs_git
def test_blame_basic(tmp_path):
    _init_repo(tmp_path)
    f = tmp_path / "main.py"
    f.write_text("line one\nline two\nline three\n", encoding="utf-8")
    subprocess.run([GIT, "add", "main.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)

    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = git_blame_line(path="main.py", line=2)
        assert "line 2:" in out
        assert "Test User" in out
        assert "line two" in out
    finally:
        _ws_current.reset(token)


@needs_git
def test_blame_invalid_line_errors():
    out = git_blame_line(path="x", line=0)
    assert out.startswith("[error]")


@needs_git
def test_blame_missing_path():
    out = git_blame_line(path="", line=1)
    assert out.startswith("[error]")


@needs_git
def test_blame_missing_file_errors(tmp_path):
    _init_repo(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = git_blame_line(path="ghost.py", line=1)
        assert out.startswith("[error]")
        assert "not a file" in out
    finally:
        _ws_current.reset(token)


@needs_git
def test_blame_outside_repo_errors(tmp_path):
    """A directory that's not a git repo should error from git itself."""
    f = tmp_path / "untracked.txt"
    f.write_text("x\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = git_blame_line(path="untracked.txt", line=1)
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


@needs_git
def test_blame_includes_iso_timestamp(tmp_path):
    _init_repo(tmp_path)
    f = tmp_path / "x.txt"
    f.write_text("hello\n", encoding="utf-8")
    subprocess.run([GIT, "add", "x.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        import re
        out = git_blame_line(path="x.txt", line=1)
        # ISO format YYYY-MM-DDTHH:MM:SS
        assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", out)
    finally:
        _ws_current.reset(token)
