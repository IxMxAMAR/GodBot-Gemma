"""Tests for sub-project 30 — workspace_context aggregator tool."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from godbot.core.registry import DEFAULT
from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import workspace_context


def _git_init(path):
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True, capture_output=True)


def test_workspace_context_no_active_workspace():
    out = workspace_context()
    assert "no active workspace" in out.lower()


def test_workspace_context_python_project_listing(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\ndependencies = ["fastapi"]\n', encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# demo\nA tiny project.", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main(): pass\n", encoding="utf-8")

    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = workspace_context()
        assert "=== project ===" in out
        assert "demo" in out.lower() or "Project: demo" in out
        assert "fastapi" in out.lower()
        assert "=== top level ===" in out
        assert "src/" in out
        assert "pyproject.toml" in out
    finally:
        _ws_current.reset(token)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_workspace_context_includes_git_when_repo(tmp_path):
    _git_init(tmp_path)
    (tmp_path / "f.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "first commit"], cwd=tmp_path, check=True, capture_output=True)

    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = workspace_context()
        assert "=== git ===" in out
        assert "main" in out
        assert "clean" in out
        assert "=== recent commits ===" in out
        assert "first commit" in out
    finally:
        _ws_current.reset(token)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_workspace_context_dirty_state_visible(tmp_path):
    _git_init(tmp_path)
    (tmp_path / "f.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "f.txt").write_text("v2\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")

    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = workspace_context()
        assert "dirty" in out
        assert "f.txt" in out
        assert "untracked.txt" in out
    finally:
        _ws_current.reset(token)


def test_workspace_context_omits_git_when_not_a_repo(tmp_path):
    """Non-git directory: the git/recent-commits sections must NOT appear."""
    (tmp_path / "thing.txt").write_text("x\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = workspace_context()
        assert "=== git ===" not in out
        assert "=== recent commits ===" not in out
        assert "=== top level ===" in out
    finally:
        _ws_current.reset(token)


def test_workspace_context_caps_output_size(tmp_path):
    """Many top-level files get capped at 30 entries."""
    for i in range(60):
        (tmp_path / f"file_{i:03d}.txt").write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = workspace_context()
        assert "(30 more)" in out or "... (" in out
    finally:
        _ws_current.reset(token)


def test_workspace_context_tool_is_registered():
    assert DEFAULT.spec("workspace_context") is not None


def test_workspace_context_tool_is_not_dangerous():
    spec = DEFAULT.spec("workspace_context")
    assert spec is not None
    assert spec.dangerous is False
