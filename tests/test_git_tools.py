"""Tests for the read-only git tools (sub-project 15).

Each test creates a tiny git repo in tmp_path and exercises one tool. We
shell out to real git because the goal is end-to-end behaviour — mocking
subprocess would test our argv assembly only and miss output-shape
regressions.

If git isn't on PATH, every test skips.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.git_tools import git_branch, git_diff, git_log, git_status


GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git binary not on PATH")


def _init_repo(path):
    subprocess.run([GIT, "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run([GIT, "config", "user.email", "test@example.com"], cwd=path, check=True, capture_output=True)
    subprocess.run([GIT, "config", "user.name", "Test"], cwd=path, check=True, capture_output=True)
    subprocess.run([GIT, "config", "commit.gpgsign", "false"], cwd=path, check=True, capture_output=True)


@needs_git
def test_status_no_workspace_no_path_errors():
    out = git_status(path="")
    assert out.startswith("[error]")


@needs_git
def test_status_clean_repo(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("hello", encoding="utf-8")
    subprocess.run([GIT, "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    out = git_status(path=str(tmp_path))
    # Branch line is always present in --short --branch even on clean repo.
    assert "main" in out or "master" in out


@needs_git
def test_status_shows_unstaged_changes(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("hello", encoding="utf-8")
    subprocess.run([GIT, "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.txt").write_text("changed", encoding="utf-8")
    out = git_status(path=str(tmp_path))
    assert "file.txt" in out


@needs_git
def test_status_shows_untracked(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")
    out = git_status(path=str(tmp_path))
    assert "new.txt" in out


@needs_git
def test_diff_unstaged(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("a\nb\nc\n", encoding="utf-8")
    subprocess.run([GIT, "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "f.txt").write_text("a\nb\nNEW\n", encoding="utf-8")
    out = git_diff(path=str(tmp_path))
    assert "NEW" in out
    assert "f.txt" in out


@needs_git
def test_diff_staged(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("a\n", encoding="utf-8")
    subprocess.run([GIT, "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
    out = git_diff(path=str(tmp_path), staged=True)
    assert "f.txt" in out


@needs_git
def test_log_lists_commits(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("v1", encoding="utf-8")
    subprocess.run([GIT, "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "first commit"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "f.txt").write_text("v2", encoding="utf-8")
    subprocess.run([GIT, "commit", "-am", "second commit"], cwd=tmp_path, check=True, capture_output=True)
    out = git_log(path=str(tmp_path))
    assert "first commit" in out
    assert "second commit" in out


@needs_git
def test_log_respects_limit(tmp_path):
    _init_repo(tmp_path)
    for i in range(5):
        (tmp_path / "f.txt").write_text(f"v{i}", encoding="utf-8")
        subprocess.run([GIT, "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run([GIT, "commit", "-m", f"commit {i}"], cwd=tmp_path, check=True, capture_output=True)
    out = git_log(path=str(tmp_path), limit=2)
    # Two newline-separated commits → at most 2 line breaks; the third
    # commit message should not appear.
    assert "commit 4" in out
    assert "commit 3" in out
    assert "commit 2" not in out


@needs_git
def test_branch_lists_current(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run([GIT, "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    out = git_branch(path=str(tmp_path))
    # `git branch --list` prefixes the current branch with `* `.
    assert "*" in out
    assert "main" in out or "master" in out


@needs_git
def test_uses_active_workspace_when_no_path_given(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run([GIT, "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)

    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = git_status(path="")
        assert "[error]" not in out
    finally:
        _ws_current.reset(token)


@needs_git
def test_invalid_path_returns_error_string(tmp_path):
    out = git_status(path=str(tmp_path / "no_such"))
    assert out.startswith("[error]") and "not a directory" in out


@needs_git
def test_diff_target_specifier(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run([GIT, "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "commit", "-m", "first"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "f.txt").write_text("v2\n", encoding="utf-8")
    subprocess.run([GIT, "commit", "-am", "second"], cwd=tmp_path, check=True, capture_output=True)
    out = git_diff(path=str(tmp_path), target="HEAD~1..HEAD")
    assert "v2" in out
    assert "v1" in out
