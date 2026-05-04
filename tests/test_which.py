"""Tests for sub-project 53 — which tool."""
from __future__ import annotations

import shutil

import pytest

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import which


@pytest.mark.skipif(shutil.which("python") is None, reason="python not on PATH")
def test_which_finds_python():
    out = which(name="python")
    assert out.startswith("found:")
    # Path should end with python.exe on Windows or just point at a real file.
    assert ".exe" in out.lower() or "python" in out.lower()


def test_which_handles_missing():
    out = which(name="this-binary-definitely-does-not-exist-on-PATH-1234")
    assert out.startswith("not found:")


def test_which_empty_name():
    out = which(name="")
    assert out.startswith("[error]")
    assert "name required" in out


def test_which_is_registered_and_non_dangerous():
    spec = DEFAULT.spec("which")
    assert spec is not None
    assert spec.dangerous is False


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_which_git():
    out = which(name="git")
    assert out.startswith("found:")
    assert "git" in out.lower()
