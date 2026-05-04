"""Tests for the run_tests tool (sub-project 16)."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.testing import _detect_runner, _summarize_pytest, run_tests


def test_detect_pytest_via_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n", encoding="utf-8")
    assert _detect_runner(tmp_path) == "pytest"


def test_detect_pytest_via_pytest_ini(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    assert _detect_runner(tmp_path) == "pytest"


def test_detect_npm_via_package_json(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"name": "demo", "scripts": {"test": "jest"}}', encoding="utf-8"
    )
    assert _detect_runner(tmp_path) == "npm"


def test_detect_unknown_when_no_markers(tmp_path):
    assert _detect_runner(tmp_path) == "unknown"


def test_package_json_without_test_script_is_unknown(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"name": "demo", "scripts": {"build": "tsc"}}', encoding="utf-8"
    )
    assert _detect_runner(tmp_path) == "unknown"


def test_summarize_pytest_extracts_passed_failed():
    stdout = (
        "...F.\n"
        "============================== short test summary info ===========\n"
        "FAILED tests/test_x.py::test_y\n"
        "==================== 4 passed, 1 failed in 0.42s =================\n"
    )
    s = _summarize_pytest(stdout, 1)
    assert "4 passed" in s
    assert "1 failed" in s


def test_summarize_pytest_falls_back_when_no_summary():
    s = _summarize_pytest("garbage output\n", 99)
    assert "exited 99" in s


def test_run_tests_unknown_runner_errors(tmp_path):
    out = run_tests(path=str(tmp_path))
    assert "could not detect test runner" in out


def test_run_tests_no_workspace_no_path_errors():
    out = run_tests(path="")
    assert out.startswith("[error]")


@pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")
def test_run_tests_pytest_smoke(tmp_path):
    """End-to-end: a tiny project with one passing test gets summarized."""
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\nasyncio_mode = 'auto'\n", encoding="utf-8"
    )
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_demo.py").write_text(
        "def test_one():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    out = run_tests(path=str(tmp_path))
    assert "runner: pytest" in out
    assert "exit: 0" in out
    assert "1 passed" in out


@pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")
def test_run_tests_uses_active_workspace(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_demo.py").write_text(
        "def test_x(): assert True\n", encoding="utf-8"
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = run_tests(path="")
        assert "runner: pytest" in out
        assert "exit: 0" in out
    finally:
        _ws_current.reset(token)


@pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")
def test_run_tests_pytest_failure_exit_code(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_fail.py").write_text(
        "def test_break(): assert False\n", encoding="utf-8"
    )
    out = run_tests(path=str(tmp_path))
    assert "exit: 1" in out
    assert "1 failed" in out
