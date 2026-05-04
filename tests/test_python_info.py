"""Tests for sub-project 64 — python_info tool."""
from __future__ import annotations

import sys

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import python_info


def test_python_info_includes_executable():
    out = python_info()
    assert "executable:" in out
    assert sys.executable in out


def test_python_info_includes_version():
    out = python_info()
    assert "version:" in out
    # Major.minor.patch substring should appear.
    parts = sys.version.split()[0].split(".")
    expected_prefix = f"{parts[0]}.{parts[1]}"
    assert expected_prefix in out


def test_python_info_includes_prefix():
    out = python_info()
    assert "prefix:" in out
    assert sys.prefix in out


def test_python_info_lists_distributions():
    out = python_info()
    # The 'distributions' line should mention SOME total count and a list.
    # Use a permissive assertion: the line exists and the count is plausible.
    line = next(ln for ln in out.splitlines() if ln.startswith("distributions"))
    assert "total" in line or "(none discovered)" in line


def test_python_info_is_registered_non_dangerous():
    spec = DEFAULT.spec("python_info")
    assert spec is not None
    assert spec.dangerous is False


def test_python_info_caps_distribution_list():
    """The distributions field caps at 30 entries."""
    out = python_info()
    line = next(ln for ln in out.splitlines() if ln.startswith("distributions"))
    # Either "(none discovered)" or "(N total, top 30): a, b, ..." — count
    # the comma-separated entries on the latter.
    if "top 30" in line:
        names_part = line.split(": ", 1)[1]
        names = [n.strip() for n in names_part.split(",")]
        assert len(names) <= 30
