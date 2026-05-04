"""Tests for sub-project 85 — system_info tool."""
from __future__ import annotations

import sys

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import system_info


def test_system_info_includes_os():
    out = system_info()
    assert "os:" in out
    assert sys.platform in out


def test_system_info_includes_python_version():
    out = system_info()
    assert "python:" in out
    parts = sys.version.split()[0].split(".")
    expected = f"{parts[0]}.{parts[1]}"
    assert expected in out


def test_system_info_includes_cpu_count():
    out = system_info()
    assert "cpu_count:" in out


def test_system_info_memory_line_present():
    """Memory line is present, regardless of whether psutil is installed."""
    out = system_info()
    assert "memory_gb:" in out


def test_system_info_disk_line_present():
    out = system_info()
    assert "disk_free_gb:" in out


def test_system_info_includes_cwd():
    out = system_info()
    assert "cwd:" in out


def test_system_info_is_registered_non_dangerous():
    spec = DEFAULT.spec("system_info")
    assert spec is not None
    assert spec.dangerous is False


def test_system_info_never_raises():
    """Running it twice in a row shouldn't crash."""
    out_a = system_info()
    out_b = system_info()
    assert out_a.startswith("os:")
    assert out_b.startswith("os:")
