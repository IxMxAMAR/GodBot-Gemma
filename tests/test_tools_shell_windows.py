"""Tests for sub-project 109 — Windows-specific bash routing.

The previous implementation called `shutil.which("bash")` which on most
Windows hosts returns `C:\\Windows\\System32\\bash.exe` — the WSL stub.
When WSL isn't installed (the common case), every `run_bash` call
returned junk that confused the agent into a max_steps loop.
"""
from __future__ import annotations

import sys

import pytest

from godbot.core.registry import DEFAULT
from godbot.tools.shell import _find_bash_windows, run_bash


pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows-only tests"
)


def test_find_bash_skips_system32_stub(monkeypatch):
    """If the only PATH bash is the WSL stub at System32, return None."""
    import os
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda name: r"C:\Windows\System32\bash.exe")
    path, src = _find_bash_windows()
    assert path is None
    assert src is None


def test_find_bash_finds_git_bash_when_present(monkeypatch):
    """When Git Bash is at the canonical path, prefer it."""
    git_path = r"C:\Program Files\Git\bin\bash.exe"
    import os
    monkeypatch.setattr(os.path, "exists", lambda p: p == git_path)
    path, src = _find_bash_windows()
    assert path == git_path
    assert src == "git-bash"


def test_find_bash_accepts_path_when_not_stub(monkeypatch):
    """When `bash` on PATH is NOT the System32 stub, use it."""
    import os
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda name: r"C:\custom\bin\bash.exe")
    path, _src = _find_bash_windows()
    assert path == r"C:\custom\bin\bash.exe"


def test_run_bash_no_real_bash_returns_useful_error(monkeypatch):
    """When no real bash is found, return a message pointing the agent
    at run_powershell — not the unhelpful WSL install instructions."""
    from godbot.tools import shell
    monkeypatch.setattr(shell, "_find_bash_windows", lambda: (None, None))
    out = DEFAULT.execute("run_bash", {"cmd": "echo hi"})
    assert out.startswith("[error]")
    assert "run_powershell" in out
    assert "WSL" in out or "wsl" in out


def test_run_bash_rewrites_wsl_distro_error(monkeypatch):
    """If we somehow still routed through the WSL stub at runtime, the
    'no installed distributions' string in the output should be rewritten
    so the agent sees actionable guidance."""
    from godbot.tools import shell
    # Pretend we found *some* bash, but make _run return WSL-stub junk.
    monkeypatch.setattr(shell, "_find_bash_windows", lambda: (r"C:\fake\bash.exe", "PATH"))
    monkeypatch.setattr(
        shell, "_run",
        lambda *a, **kw: (
            "STDOUT:\nWindows Subsystem for Linux has no installed distributions.\n"
            "exit code 1"
        ),
    )
    out = DEFAULT.execute("run_bash", {"cmd": "echo hi"})
    assert out.startswith("[error]")
    assert "run_powershell" in out


def test_run_bash_works_when_real_bash_present():
    """Smoke: with the actual installed Git Bash, a trivial echo works."""
    out = DEFAULT.execute("run_bash", {"cmd": "echo godbot-test-marker"})
    if out.startswith("[error]"):
        pytest.skip(f"no real bash on this host: {out}")
    assert "godbot-test-marker" in out
