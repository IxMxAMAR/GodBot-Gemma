from __future__ import annotations
import os
import shutil
import subprocess
import sys

from godbot.core.registry import tool
from godbot.core.workspace import current_workspace


def _run(argv: list[str], input_text: str = "", timeout: int = 60, cwd: str | None = None) -> str:
    ws = current_workspace()
    if ws is not None and cwd is None:
        cwd = str(ws.root)
    env = os.environ.copy()
    if ws is not None:
        env["WORKSPACE_ROOT"] = str(ws.root)
    try:
        proc = subprocess.run(
            argv, input=input_text, capture_output=True, text=True,
            timeout=timeout, cwd=cwd, env=env,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"[error] timeout after {timeout}s"
    parts = []
    if proc.stdout:
        parts.append("STDOUT:\n" + proc.stdout)
    if proc.stderr:
        parts.append("STDERR:\n" + proc.stderr)
    parts.append(f"exit code {proc.returncode}")
    return "\n".join(parts)


@tool(dangerous=True, timeout=120)
def run_powershell(cmd: str, cwd: str | None = None, timeout: int = 60) -> str:
    """Run a PowerShell command. Returns stdout, stderr, and exit code."""
    return _run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
        timeout=timeout, cwd=cwd,
    )


_WSL_STUB_HINT = (
    "Windows Subsystem for Linux has no installed distributions"
)


def _find_bash_windows() -> tuple[str | None, str | None]:
    """Locate a real bash on Windows, skipping the WSL stub.

    Returns (path, source_label) or (None, None) when only the WSL stub
    or nothing at all is found. Search order matches typical user
    expectations: Git Bash > MSYS2 > Cygwin > PATH (after a stub check).
    """
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\msys64\usr\bin\bash.exe",
        r"C:\cygwin64\bin\bash.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c, "git-bash" if "Git" in c else ("msys2" if "msys64" in c else "cygwin")
    # Fall back to PATH but reject the WSL stub explicitly.
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found, "PATH"
    return None, None


@tool(dangerous=True, timeout=120)
def run_bash(cmd: str, cwd: str | None = None, timeout: int = 60) -> str:
    """Run a bash command. Returns stdout, stderr, and exit code.

    On Windows, prefers Git Bash / MSYS2 / Cygwin over the WSL stub at
    ``C:\\Windows\\System32\\bash.exe`` (which fails when WSL has no
    distros installed). If no real bash is available, returns an error
    pointing at ``run_powershell`` instead — without that hint the
    agent tends to retry the same broken bash call in a loop.
    """
    if sys.platform == "win32":
        bash_path, _source = _find_bash_windows()
        if bash_path is None:
            return (
                "[error] no real bash found on this Windows host (only the "
                "WSL stub is on PATH and it has no distros installed). "
                "Use run_powershell for shell commands, or install Git for "
                "Windows / MSYS2 if you specifically need bash."
            )
        result = _run([bash_path, "-c", cmd], timeout=timeout, cwd=cwd)
        # Defensive: if somehow we still ended up routing through the WSL
        # stub (PATH change, alias, etc.), rewrite the error so the agent
        # sees actionable guidance instead of WSL install instructions.
        if _WSL_STUB_HINT in result:
            return (
                "[error] bash invocation hit the Windows WSL stub which has "
                "no distros installed. Use run_powershell instead."
            )
        return result
    return _run(["/bin/bash", "-c", cmd], timeout=timeout, cwd=cwd)
