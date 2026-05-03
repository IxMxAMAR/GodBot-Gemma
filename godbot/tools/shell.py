from __future__ import annotations
import shutil
import subprocess
import sys

from godbot.core.registry import tool


def _run(argv: list[str], input_text: str = "", timeout: int = 60, cwd: str | None = None) -> str:
    try:
        proc = subprocess.run(
            argv, input=input_text, capture_output=True, text=True,
            timeout=timeout, cwd=cwd, encoding="utf-8", errors="replace",
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


@tool(dangerous=True, timeout=120)
def run_bash(cmd: str, cwd: str | None = None, timeout: int = 60) -> str:
    """Run a bash command. Returns stdout, stderr, and exit code."""
    # Windows: prefer Git Bash via shutil.which (avoids WSL stub bash.exe shadowing);
    # else /bin/bash on POSIX.
    if sys.platform == "win32":
        bash_path = shutil.which("bash") or "bash"
        return _run([bash_path, "-c", cmd], timeout=timeout, cwd=cwd)
    return _run(["/bin/bash", "-c", cmd], timeout=timeout, cwd=cwd)
