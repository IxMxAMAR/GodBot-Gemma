from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from godbot.core.registry import tool
from godbot.core.workspace import current_workspace


@tool(dangerous=True, timeout=60)
def run_python(code: str, timeout: int = 30) -> str:
    """Run a Python snippet in a subprocess against the harness's interpreter."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    ws = current_workspace()
    cwd = str(ws.root) if ws is not None else None
    env = os.environ.copy()
    if ws is not None:
        env["WORKSPACE_ROOT"] = str(ws.root)
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True,
            timeout=timeout, cwd=cwd, env=env,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"[error] timeout after {timeout}s"
    finally:
        Path(path).unlink(missing_ok=True)
    parts = []
    if proc.stdout:
        parts.append("STDOUT:\n" + proc.stdout)
    if proc.stderr:
        parts.append("STDERR:\n" + proc.stderr)
    parts.append(f"exit code {proc.returncode}")
    return "\n".join(parts)
