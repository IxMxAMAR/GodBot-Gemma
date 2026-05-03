from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path

from godbot.core.registry import tool


@tool(dangerous=True, timeout=60)
def run_python(code: str, timeout: int = 30) -> str:
    """Run a Python snippet in a subprocess against the harness's interpreter."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
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
