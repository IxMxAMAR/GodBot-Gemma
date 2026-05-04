"""Test-runner tool (sub-project 16).

Runs the project's test suite and returns a structured summary the agent
can act on. v1 supports pytest (Python) and ``npm test`` (Node) — auto-
detected from marker files at the workspace root. Other runners (Go,
Rust, etc.) fall through to a hint asking the agent to use
``run_powershell`` / ``run_bash``.

The tool is non-dangerous: it only spawns the configured test runner and
reads stdout. The runners themselves can have side effects (e.g. tests
that write to disk), but that's the project's responsibility, not ours.
The 5-minute timeout cap keeps a hanging test suite from pinning the
agent.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from godbot.core.registry import tool
from godbot.core.workspace import current_workspace


_OUTPUT_CAP = 20_000
_RUN_TIMEOUT = 300  # 5 minutes


def _detect_runner(repo: Path) -> Optional[str]:
    """Return ``"pytest"`` | ``"npm"`` | ``"unknown"``.

    Detection is path-based: pyproject.toml or pytest.ini → pytest;
    package.json with a test script → npm; otherwise unknown.
    """
    if (repo / "pyproject.toml").is_file() or (repo / "pytest.ini").is_file():
        return "pytest"
    pkg = repo / "package.json"
    if pkg.is_file():
        try:
            import json
            data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
            scripts = data.get("scripts") or {}
            if isinstance(scripts, dict) and "test" in scripts:
                return "npm"
        except Exception:
            pass
    return "unknown"


def _resolve_repo(path: Optional[str]) -> Path | str:
    ws = current_workspace()
    if path:
        p = Path(path)
        if not p.is_absolute() and ws is not None:
            p = ws.root / p
        try:
            p = p.resolve()
        except OSError as e:
            return f"[error] cannot resolve {path!r}: {e}"
        if not p.is_dir():
            return f"[error] not a directory: {p}"
        return p
    if ws is not None:
        return Path(ws.root)
    return "[error] no path supplied and no active workspace"


def _cap(text: str) -> str:
    if len(text) <= _OUTPUT_CAP:
        return text
    head = text[: _OUTPUT_CAP // 2]
    tail = text[-_OUTPUT_CAP // 2:]
    return f"{head}\n[...output truncated; {len(text) - _OUTPUT_CAP} chars elided...]\n{tail}"


def _summarize_pytest(stdout: str, returncode: int) -> str:
    """Extract a one-line summary from pytest's output, or fall back."""
    # Pytest prints a final line like ``=== 5 passed, 2 failed in 1.20s ===``.
    m = re.search(r"=+\s+([^=]+?in\s+[\d.]+s)\s+=+\s*$", stdout.rstrip())
    if m:
        return m.group(1).strip()
    return f"(pytest exited {returncode}; no summary line found)"


@tool(timeout=_RUN_TIMEOUT + 10)
def run_tests(path: str = "", target: str = "", extra_args: str = "") -> str:
    """Run the project's test suite.

    Auto-detects pytest (pyproject.toml/pytest.ini) or npm test (package.json).
    ``target`` narrows scope: for pytest a file/node id, for npm a script
    suffix passed via ``--``. ``extra_args`` adds raw CLI args.

    Returns a header with the runner + summary line, then the captured
    stdout/stderr (capped at 20k chars). Use this instead of run_powershell
    when you want a structured summary.
    """
    repo = _resolve_repo(path or None)
    if isinstance(repo, str):
        return repo
    runner = _detect_runner(repo)
    if runner == "unknown":
        return (
            "[error] could not detect test runner (no pyproject.toml/pytest.ini/"
            "package.json with `test` script). Use run_powershell directly."
        )

    if runner == "pytest":
        cmd = ["pytest"]
        if target:
            cmd.append(target)
        if extra_args:
            cmd.extend(extra_args.split())
        # Quiet by default — pytest's verbose output dominates the agent's
        # context. The agent can pass `-v` via extra_args if it wants more.
        if not any(a in extra_args.split() for a in ("-v", "-vv", "--verbose")) if extra_args else True:
            cmd.append("-q")
    elif runner == "npm":
        cmd = ["npm", "test"]
        if extra_args or target:
            cmd.append("--")
            if target:
                cmd.append(target)
            if extra_args:
                cmd.extend(extra_args.split())
    else:
        return f"[error] unsupported runner: {runner!r}"

    bin_path = shutil.which(cmd[0])
    if bin_path is None:
        return f"[error] {cmd[0]!r} not on PATH"
    cmd[0] = bin_path

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_RUN_TIMEOUT,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return f"[error] {runner} timed out after {_RUN_TIMEOUT}s"
    except Exception as e:
        return f"[error] {runner} invocation failed: {type(e).__name__}: {e}"

    summary: str
    if runner == "pytest":
        summary = _summarize_pytest(proc.stdout, proc.returncode)
    else:
        summary = f"npm test exited {proc.returncode}"

    out_blocks: list[str] = [
        f"runner: {runner}",
        f"exit: {proc.returncode}",
        f"summary: {summary}",
    ]
    if proc.stdout:
        out_blocks.append("--- stdout ---")
        out_blocks.append(_cap(proc.stdout))
    if proc.stderr:
        out_blocks.append("--- stderr ---")
        out_blocks.append(_cap(proc.stderr))
    return "\n".join(out_blocks)
