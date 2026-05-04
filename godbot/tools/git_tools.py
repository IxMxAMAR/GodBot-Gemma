"""Read-only git tools (sub-project 15).

These run ``git`` as a subprocess and surface stdout to the agent. All four
are non-dangerous — they only read repository state. For mutating commands
(``commit``, ``push``, ``checkout``) the agent should fall through to
``run_powershell``/``run_bash`` so the existing gate mechanism takes effect.

Each tool runs against the active workspace's root by default; absolute
paths or relative paths are accepted but always confined when a workspace
is active. Callers without an active workspace get an explicit error.

All git invocations cap output at 16k bytes — long ``git log``s on large
repos would otherwise dominate the agent's context window. Truncation is
flagged inline with ``[...output truncated...]``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from godbot.core.registry import tool
from godbot.core.workspace import current_workspace


_OUTPUT_CAP = 16_000


def _resolve_repo_root(path: Optional[str] = None) -> Path | str:
    """Return the directory to run git in, or an error string.

    Resolution order: explicit ``path`` → active workspace → error. Relative
    paths resolve against the workspace when one is active.
    """
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


def _run_git(args: list[str], cwd: Path) -> str:
    """Invoke git, return stdout (or stderr on non-zero exit), capped."""
    git_bin = shutil.which("git")
    if git_bin is None:
        return "[error] git not found on PATH"
    try:
        proc = subprocess.run(
            [git_bin, *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return f"[error] git {args[0]} timed out after 30s"
    except Exception as e:
        return f"[error] git invocation failed: {type(e).__name__}: {e}"
    if proc.returncode != 0:
        body = (proc.stderr or proc.stdout or "").strip()
        return f"[error] git exit {proc.returncode}: {body[:_OUTPUT_CAP]}"
    out = proc.stdout or ""
    if len(out) > _OUTPUT_CAP:
        head = out[: _OUTPUT_CAP // 2]
        tail = out[-_OUTPUT_CAP // 2:]
        out = f"{head}\n[...output truncated; {len(out) - _OUTPUT_CAP} chars elided...]\n{tail}"
    return out or "(no output)"


@tool()
def git_status(path: str = "") -> str:
    """Show the working-tree status (porcelain v1) of a git repo.

    Equivalent to `git status --short --branch`. Capped at 16k of output.
    Use this to see staged/unstaged/untracked files at a glance.
    """
    repo = _resolve_repo_root(path or None)
    if isinstance(repo, str):
        return repo
    return _run_git(["status", "--short", "--branch"], cwd=repo)


@tool()
def git_diff(path: str = "", staged: bool = False, target: str = "") -> str:
    """Show diff in a git repo.

    By default, unstaged working-tree diff. ``staged=true`` shows the
    staging area against HEAD. ``target`` lets you diff a specific path
    or a ref like ``HEAD~1`` or ``main..HEAD``. Capped at 16k chars.
    """
    repo = _resolve_repo_root(path or None)
    if isinstance(repo, str):
        return repo
    args = ["diff"]
    if staged:
        args.append("--staged")
    if target:
        args.append(target)
    return _run_git(args, cwd=repo)


@tool()
def git_log(path: str = "", limit: int = 10) -> str:
    """Show the most recent commits in a git repo.

    One line per commit: short SHA, author date, subject line. ``limit``
    caps the count (default 10). Capped at 16k chars regardless of limit.
    """
    repo = _resolve_repo_root(path or None)
    if isinstance(repo, str):
        return repo
    if limit <= 0:
        limit = 10
    return _run_git(
        ["log", f"-{limit}", "--pretty=format:%h %ad %s", "--date=short"],
        cwd=repo,
    )


@tool()
def git_branch(path: str = "") -> str:
    """List local branches and mark the current one with ``*``.

    Useful when the agent needs to know "am I on master or a feature
    branch?" before suggesting changes.
    """
    repo = _resolve_repo_root(path or None)
    if isinstance(repo, str):
        return repo
    return _run_git(["branch", "--list"], cwd=repo)
