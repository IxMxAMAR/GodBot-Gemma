"""Workspace context aggregator (sub-project 30).

A single tool the agent can call to get an at-a-glance briefing on the
active workspace. Bundles together calls that previously took 3-4
separate tool-loop turns:

- ``project_summary`` — language, dependencies, README excerpt
- ``git_status`` — branch, dirty files (when the dir is a git repo)
- top-level directory listing (capped)
- top 5 recent commit subjects (when git is available)

This is read-only and non-dangerous. The output is plain text formatted
for the agent's context window — short headings + body, capped at 8 KB
total so it never dominates the prompt.

Use this on the agent's first turn after a workspace opens to anchor
context. After that, prefer the focused tools (``read_file``,
``search_workspace``, ``git_diff``) for specific lookups.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from godbot.core.registry import tool
from godbot.core.workspace import current_workspace


_OUTPUT_CAP = 8_000
_DIR_LIST_CAP = 30
_LOG_LIMIT = 5


def _section(heading: str, body: str) -> str:
    body = body.strip()
    if not body:
        return ""
    return f"=== {heading} ===\n{body}"


def _git_branch_dirty(repo: Path) -> str:
    """Return one-line "branch (dirty/clean) — N changes" or "" if not a repo."""
    git_bin = shutil.which("git")
    if git_bin is None:
        return ""
    if not (repo / ".git").exists():
        return ""
    try:
        branch = subprocess.run(
            [git_bin, "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5,
        )
        if branch.returncode != 0:
            return ""
        status = subprocess.run(
            [git_bin, "status", "--porcelain"],
            cwd=str(repo), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
    except Exception:
        return ""
    branch_name = branch.stdout.strip() or "?"
    lines = [ln for ln in (status.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return f"on `{branch_name}` (clean)"
    return f"on `{branch_name}` (dirty) — {len(lines)} changes:\n" + "\n".join(lines[:20])


def _git_recent_log(repo: Path) -> str:
    git_bin = shutil.which("git")
    if git_bin is None or not (repo / ".git").exists():
        return ""
    try:
        proc = subprocess.run(
            [git_bin, "log", f"-{_LOG_LIMIT}", "--pretty=format:%h %ad %s", "--date=short"],
            cwd=str(repo), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()
    except Exception:
        return ""


def _toplevel_listing(repo: Path) -> str:
    """List top-level entries, dirs first, capped at _DIR_LIST_CAP."""
    try:
        entries = sorted(repo.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        return f"(could not list: {e})"
    out: list[str] = []
    for p in entries[:_DIR_LIST_CAP]:
        if p.name.startswith("."):
            # Hidden files contribute clutter; surface only the well-known ones.
            if p.name not in {".gitignore", ".github", ".env.example"}:
                continue
        out.append(f"{p.name}/" if p.is_dir() else p.name)
    if len(entries) > _DIR_LIST_CAP:
        out.append(f"... ({len(entries) - _DIR_LIST_CAP} more)")
    return "\n".join(out)


@tool()
def tool_help(name: str = "") -> str:
    """Introspect the tool catalog (sub-project 37).

    With ``name=""`` (default), returns one line per registered tool:
    ``name [DANGEROUS] — first line of docstring``. Useful when the
    agent wants to re-check what's available without re-reading the
    full system prompt.

    With a specific tool name, returns the full description plus the
    JSON schema of its arguments. ``[error]`` lines are returned for
    unknown names.

    Always non-dangerous. Read-only.
    """
    from godbot.core.registry import DEFAULT
    import json as _json

    if not name:
        lines = ["Available tools:"]
        for spec in sorted(DEFAULT.all(), key=lambda t: t.name):
            flag = " [DANGEROUS]" if spec.dangerous else ""
            lines.append(f"  {spec.name}{flag} — {spec.description}")
        return "\n".join(lines)

    spec = DEFAULT.spec(name)
    if spec is None:
        return f"[error] unknown tool {name!r}"
    flag = " [DANGEROUS]" if spec.dangerous else ""
    schema_pretty = _json.dumps(spec.schema, indent=2)
    return (
        f"{spec.name}{flag}\n"
        f"description: {spec.description}\n"
        f"timeout: {spec.timeout}s\n"
        f"args schema:\n{schema_pretty}"
    )


@tool()
def workspace_context() -> str:
    """At-a-glance summary of the active workspace (sub-project 30).

    Bundles project type/deps (via project_summary), current git branch
    + dirty state, recent commit subjects, and a top-level dir listing
    in one tool call. Use on the first turn after a workspace opens to
    anchor context cheaply; prefer focused tools for follow-up lookups.

    Returns plain text capped at 8 KB. Empty sections are omitted.
    Outside an active workspace returns an explicit error message.
    """
    ws = current_workspace()
    if ws is None:
        return "(no active workspace; open a folder first)"
    repo = Path(ws.root)

    sections: list[str] = []

    # Project type + dependencies. Reuse the existing tool's logic via
    # direct call so we don't pay double for the heuristic walk.
    try:
        from godbot.tools.project import project_summary
        sections.append(_section("project", project_summary(root=str(repo))))
    except Exception as e:
        sections.append(_section("project", f"(project_summary failed: {e})"))

    # Git state.
    git_state = _git_branch_dirty(repo)
    if git_state:
        sections.append(_section("git", git_state))
    git_log = _git_recent_log(repo)
    if git_log:
        sections.append(_section("recent commits", git_log))

    # Top-level directory listing.
    sections.append(_section("top level", _toplevel_listing(repo)))

    text = "\n\n".join(s for s in sections if s)
    if len(text) > _OUTPUT_CAP:
        text = text[:_OUTPUT_CAP] + "\n... [truncated]"
    return text
