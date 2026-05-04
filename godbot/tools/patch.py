"""Unified-diff patch applier (sub-project 22).

A more flexible alternative to ``edit_file`` for non-trivial edits: the
agent emits a standard unified diff (``@@ -L,N +L,N @@`` hunks with
``-`` / ``+`` / `` `` lines), and we apply it against the workspace
file. Multiple hunks per file and multiple files per patch are
supported in one call.

Why bother when ``edit_file`` exists? Two reasons:
- ``edit_file`` requires a unique ``old`` substring across the file,
  which fails when a snippet repeats (e.g. import lines).
- A unified diff is the lingua franca of code edits — every git tool
  speaks it, and many models emit it natively when asked for "the
  patch".

The applier is **strict by default**: hunk context lines must match
the file exactly. ``fuzzy=True`` enables a fallback that searches the
file for the hunk's context block when the line numbers are off (common
when the model prepares a patch from a slightly stale view).

All edits are workspace-confined and gated — same machinery as
``write_file`` / ``edit_file``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from godbot.core.registry import tool
from godbot.core.workspace import current_workspace, WorkspaceEscape


# --- diff parsing -------------------------------------------------------


@dataclass
class Hunk:
    """One ``@@ -L,N +L,N @@`` block plus its line bodies."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FileDiff:
    """Diff for a single target file."""

    path: str
    hunks: list[Hunk] = field(default_factory=list)


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(text: str) -> list[FileDiff]:
    """Parse a unified-diff blob into per-file diff records.

    Tolerates the common encoding variations: ``a/`` and ``b/`` prefixes
    on path headers, missing-newline-at-end markers (``\\ No newline at
    end of file``), and trailing whitespace. Raises ``ValueError`` if
    the structure is unrecognisable.
    """
    files: list[FileDiff] = []
    current: Optional[FileDiff] = None
    current_hunk: Optional[Hunk] = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            # `--- a/path` or `--- /dev/null`. The companion `+++ b/path` on
            # the next line carries the actual target path.
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise ValueError(f"line {i+1}: expected `+++` after `---`")
            target = lines[i + 1][4:].strip()
            # Strip the conventional `b/` (or `a/`) prefix so we operate
            # on a workspace-relative path.
            if target.startswith("b/"):
                target = target[2:]
            elif target.startswith("a/"):
                target = target[2:]
            current = FileDiff(path=target)
            files.append(current)
            current_hunk = None
            i += 2
            continue
        m = _HUNK_HEADER.match(line)
        if m:
            if current is None:
                raise ValueError(f"line {i+1}: hunk header before file header")
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) else 1
            current_hunk = Hunk(
                old_start=old_start, old_count=old_count,
                new_start=new_start, new_count=new_count,
            )
            current.hunks.append(current_hunk)
            i += 1
            continue
        if current_hunk is not None and (
            line.startswith(" ") or line.startswith("-") or line.startswith("+")
            or line == ""  # blank context line
        ):
            current_hunk.lines.append(line)
            i += 1
            continue
        if line.startswith("\\ No newline at end of file"):
            i += 1
            continue
        # Anything else (file-mode markers, "diff --git", etc.) we silently skip.
        i += 1
    if not files:
        raise ValueError("no `--- /+++` file headers found in patch")
    return files


# --- application --------------------------------------------------------


def _apply_hunks(file_text: str, hunks: list[Hunk], *, fuzzy: bool = False) -> str:
    """Apply ``hunks`` to ``file_text``. Raises ``ValueError`` on mismatch.

    Strict mode: each hunk's context lines must match exactly at the
    line number declared in the hunk header. Fuzzy mode searches the
    file for the hunk's leading context block if the strict location
    fails.
    """
    src_lines = file_text.splitlines(keepends=False)
    # Sort hunks by old_start descending so applying from bottom-up keeps
    # earlier line numbers valid.
    out_lines = list(src_lines)
    for hunk in sorted(hunks, key=lambda h: h.old_start, reverse=True):
        anchor = _find_anchor(out_lines, hunk, fuzzy=fuzzy)
        if anchor is None:
            ctx_preview = "\n".join(
                ln for ln in hunk.lines[:5] if not ln.startswith("+")
            )
            raise ValueError(
                f"hunk at -{hunk.old_start},{hunk.old_count} did not match "
                f"file content (context preview: {ctx_preview!r})"
            )
        # Build the replacement region and splice it in.
        new_block: list[str] = []
        consumed = 0
        idx = anchor
        for diff_line in hunk.lines:
            if diff_line.startswith(" ") or diff_line == "":
                # Context line: must match what's at idx, then keep it.
                src_line = out_lines[idx] if idx < len(out_lines) else ""
                expected = diff_line[1:] if diff_line.startswith(" ") else diff_line
                if src_line != expected:
                    raise ValueError(
                        f"context mismatch at line {idx+1}: "
                        f"expected {expected!r}, got {src_line!r}"
                    )
                new_block.append(src_line)
                idx += 1
                consumed += 1
            elif diff_line.startswith("-"):
                expected = diff_line[1:]
                src_line = out_lines[idx] if idx < len(out_lines) else ""
                if src_line != expected:
                    raise ValueError(
                        f"removal mismatch at line {idx+1}: "
                        f"expected {expected!r}, got {src_line!r}"
                    )
                # Drop this line — don't append to new_block.
                idx += 1
                consumed += 1
            elif diff_line.startswith("+"):
                new_block.append(diff_line[1:])
        # Splice: replace [anchor, anchor+consumed) with new_block.
        out_lines[anchor:anchor + consumed] = new_block
    # Preserve trailing newline if the original had one.
    result = "\n".join(out_lines)
    if file_text.endswith("\n"):
        result += "\n"
    return result


def _find_anchor(src_lines: list[str], hunk: Hunk, *, fuzzy: bool) -> Optional[int]:
    """Return the 0-based index in ``src_lines`` where the hunk starts."""
    # Strict path: try the line number the hunk declared, minus 1.
    declared = max(0, hunk.old_start - 1)
    if _matches_at(src_lines, declared, hunk):
        return declared
    if not fuzzy:
        return None
    # Fuzzy path: scan for any position where the hunk's leading context
    # matches. Build the context-only prefix of the hunk so we have an
    # anchor we can search for verbatim.
    ctx_prefix: list[str] = []
    for diff_line in hunk.lines:
        if diff_line.startswith(" ") or diff_line == "":
            ctx_prefix.append(diff_line[1:] if diff_line.startswith(" ") else diff_line)
        elif diff_line.startswith("-"):
            ctx_prefix.append(diff_line[1:])
        else:
            break  # plus lines = stop building the prefix
    if not ctx_prefix:
        return None
    L = len(ctx_prefix)
    for i in range(len(src_lines) - L + 1):
        if src_lines[i:i + L] == ctx_prefix and _matches_at(src_lines, i, hunk):
            return i
    return None


def _matches_at(src_lines: list[str], idx: int, hunk: Hunk) -> bool:
    """Quick check that the hunk's removal+context lines match src starting at idx."""
    cursor = idx
    for diff_line in hunk.lines:
        if diff_line.startswith("+"):
            continue
        if cursor >= len(src_lines):
            return False
        expected = diff_line[1:] if diff_line and diff_line[0] in " -" else diff_line
        if src_lines[cursor] != expected:
            return False
        cursor += 1
    return True


# --- tool ---------------------------------------------------------------


def _confine_or_error(path: str) -> tuple[str, str | None]:
    """Workspace confinement helper (mirrors the one in tools/fs.py)."""
    ws = current_workspace()
    if ws is None:
        return path, None
    try:
        return str(ws.confine(path)), None
    except WorkspaceEscape as e:
        return "", f"[error] sandbox: {e}"


@tool(dangerous=True)
def apply_patch(patch: str, fuzzy: bool = False) -> str:
    """Apply a unified-diff patch to one or more workspace files.

    Patch format is the standard ``--- a/path``/``+++ b/path`` plus
    ``@@ -L,N +L,N @@`` hunks. ``fuzzy=true`` lets the applier search
    for the hunk's context block when the declared line numbers are
    off — useful when the model prepared the patch from a slightly
    stale view of the file.

    Returns a per-file summary on success, or an error string on the
    first failure (no partial application — the patch is all-or-nothing).
    Use this instead of ``edit_file`` when you have multi-line changes,
    multiple hunks, or non-unique ``old`` strings.
    """
    try:
        diffs = parse_patch(patch)
    except ValueError as e:
        return f"[error] parse: {e}"

    # Compute the planned writes first; if any file fails, abort before
    # touching disk so the patch stays atomic.
    planned: list[tuple[Path, str]] = []
    for fd in diffs:
        confined, err = _confine_or_error(fd.path)
        if err:
            return err
        p = Path(confined)
        if not p.exists():
            return f"[error] file not found: {fd.path}"
        if not p.is_file():
            return f"[error] not a regular file: {fd.path}"
        try:
            current = p.read_text(encoding="utf-8")
        except Exception as e:
            return f"[error] read {fd.path}: {type(e).__name__}: {e}"
        try:
            updated = _apply_hunks(current, fd.hunks, fuzzy=fuzzy)
        except ValueError as e:
            return f"[error] {fd.path}: {e}"
        planned.append((p, updated))

    # Commit phase.
    summary: list[str] = []
    for p, content in planned:
        p.write_text(content, encoding="utf-8")
        summary.append(f"patched {p}")
    return "ok: " + "; ".join(summary)
