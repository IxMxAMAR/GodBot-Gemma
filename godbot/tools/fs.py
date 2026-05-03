from __future__ import annotations
import fnmatch
import os
import re
from pathlib import Path

from godbot.core.registry import tool


@tool()
def read_file(path: str, max_lines: int = 2000, start_line: int = 1) -> str:
    """Read a UTF-8 text file. Returns numbered lines starting from start_line."""
    p = Path(path)
    if not p.exists():
        return f"[error] file not found: {path}"
    if not p.is_file():
        return f"[error] not a file: {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    end = min(len(lines), start_line - 1 + max_lines)
    out = []
    for i in range(start_line - 1, end):
        out.append(f"{i+1}\t{lines[i]}")
    if end < len(lines):
        out.append(f"... [{len(lines) - end} more lines]")
    return "\n".join(out)


@tool()
def list_dir(path: str = ".") -> str:
    """List the contents of a directory. Suffixes directories with '/'."""
    p = Path(path)
    if not p.exists():
        return f"[error] not found: {path}"
    if not p.is_dir():
        return f"[error] not a directory: {path}"
    out = []
    for child in sorted(p.iterdir()):
        out.append(child.name + ("/" if child.is_dir() else ""))
    return "\n".join(out) if out else "(empty)"


@tool()
def glob(pattern: str, root: str = ".") -> str:
    """Recursively match files against a glob pattern, e.g. '**/*.py'."""
    p = Path(root)
    if not p.exists():
        return f"[error] root not found: {root}"
    matches = sorted(str(m.relative_to(p)) for m in p.glob(pattern) if m.is_file())
    return "\n".join(matches) if matches else "(no matches)"


@tool()
def grep(pattern: str, root: str = ".", glob_filter: str = "*") -> str:
    """Search for a regex pattern across files under root. Returns 'path:line:match'."""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"[error] bad regex: {e}"
    p = Path(root)
    if not p.exists():
        return f"[error] root not found: {root}"
    out = []
    for f in p.rglob(glob_filter):
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                rel = f.relative_to(p)
                out.append(f"{rel}:{i}:{line.rstrip()}")
                if len(out) > 500:
                    out.append("... [truncated at 500 matches]")
                    return "\n".join(out)
    return "\n".join(out) if out else "(no matches)"
