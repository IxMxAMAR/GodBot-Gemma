from __future__ import annotations
import fnmatch
import os
import re
from pathlib import Path

from godbot.core.registry import tool
from godbot.core.workspace import current_workspace, WorkspaceEscape


def _confine_or_error(path: str, *, check_links: bool = True) -> tuple[str, str | None]:
    """Apply workspace confinement.

    Returns (resolved_path, error_message). When check_links=True and the path
    points to an existing multi-link file, refuse — hardlinks can target outside
    the workspace and bypass confinement.

    On success: (resolved_path, None).
    On escape: ("", error_string).
    No workspace active: (path, None) - passthrough.
    """
    ws = current_workspace()
    if ws is None:
        return path, None
    try:
        resolved = ws.confine(path)
    except WorkspaceEscape as e:
        return "", f"[error] sandbox: {e}"
    if check_links:
        try:
            st = os.stat(resolved)
            if st.st_nlink > 1:
                return "", (
                    f"[error] sandbox: refusing op on multi-link file "
                    f"{resolved} (nlink={st.st_nlink}); hard links can target outside the workspace"
                )
        except FileNotFoundError:
            pass  # creating a new file is fine
        except OSError:
            pass  # symlink resolution issues already caught by confine
    return str(resolved), None


@tool()
def read_file(path: str, max_lines: int = 2000, start_line: int = 1) -> str:
    """Read a UTF-8 text file. Returns numbered lines starting from start_line."""
    path, err = _confine_or_error(path, check_links=True)
    if err:
        return err
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
    path, err = _confine_or_error(path, check_links=False)
    if err:
        return err
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
    root, err = _confine_or_error(root, check_links=False)
    if err:
        return err
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
    root, err = _confine_or_error(root, check_links=False)
    if err:
        return err
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


@tool(dangerous=True)
def write_file(path: str, content: str) -> str:
    """Write UTF-8 content to a file (creates parent dirs)."""
    path, err = _confine_or_error(path, check_links=True)
    if err:
        return err
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"ok: wrote {len(content)} chars to {p}"


@tool(dangerous=True)
def edit_file(path: str, old: str, new: str) -> str:
    """Replace one unique occurrence of `old` with `new` in a file."""
    path, err = _confine_or_error(path, check_links=True)
    if err:
        return err
    p = Path(path)
    if not p.exists():
        return f"[error] file not found: {path}"
    text = p.read_text(encoding="utf-8")
    occurrences = text.count(old)
    if occurrences == 0:
        return f"[error] old string not found in {path}"
    if occurrences > 1:
        return f"[error] old string not unique ({occurrences} matches) in {path}"
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"ok: edited {p}"
