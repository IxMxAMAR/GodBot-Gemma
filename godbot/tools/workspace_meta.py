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
def which(name: str) -> str:
    """Locate a CLI binary on PATH (sub-project 53).

    Returns ``found: <full path>`` when ``name`` resolves, or
    ``not found: <name>`` otherwise. Useful when the agent wants to
    decide between two paths (e.g. "use pytest if installed, else
    fall back to run_python on the test file") or to surface a
    helpful "install X first" hint for the user.

    Non-dangerous; pure read on PATH. Resolves the *first* match —
    same behavior as POSIX which / Windows where.exe.
    """
    import shutil as _shutil
    if not name:
        return "[error] name required"
    found = _shutil.which(name)
    if found is None:
        return f"not found: {name}"
    return f"found: {found}"


@tool()
def check_python_syntax(path: str = "", code: str = "") -> str:
    """Validate that a file or snippet is syntactically valid Python (sub-project 50).

    Pass exactly one of ``path`` (workspace-confined .py file) or ``code``
    (inline source). Returns ``"ok: <label> parses cleanly"`` on success
    or ``"[error] line L col C: <SyntaxError msg>"`` on failure.

    Cheaper than ``run_python`` for "does this even parse?" sanity
    checks — non-dangerous, no subprocess, no side effects. Useful
    after applying a patch to confirm the file still compiles.
    """
    import ast
    from pathlib import Path as _Path

    if not path and not code:
        return "[error] supply either `path` or `code`"
    if path and code:
        return "[error] supply only one of `path` or `code`"

    body: str
    label: str
    if path:
        ws = current_workspace()
        p = _Path(path)
        if not p.is_absolute() and ws is not None:
            p = ws.root / p
        try:
            p = p.resolve()
        except OSError as e:
            return f"[error] cannot resolve {path!r}: {e}"
        if ws is not None:
            try:
                p.relative_to(ws.root)
            except ValueError:
                return f"[error] {p} is outside the active workspace"
        if not p.is_file():
            return f"[error] not a file: {p}"
        if p.suffix != ".py":
            return f"[error] not a Python file: {p}"
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[error] read failed: {type(e).__name__}: {e}"
        label = str(p)
    else:
        body = code
        label = "<inline>"

    try:
        ast.parse(body, filename=label)
    except SyntaxError as e:
        return f"[error] {label}: line {e.lineno} col {e.offset}: {e.msg}"
    return f"ok: {label} parses cleanly"


def _resolve_workspace_file(path: str) -> tuple[str | None, str | None]:
    """Workspace-confine and validate a file path. Returns (str(path), None)
    or (None, error_msg)."""
    from pathlib import Path as _Path
    if not path:
        return None, "[error] path required"
    ws = current_workspace()
    p = _Path(path)
    if not p.is_absolute() and ws is not None:
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return None, f"[error] cannot resolve {path!r}: {e}"
    if ws is not None:
        try:
            p.relative_to(ws.root)
        except ValueError:
            return None, f"[error] {p} is outside the active workspace"
    if not p.is_file():
        return None, f"[error] not a file: {p}"
    return str(p), None


@tool()
def directory_size(path: str = ".") -> str:
    """Report total disk usage of a directory tree (sub-project 56).

    Walks ``path`` recursively, sums file sizes, and reports the top
    10 largest files plus the grand total in human-readable units.
    Workspace-confined: ``path`` defaults to the workspace root.

    Useful for "what's hogging space here?" reconnaissance — point at
    ``.venv`` or ``node_modules`` to see the heavy hitters.
    """
    import os as _os
    from pathlib import Path as _Path

    ws = current_workspace()
    p = _Path(path)
    if not p.is_absolute() and ws is not None:
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"[error] cannot resolve {path!r}: {e}"
    if ws is not None:
        try:
            p.relative_to(ws.root)
        except ValueError:
            return f"[error] {p} is outside the active workspace"
    if not p.is_dir():
        return f"[error] not a directory: {p}"

    total_bytes = 0
    file_count = 0
    largest: list[tuple[int, _Path]] = []
    skipped_dirs = {".git", "__pycache__", ".pytest_cache"}
    try:
        for root, dirs, files in _os.walk(p):
            dirs[:] = [d for d in dirs if d not in skipped_dirs]
            for fname in files:
                fp = _Path(root) / fname
                try:
                    sz = fp.stat().st_size
                except OSError:
                    continue
                total_bytes += sz
                file_count += 1
                # Maintain top-10 by size with simple insert-and-trim.
                largest.append((sz, fp))
                if len(largest) > 50:
                    largest.sort(key=lambda x: x[0], reverse=True)
                    largest = largest[:10]
    except OSError as e:
        return f"[error] walk failed: {e}"

    largest.sort(key=lambda x: x[0], reverse=True)
    largest = largest[:10]

    def _human(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024:
                return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
            n /= 1024  # type: ignore[assignment]
        return f"{n:.1f} PB"

    out = [
        f"path: {p}",
        f"total: {_human(total_bytes)} across {file_count} file(s)",
    ]
    if largest:
        out.append("top 10 largest:")
        for sz, fp in largest:
            try:
                rel = fp.relative_to(p)
            except ValueError:
                rel = fp
            out.append(f"  {_human(sz)}  {rel}")
    return "\n".join(out)


@tool()
def head(path: str, lines: int = 50) -> str:
    """Return the first ``lines`` lines of a workspace-confined text file.

    Cheaper than ``read_file`` when you only need a peek — typical use
    case is checking the top of a log to see whether a process started
    cleanly. ``lines`` is capped at 5000 to keep responses bounded.
    """
    from pathlib import Path as _Path
    p_str, err = _resolve_workspace_file(path)
    if err:
        return err
    n = max(0, min(int(lines), 5000))
    try:
        with open(p_str, "r", encoding="utf-8", errors="replace") as f:
            out: list[str] = []
            for i, line in enumerate(f):
                if i >= n:
                    break
                out.append(f"{i + 1}\t{line.rstrip()}")
        return "\n".join(out) if out else "(empty file)"
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"


@tool()
def tail(path: str, lines: int = 50) -> str:
    """Return the last ``lines`` lines of a workspace-confined text file.

    Useful for checking the end of a log or recently-appended events.
    Reads the whole file so for huge files this isn't free; if you
    need true seek-from-end behavior, use ``run_powershell`` with
    ``Get-Content -Tail`` (Windows) or ``run_bash`` with ``tail`` (Unix).
    """
    p_str, err = _resolve_workspace_file(path)
    if err:
        return err
    n = max(0, min(int(lines), 5000))
    try:
        with open(p_str, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"
    if not all_lines:
        return "(empty file)"
    selected = all_lines[-n:]
    start = len(all_lines) - len(selected) + 1
    return "\n".join(
        f"{start + i}\t{line.rstrip()}" for i, line in enumerate(selected)
    )


@tool()
def list_functions(path: str) -> str:
    """List every top-level + class-method definition in a Python file (sub-project 54).

    Output format, one entry per line:

      <line>: <kind> <qualified_name>(<sig>)

    where ``kind`` is ``def`` / ``async def`` / ``class``. Methods are
    qualified with their containing class (``Foo.bar``). Useful for
    "what's in this file?" reconnaissance before extract_function or
    targeted edits. Workspace-confined; .py files only.
    """
    import ast
    from pathlib import Path as _Path

    if not path:
        return "[error] path required"
    ws = current_workspace()
    p = _Path(path)
    if not p.is_absolute() and ws is not None:
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"[error] cannot resolve {path!r}: {e}"
    if ws is not None:
        try:
            p.relative_to(ws.root)
        except ValueError:
            return f"[error] {p} is outside the active workspace"
    if not p.is_file():
        return f"[error] not a file: {p}"
    if p.suffix != ".py":
        return f"[error] not a Python file: {p}"
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"
    try:
        tree = ast.parse(source, filename=str(p))
    except SyntaxError as e:
        return f"[error] syntax: line {e.lineno} col {e.offset}: {e.msg}"

    entries: list[tuple[int, str]] = []

    def _sig_for(node) -> str:
        try:
            return f"({ast.unparse(node.args)})"
        except Exception:
            return "(...)"

    def visit(node, qual: str = ""):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            name = f"{qual}.{node.name}" if qual else node.name
            entries.append((node.lineno, f"{kind} {name}{_sig_for(node)}"))
        elif isinstance(node, ast.ClassDef):
            entries.append((node.lineno, f"class {qual + '.' if qual else ''}{node.name}"))
            for child in node.body:
                visit(child, qual=node.name if not qual else f"{qual}.{node.name}")
            return
        for child in ast.iter_child_nodes(node):
            visit(child, qual=qual)

    visit(tree)
    entries.sort(key=lambda e: e[0])
    if not entries:
        return f"{p}: (no top-level definitions)"
    out = [f"file: {p}"]
    out.extend(f"  L{lineno}: {desc}" for lineno, desc in entries)
    return "\n".join(out)


@tool()
def extract_function(path: str, name: str) -> str:
    """Extract a Python function or class definition by name (sub-project 48).

    Parses ``path`` with ``ast`` and returns the full source of the
    first top-level (or method) definition named ``name``, including
    decorators and docstring. Searches walks recursively so a method
    nested in a class can be extracted by its bare name.

    Output:

      file:line: <function or class signature>
      <full source slice>

    Returns ``[error] not found: <name>`` if no definition matches.
    Workspace-confined; ``.py`` files only.
    """
    import ast
    from pathlib import Path as _Path

    if not path or not name:
        return "[error] both path and name are required"
    ws = current_workspace()
    p = _Path(path)
    if not p.is_absolute() and ws is not None:
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"[error] cannot resolve {path!r}: {e}"
    if ws is not None:
        try:
            p.relative_to(ws.root)
        except ValueError:
            return f"[error] {p} is outside the active workspace"
    if not p.is_file():
        return f"[error] not a file: {p}"
    if p.suffix != ".py":
        return f"[error] not a Python file: {p}"
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"
    try:
        tree = ast.parse(source, filename=str(p))
    except SyntaxError as e:
        return f"[error] syntax: line {e.lineno} col {e.offset}: {e.msg}"

    matches: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                matches.append(node)
                break  # first hit wins
    if not matches:
        return f"[error] not found: {name}"
    node = matches[0]
    # Capture decorators above the def line.
    decorator_start = (
        min((d.lineno for d in node.decorator_list), default=node.lineno)
        if getattr(node, "decorator_list", None) else node.lineno
    )
    end_line = getattr(node, "end_lineno", node.lineno)
    src_lines = source.splitlines()
    body = "\n".join(src_lines[decorator_start - 1:end_line])
    sig = src_lines[node.lineno - 1].strip()
    return f"{p}:{decorator_start}: {sig}\n{body}"


@tool()
def find_imports(path: str) -> str:
    """List the imports in a Python file (sub-project 47).

    Parses the file with ``ast`` and reports two sections:

      direct: <one per line — `import foo` / `import foo.bar`>
      from:   <one per line — `from x import y, z`>

    Useful for "what does this file depend on?" reasoning before
    refactoring or tracing dataflow. Workspace-confined; only Python
    files (``.py``). Returns ``[error] ...`` on parse failure with
    line:col from the SyntaxError.
    """
    import ast
    from pathlib import Path as _Path

    if not path:
        return "[error] path required"
    ws = current_workspace()
    p = _Path(path)
    if not p.is_absolute() and ws is not None:
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"[error] cannot resolve {path!r}: {e}"
    if ws is not None:
        try:
            p.relative_to(ws.root)
        except ValueError:
            return f"[error] {p} is outside the active workspace"
    if not p.is_file():
        return f"[error] not a file: {p}"
    if p.suffix != ".py":
        return f"[error] not a Python file: {p}"
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"
    try:
        tree = ast.parse(source, filename=str(p))
    except SyntaxError as e:
        return f"[error] syntax: line {e.lineno} col {e.offset}: {e.msg}"

    direct: list[str] = []
    from_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                rendered = alias.name + (f" as {alias.asname}" if alias.asname else "")
                direct.append(f"import {rendered}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            dots = "." * (node.level or 0)
            target = f"{dots}{mod}"
            names = ", ".join(
                a.name + (f" as {a.asname}" if a.asname else "") for a in node.names
            )
            from_imports.append(f"from {target} import {names}")

    if not direct and not from_imports:
        return f"{p}: (no imports)"
    parts = [f"file: {p}"]
    if direct:
        parts.append(f"direct ({len(direct)}):")
        parts.extend(f"  {ln}" for ln in direct)
    if from_imports:
        parts.append(f"from ({len(from_imports)}):")
        parts.extend(f"  {ln}" for ln in from_imports)
    return "\n".join(parts)


@tool()
def compare_files(path_a: str, path_b: str, max_lines: int = 200) -> str:
    """Show a unified diff between two workspace files (sub-project 46).

    Both paths are workspace-confined when a workspace is active.
    Output is standard unified-diff format with up to ``max_lines``
    of context-and-change lines (capped to keep the agent's context
    manageable on big diffs).

    Returns a one-line "files identical" message when the contents
    match exactly. Non-dangerous; read-only.
    """
    import difflib
    from pathlib import Path as _Path

    if not path_a or not path_b:
        return "[error] both path_a and path_b are required"
    ws = current_workspace()

    def resolve(p: str) -> tuple[_Path | None, str | None]:
        path = _Path(p)
        if not path.is_absolute() and ws is not None:
            path = ws.root / path
        try:
            path = path.resolve()
        except OSError as e:
            return None, f"[error] cannot resolve {p!r}: {e}"
        if ws is not None:
            try:
                path.relative_to(ws.root)
            except ValueError:
                return None, f"[error] {path} is outside the active workspace"
        if not path.is_file():
            return None, f"[error] not a file: {path}"
        return path, None

    pa, err = resolve(path_a)
    if err:
        return err
    pb, err = resolve(path_b)
    if err:
        return err

    try:
        content_a = pa.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        content_b = pb.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"

    if content_a == content_b:
        return f"files identical: {pa} == {pb}"

    diff_lines = list(difflib.unified_diff(
        content_a, content_b,
        fromfile=str(pa), tofile=str(pb),
        lineterm="",
    ))
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines]
        diff_lines.append(f"... [truncated at {max_lines} lines]")
    return "\n".join(diff_lines)


@tool()
def parse_url(url: str) -> str:
    """Parse a URL into structured components (sub-project 74).

    Output:

      scheme: <https / http / ftp / ...>
      host: <netloc without port>
      port: <int or 'default'>
      path: <decoded path>
      query: <raw query string>
      params (N): one ``  key=value`` row per query param
      fragment: <decoded fragment>

    Returns ``[error] ...`` on parse failure or empty url.
    """
    from urllib.parse import urlparse, parse_qsl

    if not isinstance(url, str) or not url.strip():
        return "[error] url required"
    try:
        parsed = urlparse(url)
    except Exception as e:
        return f"[error] parse failed: {type(e).__name__}: {e}"
    parts = [
        f"scheme: {parsed.scheme or '(none)'}",
        f"host: {parsed.hostname or '(none)'}",
        f"port: {parsed.port if parsed.port else 'default'}",
        f"path: {parsed.path or '/'}",
        f"query: {parsed.query or '(empty)'}",
    ]
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if pairs:
        parts.append(f"params ({len(pairs)}):")
        parts.extend(f"  {k}={v}" for k, v in pairs)
    if parsed.fragment:
        parts.append(f"fragment: {parsed.fragment}")
    return "\n".join(parts)


@tool()
def string_diff(a: str, b: str, max_lines: int = 200) -> str:
    """Show a unified diff between two inline strings (sub-project 71).

    Output is standard ``difflib.unified_diff`` format with up to
    ``max_lines`` of diff lines (capped to keep responses bounded).
    Returns ``"identical"`` when the inputs are byte-equal.

    Pairs with ``compare_files`` (which does files): use string_diff
    when you have two text variables already in memory and want to
    see how they differ.
    """
    import difflib as _difflib

    if not isinstance(a, str) or not isinstance(b, str):
        return "[error] both a and b must be strings"
    if a == b:
        return "identical"
    a_lines = a.splitlines(keepends=False)
    b_lines = b.splitlines(keepends=False)
    diff = list(_difflib.unified_diff(a_lines, b_lines, fromfile="a", tofile="b", lineterm=""))
    cap = max(1, min(int(max_lines), 5000))
    if len(diff) > cap:
        diff = diff[:cap]
        diff.append(f"... [truncated at {cap} lines]")
    return "\n".join(diff)


@tool()
def regex_search(pattern: str, text: str, max_matches: int = 20, ignore_case: bool = False) -> str:
    """Find regex matches in inline text (sub-project 70).

    Returns one line per match in ``pos: <text>`` format, where ``pos``
    is the 0-based start offset. Capped at ``max_matches`` (default 20,
    hard limit 500). ``ignore_case=True`` enables case-insensitive
    matching. Returns ``[error] regex: <msg>`` on bad pattern.

    Use for "does this string match this pattern?" / "find all email
    addresses in this paragraph" reasoning. Cheaper than reaching for
    run_python.
    """
    import re as _re

    if not pattern:
        return "[error] pattern required"
    if not isinstance(text, str):
        return "[error] text must be a string"
    flags = _re.IGNORECASE if ignore_case else 0
    try:
        rx = _re.compile(pattern, flags)
    except _re.error as e:
        return f"[error] regex: {e}"
    cap = max(1, min(int(max_matches), 500))
    matches: list[str] = []
    truncated = False
    for m in rx.finditer(text):
        matches.append(f"{m.start()}: {m.group(0)}")
        if len(matches) >= cap:
            truncated = True
            break
    if not matches:
        return "(no matches)"
    out = f"{len(matches)} match(es):\n" + "\n".join(matches)
    if truncated:
        out += f"\n... [stopped at max_matches={cap}]"
    return out


@tool()
def file_info(path: str) -> str:
    """Report a file's metadata (sub-project 69).

    Output:
      path: <absolute>
      size: <N> bytes
      type: file | directory | symlink | other
      modified: <ISO-8601 mtime>
      mime: <guessed Content-Type>

    Workspace-confined. Cheaper than read_file when you only need
    "is this big?" / "when was it last touched?" / "what kind of file
    is this?" reconnaissance.
    """
    import mimetypes
    from datetime import datetime
    from pathlib import Path as _Path

    if not path:
        return "[error] path required"
    ws = current_workspace()
    p = _Path(path)
    if not p.is_absolute() and ws is not None:
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"[error] cannot resolve {path!r}: {e}"
    if ws is not None:
        try:
            p.relative_to(ws.root)
        except ValueError:
            return f"[error] {p} is outside the active workspace"
    try:
        st = p.stat()
    except FileNotFoundError:
        return f"[error] not found: {p}"
    except OSError as e:
        return f"[error] stat failed: {type(e).__name__}: {e}"

    if p.is_symlink():
        kind = "symlink"
    elif p.is_dir():
        kind = "directory"
    elif p.is_file():
        kind = "file"
    else:
        kind = "other"
    mime, _ = mimetypes.guess_type(str(p))
    mtime = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    parts = [
        f"path: {p}",
        f"size: {st.st_size} bytes",
        f"type: {kind}",
        f"modified: {mtime}",
    ]
    if mime:
        parts.append(f"mime: {mime}")
    return "\n".join(parts)


_HASH_ALGOS = {"sha256", "sha1", "md5", "sha512", "blake2b"}


@tool()
def hash_text(text: str, algorithm: str = "sha256") -> str:
    """Compute a cryptographic hash of UTF-8 text (sub-project 68).

    Algorithms supported: sha256 (default), sha1, md5, sha512, blake2b.
    Returns ``<algo>: <hex digest>`` or ``[error] ...`` on unknown
    algorithm. Useful for cache keys, content-addressed lookups, and
    integrity tags.
    """
    import hashlib as _hashlib
    if not isinstance(text, str):
        return "[error] text must be a string"
    algo = algorithm.lower()
    if algo not in _HASH_ALGOS:
        return f"[error] unknown algorithm: {algorithm!r} (try: {', '.join(sorted(_HASH_ALGOS))})"
    h = _hashlib.new(algo)
    h.update(text.encode("utf-8"))
    return f"{algo}: {h.hexdigest()}"


@tool()
def hash_file(path: str, algorithm: str = "sha256") -> str:
    """Compute a cryptographic hash of a workspace-confined file
    (sub-project 68).

    Streams the file in 64 KB chunks so big files don't blow memory.
    Same algorithm set as ``hash_text``. Returns
    ``<algo>: <hex digest>  (<path>, <bytes> bytes)``.
    """
    import hashlib as _hashlib
    from pathlib import Path as _Path

    if not path:
        return "[error] path required"
    ws = current_workspace()
    p = _Path(path)
    if not p.is_absolute() and ws is not None:
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"[error] cannot resolve {path!r}: {e}"
    if ws is not None:
        try:
            p.relative_to(ws.root)
        except ValueError:
            return f"[error] {p} is outside the active workspace"
    if not p.is_file():
        return f"[error] not a file: {p}"
    algo = algorithm.lower()
    if algo not in _HASH_ALGOS:
        return f"[error] unknown algorithm: {algorithm!r} (try: {', '.join(sorted(_HASH_ALGOS))})"
    h = _hashlib.new(algo)
    size = 0
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(64 * 1024), b""):
                h.update(chunk)
                size += len(chunk)
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"
    return f"{algo}: {h.hexdigest()}  ({p}, {size} bytes)"


@tool()
def now_iso(tz: str = "local") -> str:
    """Return the current time as an ISO-8601 string (sub-project 67).

    ``tz`` is either ``"local"`` (default) or ``"utc"``. Output format
    is seconds-resolution. Useful for stamping log entries, generating
    filenames, or pinning "what time was it when I called this tool?"
    reasoning.
    """
    from datetime import datetime, timezone
    if tz.lower() == "utc":
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    return datetime.now().isoformat(timespec="seconds")


@tool()
def parse_iso(text: str) -> str:
    """Parse an ISO-8601 timestamp and report a structured breakdown
    (sub-project 67).

    Output:
      year=YYYY month=MM day=DD weekday=Name hour=HH minute=MM second=SS
      epoch=<unix seconds>

    Returns ``[error] ...`` on parse failure. Accepts ``2026-05-05``,
    ``2026-05-05T12:34:56``, ``2026-05-05T12:34:56+00:00``, ``...Z``.
    """
    from datetime import datetime
    if not isinstance(text, str) or not text:
        return "[error] text required"
    try:
        normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
        dt = datetime.fromisoformat(normalized)
    except ValueError as e:
        return f"[error] parse: {e}"
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
    weekday = weekdays[dt.weekday()]
    epoch = int(dt.timestamp()) if dt.tzinfo else int(dt.replace(microsecond=0).timestamp())
    return (
        f"year={dt.year} month={dt.month:02d} day={dt.day:02d} "
        f"weekday={weekday} hour={dt.hour:02d} minute={dt.minute:02d} "
        f"second={dt.second:02d}\nepoch={epoch}"
    )


@tool()
def time_ago(text: str) -> str:
    """Report how long ago an ISO-8601 timestamp was, in human terms
    (sub-project 67).

    Output examples: ``"42 seconds ago"``, ``"3 minutes ago"``,
    ``"in 5 hours"`` (future timestamps render as "in"). Granularity:
    seconds → minutes → hours → days. Useful for the agent reasoning
    about how stale data is.
    """
    from datetime import datetime, timezone
    if not isinstance(text, str) or not text:
        return "[error] text required"
    try:
        normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
        dt = datetime.fromisoformat(normalized)
    except ValueError as e:
        return f"[error] parse: {e}"
    if dt.tzinfo is None:
        dt = dt.astimezone()
    now = datetime.now(tz=timezone.utc)
    delta = now - dt.astimezone(timezone.utc)
    total = delta.total_seconds()
    future = total < 0
    total = abs(total)
    if total < 60:
        amount, unit = int(total), "second"
    elif total < 3600:
        amount, unit = int(total // 60), "minute"
    elif total < 86400:
        amount, unit = int(total // 3600), "hour"
    else:
        amount, unit = int(total // 86400), "day"
    plural = "" if amount == 1 else "s"
    return f"in {amount} {unit}{plural}" if future else f"{amount} {unit}{plural} ago"


@tool()
def text_metrics(text: str = "", path: str = "") -> str:
    """Quick text statistics for inline text or a workspace file (sub-project 65).

    Reports characters, bytes (UTF-8), words, lines, and the longest
    line. Pass exactly one of ``text`` or ``path``.

    Useful for "is this prompt too big?" sanity checks and "summarize
    this file's shape" reconnaissance. Cheaper than count_tokens for
    quick metrics — count_tokens is the right tool when you want a
    token estimate; this one is the right tool for raw shape.
    """
    from pathlib import Path as _Path

    if not text and not path:
        return "[error] supply either `text` or `path`"
    if text and path:
        return "[error] supply only one of `text` or `path`"

    body: str
    label: str
    if path:
        ws = current_workspace()
        p = _Path(path)
        if not p.is_absolute() and ws is not None:
            p = ws.root / p
        try:
            p = p.resolve()
        except OSError as e:
            return f"[error] cannot resolve {path!r}: {e}"
        if ws is not None:
            try:
                p.relative_to(ws.root)
            except ValueError:
                return f"[error] {p} is outside the active workspace"
        if not p.is_file():
            return f"[error] not a file: {p}"
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[error] read failed: {type(e).__name__}: {e}"
        label = str(p)
    else:
        body = text
        label = "<inline>"

    lines = body.splitlines()
    chars = len(body)
    bytes_ = len(body.encode("utf-8"))
    words = len(body.split())
    longest = max((len(ln) for ln in lines), default=0)
    return (
        f"{label}: {chars} chars, {bytes_} bytes, {words} words, "
        f"{len(lines)} lines, longest line {longest} chars"
    )


@tool()
def python_info() -> str:
    """Report the Python interpreter, version, prefix, and key installed
    packages (sub-project 64).

    Useful for "is this the venv I expect?" checks. Lists up to 30
    distribution names that are installed, in alphabetical order.
    Non-dangerous; in-process introspection only.
    """
    import sys
    try:
        from importlib.metadata import distributions
        names = sorted({d.metadata["Name"] for d in distributions() if d.metadata.get("Name")})
    except Exception:
        names = []
    parts: list[str] = []
    parts.append(f"executable: {sys.executable}")
    parts.append(f"version: {sys.version.split()[0]}")
    parts.append(f"prefix: {sys.prefix}")
    parts.append(f"platform: {sys.platform}")
    if names:
        head = names[:30]
        parts.append(f"distributions ({len(names)} total, top 30): {', '.join(head)}")
    else:
        parts.append("distributions: (none discovered)")
    return "\n".join(parts)


@tool()
def random_id(kind: str = "uuid", length: int = 16) -> str:
    """Generate a random identifier (sub-project 62).

    ``kind`` selects the shape:
      - ``"uuid"`` (default) — RFC 4122 UUID4 string.
      - ``"hex"``  — random hex of ``length`` chars (1..256).
      - ``"slug"`` — URL-safe base64 of ``length`` bytes (1..64).
      - ``"int"``  — random integer in [0, 2**63).

    Useful when the agent needs a non-colliding name for a temp file,
    a request id, an API key placeholder, etc.
    """
    import secrets as _secrets
    import uuid as _uuid

    kind = (kind or "uuid").lower()
    if kind == "uuid":
        return str(_uuid.uuid4())
    if kind == "hex":
        n = max(1, min(int(length), 256))
        return _secrets.token_hex(max(1, n // 2))[:n]
    if kind == "slug":
        n = max(1, min(int(length), 64))
        return _secrets.token_urlsafe(n)
    if kind == "int":
        return str(_secrets.randbelow(2**63))
    return f"[error] unknown kind: {kind!r}"


@tool()
def urlquote(text: str, safe: str = "") -> str:
    """Percent-encode ``text`` for use in a URL (sub-project 62).

    ``safe`` is a string of characters left unescaped (default: none).
    Useful when the agent constructs an API URL with user content.
    """
    from urllib.parse import quote as _quote
    if not isinstance(text, str):
        return "[error] text must be a string"
    return _quote(text, safe=safe)


@tool()
def urlunquote(text: str) -> str:
    """Decode a percent-encoded URL string (sub-project 62)."""
    from urllib.parse import unquote as _unquote
    if not isinstance(text, str):
        return "[error] text must be a string"
    return _unquote(text)


@tool()
def b64encode(text: str, urlsafe: bool = False) -> str:
    """Base64-encode UTF-8 text (sub-project 63).

    ``urlsafe=True`` uses the URL-safe alphabet (``-`` / ``_`` instead
    of ``+`` / ``/``) and strips padding ``=`` so the result is safe
    in query strings or filenames. Standard base64 keeps padding.
    """
    import base64 as _b64
    if not isinstance(text, str):
        return "[error] text must be a string"
    raw = text.encode("utf-8")
    if urlsafe:
        return _b64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    return _b64.b64encode(raw).decode("ascii")


@tool()
def b64decode(text: str, urlsafe: bool = False) -> str:
    """Decode a base64-encoded string back to UTF-8 (sub-project 63).

    ``urlsafe=True`` accepts URL-safe alphabet input AND auto-pads any
    missing ``=`` so the round-trip with ``b64encode(urlsafe=True)``
    just works. Standard base64 expects properly-padded input.
    """
    import base64 as _b64
    if not isinstance(text, str):
        return "[error] text must be a string"
    try:
        if urlsafe:
            # Auto-pad to a multiple of 4.
            pad = (-len(text)) % 4
            return _b64.urlsafe_b64decode(text + "=" * pad).decode(
                "utf-8", errors="replace"
            )
        return _b64.b64decode(text, validate=False).decode("utf-8", errors="replace")
    except Exception as e:
        return f"[error] decode failed: {type(e).__name__}: {e}"


@tool()
def format_json(text: str, indent: int = 2, sort_keys: bool = False) -> str:
    """Pretty-print a JSON blob (sub-project 60).

    Parses ``text``, re-serialises with ``indent`` spaces (1..8), and
    returns the formatted output. Returns ``[error] ...`` on parse
    failure. Useful when the agent generates a config / payload and
    wants it readable before writing to disk. Optional ``sort_keys``
    applies a deterministic key order.

    Non-dangerous, in-memory only.
    """
    import json as _json

    if not text:
        return "[error] text required"
    try:
        parsed = _json.loads(text)
    except _json.JSONDecodeError as e:
        return f"[error] line {e.lineno} col {e.colno}: {e.msg}"
    indent = max(1, min(int(indent), 8))
    return _json.dumps(parsed, indent=indent, sort_keys=bool(sort_keys), ensure_ascii=False)


@tool()
def format_python(text: str = "", path: str = "") -> str:
    """Re-serialise Python source via ast.unparse (sub-project 60).

    Round-trips ``text`` (or the contents of ``path``) through
    ``ast.parse`` + ``ast.unparse`` — normalises whitespace, removes
    redundant parentheses, and matches the stdlib's canonical
    formatting. **Comments are NOT preserved** (a known limitation
    of ast.unparse); callers who need them should reach for a real
    formatter via run_powershell + black.

    Returns the formatted source, or ``[error] line:col: msg`` on
    syntax failure. Non-dangerous; in-memory only.
    """
    import ast
    from pathlib import Path as _Path

    if not text and not path:
        return "[error] supply either `text` or `path`"
    if text and path:
        return "[error] supply only one of `text` or `path`"

    body: str
    label: str
    if path:
        ws = current_workspace()
        p = _Path(path)
        if not p.is_absolute() and ws is not None:
            p = ws.root / p
        try:
            p = p.resolve()
        except OSError as e:
            return f"[error] cannot resolve {path!r}: {e}"
        if ws is not None:
            try:
                p.relative_to(ws.root)
            except ValueError:
                return f"[error] {p} is outside the active workspace"
        if not p.is_file():
            return f"[error] not a file: {p}"
        if p.suffix != ".py":
            return f"[error] not a Python file: {p}"
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[error] read failed: {type(e).__name__}: {e}"
        label = str(p)
    else:
        body = text
        label = "<inline>"

    try:
        tree = ast.parse(body, filename=label)
    except SyntaxError as e:
        return f"[error] {label}: line {e.lineno} col {e.offset}: {e.msg}"
    try:
        return ast.unparse(tree)
    except Exception as e:
        return f"[error] {label}: unparse failed: {type(e).__name__}: {e}"


@tool()
def validate_json(text: str = "", path: str = "") -> str:
    """Validate that ``text`` (or the contents of ``path``) is well-formed JSON.

    Returns ``"ok: <type> with N keys/items"`` on success, or
    ``"[error] line L col C: <msg>"`` on parse failure. Use this before
    writing a JSON config file or sending a JSON payload through a
    tool that doesn't validate its inputs.

    Pass exactly one of ``text`` (inline) or ``path`` (workspace-confined
    file). Non-dangerous, read-only.
    """
    import json as _json
    from pathlib import Path as _Path

    if not text and not path:
        return "[error] supply either `text` or `path`"
    if text and path:
        return "[error] supply only one of `text` or `path`"

    body: str
    label: str
    if path:
        ws = current_workspace()
        p = _Path(path)
        if not p.is_absolute() and ws is not None:
            p = ws.root / p
        try:
            p = p.resolve()
        except OSError as e:
            return f"[error] cannot resolve {path!r}: {e}"
        if ws is not None:
            try:
                p.relative_to(ws.root)
            except ValueError:
                return f"[error] {p} is outside the active workspace"
        if not p.is_file():
            return f"[error] not a file: {p}"
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[error] read failed: {type(e).__name__}: {e}"
        label = str(p)
    else:
        body = text
        label = "<inline>"

    try:
        parsed = _json.loads(body)
    except _json.JSONDecodeError as e:
        return f"[error] {label}: line {e.lineno} col {e.colno}: {e.msg}"

    kind = type(parsed).__name__
    if isinstance(parsed, dict):
        return f"ok: {label} → object with {len(parsed)} key(s)"
    if isinstance(parsed, list):
        return f"ok: {label} → array of {len(parsed)} item(s)"
    return f"ok: {label} → {kind} ({parsed!r})"


@tool()
def count_tokens(text: str = "", path: str = "") -> str:
    """Estimate token count for a piece of text or a file (sub-project 39).

    Heuristic: ~4 characters per token (matches the chars/4 estimator the
    daemon already uses for context-window math). Not exact; OpenAI BPE
    runs ~3.3-4.0 chars/token depending on language. For accurate counts
    use the model provider's own tokenizer.

    Pass exactly one of ``text`` (inline) or ``path`` (workspace-relative
    or absolute file inside the workspace). Returns
    ``"approx N tokens (M chars, K lines)"``.
    """
    from godbot.core.session import estimate_tokens

    if not text and not path:
        return "[error] supply either `text` or `path`"
    if text and path:
        return "[error] supply only one of `text` or `path`"

    body: str
    label: str
    if path:
        from pathlib import Path as _Path
        ws = current_workspace()
        p = _Path(path)
        if not p.is_absolute() and ws is not None:
            p = ws.root / p
        try:
            p = p.resolve()
        except OSError as e:
            return f"[error] cannot resolve {path!r}: {e}"
        # Workspace confinement when active.
        if ws is not None:
            try:
                p.relative_to(ws.root)
            except ValueError:
                return f"[error] {p} is outside the active workspace"
        if not p.is_file():
            return f"[error] not a file: {p}"
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[error] read failed: {type(e).__name__}: {e}"
        label = str(p)
    else:
        body = text
        label = "<inline text>"

    tokens = estimate_tokens(body)
    chars = len(body)
    lines = body.count("\n") + (0 if body.endswith("\n") or not body else 1)
    return f"{label}: approx {tokens} tokens ({chars} chars, {lines} lines)"


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
