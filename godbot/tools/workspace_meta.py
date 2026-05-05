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
def read_lines_around(path: str, line: int, context: int = 3) -> str:
    """Read a line plus N lines of context above and below (sub-project 97).

    Output is one row per line in ``MARKER LINENO\\tcontent`` format
    with the target line marked by a leading ``>`` instead of a space.
    ``context`` clamped to [0, 50].

    Useful for "show me line 42 with surroundings" anchored views,
    pairs with regex_search / git_blame_line that point at specific
    lines.
    """
    p_str, err = _resolve_workspace_file(path)
    if err:
        return err
    if line < 1:
        return "[error] line must be >= 1"
    ctx = max(0, min(int(context), 50))
    try:
        with open(p_str, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"
    if not all_lines:
        return "(empty file)"
    if line > len(all_lines):
        return f"[error] line {line} > total {len(all_lines)}"
    start = max(1, line - ctx)
    end = min(len(all_lines), line + ctx)
    out = []
    for ln in range(start, end + 1):
        marker = ">" if ln == line else " "
        out.append(f"{marker}{ln}\t{all_lines[ln - 1].rstrip()}")
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


_ENV_REDACT_PATTERNS = (
    "key", "token", "secret", "password", "passwd", "auth",
    "credential", "private", "session",
)


def _redact(name: str, value: str) -> str:
    """Mask the value if the name looks like a secret."""
    lower = name.lower()
    if any(p in lower for p in _ENV_REDACT_PATTERNS):
        if not value:
            return "(empty)"
        if len(value) <= 6:
            return "***"
        return f"{value[:2]}***{value[-2:]} (len={len(value)})"
    return value


@tool()
def get_env(name: str) -> str:
    """Read one environment variable (sub-project 75).

    Names that look secret-bearing (contain key / token / secret /
    password / auth / credential / private / session) are redacted in
    the response — the value is shown as ``ab***yz (len=N)`` so the
    agent can verify presence without leaking the secret into logs.

    Returns ``"<name>: (unset)"`` when missing. Non-dangerous,
    read-only.
    """
    import os as _os
    if not name:
        return "[error] name required"
    v = _os.environ.get(name)
    if v is None:
        return f"{name}: (unset)"
    return f"{name}: {_redact(name, v)}"


@tool()
def list_env(prefix: str = "", max_results: int = 30) -> str:
    """List environment variables, optionally filtered by name prefix
    (sub-project 75).

    Same redaction rules as ``get_env``. Returns up to ``max_results``
    entries (default 30, hard cap 200) sorted alphabetically. Useful
    for "what API keys are configured?" without leaking values.
    """
    import os as _os
    cap = max(1, min(int(max_results), 200))
    items = sorted(
        (n for n in _os.environ if n.startswith(prefix)),
        key=str.lower,
    )
    if not items:
        return f"(no env vars matching prefix {prefix!r})"
    truncated = len(items) > cap
    items = items[:cap]
    rows = [f"{n}={_redact(n, _os.environ[n])}" for n in items]
    out = "\n".join(rows)
    if truncated:
        out += f"\n... [truncated at max_results={cap}]"
    return out


@tool()
def summarize_logs(path: str, levels: str = "ERROR,WARNING", max_lines: int = 50) -> str:
    """Pull log lines matching one or more level keywords (sub-project 82).

    ``levels`` is a comma-separated list of keywords (case-insensitive)
    matched as substrings on each line — ``ERROR``, ``CRITICAL``,
    ``Traceback`` are typical picks. ``max_lines`` caps results
    (default 50, hard cap 1000). Workspace-confined.

    Output:

      file: <abs path>
      total lines: <N>
      matched (M):
        L42: ERROR something broke
        L78: WARNING fallback used
        ...

    Useful as first-look log triage: agent points at a 100k-line log
    and gets just the stuff worth reading. Pairs with tail (SP55) for
    "what's happening right now?" while this is "what went wrong?"
    """
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

    keywords = [k.strip().lower() for k in (levels or "").split(",") if k.strip()]
    if not keywords:
        return "[error] levels must contain at least one keyword"
    cap = max(1, min(int(max_lines), 1000))

    matched: list[str] = []
    total = 0
    truncated = False
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                total += 1
                lower = line.lower()
                if any(k in lower for k in keywords):
                    matched.append(f"  L{lineno}: {line.rstrip()}")
                    if len(matched) >= cap:
                        truncated = True
                        # Drain the rest just for the count without storing.
                        for _ in f:
                            total += 1
                        break
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"

    parts = [
        f"file: {p}",
        f"total lines: {total}",
        f"matched ({len(matched)}{' capped' if truncated else ''}):",
    ]
    parts.extend(matched)
    if truncated:
        parts.append(f"  ... [stopped at max_lines={cap}]")
    return "\n".join(parts)


@tool()
def tree(path: str = ".", max_depth: int = 2, max_entries: int = 200) -> str:
    """Render a directory tree as ASCII (sub-project 81).

    Workspace-confined. ``max_depth`` caps recursion depth (default 2,
    hard cap 6). ``max_entries`` caps total lines so huge trees stay
    scannable. Skips ``.git``/__pycache__/.venv/node_modules etc.

    Output uses ``├── name/`` / ``└── leaf`` glyphs. Useful for first-
    look "show me the project layout" reconnaissance.
    """
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

    SKIPS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             "dist", "build", ".pytest_cache", ".mypy_cache",
             ".ruff_cache", ".tox"}
    depth_cap = max(0, min(int(max_depth), 6))
    entry_cap = max(1, min(int(max_entries), 5000))

    lines: list[str] = [str(p)]
    truncated = False

    def walk(directory: _Path, prefix: str, depth: int) -> None:
        nonlocal truncated
        if truncated or depth > depth_cap:
            return
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )
        except OSError:
            return
        children = [c for c in children if c.name not in SKIPS]
        last_idx = len(children) - 1
        for i, child in enumerate(children):
            if len(lines) >= entry_cap:
                lines.append(prefix + "... [truncated]")
                truncated = True
                return
            connector = "└── " if i == last_idx else "├── "
            suffix = "/" if child.is_dir() else ""
            lines.append(prefix + connector + child.name + suffix)
            if child.is_dir():
                next_prefix = prefix + ("    " if i == last_idx else "│   ")
                walk(child, next_prefix, depth + 1)

    walk(p, "", 1)
    return "\n".join(lines)


@tool()
def parse_csv(path: str, max_rows: int = 100) -> str:
    """Parse a CSV file and return rows as JSON dicts (sub-project 99).

    Output is a JSON array of objects keyed by column name. Auto-sniffs
    delimiter (same heuristic as csv_summary). Workspace-confined;
    max_rows capped at 1000 to keep responses bounded.

    Use this when the agent needs to *act on* CSV data — pairs with
    csv_summary which just shows shape. Returns ``[error] ...`` on
    read/parse failure.
    """
    import csv as _csv
    import json as _json
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

    cap = max(1, min(int(max_rows), 1000))
    try:
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            head_sample = f.read(8192)
            try:
                dialect = _csv.Sniffer().sniff(head_sample, delimiters=",;\t|")
            except _csv.Error:
                dialect = _csv.excel
            f.seek(0)
            reader = _csv.DictReader(f, dialect=dialect)
            rows: list[dict] = []
            truncated = False
            for r in reader:
                rows.append({k: v for k, v in r.items() if k is not None})
                if len(rows) >= cap:
                    if next(reader, None) is not None:
                        truncated = True
                    break
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"

    payload = _json.dumps(rows, indent=2, ensure_ascii=False)
    if truncated:
        payload += f"\n... [truncated at max_rows={cap}]"
    return payload


@tool()
def csv_summary(path: str, sample_rows: int = 5) -> str:
    """Quick overview of a CSV file (sub-project 78).

    Reports delimiter (auto-sniffed), column names, row count, and a
    sample of the first N rows (default 5, capped at 50). Workspace-
    confined.

    Output:

      file: <abs path>
      delimiter: ','
      columns (5): a, b, c, d, e
      rows: 1234
      sample (3 of 1234):
        row 1: a=1, b=2, c=3, d=4, e=5
        row 2: ...

    Useful as first-look reconnaissance before deciding how to process
    a data file.
    """
    import csv as _csv
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

    try:
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            head_sample = f.read(8192)
            try:
                dialect = _csv.Sniffer().sniff(head_sample, delimiters=",;\t|")
                delimiter = dialect.delimiter
            except _csv.Error:
                dialect = _csv.excel
                delimiter = ","
            f.seek(0)
            reader = _csv.reader(f, dialect=dialect)
            try:
                header = next(reader)
            except StopIteration:
                return f"file: {p}\n(empty)"
            cap = max(1, min(int(sample_rows), 50))
            rows: list[list[str]] = []
            row_count = 0
            for r in reader:
                row_count += 1
                if len(rows) < cap:
                    rows.append(r)
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"

    parts = [
        f"file: {p}",
        f"delimiter: {delimiter!r}",
        f"columns ({len(header)}): {', '.join(header)}",
        f"rows: {row_count}",
    ]
    if rows:
        parts.append(f"sample ({len(rows)} of {row_count}):")
        for i, r in enumerate(rows, 1):
            kv = ", ".join(
                f"{header[j]}={r[j]}" for j in range(min(len(header), len(r)))
            )
            parts.append(f"  row {i}: {kv}")
    return "\n".join(parts)


@tool()
def sort_lines(text: str, reverse: bool = False, numeric: bool = False) -> str:
    """Sort lines of inline text (sub-project 84).

    ``reverse=True`` flips the order. ``numeric=True`` sorts by leading
    integer (lines without a leading number sort to the bottom). Lines
    are split on '\\n' and rejoined; trailing newline is preserved.

    Useful when the agent has stitched together output and wants it
    canonical for diffing or display.
    """
    if not isinstance(text, str):
        return "[error] text must be a string"
    had_trailing_nl = text.endswith("\n")
    lines = text.splitlines()
    if numeric:
        def _key(line: str) -> tuple[int, int, str]:
            stripped = line.lstrip()
            i = 0
            while i < len(stripped) and (stripped[i].isdigit() or (i == 0 and stripped[i] == "-")):
                i += 1
            try:
                num = int(stripped[:i]) if i else None
            except ValueError:
                num = None
            return (1 if num is None else 0, num if num is not None else 0, line)
        lines.sort(key=_key, reverse=reverse)
    else:
        lines.sort(reverse=reverse)
    out = "\n".join(lines)
    if had_trailing_nl:
        out += "\n"
    return out


@tool()
def unique_lines(text: str, preserve_order: bool = True) -> str:
    """Return distinct lines from inline text (sub-project 84).

    ``preserve_order=True`` (default) keeps first-occurrence ordering;
    ``False`` returns unique lines sorted alphabetically. Trailing
    newline behavior matches the input.

    Useful for dedup'ing log output the agent has stitched together.
    """
    if not isinstance(text, str):
        return "[error] text must be a string"
    had_trailing_nl = text.endswith("\n")
    lines = text.splitlines()
    if preserve_order:
        seen: set[str] = set()
        kept: list[str] = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                kept.append(line)
    else:
        kept = sorted(set(lines))
    out = "\n".join(kept)
    if had_trailing_nl:
        out += "\n"
    return out


@tool()
def text_replace(text: str, old: str, new: str, count: int = -1) -> str:
    """Replace ``old`` with ``new`` in ``text`` (sub-project 77).

    ``count`` caps the number of replacements (negative = all). Returns
    ``"replaced N: <result>"`` on success, ``[error] ...`` on bad input.
    Useful when the agent has text in a variable and wants to transform
    it without a run_python detour.

    Plain string replacement, not regex. Use ``run_python`` for regex.
    """
    if not isinstance(text, str) or not isinstance(old, str) or not isinstance(new, str):
        return "[error] text, old, new must all be strings"
    if not old:
        return "[error] old must be non-empty"
    occurrences = text.count(old)
    n_to_do = occurrences if count < 0 else min(int(count), occurrences)
    out = text.replace(old, new, n_to_do if count >= 0 else -1)
    return f"replaced {n_to_do}: {out}"


@tool()
def text_truncate(text: str, max_chars: int = 1000, suffix: str = "...") -> str:
    """Truncate ``text`` to at most ``max_chars`` characters (sub-project 77).

    Appends ``suffix`` (default ``"..."``) when truncation occurs.
    Returns the input unchanged when already short enough. ``max_chars``
    clamped to [1, 1_000_000].
    """
    if not isinstance(text, str):
        return "[error] text must be a string"
    cap = max(1, min(int(max_chars), 1_000_000))
    if len(text) <= cap:
        return text
    suffix_str = str(suffix)
    keep = max(1, cap - len(suffix_str))
    return text[:keep] + suffix_str


@tool()
def find_files_by_age(
    older_than_days: int = 0, newer_than_days: int = 0,
    pattern: str = "**/*", root: str = ".", max_results: int = 100,
) -> str:
    """Find files matching a glob, filtered by modification age (sub-project 86).

    Workspace-confined; ``root`` defaults to the workspace root.
    ``older_than_days`` returns files mtime'd before that cutoff;
    ``newer_than_days`` returns files mtime'd after. Pass exactly one
    (set the other to 0). ``max_results`` capped at 1000.

    Output:

      pattern: <pattern>
      mode: older_than 30 / newer_than 7
      total: N matches
      files (oldest first):
        2025-04-12  /path/to/file.txt
        ...

    Use cases: cleanup ("what hasn't been touched in 90 days?"),
    triage ("what changed in the last week?"), staleness checks.
    """
    from datetime import datetime, timedelta
    from pathlib import Path as _Path

    if older_than_days < 0 or newer_than_days < 0:
        return "[error] day counts must be non-negative"
    if older_than_days == 0 and newer_than_days == 0:
        return "[error] supply older_than_days or newer_than_days (> 0)"
    if older_than_days > 0 and newer_than_days > 0:
        return "[error] use exactly one of older_than_days / newer_than_days"

    ws = current_workspace()
    p = _Path(root)
    if not p.is_absolute() and ws is not None:
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"[error] cannot resolve {root!r}: {e}"
    if ws is not None:
        try:
            p.relative_to(ws.root)
        except ValueError:
            return f"[error] {p} is outside the active workspace"
    if not p.is_dir():
        return f"[error] not a directory: {p}"

    SKIPS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             "dist", "build", ".pytest_cache"}
    cap = max(1, min(int(max_results), 1000))
    cutoff_secs = (
        (datetime.now() - timedelta(days=older_than_days)).timestamp()
        if older_than_days > 0
        else (datetime.now() - timedelta(days=newer_than_days)).timestamp()
    )
    matches: list[tuple[float, _Path]] = []
    try:
        for path in p.rglob(pattern):
            if not path.is_file():
                continue
            if any(part in SKIPS for part in path.parts):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if older_than_days > 0 and mtime < cutoff_secs:
                matches.append((mtime, path))
            elif newer_than_days > 0 and mtime > cutoff_secs:
                matches.append((mtime, path))
    except OSError as e:
        return f"[error] glob failed: {e}"

    matches.sort(key=lambda x: x[0])
    truncated = len(matches) > cap
    matches = matches[:cap]

    mode_label = (
        f"older_than {older_than_days}" if older_than_days > 0
        else f"newer_than {newer_than_days}"
    )
    parts = [
        f"pattern: {pattern}",
        f"mode: {mode_label}",
        f"total: {len(matches)} match(es){' (capped)' if truncated else ''}",
    ]
    if matches:
        parts.append("files (oldest first):")
        for mtime, path in matches:
            ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            parts.append(f"  {ts}  {path}")
    return "\n".join(parts)


@tool()
def count_files(pattern: str = "**/*", root: str = ".", max_count: int = 10_000) -> str:
    """Count files matching a recursive glob pattern (sub-project 76).

    Workspace-confined; ``root`` defaults to the workspace root.
    ``pattern`` is a glob (default ``**/*`` for recursive everything).
    Skips ``.git``, ``__pycache__``, ``.venv``, ``node_modules``, and
    similar high-volume noise dirs.

    Output:

      pattern: <pattern>
      root: <abs path>
      total: <N> file(s)
      by extension (top 10):
        .py: 42
        .md: 12
        ...

    Cheaper than directory_size when only counts (not bytes) matter.
    """
    from collections import Counter
    from pathlib import Path as _Path

    ws = current_workspace()
    p = _Path(root)
    if not p.is_absolute() and ws is not None:
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"[error] cannot resolve {root!r}: {e}"
    if ws is not None:
        try:
            p.relative_to(ws.root)
        except ValueError:
            return f"[error] {p} is outside the active workspace"
    if not p.is_dir():
        return f"[error] not a directory: {p}"

    SKIPS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             "dist", "build", ".pytest_cache"}
    cap = max(1, min(int(max_count), 200_000))
    total = 0
    by_ext: Counter[str] = Counter()
    truncated = False
    try:
        for path in p.rglob(pattern):
            if not path.is_file():
                continue
            if any(part in SKIPS for part in path.parts):
                continue
            total += 1
            ext = path.suffix.lower() or "(no ext)"
            by_ext[ext] += 1
            if total >= cap:
                truncated = True
                break
    except OSError as e:
        return f"[error] glob failed: {e}"

    parts = [
        f"pattern: {pattern}",
        f"root: {p}",
        f"total: {total} file(s){' (capped)' if truncated else ''}",
    ]
    if by_ext:
        parts.append("by extension (top 10):")
        for ext, n in by_ext.most_common(10):
            parts.append(f"  {ext}: {n}")
    return "\n".join(parts)


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
def is_binary_file(path: str) -> str:
    """Heuristic check whether a file is binary (sub-project 95).

    Reads the first 8 KB and reports ``yes``/``no`` based on:
    presence of NUL bytes, or > 30% non-printable bytes outside the
    standard whitespace set. Workspace-confined.

    Returns one of:

      yes: <path> (reason: null-byte / non-printable ratio X.XX)
      no:  <path>

    Useful before ``read_file``/``head``/``tail`` on an unknown blob —
    avoids spamming the agent's context with mojibake from a JPEG.
    """
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

    try:
        with open(p, "rb") as f:
            head = f.read(8192)
    except Exception as e:
        return f"[error] read failed: {type(e).__name__}: {e}"
    if not head:
        return f"no:  {p} (empty)"
    if b"\x00" in head:
        return f"yes: {p} (reason: null-byte)"
    textish = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13))
    ratio = textish / len(head)
    if ratio < 0.7:
        return f"yes: {p} (reason: non-printable ratio {1 - ratio:.2f})"
    return f"no:  {p}"


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


@tool(timeout=70)
def sleep(seconds: float = 1.0) -> str:
    """Pause for ``seconds`` seconds (sub-project 91).

    Bounded at [0.0, 60.0]. Useful when the agent needs to pace itself
    while waiting for an external state to settle (e.g. file write to
    flush, a background task to make progress) without busy-looping.
    Returns ``"slept N.NNs"``.

    Non-dangerous; pure time delay. The 60-second cap prevents the
    agent from pinning a turn for an unbounded interval.
    """
    import time as _time
    try:
        s = max(0.0, min(float(seconds), 60.0))
    except (TypeError, ValueError):
        return "[error] seconds must be a number"
    _time.sleep(s)
    return f"slept {s:.2f}s"


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
def system_info() -> str:
    """Report host hardware + OS at a glance (sub-project 85).

    Output:

      os: <platform string>
      python: <version>
      cpu_count: <logical cores>
      memory_gb: <total / available>     (when psutil is available)
      disk_free_gb: <root free space>    (when shutil.disk_usage works)
      cwd: <current working directory>

    Useful when the agent needs to make resource-aware decisions:
    "do I have enough RAM to load this model?", "is the disk full
    before I run this build?". psutil is a soft dep — we degrade
    gracefully when it isn't installed.
    """
    import os as _os
    import shutil as _shutil
    import sys as _sys

    parts: list[str] = []
    parts.append(f"os: {_sys.platform}")
    parts.append(f"python: {_sys.version.split()[0]}")
    parts.append(f"cpu_count: {_os.cpu_count() or 'unknown'}")

    try:
        import psutil  # type: ignore[import-not-found]
        vm = psutil.virtual_memory()
        parts.append(
            f"memory_gb: {vm.total / 1024**3:.1f} total / {vm.available / 1024**3:.1f} available"
        )
    except ImportError:
        parts.append("memory_gb: (psutil not installed)")
    except Exception as e:
        parts.append(f"memory_gb: (psutil error: {e})")

    try:
        from pathlib import Path as _Path
        anchor = _Path(_os.environ.get("SystemDrive", "/") + _os.sep) if _sys.platform == "win32" else _Path("/")
        usage = _shutil.disk_usage(str(anchor))
        parts.append(f"disk_free_gb: {usage.free / 1024**3:.1f} (anchor {anchor})")
    except Exception as e:
        parts.append(f"disk_free_gb: (lookup failed: {e})")

    parts.append(f"cwd: {_os.getcwd()}")
    return "\n".join(parts)


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


_URL_RE = r"https?://[^\s<>\")\]}]+|ftp://[^\s<>\")\]}]+"
_EMAIL_RE = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"


@tool()
def summarize_diff(diff: str) -> str:
    """Summarise a unified diff as +N -M lines across K files (sub-project 104).

    Output:

      summary: K files, +N -M lines
      per file:
        path/to/a.py  +12 -3
        path/to/b.py  +0  -7

    Pairs with apply_patch / preview_patch (SP22+24) — agent can call
    summarize_diff first to give the user a one-line pitch before
    asking to apply. Returns ``[error] ...`` for non-diff input.
    """
    if not isinstance(diff, str):
        return "[error] diff must be a string"
    files: list[tuple[str, int, int]] = []
    cur_path = ""
    cur_add = 0
    cur_del = 0
    has_any = False
    for line in diff.splitlines():
        if line.startswith("+++ "):
            if cur_path:
                files.append((cur_path, cur_add, cur_del))
            target = line[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            elif target.startswith("a/"):
                target = target[2:]
            cur_path = target
            cur_add = 0
            cur_del = 0
            has_any = True
        elif line.startswith("+") and not line.startswith("+++"):
            cur_add += 1
        elif line.startswith("-") and not line.startswith("---"):
            cur_del += 1
    if cur_path:
        files.append((cur_path, cur_add, cur_del))
    if not has_any:
        return "[error] no `+++` file headers found — does not look like a unified diff"

    total_add = sum(a for _, a, _ in files)
    total_del = sum(d for _, _, d in files)
    parts = [f"summary: {len(files)} file(s), +{total_add} -{total_del} lines"]
    if files:
        parts.append("per file:")
        width = min(60, max(len(p) for p, _, _ in files))
        for path, a, d in files:
            parts.append(f"  {path[:60]:<{width}}  +{a:<3} -{d}")
    return "\n".join(parts)


@tool()
def extract_links(text: str, kind: str = "all") -> str:
    """Extract URLs and/or email addresses from text (sub-project 103).

    ``kind`` selects: ``"url"`` (http/https/ftp), ``"email"``, or
    ``"all"`` (default; both, in order of appearance). Returns one
    match per line:

      url: https://example.com
      email: contact@example.com
      url: ftp://files.example.com

    Returns ``"(no matches)"`` when nothing found. Useful when the
    agent processes scraped pages or unstructured text and needs a
    structured list of links.
    """
    import re as _re
    if not isinstance(text, str):
        return "[error] text must be a string"
    kind = (kind or "all").lower()
    if kind not in {"url", "email", "all"}:
        return f"[error] kind must be 'url', 'email', or 'all', got {kind!r}"

    out: list[tuple[int, str, str]] = []
    if kind in ("url", "all"):
        for m in _re.finditer(_URL_RE, text):
            out.append((m.start(), "url", m.group(0).rstrip(".,;:")))
    if kind in ("email", "all"):
        for m in _re.finditer(_EMAIL_RE, text):
            out.append((m.start(), "email", m.group(0)))
    if not out:
        return "(no matches)"
    out.sort(key=lambda x: x[0])
    if kind == "all":
        return "\n".join(f"{k}: {v}" for _, k, v in out)
    return "\n".join(v for _, _, v in out)


@tool()
def levenshtein(a: str, b: str) -> str:
    """Compute Levenshtein edit distance between two strings (sub-project 102).

    Returns ``"distance: N (similarity: P.PP)"`` where similarity is
    ``1 - N/max(len(a), len(b))``. Useful for fuzzy matching: "did the
    user mean X?", typo tolerance, near-dup detection.

    Both inputs must be strings; truncated to 5000 chars each so the
    O(len(a)*len(b)) DP table stays bounded.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return "[error] both a and b must be strings"
    a_t = a[:5000]
    b_t = b[:5000]
    if a_t == b_t:
        return "distance: 0 (similarity: 1.00)"
    if not a_t:
        return f"distance: {len(b_t)} (similarity: 0.00)"
    if not b_t:
        return f"distance: {len(a_t)} (similarity: 0.00)"
    prev = list(range(len(b_t) + 1))
    for i, ca in enumerate(a_t, 1):
        cur = [i] + [0] * len(b_t)
        for j, cb in enumerate(b_t, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = cur
    dist = prev[-1]
    longer = max(len(a_t), len(b_t))
    sim = 1 - dist / longer if longer else 1.0
    return f"distance: {dist} (similarity: {sim:.2f})"


@tool()
def slugify(text: str, max_length: int = 80) -> str:
    """Convert text to a URL/filename-safe slug (sub-project 101).

    Lowercases, replaces non-alphanumerics with ``-``, collapses
    consecutive dashes, strips leading/trailing dashes, and truncates
    to ``max_length`` (clamped to [1, 200]). Useful for the agent
    generating filenames from titles or constructing URLs.

    Returns ``"-"`` when the input has no alphanumerics so the caller
    always gets a usable string instead of empty.
    """
    import re as _re
    if not isinstance(text, str):
        return "[error] text must be a string"
    cap = max(1, min(int(max_length), 200))
    s = text.lower()
    s = _re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    s = _re.sub(r"-{2,}", "-", s)
    if not s:
        return "-"
    return s[:cap]


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
