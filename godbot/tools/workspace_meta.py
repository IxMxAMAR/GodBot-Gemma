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
