"""Heuristic project summary tool.

Scans a workspace for marker files (README, package.json, pyproject.toml, etc.)
and produces a structured plain-text summary. NO LLM is invoked — this tool
returns a digest of the marker files so the agent (or a one-turn summarizer)
can produce a natural-language overview.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from godbot.core.registry import tool
from godbot.core.workspace import current_workspace


# Marker files we look for at workspace root + first two depth levels.
# Order matters: earlier patterns win the "language" attribution.
_MARKER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("requirements.txt", "Python"),
    ("package.json", "JavaScript/TypeScript"),
    ("Cargo.toml", "Rust"),
    ("go.mod", "Go"),
    ("mix.exs", "Elixir"),
    ("Gemfile", "Ruby"),
    ("pom.xml", "Java"),
    ("build.gradle", "Java/Kotlin"),
)


def _shallow_glob(root: Path, name: str, max_depth: int = 2) -> list[Path]:
    """Find files named exactly ``name`` at depth <= max_depth.

    Capped depth keeps the scan cheap on large monorepos. Returns paths
    relative to root, sorted shortest-first so root-level wins.
    """
    matches: list[Path] = []
    if (root / name).is_file():
        matches.append(root / name)
    if max_depth >= 1:
        for child in root.iterdir() if root.is_dir() else []:
            if not child.is_dir() or child.name.startswith("."):
                continue
            if (child / name).is_file():
                matches.append(child / name)
    return matches


def _find_readme(root: Path) -> Path | None:
    """Prefer README.md, fall back to README.* (any extension or none)."""
    for cand in ("README.md", "README.MD", "README.rst", "README.txt", "README"):
        p = root / cand
        if p.is_file():
            return p
    # Glob anything starting with README at the root.
    for p in sorted(root.glob("README*")):
        if p.is_file():
            return p
    return None


def _safe_read(p: Path, max_lines: int = 2000) -> str:
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return "\n".join(lines)


def _extract_pyproject_info(text: str) -> tuple[str | None, list[str], str | None]:
    """Return (project_name, top_dependencies, entry_hint) from a pyproject.

    Uses tomllib if available; falls back to regex extraction. Caps deps at 10.
    """
    name: str | None = None
    deps: list[str] = []
    entry: str | None = None
    try:
        import tomllib  # type: ignore[import-not-found]
        data = tomllib.loads(text)
        proj = data.get("project") or {}
        if isinstance(proj.get("name"), str):
            name = proj["name"]
        raw_deps = proj.get("dependencies") or []
        if isinstance(raw_deps, list):
            for d in raw_deps:
                if isinstance(d, str):
                    deps.append(d)
        scripts = proj.get("scripts") or {}
        if isinstance(scripts, dict) and scripts:
            # Take the first script name as a hint at the entry point.
            first_key = next(iter(scripts))
            entry = f"{first_key} -> {scripts[first_key]}"
    except Exception:
        # Regex fallback: name = "..."
        m = re.search(r'^\s*name\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m:
            name = m.group(1)
    return name, deps[:10], entry


def _extract_package_json_info(text: str) -> tuple[str | None, list[str], str | None]:
    name: str | None = None
    deps: list[str] = []
    entry: str | None = None
    try:
        data = json.loads(text)
    except Exception:
        return name, deps, entry
    if isinstance(data.get("name"), str):
        name = data["name"]
    if isinstance(data.get("main"), str):
        entry = data["main"]
    elif isinstance(data.get("module"), str):
        entry = data["module"]
    for key in ("dependencies", "devDependencies"):
        d = data.get(key) or {}
        if isinstance(d, dict):
            for dep_name in list(d.keys())[: 10 - len(deps)]:
                deps.append(dep_name)
        if len(deps) >= 10:
            break
    return name, deps[:10], entry


def _extract_cargo_info(text: str) -> tuple[str | None, list[str]]:
    name: str | None = None
    deps: list[str] = []
    try:
        import tomllib  # type: ignore[import-not-found]
        data = tomllib.loads(text)
        pkg = data.get("package") or {}
        if isinstance(pkg.get("name"), str):
            name = pkg["name"]
        raw_deps = data.get("dependencies") or {}
        if isinstance(raw_deps, dict):
            deps = list(raw_deps.keys())[:10]
    except Exception:
        m = re.search(r'^\s*name\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m:
            name = m.group(1)
    return name, deps


def _find_entry(root: Path, language: str) -> str | None:
    """Best-effort: pick a likely entry-point file."""
    if language.startswith("Python"):
        for cand in ("main.py", "__main__.py", "__init__.py", "app.py", "run.py"):
            if (root / cand).is_file():
                return cand
            # Also check one level deep under packages.
            for child in root.iterdir() if root.is_dir() else []:
                if child.is_dir() and not child.name.startswith(".") and (child / cand).is_file():
                    return f"{child.name}/{cand}"
    if "JavaScript" in language or "TypeScript" in language:
        for cand in ("index.js", "index.ts", "src/index.js", "src/index.ts", "main.js", "main.ts"):
            if (root / cand).is_file():
                return cand
    if language == "Rust":
        for cand in ("src/main.rs", "src/lib.rs"):
            if (root / cand).is_file():
                return cand
    if language == "Go":
        if (root / "main.go").is_file():
            return "main.go"
    return None


def _readme_excerpt(text: str, max_chars: int = 600) -> str:
    """Return first non-empty paragraph(s) up to max_chars, stripped of headers."""
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        # Skip badge-only lines (common in OSS READMEs) — heuristic: leading [![
        if line.lstrip().startswith("[!["):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + "..."
    return cleaned


def _resolve_target_root(root: str | None) -> Path | None:
    """Pick the directory to summarize.

    Prefer (in order): explicit ``root`` arg, the active workspace, cwd."""
    if root:
        p = Path(root)
        if p.is_dir():
            return p
    ws = current_workspace()
    if ws is not None:
        return Path(ws.root)
    return None


@tool()
def project_summary(root: str = "") -> str:
    """Summarize the project at `root` (or the active workspace) by reading marker files.

    Heuristic-only: scans for README, pyproject.toml, package.json, Cargo.toml,
    go.mod, setup.py, mix.exs (capped to top two directory levels), reads each
    file (cap 2000 lines), and returns a structured text block:

        Project: <name>
        Language: <Python/JS/Rust/...>
        Entry: <main file or script hint>
        Dependencies (top 10): a, b, c, ...
        README excerpt:
        <first paragraph>

    Use this once at workspace open to anchor system context. The tool runs
    synchronously and never calls an LLM; downstream callers may pass the
    output to an LLM for natural-language summarization.
    """
    target = _resolve_target_root(root)
    if target is None:
        return "[error] no workspace active and no root supplied"
    if not target.is_dir():
        return f"[error] not a directory: {target}"

    found_markers: list[tuple[str, Path]] = []
    for marker, _lang in _MARKER_PATTERNS:
        for hit in _shallow_glob(target, marker):
            found_markers.append((marker, hit))

    name: str | None = None
    language: str | None = None
    entry: str | None = None
    deps: list[str] = []

    # Walk markers in priority order to pick the canonical project metadata.
    for marker, lang in _MARKER_PATTERNS:
        hits = [h for (m, h) in found_markers if m == marker]
        if not hits:
            continue
        if language is None:
            language = lang
        # Read the first (root-most) marker for metadata extraction.
        path = hits[0]
        text = _safe_read(path)
        if marker == "pyproject.toml":
            n, d, e = _extract_pyproject_info(text)
            if name is None and n:
                name = n
            if not deps and d:
                deps = d
            if entry is None and e:
                entry = e
        elif marker == "package.json":
            n, d, e = _extract_package_json_info(text)
            if name is None and n:
                name = n
            if not deps and d:
                deps = d
            if entry is None and e:
                entry = e
        elif marker == "Cargo.toml":
            n, d = _extract_cargo_info(text)
            if name is None and n:
                name = n
            if not deps and d:
                deps = d
        elif marker == "requirements.txt":
            if not deps:
                lines = [
                    ln.strip().split("=")[0].split(">")[0].split("<")[0].split(";")[0].strip()
                    for ln in text.splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")
                ]
                deps = [d for d in lines if d][:10]

    # Fallback: directory name as project name.
    if name is None:
        name = target.name

    if entry is None and language is not None:
        entry = _find_entry(target, language)

    readme_path = _find_readme(target)
    readme_excerpt = ""
    if readme_path is not None:
        readme_excerpt = _readme_excerpt(_safe_read(readme_path))

    parts: list[str] = []
    parts.append(f"Project: {name}")
    parts.append(f"Language: {language or 'unknown'}")
    parts.append(f"Entry: {entry or '(none detected)'}")
    if deps:
        parts.append(f"Dependencies (top {len(deps)}): {', '.join(deps)}")
    else:
        parts.append("Dependencies (top 10): (none detected)")
    marker_names = sorted({m for (m, _h) in found_markers})
    if marker_names:
        parts.append(f"Markers found: {', '.join(marker_names)}")
    if readme_excerpt:
        parts.append("README excerpt:")
        parts.append(readme_excerpt)
    else:
        parts.append("README excerpt: (no README found)")
    return "\n".join(parts)
