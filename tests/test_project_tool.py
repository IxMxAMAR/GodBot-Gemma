"""Tests for the heuristic project_summary tool."""
from __future__ import annotations
import json
from pathlib import Path

import godbot.tools  # noqa: F401 — trigger discovery
from godbot.core.registry import DEFAULT
from godbot.core.workspace import Workspace, _current


def _exec(root: str | None = None) -> str:
    args = {"root": root} if root is not None else {}
    return DEFAULT.execute("project_summary", args)


def test_project_summary_registered():
    assert "project_summary" in DEFAULT.names()


def test_project_summary_python_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\n'
        'name = "thunderbolt"\n'
        'dependencies = ["httpx>=0.25", "pydantic", "fastapi"]\n'
        '[project.scripts]\n'
        'thunder = "thunderbolt.cli:main"\n',
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Thunderbolt\n\nA fast lightning bolt simulator.\n", encoding="utf-8"
    )

    out = _exec(str(tmp_path))
    assert "Project: thunderbolt" in out
    assert "Language: Python" in out
    # main.py at root should be picked up as entry (script hint may also exist).
    assert "main.py" in out or "thunder" in out
    assert "httpx>=0.25" in out
    assert "fastapi" in out
    assert "Thunderbolt" in out
    assert "lightning bolt simulator" in out


def test_project_summary_javascript_package_json(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "name": "studio-app",
                "main": "src/index.ts",
                "dependencies": {"react": "^18", "vite": "^5"},
                "devDependencies": {"typescript": "^5"},
            }
        ),
        encoding="utf-8",
    )
    out = _exec(str(tmp_path))
    assert "Project: studio-app" in out
    assert "JavaScript" in out or "TypeScript" in out
    assert "src/index.ts" in out
    assert "react" in out
    assert "vite" in out


def test_project_summary_rust_cargo(tmp_path):
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "rusty"\nversion = "0.1.0"\n\n'
        '[dependencies]\nserde = "1"\ntokio = "1"\n',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    out = _exec(str(tmp_path))
    assert "Project: rusty" in out
    assert "Language: Rust" in out
    assert "src/main.rs" in out
    assert "serde" in out


def test_project_summary_no_workspace_no_arg(tmp_path, monkeypatch):
    # Ensure no workspace active.
    token = _current.set(None)
    try:
        out = DEFAULT.execute("project_summary", {})
    finally:
        _current.reset(token)
    assert out.startswith("[error]")


def test_project_summary_uses_active_workspace(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "ws-active"\n', encoding="utf-8"
    )
    ws = Workspace.of(str(tmp_path))
    token = _current.set(ws)
    try:
        out = DEFAULT.execute("project_summary", {})
    finally:
        _current.reset(token)
    assert "Project: ws-active" in out


def test_project_summary_falls_back_to_dirname(tmp_path):
    # No marker files at all — should still succeed using dirname.
    (tmp_path / "src").mkdir()
    out = _exec(str(tmp_path))
    assert f"Project: {tmp_path.name}" in out
    assert "Language: unknown" in out


def test_project_summary_requirements_txt(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "# top deps\nrequests>=2.0\nclick==8.1\n# comment\nrich\n",
        encoding="utf-8",
    )
    out = _exec(str(tmp_path))
    assert "Language: Python" in out
    assert "requests" in out
    assert "click" in out


def test_project_summary_readme_truncated(tmp_path):
    long_readme = "Hello.\n\n" + ("word " * 500)
    (tmp_path / "README.md").write_text(long_readme, encoding="utf-8")
    out = _exec(str(tmp_path))
    # README excerpt should be present and not exceed our cap (~600 chars + suffix).
    excerpt_idx = out.find("README excerpt:")
    assert excerpt_idx >= 0
    excerpt = out[excerpt_idx:]
    assert len(excerpt) < 1500


def test_project_summary_bad_root_returns_error(tmp_path):
    out = _exec(str(tmp_path / "does-not-exist"))
    assert out.startswith("[error]")
