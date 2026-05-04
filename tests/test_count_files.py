"""Tests for sub-project 76 — count_files tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import count_files


def test_count_files_basic(tmp_path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.py").write_text("y", encoding="utf-8")
    (tmp_path / "c.md").write_text("z", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_files()
        assert "total: 3 file" in out
        assert ".py: 2" in out
        assert ".md: 1" in out
    finally:
        _ws_current.reset(token)


def test_count_files_pattern_filter(tmp_path):
    (tmp_path / "src.py").write_text("x", encoding="utf-8")
    (tmp_path / "doc.md").write_text("y", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_files(pattern="**/*.py")
        assert "total: 1 file" in out
    finally:
        _ws_current.reset(token)


def test_count_files_skips_noise_dirs(tmp_path):
    """Files under .git / __pycache__ / .venv etc. are excluded."""
    real = tmp_path / "real.py"
    real.write_text("x", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "huge.bin").write_text("y", encoding="utf-8")
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "x.pyc").write_text("z", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_files()
        assert "total: 1 file" in out
    finally:
        _ws_current.reset(token)


def test_count_files_recursive(tmp_path):
    nested = tmp_path / "deep" / "deeper"
    nested.mkdir(parents=True)
    (nested / "file.py").write_text("x", encoding="utf-8")
    (tmp_path / "top.py").write_text("y", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_files()
        assert "total: 2 file" in out
    finally:
        _ws_current.reset(token)


def test_count_files_no_extension(tmp_path):
    (tmp_path / "noext").write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_files()
        assert "(no ext): 1" in out
    finally:
        _ws_current.reset(token)


def test_count_files_max_count_cap(tmp_path):
    for i in range(20):
        (tmp_path / f"f{i:02d}.py").write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_files(max_count=5)
        assert "total: 5 file(s) (capped)" in out
    finally:
        _ws_current.reset(token)


def test_count_files_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = count_files(root=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_count_files_not_a_directory(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_files(root="file.txt")
        assert out.startswith("[error]")
        assert "not a directory" in out
    finally:
        _ws_current.reset(token)


def test_count_files_empty_dir(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = count_files()
        assert "total: 0 file" in out
    finally:
        _ws_current.reset(token)
