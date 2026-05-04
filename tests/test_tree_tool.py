"""Tests for sub-project 81 — tree directory visualizer."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import tree


def test_tree_basic(tmp_path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.py").write_text("y", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("z", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tree(path=".")
        assert "sub/" in out
        assert "a.py" in out
        assert "b.py" in out
        # Connectors used:
        assert "├── " in out or "└── " in out
    finally:
        _ws_current.reset(token)


def test_tree_max_depth(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "leaf.py").write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        # max_depth=1 should NOT descend into a/b/c.
        out = tree(path=".", max_depth=1)
        assert "a/" in out
        assert "leaf.py" not in out
        # max_depth=4 should reveal everything.
        out_deep = tree(path=".", max_depth=4)
        assert "leaf.py" in out_deep
    finally:
        _ws_current.reset(token)


def test_tree_skips_noise_dirs(tmp_path):
    (tmp_path / "real.py").write_text("x", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("q", encoding="utf-8")
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tree(path=".")
        assert ".git" not in out
        assert "__pycache__" not in out
        assert "real.py" in out
    finally:
        _ws_current.reset(token)


def test_tree_max_entries_cap(tmp_path):
    """Producing > max_entries lines should append a truncation marker."""
    for i in range(50):
        (tmp_path / f"f{i:02d}.txt").write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tree(path=".", max_entries=10)
        assert "[truncated]" in out
    finally:
        _ws_current.reset(token)


def test_tree_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = tree(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_tree_not_a_directory(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tree(path="file.txt")
        assert out.startswith("[error]")
        assert "not a directory" in out
    finally:
        _ws_current.reset(token)


def test_tree_empty_directory(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tree(path=".")
        # Just the root path on the first line.
        assert str(tmp_path.resolve()) in out
    finally:
        _ws_current.reset(token)


def test_tree_directories_render_with_trailing_slash(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "leaf").write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = tree(path=".")
        assert "subdir/" in out
    finally:
        _ws_current.reset(token)
