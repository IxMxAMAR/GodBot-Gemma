"""Tests for sub-project 56 — directory_size tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import directory_size


def test_directory_size_basic(tmp_path):
    (tmp_path / "small.txt").write_text("a" * 100, encoding="utf-8")
    (tmp_path / "medium.txt").write_text("b" * 1024, encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = directory_size(path=".")
        assert "total:" in out
        assert "2 file(s)" in out
        # Both files should appear in top-10 list.
        assert "small.txt" in out
        assert "medium.txt" in out
    finally:
        _ws_current.reset(token)


def test_directory_size_top_10_cap(tmp_path):
    """20 files → top-10 list shows at most 10 entries."""
    for i in range(20):
        (tmp_path / f"f{i:02d}.txt").write_text("x" * (i + 1), encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = directory_size(path=".")
        # Count lines starting with two-space indent (top-10 entries).
        listed = [ln for ln in out.splitlines() if ln.startswith("  ") and "f" in ln]
        assert len(listed) == 10
        # Largest file (f19.txt, 20 bytes) should be at the top.
        assert "f19.txt" in listed[0]
    finally:
        _ws_current.reset(token)


def test_directory_size_skips_git_dir(tmp_path):
    """The .git directory is excluded from the walk."""
    (tmp_path / "real.txt").write_text("x" * 50, encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "huge.bin").write_text("x" * 999_999, encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = directory_size(path=".")
        # The 999k .git/huge.bin should NOT be in the walk total.
        assert ".git" not in out
        assert "huge.bin" not in out
        assert "1 file(s)" in out
    finally:
        _ws_current.reset(token)


def test_directory_size_recursive(tmp_path):
    """Files in nested directories are counted."""
    nested = tmp_path / "deep" / "deeper"
    nested.mkdir(parents=True)
    (nested / "f.txt").write_text("x" * 200, encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = directory_size(path=".")
        assert "1 file(s)" in out
        assert "f.txt" in out
    finally:
        _ws_current.reset(token)


def test_directory_size_empty_dir(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = directory_size(path=".")
        assert "0 file(s)" in out
        assert "0 B" in out
    finally:
        _ws_current.reset(token)


def test_directory_size_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = directory_size(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_directory_size_not_a_directory(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = directory_size(path="file.txt")
        assert out.startswith("[error]")
        assert "not a directory" in out
    finally:
        _ws_current.reset(token)


def test_directory_size_human_readable_units(tmp_path):
    """Files are reported in B/KB/MB/etc."""
    (tmp_path / "kbsize.txt").write_text("x" * 5000, encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = directory_size(path=".")
        # 5000 bytes = ~4.9 KB, should render with KB suffix.
        assert "KB" in out
    finally:
        _ws_current.reset(token)
