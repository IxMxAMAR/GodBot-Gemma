from pathlib import Path
import pytest
from godbot.core.workspace import (
    Workspace, WorkspaceEscape, current_workspace, set_workspace,
)


def test_of_with_none_returns_none():
    assert Workspace.of(None) is None


def test_of_with_str_resolves_to_absolute(tmp_path):
    ws = Workspace.of(str(tmp_path))
    assert ws is not None
    assert ws.root.is_absolute()
    assert ws.root == tmp_path.resolve()


def test_of_with_nonexistent_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Workspace.of(str(tmp_path / "does_not_exist"))


def test_of_with_file_not_dir_raises(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        Workspace.of(str(f))


def test_confine_relative_path(tmp_path):
    ws = Workspace.of(str(tmp_path))
    out = ws.confine("a/b.txt")
    assert out == (tmp_path / "a" / "b.txt").resolve()


def test_confine_absolute_inside(tmp_path):
    ws = Workspace.of(str(tmp_path))
    target = tmp_path / "x.txt"
    out = ws.confine(str(target))
    assert out == target.resolve()


def test_confine_absolute_outside_raises(tmp_path):
    ws = Workspace.of(str(tmp_path))
    with pytest.raises(WorkspaceEscape):
        ws.confine("C:/Windows" if str(tmp_path).startswith("C:") else "/etc")


def test_confine_dotdot_escape_raises(tmp_path):
    ws = Workspace.of(str(tmp_path))
    with pytest.raises(WorkspaceEscape):
        ws.confine("../../escape.txt")


def test_confine_dotdot_inside_ok(tmp_path):
    ws = Workspace.of(str(tmp_path))
    sub = tmp_path / "sub"
    sub.mkdir()
    out = ws.confine("sub/../within.txt")
    assert out == (tmp_path / "within.txt").resolve()


def test_confine_symlink_to_outside_raises(tmp_path):
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlinks unavailable")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x")
    inside_link = tmp_path / "link"
    try:
        inside_link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks require admin on Windows")
    ws = Workspace.of(str(tmp_path))
    with pytest.raises(WorkspaceEscape):
        ws.confine("link")


def test_is_inside(tmp_path):
    ws = Workspace.of(str(tmp_path))
    assert ws.is_inside(str(tmp_path / "x"))
    assert not ws.is_inside(str(tmp_path.parent / "x"))


def test_contextvar_set_and_get(tmp_path):
    assert current_workspace() is None
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        assert current_workspace() is ws
    finally:
        import contextvars
        # Reset by setting back to None.
        set_workspace(None)
    assert current_workspace() is None


def test_auto_approve_default_false(tmp_path):
    ws = Workspace.of(str(tmp_path))
    assert ws.auto_approve_in_sandbox is False


def test_auto_approve_true(tmp_path):
    ws = Workspace.of(str(tmp_path), auto_approve=True)
    assert ws.auto_approve_in_sandbox is True
