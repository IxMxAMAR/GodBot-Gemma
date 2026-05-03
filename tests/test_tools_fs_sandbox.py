import pytest
from godbot.core.registry import DEFAULT
from godbot.core.workspace import Workspace, set_workspace
import godbot.tools  # discovery


@pytest.fixture
def workspace_token(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    yield ws
    set_workspace(None)


def test_read_file_inside_workspace_works(tmp_path, workspace_token):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    out = DEFAULT.execute("read_file", {"path": "x.txt"})
    assert "hello" in out


def test_read_file_outside_workspace_refused(tmp_path, workspace_token):
    out = DEFAULT.execute("read_file", {"path": "../escape.txt"})
    assert "[error] sandbox" in out


def test_read_file_absolute_outside_refused(tmp_path, workspace_token):
    target = "C:/Windows/notepad.exe" if str(tmp_path).startswith("C:") else "/etc/passwd"
    out = DEFAULT.execute("read_file", {"path": target})
    assert "[error] sandbox" in out


def test_write_file_inside_workspace_works(tmp_path, workspace_token):
    out = DEFAULT.execute("write_file", {"path": "new.txt", "content": "x"})
    assert (tmp_path / "new.txt").read_text() == "x"
    assert "ok" in out.lower()


def test_write_file_outside_refused(tmp_path, workspace_token):
    out = DEFAULT.execute("write_file", {"path": "../escape.txt", "content": "x"})
    assert "[error] sandbox" in out


def test_list_dir_inside_works(tmp_path, workspace_token):
    (tmp_path / "a.txt").write_text("x")
    out = DEFAULT.execute("list_dir", {"path": "."})
    assert "a.txt" in out


def test_list_dir_outside_refused(tmp_path, workspace_token):
    out = DEFAULT.execute("list_dir", {"path": ".."})
    assert "[error] sandbox" in out


def test_glob_root_outside_refused(tmp_path, workspace_token):
    out = DEFAULT.execute("glob", {"pattern": "*", "root": ".."})
    assert "[error] sandbox" in out


def test_grep_root_outside_refused(tmp_path, workspace_token):
    out = DEFAULT.execute("grep", {"pattern": "x", "root": ".."})
    assert "[error] sandbox" in out


def test_edit_file_outside_refused(tmp_path, workspace_token):
    out = DEFAULT.execute("edit_file", {"path": "../escape.txt", "old": "a", "new": "b"})
    assert "[error] sandbox" in out


def test_no_workspace_means_no_confinement(tmp_path):
    # No fixture: contextvar is None.
    target = tmp_path / "x.txt"
    target.write_text("hello")
    out = DEFAULT.execute("read_file", {"path": str(target)})
    assert "hello" in out


def test_hardlink_escape_refused_on_write(tmp_path, workspace_token):
    import os
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("SECRET")
    inside = tmp_path / "innocent.txt"
    try:
        os.link(str(outside), str(inside))
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hardlinks unavailable on this filesystem")
    out = DEFAULT.execute("write_file", {"path": "innocent.txt", "content": "OVERWRITTEN"})
    assert "[error] sandbox" in out
    assert "multi-link" in out or "nlink" in out
    assert outside.read_text() == "SECRET"  # outside untouched


def test_hardlink_escape_refused_on_read(tmp_path, workspace_token):
    import os
    outside = tmp_path.parent / "outside_secret_read.txt"
    outside.write_text("SECRET_CONTENT")
    inside = tmp_path / "innocent_read.txt"
    try:
        os.link(str(outside), str(inside))
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hardlinks unavailable")
    out = DEFAULT.execute("read_file", {"path": "innocent_read.txt"})
    assert "[error] sandbox" in out
    assert "SECRET_CONTENT" not in out
