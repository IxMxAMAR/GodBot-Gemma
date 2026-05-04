"""Tests for the preview_patch tool (sub-project 24)."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.patch import preview_patch


def test_preview_patch_invalid_parse_returns_error():
    out = preview_patch(patch="garbage")
    assert out.startswith("[error]")


def test_preview_patch_reports_ok_and_does_not_write(tmp_path):
    f = tmp_path / "main.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = preview_patch(patch=(
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def foo():\n"
            "-    return 1\n"
            "+    return 42\n"
        ))
        assert out.startswith("ok main.py")
        assert "+1" in out
        assert "-1" in out
        assert "1 hunk" in out
        # The file content must NOT have changed.
        assert f.read_text(encoding="utf-8") == "def foo():\n    return 1\n"
    finally:
        _ws_current.reset(token)


def test_preview_patch_reports_failure_per_file_and_summary(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("a1\n", encoding="utf-8")
    b.write_text("b1\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = preview_patch(patch=(
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-a1\n"
            "+A1\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1 +1 @@\n"
            "-WRONG\n"
            "+X\n"
        ))
        # First file ok, second file fails.
        assert "ok a.py" in out
        assert "FAIL b.py" in out
        assert "summary: 1 file ok, 1 failed" in out
        # Both files untouched.
        assert a.read_text(encoding="utf-8") == "a1\n"
        assert b.read_text(encoding="utf-8") == "b1\n"
    finally:
        _ws_current.reset(token)


def test_preview_patch_missing_file_fails_cleanly(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = preview_patch(patch=(
            "--- a/no_such.py\n"
            "+++ b/no_such.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        ))
        assert "FAIL no_such.py" in out
        assert "0 file" in out and "1 failed" in out
    finally:
        _ws_current.reset(token)


def test_preview_patch_aggregates_totals_across_files(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("a1\na2\n", encoding="utf-8")
    b.write_text("b1\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = preview_patch(patch=(
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,2 +1,3 @@\n"
            "-a1\n"
            "+A1\n"
            "+ANEW\n"
            " a2\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1 +1 @@\n"
            "-b1\n"
            "+B1\n"
        ))
        # a.py contributes +2 -1, b.py contributes +1 -1, total +3 -2.
        assert "+3 -2 total" in out
        assert "2 files ok" in out
    finally:
        _ws_current.reset(token)


def test_preview_patch_workspace_escape_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("x\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = preview_patch(patch=(
            "--- a/../outside.py\n"
            "+++ b/../outside.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        ))
        assert "FAIL" in out
        assert "sandbox" in out.lower()
    finally:
        _ws_current.reset(token)
