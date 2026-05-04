"""Tests for the apply_patch tool (sub-project 22)."""
from __future__ import annotations

import pytest

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.patch import (
    _apply_hunks,
    _find_anchor,
    Hunk,
    apply_patch,
    parse_patch,
)


# --- parser tests ----


def test_parse_patch_single_file_single_hunk():
    text = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        " a\n"
        "-b\n"
        "+c\n"
    )
    diffs = parse_patch(text)
    assert len(diffs) == 1
    assert diffs[0].path == "foo.py"
    assert len(diffs[0].hunks) == 1
    h = diffs[0].hunks[0]
    assert h.old_start == 1
    assert h.old_count == 2
    assert h.new_start == 1
    assert h.new_count == 2
    assert h.lines == [" a", "-b", "+c"]


def test_parse_patch_strips_a_prefix_too():
    text = (
        "--- a/foo.py\n"
        "+++ a/foo.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    diffs = parse_patch(text)
    assert diffs[0].path == "foo.py"


def test_parse_patch_multi_file():
    text = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -1 +1 @@\n"
        "-foo\n"
        "+bar\n"
    )
    diffs = parse_patch(text)
    assert [d.path for d in diffs] == ["a.py", "b.py"]


def test_parse_patch_default_count_is_one():
    """`@@ -1 +1 @@` (no `,N`) means count of 1 on each side."""
    text = (
        "--- a/x\n"
        "+++ b/x\n"
        "@@ -3 +3 @@\n"
        "-old\n"
        "+new\n"
    )
    diffs = parse_patch(text)
    h = diffs[0].hunks[0]
    assert h.old_count == 1
    assert h.new_count == 1


def test_parse_patch_handles_no_newline_marker():
    text = (
        "--- a/x\n"
        "+++ b/x\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "\\ No newline at end of file\n"
    )
    diffs = parse_patch(text)
    assert len(diffs) == 1
    assert diffs[0].hunks[0].lines == ["-old", "+new"]


def test_parse_patch_no_file_headers_raises():
    with pytest.raises(ValueError):
        parse_patch("just some text\n")


def test_parse_patch_dangling_minus_raises():
    with pytest.raises(ValueError):
        parse_patch("--- a/x\n")  # no +++


# --- application tests ----


def test_apply_hunks_basic_replacement():
    src = "a\nb\nc\n"
    hunk = Hunk(old_start=2, old_count=1, new_start=2, new_count=1, lines=["-b", "+B"])
    out = _apply_hunks(src, [hunk])
    assert out == "a\nB\nc\n"


def test_apply_hunks_with_context():
    src = "a\nb\nc\nd\n"
    hunk = Hunk(
        old_start=1, old_count=4, new_start=1, new_count=4,
        lines=[" a", "-b", "+B", " c", " d"],
    )
    out = _apply_hunks(src, [hunk])
    assert out == "a\nB\nc\nd\n"


def test_apply_hunks_pure_insert():
    src = "a\nc\n"
    hunk = Hunk(
        old_start=1, old_count=2, new_start=1, new_count=3,
        lines=[" a", "+b", " c"],
    )
    out = _apply_hunks(src, [hunk])
    assert out == "a\nb\nc\n"


def test_apply_hunks_pure_delete():
    src = "a\nb\nc\n"
    hunk = Hunk(
        old_start=1, old_count=3, new_start=1, new_count=2,
        lines=[" a", "-b", " c"],
    )
    out = _apply_hunks(src, [hunk])
    assert out == "a\nc\n"


def test_apply_hunks_context_mismatch_raises():
    src = "x\ny\n"
    hunk = Hunk(old_start=1, old_count=1, new_start=1, new_count=1, lines=["-NOT_X", "+Z"])
    with pytest.raises(ValueError):
        _apply_hunks(src, [hunk])


def test_apply_hunks_multiple_hunks_apply_bottom_up():
    src = "a\nb\nc\nd\n"
    h1 = Hunk(old_start=1, old_count=1, new_start=1, new_count=1, lines=["-a", "+A"])
    h2 = Hunk(old_start=4, old_count=1, new_start=4, new_count=1, lines=["-d", "+D"])
    out = _apply_hunks(src, [h1, h2])
    assert out == "A\nb\nc\nD\n"


def test_apply_hunks_preserves_no_trailing_newline():
    src = "a\nb"
    hunk = Hunk(old_start=2, old_count=1, new_start=2, new_count=1, lines=["-b", "+B"])
    out = _apply_hunks(src, [hunk])
    assert out == "a\nB"


def test_find_anchor_strict():
    src = ["a", "b", "c"]
    hunk = Hunk(old_start=2, old_count=1, new_start=2, new_count=1, lines=["-b", "+B"])
    assert _find_anchor(src, hunk, fuzzy=False) == 1


def test_find_anchor_fuzzy_finds_when_lineno_stale():
    src = ["pad1", "pad2", "a", "b", "c", "d"]
    # Hunk thinks the file starts at line 1 with `a` — but in this file
    # `a` is at line 3.
    hunk = Hunk(
        old_start=1, old_count=2, new_start=1, new_count=2,
        lines=[" a", "-b", "+B"],
    )
    assert _find_anchor(src, hunk, fuzzy=False) is None
    assert _find_anchor(src, hunk, fuzzy=True) == 2


# --- end-to-end via apply_patch ----


def test_apply_patch_end_to_end_workspace_confined(tmp_path):
    f = tmp_path / "main.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = apply_patch(patch=(
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def foo():\n"
            "-    return 1\n"
            "+    return 42\n"
        ))
        assert out.startswith("ok:")
        assert "main.py" in out
        new_content = f.read_text(encoding="utf-8")
        assert "return 42" in new_content
    finally:
        _ws_current.reset(token)


def test_apply_patch_atomic_on_failure(tmp_path):
    """If any file's hunk fails, NONE of the files are written."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("a1\na2\n", encoding="utf-8")
    b.write_text("b1\nb2\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        # Patch for a.py is fine; patch for b.py has a context mismatch.
        out = apply_patch(patch=(
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-a1\n"
            "+A1\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1 +1 @@\n"
            "-NOT_THERE\n"
            "+X\n"
        ))
        assert out.startswith("[error]")
        # a.py must not have been touched.
        assert a.read_text(encoding="utf-8") == "a1\na2\n"
        assert b.read_text(encoding="utf-8") == "b1\nb2\n"
    finally:
        _ws_current.reset(token)


def test_apply_patch_missing_file_errors(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = apply_patch(patch=(
            "--- a/no_such.py\n"
            "+++ b/no_such.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        ))
        assert out.startswith("[error]")
        assert "not found" in out
    finally:
        _ws_current.reset(token)


def test_apply_patch_workspace_escape_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    ws = Workspace.of(str(ws_dir))
    outside = tmp_path / "outside.py"
    outside.write_text("x\n", encoding="utf-8")
    token = set_workspace(ws)
    try:
        out = apply_patch(patch=(
            "--- a/../outside.py\n"
            "+++ b/../outside.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        ))
        assert out.startswith("[error]")
        assert "sandbox" in out.lower()
    finally:
        _ws_current.reset(token)


def test_apply_patch_multi_file(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("a1\n", encoding="utf-8")
    b.write_text("b1\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = apply_patch(patch=(
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-a1\n"
            "+A1\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1 +1 @@\n"
            "-b1\n"
            "+B1\n"
        ))
        assert out.startswith("ok:")
        assert a.read_text(encoding="utf-8") == "A1\n"
        assert b.read_text(encoding="utf-8") == "B1\n"
    finally:
        _ws_current.reset(token)
