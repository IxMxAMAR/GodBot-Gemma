"""Tests for sub-project 104 — summarize_diff tool."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import summarize_diff


def test_basic_single_file():
    diff = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def foo():\n"
        "+    print('hi')\n"
        "     return 1\n"
    )
    out = summarize_diff(diff=diff)
    assert "1 file(s)" in out
    assert "+1" in out
    assert "-0" in out
    assert "foo.py" in out


def test_multi_file():
    diff = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -1,2 +1,1 @@\n"
        "-line1\n"
        "-line2\n"
        "+merged\n"
    )
    out = summarize_diff(diff=diff)
    assert "2 file(s)" in out
    # a.py contributes +1 -1, b.py contributes +1 -2; total +2 -3.
    assert "+2 -3" in out


def test_strips_a_b_prefix():
    diff = (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    out = summarize_diff(diff=diff)
    # Path should be normalised without the "b/" prefix.
    assert "src/main.py" in out
    assert "b/src/main.py" not in out


def test_pure_addition():
    diff = (
        "--- /dev/null\n"
        "+++ b/new.txt\n"
        "@@ -0,0 +1,3 @@\n"
        "+line1\n"
        "+line2\n"
        "+line3\n"
    )
    out = summarize_diff(diff=diff)
    assert "+3" in out
    assert "-0" in out


def test_pure_deletion():
    diff = (
        "--- a/old.txt\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-line1\n"
        "-line2\n"
    )
    out = summarize_diff(diff=diff)
    assert "+0" in out
    assert "-2" in out


def test_no_diff_headers_errors():
    out = summarize_diff(diff="just plain text\nno diff headers here")
    assert out.startswith("[error]")
    assert "+++" in out


def test_empty_input_errors():
    out = summarize_diff(diff="")
    assert out.startswith("[error]")


def test_non_string_errors():
    out = summarize_diff(diff=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_summarize_diff_registered_non_dangerous():
    spec = DEFAULT.spec("summarize_diff")
    assert spec is not None
    assert spec.dangerous is False
