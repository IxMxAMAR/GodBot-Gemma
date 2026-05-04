"""Tests for sub-project 84 — sort_lines + unique_lines."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import sort_lines, unique_lines


# --- sort_lines ----


def test_sort_lines_alphabetical():
    out = sort_lines(text="banana\napple\ncherry")
    assert out == "apple\nbanana\ncherry"


def test_sort_lines_reverse():
    out = sort_lines(text="apple\nbanana\ncherry", reverse=True)
    assert out == "cherry\nbanana\napple"


def test_sort_lines_numeric():
    out = sort_lines(text="20 items\n3 items\n100 items", numeric=True)
    assert out == "3 items\n20 items\n100 items"


def test_sort_lines_numeric_with_non_numeric_lines():
    """Non-numeric lines sort to the bottom."""
    out = sort_lines(text="10 a\nbanana\n5 b", numeric=True)
    lines = out.splitlines()
    assert lines[0] == "5 b"
    assert lines[1] == "10 a"
    # banana (non-numeric) comes last.
    assert lines[2] == "banana"


def test_sort_lines_preserves_trailing_newline():
    out = sort_lines(text="b\na\n")
    assert out == "a\nb\n"


def test_sort_lines_no_trailing_newline():
    out = sort_lines(text="b\na")
    assert out == "a\nb"


def test_sort_lines_non_string_errors():
    out = sort_lines(text=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


# --- unique_lines ----


def test_unique_lines_preserves_first_occurrence_order():
    out = unique_lines(text="b\na\nb\nc\na")
    assert out == "b\na\nc"


def test_unique_lines_preserve_order_false_sorts():
    out = unique_lines(text="banana\napple\ncherry\napple", preserve_order=False)
    assert out == "apple\nbanana\ncherry"


def test_unique_lines_preserves_trailing_newline():
    out = unique_lines(text="a\nb\na\n")
    assert out == "a\nb\n"


def test_unique_lines_empty():
    out = unique_lines(text="")
    assert out == ""


def test_unique_lines_non_string_errors():
    out = unique_lines(text=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_line_tools_registered_non_dangerous():
    for name in ("sort_lines", "unique_lines"):
        spec = DEFAULT.spec(name)
        assert spec is not None
        assert spec.dangerous is False
