"""Tests for sub-project 71 — string_diff tool."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import string_diff


def test_string_diff_identical():
    out = string_diff(a="hello\nworld\n", b="hello\nworld\n")
    assert out == "identical"


def test_string_diff_basic_change():
    out = string_diff(a="apple\nbanana\ncherry\n", b="apple\nBANANA\ncherry\n")
    assert "---" in out
    assert "+++" in out
    assert "-banana" in out
    assert "+BANANA" in out


def test_string_diff_pure_insertion():
    out = string_diff(a="a\nc\n", b="a\nb\nc\n")
    assert "+b" in out


def test_string_diff_pure_deletion():
    out = string_diff(a="a\nb\nc\n", b="a\nc\n")
    assert "-b" in out


def test_string_diff_truncates_huge_diffs():
    a = "\n".join(f"line {i}" for i in range(200))
    b = "\n".join(f"DIFF {i}" for i in range(200))
    out = string_diff(a=a, b=b, max_lines=20)
    assert "truncated at 20 lines" in out


def test_string_diff_non_string_errors():
    out = string_diff(a=123, b="x")  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_string_diff_empty_strings():
    """Both empty → 'identical'."""
    assert string_diff(a="", b="") == "identical"


def test_string_diff_one_empty():
    out = string_diff(a="", b="hello\n")
    assert "+hello" in out


def test_string_diff_is_registered_non_dangerous():
    spec = DEFAULT.spec("string_diff")
    assert spec is not None
    assert spec.dangerous is False
