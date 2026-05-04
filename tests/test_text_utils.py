"""Tests for sub-project 77 — text_replace + text_truncate."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import text_replace, text_truncate


# --- text_replace ----


def test_text_replace_basic():
    out = text_replace(text="hello world", old="world", new="there")
    assert "replaced 1" in out
    assert "hello there" in out


def test_text_replace_multiple_occurrences():
    out = text_replace(text="a a a a", old="a", new="b")
    assert "replaced 4" in out
    assert "b b b b" in out


def test_text_replace_count_cap():
    out = text_replace(text="a a a a", old="a", new="b", count=2)
    assert "replaced 2" in out
    assert "b b a a" in out


def test_text_replace_no_matches():
    out = text_replace(text="hello", old="world", new="x")
    assert "replaced 0" in out
    assert "hello" in out


def test_text_replace_empty_old_errors():
    out = text_replace(text="hi", old="", new="x")
    assert out.startswith("[error]")


def test_text_replace_non_string_errors():
    out = text_replace(text=123, old="x", new="y")  # type: ignore[arg-type]
    assert out.startswith("[error]")


# --- text_truncate ----


def test_text_truncate_short_passthrough():
    assert text_truncate(text="short") == "short"


def test_text_truncate_with_default_suffix():
    out = text_truncate(text="x" * 200, max_chars=20)
    assert len(out) == 20
    assert out.endswith("...")


def test_text_truncate_custom_suffix():
    out = text_truncate(text="abcdefghij", max_chars=6, suffix="[…]")
    assert out.endswith("[…]")
    # Total length should be exactly 6 (3 chars + 3-char suffix).
    assert len(out) == 6


def test_text_truncate_no_suffix():
    out = text_truncate(text="abcdef", max_chars=3, suffix="")
    assert out == "abc"


def test_text_truncate_max_chars_clamps():
    """max_chars below 1 is clamped to 1."""
    out = text_truncate(text="hello", max_chars=0)
    assert len(out) >= 1


def test_text_truncate_non_string_errors():
    out = text_truncate(text=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_text_utils_registered_non_dangerous():
    for name in ("text_replace", "text_truncate"):
        spec = DEFAULT.spec(name)
        assert spec is not None
        assert spec.dangerous is False
