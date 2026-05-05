"""Tests for sub-project 102 — levenshtein tool."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import levenshtein


def test_identical_strings():
    out = levenshtein(a="hello", b="hello")
    assert "distance: 0" in out
    assert "similarity: 1.00" in out


def test_one_substitution():
    out = levenshtein(a="hello", b="hallo")
    assert "distance: 1" in out


def test_one_insertion():
    out = levenshtein(a="cat", b="cats")
    assert "distance: 1" in out


def test_one_deletion():
    out = levenshtein(a="cats", b="cat")
    assert "distance: 1" in out


def test_completely_different():
    out = levenshtein(a="abc", b="xyz")
    assert "distance: 3" in out
    assert "similarity: 0.00" in out


def test_empty_left():
    out = levenshtein(a="", b="hello")
    assert "distance: 5" in out


def test_empty_right():
    out = levenshtein(a="world", b="")
    assert "distance: 5" in out


def test_both_empty():
    out = levenshtein(a="", b="")
    assert "distance: 0" in out


def test_classic_kitten_to_sitting():
    """Classic Levenshtein example: kitten -> sitting = distance 3."""
    out = levenshtein(a="kitten", b="sitting")
    assert "distance: 3" in out


def test_non_string_errors():
    out = levenshtein(a="hi", b=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_levenshtein_registered_non_dangerous():
    spec = DEFAULT.spec("levenshtein")
    assert spec is not None
    assert spec.dangerous is False


def test_long_strings_truncated():
    """Inputs longer than 5000 chars get truncated; the call doesn't crash."""
    long_a = "a" * 6000
    long_b = "a" * 6000
    out = levenshtein(a=long_a, b=long_b)
    # After truncation both are "a"*5000 so distance is 0.
    assert "distance: 0" in out
