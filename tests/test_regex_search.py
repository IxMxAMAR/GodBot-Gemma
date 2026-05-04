"""Tests for sub-project 70 — regex_search tool."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import regex_search


def test_regex_search_basic():
    out = regex_search(pattern=r"\d+", text="abc 123 def 456 ghi")
    assert "2 match(es)" in out
    assert "4: 123" in out
    assert "12: 456" in out


def test_regex_search_no_matches():
    out = regex_search(pattern="nope", text="hello world")
    assert "(no matches)" in out


def test_regex_search_invalid_pattern():
    out = regex_search(pattern="(unclosed", text="hi")
    assert out.startswith("[error]")
    assert "regex" in out


def test_regex_search_empty_pattern():
    out = regex_search(pattern="", text="hi")
    assert out.startswith("[error]")
    assert "pattern required" in out


def test_regex_search_ignore_case():
    out = regex_search(pattern="hello", text="HELLO World", ignore_case=True)
    assert "1 match" in out
    assert "0: HELLO" in out


def test_regex_search_max_matches_cap():
    """Asking for more matches than max_matches should truncate."""
    text = "a a a a a a a a a a a"  # 11 a's
    out = regex_search(pattern="a", text=text, max_matches=3)
    assert "3 match" in out
    assert "stopped at max_matches=3" in out


def test_regex_search_hard_cap_500():
    """max_matches > 500 is clamped to 500."""
    text = "x" * 1000
    out = regex_search(pattern="x", text=text, max_matches=99999)
    # Output reports stopped at 500.
    assert "stopped at max_matches=500" in out


def test_regex_search_capture_group_uses_full_match():
    """Output uses group(0) — the full match, not group(1)."""
    out = regex_search(pattern=r"(\w+)@(\w+)", text="contact me at alice@example or bob@beta")
    # Both full email-like matches should appear, not just the captured groups.
    assert "alice@example" in out
    assert "bob@beta" in out


def test_regex_search_unicode_safe():
    out = regex_search(pattern=r"\w+", text="héllo wörld")
    # Both words match in Unicode-aware mode.
    assert "héllo" in out
    assert "wörld" in out


def test_regex_search_is_registered_non_dangerous():
    spec = DEFAULT.spec("regex_search")
    assert spec is not None
    assert spec.dangerous is False
