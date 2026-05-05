"""Tests for sub-project 103 — extract_links tool."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import extract_links


def test_extract_urls_only():
    out = extract_links(text="visit https://example.com or http://a.io for more", kind="url")
    assert "https://example.com" in out
    assert "http://a.io" in out


def test_extract_emails_only():
    out = extract_links(text="contact alice@example.com or bob@beta.org", kind="email")
    assert "alice@example.com" in out
    assert "bob@beta.org" in out
    # No url tag prefix in single-kind mode.
    assert "url:" not in out
    assert "email:" not in out


def test_extract_all_default():
    out = extract_links(
        text="see https://docs.example.com or email me at admin@example.com",
    )
    assert "url: https://docs.example.com" in out
    assert "email: admin@example.com" in out


def test_extract_no_matches():
    out = extract_links(text="just plain prose with nothing in it")
    assert "(no matches)" in out


def test_extract_strips_trailing_punct():
    out = extract_links(text="See https://example.com.")
    # Trailing period should be stripped from the URL match.
    assert "https://example.com" in out
    assert "https://example.com.\n" not in out


def test_extract_ftp():
    out = extract_links(text="files at ftp://files.example.com/pub/")
    assert "ftp://files.example.com" in out


def test_extract_invalid_kind_errors():
    out = extract_links(text="x", kind="not-a-kind")
    assert out.startswith("[error]")


def test_extract_non_string_errors():
    out = extract_links(text=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_extract_preserves_appearance_order():
    text = "first https://a.com then bob@b.com then https://c.com"
    out = extract_links(text=text)
    lines = out.splitlines()
    # Order in output should mirror order in text.
    assert "a.com" in lines[0]
    assert "b.com" in lines[1]
    assert "c.com" in lines[2]


def test_extract_links_registered_non_dangerous():
    spec = DEFAULT.spec("extract_links")
    assert spec is not None
    assert spec.dangerous is False
