"""Tests for sub-project 74 — parse_url tool."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import parse_url


def test_parse_url_https_with_path_and_query():
    out = parse_url(url="https://example.com/foo/bar?x=1&y=2#section")
    assert "scheme: https" in out
    assert "host: example.com" in out
    assert "port: default" in out
    assert "path: /foo/bar" in out
    assert "query: x=1&y=2" in out
    assert "x=1" in out
    assert "y=2" in out
    assert "fragment: section" in out


def test_parse_url_explicit_port():
    out = parse_url(url="http://localhost:8080/api")
    assert "port: 8080" in out


def test_parse_url_no_scheme_relative():
    out = parse_url(url="/just/a/path")
    assert "scheme: (none)" in out
    assert "path: /just/a/path" in out


def test_parse_url_no_query_or_fragment():
    out = parse_url(url="https://example.com/")
    assert "query: (empty)" in out
    assert "fragment:" not in out  # only present when there IS one


def test_parse_url_repeated_query_keys():
    out = parse_url(url="https://example.com/?tag=a&tag=b")
    assert "params (2)" in out
    # Both occurrences should appear.
    assert "tag=a" in out
    assert "tag=b" in out


def test_parse_url_empty_value():
    """Blank query values are kept (parse_qsl with keep_blank_values=True)."""
    out = parse_url(url="https://example.com/?empty=&keyed=1")
    assert "params (2)" in out
    assert "empty=" in out


def test_parse_url_empty_input_errors():
    out = parse_url(url="")
    assert out.startswith("[error]")


def test_parse_url_whitespace_only_errors():
    out = parse_url(url="   ")
    assert out.startswith("[error]")


def test_parse_url_is_registered_non_dangerous():
    spec = DEFAULT.spec("parse_url")
    assert spec is not None
    assert spec.dangerous is False
