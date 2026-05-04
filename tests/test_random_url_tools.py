"""Tests for sub-project 62 — random_id + urlquote/unquote."""
from __future__ import annotations

import re

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import random_id, urlquote, urlunquote


# --- random_id ----


def test_random_id_uuid_default():
    out = random_id()
    # UUID4: 8-4-4-4-12 hex chars.
    assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", out)


def test_random_id_uuid_distinct():
    out_a = random_id()
    out_b = random_id()
    assert out_a != out_b


def test_random_id_hex():
    out = random_id(kind="hex", length=16)
    assert len(out) == 16
    assert all(c in "0123456789abcdef" for c in out)


def test_random_id_hex_clamps_length():
    out = random_id(kind="hex", length=99999)
    assert len(out) <= 256


def test_random_id_slug():
    out = random_id(kind="slug", length=8)
    # URL-safe base64 alphabet.
    assert re.match(r"^[A-Za-z0-9_-]+$", out)


def test_random_id_int():
    out = random_id(kind="int")
    assert out.isdigit()
    n = int(out)
    assert 0 <= n < 2**63


def test_random_id_unknown_kind():
    out = random_id(kind="not-a-kind")
    assert out.startswith("[error]")


def test_random_id_is_registered_non_dangerous():
    spec = DEFAULT.spec("random_id")
    assert spec is not None
    assert spec.dangerous is False


# --- urlquote / urlunquote ----


def test_urlquote_basic():
    out = urlquote(text="hello world")
    assert out == "hello%20world"


def test_urlquote_safe_chars():
    out = urlquote(text="a/b?c", safe="/")
    assert out == "a/b%3Fc"


def test_urlquote_no_unicode_escape_for_safe_chars():
    out = urlquote(text="abc")
    assert out == "abc"


def test_urlunquote_basic():
    out = urlunquote(text="hello%20world")
    assert out == "hello world"


def test_urlquote_round_trip():
    raw = "complex string with /spaces/ & symbols ?query=1"
    encoded = urlquote(text=raw)
    assert urlunquote(text=encoded) == raw
