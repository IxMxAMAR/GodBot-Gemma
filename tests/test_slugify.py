"""Tests for sub-project 101 — slugify tool."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import slugify


def test_slugify_basic():
    assert slugify(text="Hello World") == "hello-world"


def test_slugify_strips_punctuation():
    assert slugify(text="It's a test!") == "it-s-a-test"


def test_slugify_collapses_dashes():
    assert slugify(text="a---b") == "a-b"
    assert slugify(text="a !@# b") == "a-b"


def test_slugify_strips_leading_trailing():
    assert slugify(text="--hello--") == "hello"


def test_slugify_unicode_stripped():
    """Non-ascii alphanumerics aren't preserved in this simple slugger."""
    out = slugify(text="héllo wörld")
    # Becomes h-llo-w-rld since é/ö are non-[a-z0-9] under our regex.
    assert out == "h-llo-w-rld"


def test_slugify_empty_input():
    assert slugify(text="") == "-"


def test_slugify_only_punctuation():
    assert slugify(text="!@#$%^&*()") == "-"


def test_slugify_max_length():
    long = "a" * 200
    out = slugify(text=long, max_length=20)
    assert len(out) == 20


def test_slugify_max_length_clamped():
    """Asking for huge max_length is fine; clamps at 200."""
    long = "a" * 500
    out = slugify(text=long, max_length=99999)
    assert len(out) <= 200


def test_slugify_non_string_errors():
    out = slugify(text=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_slugify_is_registered_non_dangerous():
    spec = DEFAULT.spec("slugify")
    assert spec is not None
    assert spec.dangerous is False
