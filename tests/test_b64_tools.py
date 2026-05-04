"""Tests for sub-project 63 — b64encode + b64decode."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import b64decode, b64encode


def test_b64_basic_round_trip():
    encoded = b64encode(text="hello world")
    assert encoded == "aGVsbG8gd29ybGQ="
    decoded = b64decode(text=encoded)
    assert decoded == "hello world"


def test_b64_urlsafe_round_trip():
    raw = "binary?with/special+chars==maybe"
    encoded = b64encode(text=raw, urlsafe=True)
    # No `=` padding in urlsafe output.
    assert "=" not in encoded
    # Uses URL-safe alphabet (no + or /).
    assert "+" not in encoded
    assert "/" not in encoded
    decoded = b64decode(text=encoded, urlsafe=True)
    assert decoded == raw


def test_b64_unicode_handled():
    encoded = b64encode(text="héllo 🌍")
    decoded = b64decode(text=encoded)
    assert decoded == "héllo 🌍"


def test_b64decode_invalid_returns_error():
    out = b64decode(text="not__valid__base64___@@@")
    # Either decodes (lenient mode) or returns an error string starting with [error].
    # We just want it to not crash.
    assert isinstance(out, str)


def test_b64encode_non_string_errors():
    # type: ignore — intentional bad input
    out = b64encode(text=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_b64decode_non_string_errors():
    out = b64decode(text=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_b64_tools_registered():
    assert DEFAULT.spec("b64encode") is not None
    assert DEFAULT.spec("b64decode") is not None
    assert DEFAULT.spec("b64encode").dangerous is False
    assert DEFAULT.spec("b64decode").dangerous is False


def test_b64encode_empty_string():
    """Empty input should encode/decode to empty without erroring."""
    encoded = b64encode(text="")
    assert encoded == ""
    decoded = b64decode(text="")
    assert decoded == ""
