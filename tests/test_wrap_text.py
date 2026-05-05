"""Tests for sub-project 106 — wrap_text tool."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import wrap_text


def test_wrap_basic():
    text = "This is a long line that should get wrapped at the given width."
    out = wrap_text(text=text, width=20)
    for line in out.splitlines():
        assert len(line) <= 20


def test_wrap_short_passthrough():
    """Text already under width returns unchanged."""
    assert wrap_text(text="short", width=80) == "short"


def test_wrap_preserves_paragraphs():
    text = "First paragraph here.\n\nSecond paragraph here."
    out = wrap_text(text=text, width=80)
    assert "\n\n" in out
    assert "First paragraph" in out
    assert "Second paragraph" in out


def test_wrap_indent():
    text = "This is one long sentence that wraps."
    out = wrap_text(text=text, width=20, indent="  > ")
    for line in out.splitlines():
        assert line.startswith("  > ")


def test_wrap_width_clamped_low():
    """width below 10 clamps up."""
    out = wrap_text(text="a b c d e f g", width=1)
    # If it actually used width=1 we'd get one word per line; with the
    # clamp at 10 we should fit at least a few words on a line.
    lines = out.splitlines()
    assert any(len(ln) > 1 for ln in lines)


def test_wrap_empty():
    assert wrap_text(text="") == ""


def test_wrap_non_string_errors():
    out = wrap_text(text=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_wrap_non_string_indent_errors():
    out = wrap_text(text="hi", indent=123)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_wrap_text_registered_non_dangerous():
    spec = DEFAULT.spec("wrap_text")
    assert spec is not None
    assert spec.dangerous is False
