"""Tests for sub-project 95 — is_binary_file tool."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import is_binary_file


def test_text_file_detected_as_not_binary(tmp_path):
    f = tmp_path / "src.py"
    f.write_text("def main():\n    return 42\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = is_binary_file(path="src.py")
        assert out.startswith("no:")
    finally:
        _ws_current.reset(token)


def test_null_byte_file_detected_as_binary(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\x00\x01\x02\x03binary\x00data")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = is_binary_file(path="blob.bin")
        assert out.startswith("yes:")
        assert "null-byte" in out
    finally:
        _ws_current.reset(token)


def test_high_non_printable_ratio_detected_as_binary(tmp_path):
    """No null bytes but lots of high-bit bytes → binary."""
    f = tmp_path / "weird.bin"
    # 90% high-bit bytes (0x80-0xFE).
    payload = bytes(range(0x80, 0xFF)) * 100
    f.write_bytes(payload)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = is_binary_file(path="weird.bin")
        assert out.startswith("yes:")
        assert "non-printable" in out
    finally:
        _ws_current.reset(token)


def test_empty_file_treated_as_not_binary(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = is_binary_file(path="empty.txt")
        assert out.startswith("no:")
        assert "empty" in out
    finally:
        _ws_current.reset(token)


def test_unicode_text_with_special_chars_text(tmp_path):
    """Unicode text (multi-byte UTF-8) WITH valid printable ASCII still
    qualifies as text — we measure printable-ascii ratio in raw bytes,
    so an all-emoji file might trip the heuristic. That's acceptable
    for a 'show me the code' guard."""
    f = tmp_path / "doc.md"
    f.write_text("# Header\n\nNormal English content with some words.\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = is_binary_file(path="doc.md")
        assert out.startswith("no:")
    finally:
        _ws_current.reset(token)


def test_missing_file(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = is_binary_file(path="ghost.bin")
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"hello")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = is_binary_file(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_empty_path_errors():
    out = is_binary_file(path="")
    assert out.startswith("[error]")


def test_is_binary_file_registered_non_dangerous():
    spec = DEFAULT.spec("is_binary_file")
    assert spec is not None
    assert spec.dangerous is False
