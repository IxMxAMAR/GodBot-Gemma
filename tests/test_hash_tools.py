"""Tests for sub-project 68 — hash_text + hash_file."""
from __future__ import annotations

import hashlib

from godbot.core.registry import DEFAULT
from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import hash_file, hash_text


def test_hash_text_sha256_default():
    out = hash_text(text="hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert out == f"sha256: {expected}"


def test_hash_text_md5():
    out = hash_text(text="hello", algorithm="md5")
    expected = hashlib.md5(b"hello").hexdigest()
    assert out == f"md5: {expected}"


def test_hash_text_unknown_algo():
    out = hash_text(text="x", algorithm="nope")
    assert out.startswith("[error]")
    assert "unknown algorithm" in out


def test_hash_text_non_string_errors():
    out = hash_text(text=12345)  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_hash_text_unicode_consistent():
    """Same input always hashes to the same digest."""
    a = hash_text(text="héllo")
    b = hash_text(text="héllo")
    assert a == b


def test_hash_file_basic(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello world", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = hash_file(path="data.txt")
        # SHA256 of "hello world".
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert expected in out
        assert "11 bytes" in out
    finally:
        _ws_current.reset(token)


def test_hash_file_streaming_large(tmp_path):
    """A file larger than the 64 KB chunk size should still hash correctly."""
    f = tmp_path / "big.bin"
    payload = (b"abc" * 100_000)  # ~300 KB
    f.write_bytes(payload)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = hash_file(path="big.bin")
        expected = hashlib.sha256(payload).hexdigest()
        assert expected in out
        assert f"{len(payload)} bytes" in out
    finally:
        _ws_current.reset(token)


def test_hash_file_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = hash_file(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_hash_file_missing_errors(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = hash_file(path="ghost.bin")
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_hash_file_empty_path():
    out = hash_file(path="")
    assert out.startswith("[error]")
    assert "path required" in out


def test_hash_tools_registered_non_dangerous():
    for name in ("hash_text", "hash_file"):
        spec = DEFAULT.spec(name)
        assert spec is not None
        assert spec.dangerous is False
