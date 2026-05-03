from pathlib import Path
import godbot.tools  # discovery
from godbot.core.registry import DEFAULT


def test_write_file_creates(tmp_path):
    p = tmp_path / "new.txt"
    out = DEFAULT.execute("write_file", {"path": str(p), "content": "hi"})
    assert p.read_text() == "hi"
    assert "wrote" in out.lower() or "ok" in out.lower()


def test_write_file_creates_parent(tmp_path):
    p = tmp_path / "a" / "b" / "x.txt"
    DEFAULT.execute("write_file", {"path": str(p), "content": "hi"})
    assert p.read_text() == "hi"


def test_edit_file_replaces(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello world\n")
    DEFAULT.execute("edit_file", {"path": str(p), "old": "world", "new": "earth"})
    assert p.read_text() == "hello earth\n"


def test_edit_file_old_not_found_returns_error(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello\n")
    out = DEFAULT.execute("edit_file", {"path": str(p), "old": "xxx", "new": "yyy"})
    assert "not found" in out.lower()
    assert p.read_text() == "hello\n"


def test_edit_file_old_not_unique_returns_error(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("foo bar foo\n")
    out = DEFAULT.execute("edit_file", {"path": str(p), "old": "foo", "new": "X"})
    assert "unique" in out.lower() or "multiple" in out.lower()
    assert p.read_text() == "foo bar foo\n"
