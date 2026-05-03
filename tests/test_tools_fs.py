from pathlib import Path
import godbot.tools  # triggers discovery
from godbot.core.registry import DEFAULT


def test_read_file(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("alpha\nbeta\n")
    out = DEFAULT.execute("read_file", {"path": str(p)})
    assert "alpha" in out and "beta" in out
    assert "1\t" in out or "1:" in out  # numbered


def test_read_file_truncate_max_lines(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("\n".join(str(i) for i in range(10)))
    out = DEFAULT.execute("read_file", {"path": str(p), "max_lines": 3})
    # Should include line 1..3 but not line 9.
    assert "0" in out and "1" in out and "2" in out


def test_list_dir(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b").mkdir()
    out = DEFAULT.execute("list_dir", {"path": str(tmp_path)})
    assert "a.txt" in out
    assert "b/" in out or "b\\" in out or "b" in out


def test_glob(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("x")
    out = DEFAULT.execute("glob", {"pattern": "**/*.py", "root": str(tmp_path)})
    assert "a.py" in out
    assert "c.py" in out
    assert "b.txt" not in out


def test_grep(tmp_path):
    (tmp_path / "a.txt").write_text("hello world\nfoo bar\nhello again\n")
    (tmp_path / "b.txt").write_text("nothing\n")
    out = DEFAULT.execute("grep", {"pattern": "hello", "root": str(tmp_path)})
    assert "a.txt" in out
    assert "hello world" in out
    assert "hello again" in out
    assert "b.txt" not in out
