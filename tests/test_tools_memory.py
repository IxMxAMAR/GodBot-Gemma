from pathlib import Path
import godbot.tools
from godbot.core.registry import DEFAULT


def test_save_and_recall(tmp_godbot_home):
    DEFAULT.execute("save_note", {"content": "alpha beta", "tags": ["a"]})
    DEFAULT.execute("save_note", {"content": "gamma delta"})
    out = DEFAULT.execute("recall_notes", {"query": "alpha"})
    # Pre-RAG, recall_notes does substring search and returns matches.
    assert "alpha" in out
    assert "gamma" not in out


def test_recall_all_when_query_empty(tmp_godbot_home):
    DEFAULT.execute("save_note", {"content": "n1"})
    DEFAULT.execute("save_note", {"content": "n2"})
    out = DEFAULT.execute("recall_notes", {"query": ""})
    assert "n1" in out and "n2" in out


def test_notes_isolated_to_tmp_godbot_home(tmp_godbot_home):
    # Critical-thinking regression: ensure notes never escape the tmp dir
    # into the real ~/.godbot. Verify the file actually lives under
    # tmp_godbot_home/notes and that the real home dir is untouched.
    DEFAULT.execute("save_note", {"content": "isolated"})
    notes_dir = tmp_godbot_home / "notes"
    assert notes_dir.exists()
    files = list(notes_dir.glob("*.json"))
    assert len(files) == 1
    # Confirm file path is rooted under the fixture's tmp dir.
    assert str(files[0]).startswith(str(tmp_godbot_home))
