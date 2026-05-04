"""Pin/unpin tests for the workspace memory store."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

import godbot.tools  # noqa: F401 — trigger discovery
from godbot.core.registry import DEFAULT
from godbot.core.workspace import Workspace, _current
from godbot.tools.memory import (
    delete_note,
    list_notes,
    load_recent_workspace_notes,
    set_pin,
)


@pytest.fixture
def workspace_token(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    ws = Workspace.of(str(tmp_path))
    token = _current.set(ws)
    yield ws
    _current.reset(token)


def _save(content: str) -> str:
    """Save a note via the registered tool, return the timestamp field."""
    DEFAULT.execute("save_note", {"content": content, "tags": []})
    notes_dir = Path.cwd()  # placeholder — not used; we re-read from store
    # Re-read newest file to find its timestamp.
    import os as _os
    home = Path(_os.environ["GODBOT_HOME"])
    files = sorted((home / "notes").glob("*.json"))
    data = json.loads(files[-1].read_text())
    return data["timestamp"]


def test_set_pin_adds_pinned_tag(workspace_token):
    ts = _save("first")
    assert set_pin(ts, True) is True
    notes = list_notes(workspace=str(workspace_token.root))
    assert len(notes) == 1
    assert "pinned" in notes[0]["tags"]


def test_set_pin_removes_pinned_tag(workspace_token):
    ts = _save("second")
    set_pin(ts, True)
    set_pin(ts, False)
    notes = list_notes(workspace=str(workspace_token.root))
    assert "pinned" not in notes[0]["tags"]


def test_set_pin_unknown_timestamp_returns_false(workspace_token):
    assert set_pin("does-not-exist", True) is False


def test_pinned_notes_are_sticky_in_load_recent(workspace_token):
    """A note pinned long ago must still show up even after newer notes
    push past the recency window."""
    pinned_ts = _save("OLD-PINNED-IMPORTANT")
    set_pin(pinned_ts, True)
    # Add many newer notes that would normally push the pinned one out.
    for i in range(8):
        _save(f"recent-{i}")
    block = load_recent_workspace_notes(str(workspace_token.root), limit=3)
    assert "OLD-PINNED-IMPORTANT" in block
    # The [pinned] marker should be visible to the model.
    assert "[pinned]" in block


def test_unpinned_notes_recency_capped_when_no_pins(workspace_token):
    for i in range(10):
        _save(f"plain-{i}")
    block = load_recent_workspace_notes(str(workspace_token.root), limit=3)
    # Limit=3 with no pins should yield exactly the last 3 plain notes.
    assert "plain-9" in block
    assert "plain-7" in block
    assert "plain-0" not in block


def test_pinned_cap_at_five(workspace_token):
    """If a workspace somehow has 7 pinned notes, only 5 (most recent) are
    embedded into the prompt — bounds prompt size."""
    pinned_ts = []
    for i in range(7):
        ts = _save(f"pin-{i}")
        set_pin(ts, True)
        pinned_ts.append(ts)
    block = load_recent_workspace_notes(str(workspace_token.root), limit=2)
    # Latest 5 of the 7 must be present (pin-2..pin-6); pin-0 and pin-1 should
    # have been bumped out by the cap.
    assert "pin-6" in block
    assert "pin-2" in block
    assert "pin-0" not in block
    assert "pin-1" not in block


def test_delete_note_removes_file(workspace_token, tmp_path):
    ts = _save("ephemeral")
    notes_dir = Path(tmp_path) / ".godbot" / "notes"
    assert len(list(notes_dir.glob("*.json"))) == 1
    assert delete_note(ts) is True
    assert len(list(notes_dir.glob("*.json"))) == 0


def test_delete_note_unknown_returns_false(workspace_token):
    assert delete_note("never") is False


def test_list_notes_filters_by_workspace_and_query(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ws_a = Workspace.of(str(tmp_path / "a"))
    ws_b = Workspace.of(str(tmp_path / "b"))

    t = _current.set(ws_a)
    DEFAULT.execute("save_note", {"content": "alpha-A", "tags": []})
    DEFAULT.execute("save_note", {"content": "beta-A", "tags": []})
    _current.reset(t)

    t = _current.set(ws_b)
    DEFAULT.execute("save_note", {"content": "alpha-B", "tags": []})
    _current.reset(t)

    only_a = list_notes(workspace=str(ws_a.root))
    contents_a = [n["content"] for n in only_a]
    assert "alpha-A" in contents_a
    assert "beta-A" in contents_a
    assert "alpha-B" not in contents_a

    # Newest-first ordering: beta-A was saved after alpha-A.
    assert contents_a[0] == "beta-A"

    filtered = list_notes(workspace=str(ws_a.root), query="alpha")
    assert len(filtered) == 1
    assert filtered[0]["content"] == "alpha-A"


def test_recall_notes_after_pin_still_returns_pinned_at_top(workspace_token):
    """Critical: after pinning, the prompt-injection block keeps the pinned
    note even when newer ones have arrived. Mirrors real session reuse."""
    pinned_ts = _save("KEY-DECISION: use react query")
    set_pin(pinned_ts, True)
    # Newer notes that aren't important.
    for i in range(6):
        _save(f"junk-{i}")
    block = load_recent_workspace_notes(str(workspace_token.root), limit=4)
    lines = [ln for ln in block.splitlines() if ln.startswith("- [")]
    assert lines, "should have at least one note line"
    # Pinned line is first.
    assert "KEY-DECISION" in lines[0]
    assert "[pinned]" in lines[0]
