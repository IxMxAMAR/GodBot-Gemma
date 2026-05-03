import json
import pytest
from godbot.interfaces.discord_bot.session_map import ChannelSessionMap


def test_set_and_get(tmp_path):
    m = ChannelSessionMap(tmp_path / "discord-sessions.json")
    assert m.get(1234) is None
    m.set(1234, "sid-A")
    assert m.get(1234) == "sid-A"


def test_persists_to_disk(tmp_path):
    p = tmp_path / "discord-sessions.json"
    m = ChannelSessionMap(p)
    m.set(1234, "sid-A")
    raw = json.loads(p.read_text())
    assert raw == {"1234": "sid-A"}


def test_loads_existing_file(tmp_path):
    p = tmp_path / "discord-sessions.json"
    p.write_text(json.dumps({"5678": "sid-B"}))
    m = ChannelSessionMap(p)
    assert m.get(5678) == "sid-B"


def test_pop_clears_entry(tmp_path):
    m = ChannelSessionMap(tmp_path / "discord-sessions.json")
    m.set(1234, "sid")
    popped = m.pop(1234)
    assert popped == "sid"
    assert m.get(1234) is None


def test_pop_unknown_returns_none(tmp_path):
    m = ChannelSessionMap(tmp_path / "discord-sessions.json")
    assert m.pop(9999) is None


def test_atomic_write_does_not_leave_tmp(tmp_path):
    p = tmp_path / "discord-sessions.json"
    m = ChannelSessionMap(p)
    m.set(1, "a")
    m.set(2, "b")
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_corrupt_file_treated_as_empty(tmp_path):
    p = tmp_path / "discord-sessions.json"
    p.write_text("{not valid json")
    m = ChannelSessionMap(p)
    assert m.get(1) is None
    m.set(1, "sid")
    # Corrupt file is overwritten by next set.
    assert json.loads(p.read_text())["1"] == "sid"
