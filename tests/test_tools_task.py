import os
import godbot.tools
from godbot.core.registry import DEFAULT


def test_todo_set_and_check(tmp_godbot_home, monkeypatch):
    DEFAULT.execute("todo_set", {"items": ["a", "b", "c"]})
    out = DEFAULT.execute("todo_check", {"index": 1})
    assert "[x]" in out
    assert "b" in out
