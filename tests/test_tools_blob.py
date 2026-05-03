import os
import pytest
import godbot.tools
from godbot.core.registry import DEFAULT
from godbot.core.session import Session


def test_read_blob_via_env(tmp_path, monkeypatch):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("ok")
    s.append_assistant_tool_call("c1", "echo", {}, raw="{}")
    s.record_tool_result("c1", "Z" * 9000)
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", str(s.dir))
    out = DEFAULT.execute("read_blob", {"call_id": "c1", "start": 0, "lines": 3})
    assert out.startswith("Z")
