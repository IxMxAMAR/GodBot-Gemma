import json

import pytest

from godbot.core.events import (
    TokenEvent, ToolCallEvent, ToolResultEvent,
    GateEvent, ErrorEvent, DoneEvent, event_to_dict,
)


def test_token_event_serializes():
    e = TokenEvent(text="hi")
    assert event_to_dict(e) == {"type": "token", "text": "hi"}


def test_tool_call_event_serializes():
    e = ToolCallEvent(id="c1", name="read_file", args={"path": "x.py"})
    d = event_to_dict(e)
    assert d == {"type": "tool_call", "id": "c1", "name": "read_file", "args": {"path": "x.py"}}


def test_tool_result_event_with_blob():
    e = ToolResultEvent(id="c1", preview="abc", blob="c1", duration_ms=12)
    d = event_to_dict(e)
    assert d == {"type": "tool_result", "id": "c1", "preview": "abc", "blob": "c1", "duration_ms": 12}


def test_tool_result_event_no_blob():
    e = ToolResultEvent(id="c1", preview="abc", blob=None, duration_ms=5)
    assert event_to_dict(e)["blob"] is None


def test_gate_event_serializes():
    e = GateEvent(id="c1", name="run_powershell", args={"cmd": "ls"})
    assert event_to_dict(e) == {
        "type": "gate", "id": "c1", "name": "run_powershell",
        "args": {"cmd": "ls"}, "fs_diff": None,
    }


def test_gate_event_with_fs_diff_serializes():
    diff = {"path": "x.py", "before": "old\n", "after": "new\n"}
    e = GateEvent(id="c1", name="write_file", args={"path": "x.py", "content": "new\n"}, fs_diff=diff)
    d = event_to_dict(e)
    assert d["type"] == "gate"
    assert d["fs_diff"] == diff


def test_error_event_serializes():
    e = ErrorEvent(message="oops", recoverable=True)
    assert event_to_dict(e) == {"type": "agent_error", "message": "oops", "recoverable": True}


def test_done_event_serializes():
    e = DoneEvent(step_count=3)
    assert event_to_dict(e) == {"type": "done", "step_count": 3}


@pytest.mark.parametrize(
    "event",
    [
        TokenEvent(text="hi"),
        ToolCallEvent(id="c1", name="read_file", args={"path": "x.py", "n": 3}),
        ToolResultEvent(id="c1", preview="ok", blob="c1", duration_ms=12),
        ToolResultEvent(id="c2", preview="ok", blob=None, duration_ms=0),
        GateEvent(id="c1", name="run_powershell", args={"cmd": "ls"}),
        GateEvent(
            id="c2", name="write_file", args={"path": "x.py", "content": "y"},
            fs_diff={"path": "x.py", "before": "z", "after": "y"},
        ),
        ErrorEvent(message="boom", recoverable=False),
        ErrorEvent(message="retry me", recoverable=True),
        DoneEvent(step_count=3),
    ],
)
def test_event_serializes_to_json(event):
    payload = event_to_dict(event)
    roundtrip = json.loads(json.dumps(payload))
    assert roundtrip == payload
