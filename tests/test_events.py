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
    assert event_to_dict(e) == {"type": "gate", "id": "c1", "name": "run_powershell", "args": {"cmd": "ls"}}


def test_error_event_serializes():
    e = ErrorEvent(message="oops", recoverable=True)
    assert event_to_dict(e) == {"type": "error", "message": "oops", "recoverable": True}


def test_done_event_serializes():
    e = DoneEvent(step_count=3)
    assert event_to_dict(e) == {"type": "done", "step_count": 3}
