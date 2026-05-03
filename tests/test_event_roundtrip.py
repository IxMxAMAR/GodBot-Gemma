import pytest
from godbot.core.events import (
    TokenEvent, ToolCallEvent, ToolResultEvent,
    GateEvent, ErrorEvent, DoneEvent,
    event_to_dict, dict_to_event,
)


@pytest.mark.parametrize("event", [
    TokenEvent(text="hi"),
    ToolCallEvent(id="c1", name="read_file", args={"path": "x.py"}),
    ToolResultEvent(id="c1", preview="ok", blob="c1", duration_ms=12),
    ToolResultEvent(id="c2", preview="ok", blob=None, duration_ms=0),
    GateEvent(id="c1", name="run_powershell", args={"cmd": "ls"}),
    ErrorEvent(message="boom", recoverable=False),
    ErrorEvent(message="retry", recoverable=True),
    DoneEvent(step_count=3),
])
def test_event_dict_roundtrip(event):
    d = event_to_dict(event)
    back = dict_to_event(d)
    assert back == event


def test_dict_to_event_filters_unknown_fields():
    d = {"type": "token", "text": "hi", "future_field": "ignored"}
    ev = dict_to_event(d)
    assert ev == TokenEvent(text="hi")


def test_dict_to_event_does_not_mutate_input():
    d = {"type": "token", "text": "hi"}
    snapshot = dict(d)
    dict_to_event(d)
    assert d == snapshot
