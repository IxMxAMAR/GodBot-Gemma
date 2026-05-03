import pytest
from godbot.core.events import (
    TokenEvent, ToolCallEvent, ToolResultEvent,
    GateEvent, ErrorEvent, DoneEvent,
)
from godbot.interfaces.tui_widgets.conversation import ConversationModel


def test_token_creates_assistant_bubble_then_appends():
    m = ConversationModel()
    m.handle(TokenEvent(text="hel"))
    m.handle(TokenEvent(text="lo"))
    assert m.last_assistant_text() == "hello"


def test_done_closes_assistant_bubble():
    m = ConversationModel()
    m.handle(TokenEvent(text="hi"))
    m.handle(DoneEvent(step_count=1))
    # Next token starts a NEW bubble
    m.handle(TokenEvent(text="again"))
    assert m.assistant_count() == 2


def test_tool_call_records_card():
    m = ConversationModel()
    m.handle(ToolCallEvent(id="c1", name="read_file", args={"path": "x.py"}))
    assert m.has_tool_card("c1")
    assert m.tool_card("c1").name == "read_file"
    assert m.tool_card("c1").result is None


def test_tool_result_attaches_to_card():
    m = ConversationModel()
    m.handle(ToolCallEvent(id="c1", name="read_file", args={}))
    m.handle(ToolResultEvent(id="c1", preview="412 lines", blob=None, duration_ms=12))
    card = m.tool_card("c1")
    assert card.result == "412 lines"
    assert card.duration_ms == 12


def test_gate_records_card():
    m = ConversationModel()
    m.handle(GateEvent(id="c1", name="run_powershell", args={"cmd": "ls"}))
    gates = m.pending_gates()
    assert len(gates) == 1 and gates[0].name == "run_powershell"


def test_error_creates_banner():
    m = ConversationModel()
    m.handle(ErrorEvent(message="boom"))
    assert m.has_error("boom")


def test_token_after_tool_call_starts_new_assistant():
    m = ConversationModel()
    m.handle(TokenEvent(text="thinking"))
    m.handle(ToolCallEvent(id="c1", name="x", args={}))
    m.handle(TokenEvent(text="back"))
    assert m.assistant_count() == 2
    assert m.last_assistant_text() == "back"
