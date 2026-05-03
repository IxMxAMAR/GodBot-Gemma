from io import StringIO
from rich.console import Console
from godbot.interfaces.cli import CliRenderer
from godbot.core.events import (
    TokenEvent, ToolCallEvent, ToolResultEvent, GateEvent, DoneEvent, ErrorEvent
)


def _renderer():
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=80, record=True)
    return CliRenderer(console=console), console


def test_token_appends_to_assistant_bubble():
    r, console = _renderer()
    r.handle(TokenEvent("hel"))
    r.handle(TokenEvent("lo"))
    r.handle(DoneEvent(step_count=1))
    out = console.export_text()
    assert "hello" in out


def test_tool_call_renders_one_line():
    r, console = _renderer()
    r.handle(ToolCallEvent(id="c1", name="read_file", args={"path": "x.py"}))
    r.handle(ToolResultEvent(id="c1", preview="412 lines", blob=None, duration_ms=12))
    out = console.export_text()
    assert "read_file" in out
    assert "12ms" in out or "12 ms" in out


def test_error_rendered():
    r, console = _renderer()
    r.handle(ErrorEvent(message="boom"))
    assert "boom" in console.export_text()
