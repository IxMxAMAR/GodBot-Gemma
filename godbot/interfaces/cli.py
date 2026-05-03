from __future__ import annotations
import asyncio
import sys
from typing import Optional

from rich.console import Console
from rich.text import Text

from godbot.core.events import (
    DoneEvent, ErrorEvent, Event, GateEvent,
    TokenEvent, ToolCallEvent, ToolResultEvent,
)


class CliRenderer:
    """Minimal renderer used both by the REPL and by tests."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self._assistant_buf: list[str] = []
        self._tool_calls: dict[str, dict] = {}

    def handle(self, ev: Event) -> None:
        if isinstance(ev, TokenEvent):
            self._assistant_buf.append(ev.text)
            self.console.print(ev.text, end="", soft_wrap=True, highlight=False)
        elif isinstance(ev, ToolCallEvent):
            self._flush_assistant()
            self._tool_calls[ev.id] = {"name": ev.name, "args": ev.args}
            args_preview = ", ".join(f"{k}={v!r}" for k, v in list(ev.args.items())[:3])
            self.console.print(
                f"[bold cyan]▸ {ev.name}[/]([dim]{args_preview}[/]) ...",
            )
        elif isinstance(ev, ToolResultEvent):
            preview = ev.preview.splitlines()[0][:80]
            blob = f" (blob:{ev.blob})" if ev.blob else ""
            self.console.print(
                f"  [dim]↳ {preview}{blob}  [{ev.duration_ms}ms][/]",
            )
        elif isinstance(ev, GateEvent):
            self._flush_assistant()
            args_preview = ", ".join(f"{k}={v!r}" for k, v in ev.args.items())
            self.console.print(
                f"[bold yellow]GATE[/] {ev.name}({args_preview}) approve? [y/N/a] ",
                end="",
            )
        elif isinstance(ev, ErrorEvent):
            self._flush_assistant()
            self.console.print(f"[red]error: {ev.message}[/]")
        elif isinstance(ev, DoneEvent):
            self._flush_assistant()

    def _flush_assistant(self) -> None:
        if self._assistant_buf:
            self.console.print()  # newline after stream
            self._assistant_buf = []


async def _gate_prompt() -> str:
    """Read a single character from stdin without echo. y / n / a."""
    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    line = line.strip().lower()
    if line.startswith("y"):
        return "allow"
    if line.startswith("a"):
        return "always"
    return "deny"


def main() -> int:
    """Entry point - interactive REPL. Tests cover the renderer only."""
    print("CLI REPL - implemented in 8.2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
