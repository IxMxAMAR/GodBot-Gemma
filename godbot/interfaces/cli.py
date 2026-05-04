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


def handle_slash(line: str, state: dict) -> tuple[str, object]:
    """Dispatch a CLI input line.

    Returns (kind, payload):
      - ("message", text)        - a normal user prompt
      - ("ok", None)             - handled internally (e.g. toggle)
      - ("quit", None)
      - ("unknown", line)
      - ("output", str)          - caller should print str
    """
    if not line.startswith("/"):
        return ("message", line)
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""
    if cmd == "/exit":
        return ("quit", None)
    if cmd == "/yolo":
        state["yolo"] = not state.get("yolo", False)
        return ("ok", None)
    if cmd == "/new":
        state["__new__"] = True
        return ("ok", None)
    if cmd == "/sessions":
        return ("output", "(see ~/.godbot - full impl in 8.3)")
    if cmd == "/load":
        state["__load__"] = arg
        return ("ok", None)
    if cmd == "/tools":
        return ("output", "tools toggle UI - see web; CLI prints names list")
    if cmd == "/rag":
        sub = arg.split(maxsplit=1)
        if not sub:
            return ("unknown", line)
        if sub[0] == "use":
            state["rag_collection"] = sub[1] if len(sub) > 1 else None
            return ("ok", None)
        if sub[0] == "list":
            return ("output", "(rag list - wired up in Phase 10)")
        return ("unknown", line)
    if cmd == "/stop":
        cancel = state.get("cancel")
        if cancel is not None:
            cancel.set()
        return ("ok", None)
    return ("unknown", line)


async def run_repl(*, model: str = "auto", resume: Optional[str] = None, yolo: bool = False,
                   workspace: Optional[str] = None, auto_approve: bool = False) -> int:
    """Interactive REPL. Imports kept lazy to avoid breaking unit tests."""
    from prompt_toolkit import PromptSession
    from godbot.config import load_config
    from godbot.core.llm import LLMClient
    from godbot.core.session import Session
    from godbot.core.registry import DEFAULT
    from godbot.core.agent import run_turn
    from godbot.prompts import build_system_prompt
    import godbot.tools  # discovery
    from godbot.mcp import boot_mcp as _boot_mcp
    _boot_mcp()
    import os
    from pathlib import Path

    cfg = load_config()
    sessions_root = Path.cwd() / "sessions"
    sessions_root.mkdir(exist_ok=True)
    if resume == "last":
        existing = sorted(sessions_root.iterdir())
        if not existing:
            print("no sessions to resume")
            return 1
        session = Session.load(sessions_root, existing[-1].name)
        if workspace:
            session.set_workspace(workspace, auto_approve=auto_approve)
    elif resume:
        session = Session.load(sessions_root, resume)
        if workspace:
            session.set_workspace(workspace, auto_approve=auto_approve)
    else:
        session = Session.create(
            sessions_root, model=cfg.llm.model,
            workspace_root=workspace, auto_approve_in_sandbox=auto_approve,
        )
    if yolo:
        session.set_yolo(True)
    os.environ["GODBOT_ACTIVE_SESSION"] = str(session.dir)

    llm = LLMClient(base_url=cfg.llm.base_url, model=cfg.llm.model)
    info = llm.probe()
    if info.context_length and info.context_length < cfg.llm.max_context:
        cfg.llm.max_context = info.context_length

    renderer = CliRenderer()
    state: dict = {"yolo": session.yolo}
    pt_session = PromptSession()

    print(f"GodBot · model: {info.id} · session: {session.id}")
    while True:
        try:
            line = await pt_session.prompt_async("> ")
        except (EOFError, KeyboardInterrupt):
            return 0
        kind, payload = handle_slash(line, state)
        if kind == "quit":
            session.end()
            return 0
        if kind == "output":
            print(payload)
            continue
        if kind == "unknown":
            print(f"unknown command: {payload}")
            continue
        if kind == "ok":
            if state.pop("__new__", False):
                session = Session.create(sessions_root, model=cfg.llm.model)
                os.environ["GODBOT_ACTIVE_SESSION"] = str(session.dir)
            sid = state.pop("__load__", None)
            if sid:
                session = Session.load(sessions_root, sid)
                os.environ["GODBOT_ACTIVE_SESSION"] = str(session.dir)
            continue

        # Normal user message.
        session.append_user(payload)
        cancel = asyncio.Event()
        state["cancel"] = cancel

        async def emit(ev):
            renderer.handle(ev)
            if isinstance(ev, GateEvent):
                # Inline blocking prompt for CLI.
                decision = await _gate_prompt()
                session.resolve_gate(ev.id, decision)

        try:
            await run_turn(
                llm=llm, session=session, registry=DEFAULT, emit=emit,
                cancel=cancel, max_steps=cfg.agent.max_steps,
                max_context=cfg.llm.max_context, system_prompt="",
                system_prompt_builder=build_system_prompt,
            )
        except Exception as e:
            renderer.console.print(f"[red]error: {type(e).__name__}: {e}[/]")


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--resume")
    p.add_argument("--yolo", action="store_true")
    p.add_argument("--workspace", help="Confine FS tools to this directory")
    p.add_argument("--auto-approve", action="store_true",
                   help="Auto-approve FS-safe dangerous tools when --workspace is set")
    args = p.parse_args()
    return asyncio.run(run_repl(
        resume=args.resume, yolo=args.yolo,
        workspace=args.workspace, auto_approve=args.auto_approve,
    ))


if __name__ == "__main__":
    sys.exit(main())
