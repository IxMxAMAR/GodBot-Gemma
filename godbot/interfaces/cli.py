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


DEFAULT_DAEMON_URL = "http://127.0.0.1:7878"


async def _cmd_quickrun(args) -> int:
    """Subcommand: one-shot agent run via POST /api/agent/quickrun."""
    from godbot.client import Client, ClientError
    console = Console()
    async with Client(base_url=args.base_url) as c:
        try:
            resp = await c.quickrun(
                args.goal,
                workspace=args.workspace,
                model=args.model,
                max_steps=args.max_steps,
                max_wait_seconds=args.max_wait_seconds,
                safe_only=not args.unsafe,
            )
        except ClientError as e:
            console.print(f"[red]error: {e}[/]")
            return 1
    status = resp.get("status", "unknown")
    result = resp.get("result") or ""
    console.print(result)
    meta = (
        f"[dim]status={status}  steps={resp.get('step_count', 0)}  "
        f"tool_calls={resp.get('tool_calls', 0)}  "
        f"elapsed={resp.get('elapsed_ms', 0)}ms  "
        f"session={resp.get('session_id', '?')}[/]"
    )
    console.print(meta)
    # Non-zero exit if the run didn't reach a clean terminal state.
    return 0 if status == "done" else 2


async def _cmd_stats(args) -> int:
    """Subcommand: tabular summary of GET /api/stats."""
    from rich.table import Table
    from godbot.client import Client, ClientError
    console = Console()
    async with Client(base_url=args.base_url) as c:
        try:
            data = await c.stats()
        except ClientError as e:
            console.print(f"[red]error: {e}[/]")
            return 1

    summary = Table(title="GodBot stats", show_header=True, header_style="bold cyan")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    usage = data.get("usage", {}) or {}
    tasks = data.get("tasks", {}) or {}
    summary.add_row("sessions", str(data.get("sessions", 0)))
    summary.add_row("turns", str(data.get("turns", 0)))
    summary.add_row("tool_calls", str(data.get("tool_calls_total", 0)))
    summary.add_row("input_tokens", str(usage.get("input_tokens", 0)))
    summary.add_row("output_tokens", str(usage.get("output_tokens", 0)))
    summary.add_row("estimated_cost_usd", f"${data.get('estimated_cost_usd', 0.0):.4f}")
    summary.add_row("tasks_total", str(tasks.get("total", 0)))
    console.print(summary)

    top_tools = data.get("top_tools") or []
    if top_tools:
        tools = Table(title="Top tools", show_header=True, header_style="bold cyan")
        tools.add_column("tool")
        tools.add_column("calls", justify="right")
        for entry in top_tools:
            tools.add_row(str(entry.get("name", "?")), str(entry.get("count", 0)))
        console.print(tools)
    return 0


async def _cmd_sessions(args) -> int:
    """Subcommand: one-line-per-session table from GET /api/sessions."""
    from rich.table import Table
    from godbot.client import Client, ClientError
    console = Console()
    async with Client(base_url=args.base_url) as c:
        try:
            sessions = await c.list_sessions(limit=args.limit)
        except ClientError as e:
            console.print(f"[red]error: {e}[/]")
            return 1

    if not sessions:
        console.print("[dim](no sessions)[/]")
        return 0

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("id")
    table.add_column("started_at")
    table.add_column("provider")
    table.add_column("model")
    table.add_column("workspace")
    table.add_column("pin", justify="center")
    for s in sessions:
        ws = s.get("workspace_root") or ""
        # Truncate long workspace paths so the row stays one line wide.
        if len(ws) > 40:
            ws = "…" + ws[-39:]
        table.add_row(
            str(s.get("id", "?")),
            str(s.get("started_at", "") or ""),
            str(s.get("provider", "") or ""),
            str(s.get("model_name") or s.get("model", "") or ""),
            ws,
            "★" if s.get("pinned") else "",
        )
    console.print(table)
    return 0


_SUBCOMMANDS = {"repl", "quickrun", "stats", "sessions"}


def build_parser() -> "argparse.ArgumentParser":
    import argparse
    p = argparse.ArgumentParser(prog="godbot-cli")
    sub = p.add_subparsers(dest="cmd")

    # `godbot-cli repl` (also the default when no subcommand is given —
    # see _normalize_argv which inserts "repl" up front).
    p_repl = sub.add_parser("repl", help="Interactive REPL (default)")
    p_repl.add_argument("--resume")
    p_repl.add_argument("--yolo", action="store_true")
    p_repl.add_argument("--workspace", help="Confine FS tools to this directory")
    p_repl.add_argument("--auto-approve", action="store_true",
                        help="Auto-approve FS-safe dangerous tools when --workspace is set")

    # Shared --base-url for daemon-talking subcommands.
    def _add_base_url(sp):
        sp.add_argument("--base-url", default=DEFAULT_DAEMON_URL,
                        help=f"Daemon base URL (default: {DEFAULT_DAEMON_URL})")

    p_qr = sub.add_parser("quickrun", help="One-shot agent run (POST /api/agent/quickrun)")
    _add_base_url(p_qr)
    p_qr.add_argument("goal", help="The goal string to hand the agent")
    p_qr.add_argument("--workspace", help="Confine FS tools to this directory")
    p_qr.add_argument("--model", help="Override model (default: daemon picks)")
    p_qr.add_argument("--max-steps", type=int, default=12)
    p_qr.add_argument("--max-wait-seconds", type=float, default=60.0)
    p_qr.add_argument("--unsafe", action="store_true",
                      help="Allow non-safe tools (default: safe-only)")

    p_stats = sub.add_parser("stats", help="Aggregate metrics summary (GET /api/stats)")
    _add_base_url(p_stats)

    p_sess = sub.add_parser("sessions", help="List recent sessions (GET /api/sessions)")
    _add_base_url(p_sess)
    p_sess.add_argument("--limit", type=int, default=20)

    return p


def _normalize_argv(argv: list[str]) -> list[str]:
    """Insert ``repl`` as the implicit subcommand.

    Preserves the pre-subparser CLI shape (``godbot-cli --resume=last
    --yolo`` etc.) so existing scripts and tests keep working.
    """
    if not argv:
        return ["repl"]
    first = argv[0]
    if first in _SUBCOMMANDS:
        return list(argv)
    if first in ("-h", "--help"):
        return list(argv)
    return ["repl", *argv]


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    cmd = args.cmd or "repl"

    if cmd == "quickrun":
        return asyncio.run(_cmd_quickrun(args))
    if cmd == "stats":
        return asyncio.run(_cmd_stats(args))
    if cmd == "sessions":
        return asyncio.run(_cmd_sessions(args))

    # Default: REPL.
    return asyncio.run(run_repl(
        resume=args.resume, yolo=args.yolo,
        workspace=args.workspace, auto_approve=args.auto_approve,
    ))


if __name__ == "__main__":
    sys.exit(main())
