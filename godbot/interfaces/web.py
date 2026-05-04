from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from godbot.config import load_config, providers_to_configs
from godbot.core.agent import run_turn
from godbot.core.events import event_to_dict
from godbot.core.llm import LLMClient
from godbot.core.providers import (
    get_provider as get_provider_factory,
    load_providers_from_config,
    list_registered as list_registered_providers,
)
from godbot.core.providers.base import ProviderConfig as ProviderConfigType
from godbot.core.registry import DEFAULT
from godbot.core.session import Session
from godbot.prompts import build_system_prompt
import godbot.tools  # noqa: F401 — triggers @tool auto-discovery so /api/tools is populated

# Bring up MCP servers configured in ~/.godbot/config.toml. Best-effort:
# unreachable servers are logged and skipped, so this never breaks startup
# even with a malformed [mcp.servers.*] block. Runs at module-import time
# (same pattern as godbot.tools auto-discovery above) so the daemon, CLI
# and TUI all surface MCP tools the moment they import this module.
from godbot.mcp import boot_mcp as _boot_mcp  # noqa: E402
_boot_mcp()


# Daemon start time — used by /api/health/details to report uptime.
import time as _time
_DAEMON_STARTED_AT = _time.time()


# Shared per-process state.
# Per-session event logs replace the legacy single-consumer queue
# (sub-project 21). EventLog is a sequence-numbered ring buffer with
# multi-subscriber fanout, so a Studio reconnect after a network blip
# replays buffered events from its last-seen seq instead of seeing 409.
from godbot.core.event_log import EventLog as _EventLog
_logs: dict[str, _EventLog] = {}
_cancels: dict[str, asyncio.Event] = {}
_sessions_cache: dict[str, Session] = {}


def get_llm() -> LLMClient:
    cfg = load_config()
    client = LLMClient(base_url=cfg.llm.base_url, model=cfg.llm.model)
    client.probe()
    return client


def _provider_for_session(session: Session):
    """Resolve a :class:`Provider` for ``session`` from current config.

    Reads ``[providers.<name>]`` from config.toml each call so the daemon
    picks up edits without restart. ``None`` is returned for the default
    "lmstudio" path so existing call sites keep using the legacy
    :class:`LLMClient` shim — this preserves byte-for-byte backward
    compatibility for the most common deployment.
    """
    name = session.provider or "lmstudio"
    if name == "lmstudio":
        # Special case: stay on the legacy shim so existing tests/UI keep
        # working unchanged. The provider abstraction is identical in
        # effect for LM Studio anyway.
        return None
    cfg = load_config()
    raw = providers_to_configs(cfg.providers)
    parsed = load_providers_from_config(raw)
    pcfg = parsed.get(name) or ProviderConfigType(name=name)
    return get_provider_factory(name, pcfg)


def _ensure_log(sid: str) -> _EventLog:
    """Get or create the per-session event log.

    Returns the existing log even when it's closed — closed logs still
    hold the replay buffer that late subscribers want to read. A fresh
    log is created only when none has ever existed for this session.
    For mid-turn rotation use :func:`_replace_log`.
    """
    log = _logs.get(sid)
    if log is None:
        log = _EventLog()
        _logs[sid] = log
    return log


def _replace_log(sid: str) -> _EventLog:
    """Force-create a fresh EventLog for ``sid``, closing any existing one.

    Called at POST /api/chat so each turn gets its own seq numbering.
    Subscribers from the previous turn's log will see ``close()`` fire
    and exit; the new log starts empty at seq 0.
    """
    old = _logs.get(sid)
    if old is not None and not old.closed:
        old.close()
    fresh = _EventLog()
    _logs[sid] = fresh
    return fresh


def _ensure_cancel(sid: str) -> asyncio.Event:
    ev = _cancels.get(sid)
    if ev is None:
        ev = asyncio.Event()
        _cancels[sid] = ev
    return ev


def _loop_now() -> float:
    """asyncio.get_event_loop() is deprecated in 3.12+ when there's no running
    loop; inside an async generator we always have one."""
    return asyncio.get_running_loop().time()


def _render_session_markdown(s, events, cost_dict) -> str:
    """Render a session as a human-readable markdown transcript.

    Groups events by turn boundary (each user message starts one). Tool
    calls are inlined under their assistant context with collapsible
    code-fence blocks for the result. Used by GET /api/sessions/{sid}/
    export?format=markdown.
    """
    lines: list[str] = []
    lines.append(f"# Session `{s.id}`")
    lines.append("")
    meta_bits = []
    if s.provider:
        meta_bits.append(f"provider=`{s.provider}`")
    if s.model_name:
        meta_bits.append(f"model=`{s.model_name}`")
    if s._meta.get("workspace_root"):
        meta_bits.append(f"workspace=`{s._meta['workspace_root']}`")
    if meta_bits:
        lines.append("> " + " · ".join(meta_bits))
        lines.append("")
    u = s.usage
    if u["turns"]:
        lines.append(
            f"**Usage:** {u['turns']} turns, "
            f"{u['input_tokens']:,} in / {u['output_tokens']:,} out tokens "
            f"(total {u['total_tokens']:,})"
        )
        if cost_dict.get("matched"):
            lines.append(f"**Estimated cost:** ${cost_dict['usd']:.4f}")
        lines.append("")

    import json as _json
    for ev in events:
        t = ev.get("type")
        if t == "user":
            lines.append("## User")
            lines.append("")
            lines.append(ev.get("content", ""))
            lines.append("")
        elif t == "assistant_final":
            lines.append("## Assistant")
            lines.append("")
            lines.append(ev.get("content", ""))
            lines.append("")
        elif t == "assistant_tool_call":
            lines.append(f"### Tool call: `{ev.get('name', '?')}`")
            args_pretty = _json.dumps(ev.get("args") or {}, indent=2)
            lines.append("```json")
            lines.append(args_pretty)
            lines.append("```")
        elif t == "tool_result":
            lines.append("**Tool result:**")
            lines.append("")
            content = ev.get("content", "")
            lines.append("```")
            lines.append(content[:2000])
            if len(content) > 2000:
                lines.append(f"... [truncated, full blob {ev.get('blob') or '—'}]")
            lines.append("```")
            lines.append("")
        elif t == "synthetic_tool_result":
            lines.append(f"_<system nudge: {ev.get('content', '')[:200]}>_")
            lines.append("")
    return "\n".join(lines)


def _register_endpoints(app: FastAPI, sessions_root: Path) -> None:
    @app.post("/api/chat")
    async def chat(request: Request, llm: LLMClient = Depends(get_llm)):
        body = await request.json()
        sid = body["session_id"]
        msg = body["message"]
        try:
            session = _sessions_cache.get(sid) or Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        _sessions_cache[sid] = session
        os.environ["GODBOT_ACTIVE_SESSION"] = str(session.dir)
        session.append_user(msg)

        cancel = _ensure_cancel(sid)
        cancel.clear()
        # Fresh log per turn so the per-turn seq counter starts at 0 and
        # any stale subscribers from a previous turn exit cleanly.
        log = _replace_log(sid)

        async def emit(ev):
            d = event_to_dict(ev)
            log.publish(d["type"], d)

        async def runner():
            try:
                cfg = load_config()
                provider = _provider_for_session(session)
                kwargs = dict(
                    session=session, registry=DEFAULT, emit=emit,
                    cancel=cancel, max_steps=cfg.agent.max_steps,
                    max_context=cfg.llm.max_context, system_prompt="",
                    system_prompt_builder=build_system_prompt,
                )
                if provider is not None:
                    await run_turn(provider=provider, **kwargs)
                else:
                    await run_turn(llm=llm, **kwargs)
            finally:
                # Close the log so subscribers exit; we keep the closed log
                # in _logs so a late subscriber can still replay buffered
                # events (e.g. for "did the previous turn finish?" queries).
                log.close()

        asyncio.create_task(runner())
        return {"session_id": sid}

    @app.get("/api/chat/stream")
    async def chat_stream(request: Request, session_id: str, last_event_id: int = 0):
        """SSE stream for a session's chat events (sub-project 21).

        Multi-subscriber: any number of clients can attach to the same
        session simultaneously. Reconnecting clients should send the
        ``Last-Event-ID`` header (or ``last_event_id`` query param) with
        the seq of the last event they processed; the server replays
        buffered events past that cursor before tailing for new ones.

        Each emitted SSE event carries an ``id:`` line set to its seq, so
        browsers automatically resend Last-Event-ID on reconnect.
        """
        from sse_starlette.sse import EventSourceResponse
        # Honor the SSE-standard Last-Event-ID header if present; the
        # query-string fallback supports clients that can't set headers
        # (e.g. EventSource in browsers without polyfills).
        hdr = request.headers.get("last-event-id")
        if hdr:
            try:
                last_event_id = int(hdr)
            except ValueError:
                pass
        log = _ensure_log(session_id)

        async def gen():
            heartbeat_at = _loop_now() + 15
            cursor = last_event_id
            async for ev in log.follow(since_seq=cursor, idle_timeout=1.0):
                # `__idle__` is the EventLog's keepalive sentinel; convert
                # it to an SSE comment line if it's time to ping.
                if ev.name == "__idle__":
                    if _loop_now() >= heartbeat_at:
                        yield {"event": "ping", "data": "{}"}
                        heartbeat_at = _loop_now() + 15
                    continue
                yield {
                    "id": str(ev.seq),
                    "event": ev.name,
                    "data": json.dumps(ev.data),
                }
                cursor = ev.seq
                # `done` is a logical terminator — close the SSE stream
                # so the client doesn't hang waiting for keepalives on a
                # finished log.
                if ev.name == "done":
                    return

        return EventSourceResponse(gen())

    @app.post("/api/gate/{call_id}")
    async def gate_resolve(call_id: str, request: Request):
        body = await request.json()
        sid = body["session_id"]
        decision = body["decision"]
        args_override = body.get("args_override")
        if args_override is not None and not isinstance(args_override, dict):
            raise HTTPException(400, "args_override must be an object")
        session = _sessions_cache.get(sid)
        if session is None:
            raise HTTPException(404, "no active session")
        ok = session.resolve_gate(call_id, decision, args_override=args_override)
        return {"ok": ok}

    @app.post("/api/stop")
    async def stop(request: Request):
        body = await request.json()
        sid = body["session_id"]
        ev = _cancels.get(sid)
        if ev is not None:
            ev.set()
        return {"stopped": True}

    @app.get("/api/providers")
    async def providers_list():
        """Configured providers + their connection state.

        ``configured`` mirrors the ``[providers.<name>]`` section. ``state``
        tells the UI whether a provider is reachable; we don't probe here
        (would block the daemon); the UI can call
        ``/api/providers/<name>/models`` to drive a real check.
        """
        cfg = load_config()
        registered = list_registered_providers()
        configured = list(cfg.providers.items.keys())
        out: list[dict] = []
        for name in sorted(set(registered) | set(configured)):
            item = cfg.providers.items.get(name)
            entry: dict = {
                "name": name,
                "configured": item is not None,
                "registered": name in registered,
                "default_model": item.default_model if item else "auto",
                "base_url": item.base_url if item else "",
                "has_api_key": bool(
                    (item and (item.api_key or (item.api_key_env and os.environ.get(item.api_key_env))))
                ) if item else False,
            }
            out.append(entry)
        return {"default": cfg.providers.default, "providers": out}

    @app.get("/api/providers/{name}/models")
    async def providers_models(name: str):
        """List models a provider exposes. For cloud providers this returns
        the curated set we know about (no live probe). For OpenAI-compat
        providers we hit the configured ``/models`` endpoint."""
        cfg = load_config()
        raw = providers_to_configs(cfg.providers)
        parsed = load_providers_from_config(raw)
        pcfg = parsed.get(name) or ProviderConfigType(name=name)
        try:
            provider = get_provider_factory(name, pcfg)
        except KeyError:
            raise HTTPException(404, f"unknown provider {name!r}")
        try:
            models = await provider.list_models()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "models": []}
        return {
            "models": [
                {
                    "id": m.id,
                    "context_length": m.context_length,
                    "supports_native_tools": m.supports_native_tools,
                }
                for m in models
            ]
        }

    @app.get("/api/tools")
    async def tools_list():
        return [
            {"name": t.name, "description": t.description, "dangerous": t.dangerous}
            for t in DEFAULT.all()
        ]

    @app.post("/api/tools/reload")
    async def tools_reload():
        """Re-import every godbot.tools.* module (sub-project 27).

        Picks up code changes to existing tool modules without a daemon
        restart. Useful while developing a custom tool — edit the file,
        POST here, your @tool decorators re-fire and overwrite their
        registry entries. Returns the new tool count.

        Modules whose files were *deleted* are NOT pruned from the
        registry — that's intentional for v1 since "tool I'm working on
        threw on import once" shouldn't permanently lose the prior
        registration. Restart the daemon to GC dead entries.
        """
        from godbot.tools import reload_all
        try:
            count = reload_all()
        except Exception as e:
            raise HTTPException(500, f"reload failed: {type(e).__name__}: {e}")
        return {"ok": True, "modules_reloaded": count, "tools_registered": len(DEFAULT.all())}

    @app.post("/api/tools/toggle")
    async def tool_toggle(request: Request):
        body = await request.json()
        sid = body["session_id"]
        name = body["name"]
        enabled = bool(body["enabled"])
        try:
            session = _sessions_cache.get(sid) or Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        _sessions_cache[sid] = session
        current = list(session.tool_overrides or [t.name for t in DEFAULT.all()])
        if enabled and name not in current:
            current.append(name)
        elif not enabled and name in current:
            current.remove(name)
        session.set_tool_overrides(sorted(current))
        return {"ok": True, "tool_overrides": session.tool_overrides}

    @app.get("/api/memory")
    async def memory_list(workspace: Optional[str] = None, q: str = ""):
        """List workspace-scoped notes (newest first).

        ``workspace`` is the absolute path stored on each note record. When
        omitted the endpoint returns every note in the global pool.
        """
        from godbot.tools.memory import list_notes
        notes = list_notes(workspace=workspace, query=q)
        return {"notes": notes}

    @app.post("/api/memory/pin")
    async def memory_pin(request: Request):
        """Pin or unpin a note. Body: ``{timestamp, pin: bool}``.

        Pinned notes always appear in ``load_recent_workspace_notes`` (capped
        at 5) so the user can anchor important context across sessions.
        """
        body = await request.json()
        ts = body.get("timestamp")
        pin = bool(body.get("pin", True))
        if not ts:
            raise HTTPException(400, "timestamp required")
        from godbot.tools.memory import set_pin
        ok = set_pin(ts, pin)
        if not ok:
            raise HTTPException(404, "note not found")
        return {"ok": True}

    @app.post("/api/memory/delete")
    async def memory_delete(request: Request):
        """Delete a note. Body: ``{timestamp}``."""
        body = await request.json()
        ts = body.get("timestamp")
        if not ts:
            raise HTTPException(400, "timestamp required")
        from godbot.tools.memory import delete_note
        ok = delete_note(ts)
        if not ok:
            raise HTTPException(404, "note not found")
        return {"ok": True}

    @app.post("/api/memory/auto_summarize")
    async def memory_auto_summarize(request: Request):
        """Run ``project_summary`` for ``workspace`` and persist as a note.

        Body: ``{workspace: <abs path>, force?: bool}``. When ``force`` is
        false (default) and a recent ``project_summary`` note already exists
        for this workspace, returns the cached one without re-running the
        tool. The summary is saved with tag ``project_summary`` so
        ``load_recent_workspace_notes`` can prefer it.
        """
        body = await request.json()
        workspace = body.get("workspace")
        force = bool(body.get("force", False))
        if not workspace:
            raise HTTPException(400, "workspace required")

        from godbot.tools.memory import _notes_for_workspace, save_note
        from godbot.tools.project import project_summary
        from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current

        # Cached path: return the most-recent project_summary note unless caller forces refresh.
        if not force:
            existing = [
                n for n in _notes_for_workspace(workspace)
                if "project_summary" in (n.get("tags") or [])
            ]
            if existing:
                return {"summary": existing[-1]["content"], "cached": True}

        # Activate workspace contextvar so save_note tags correctly.
        try:
            ws = Workspace.of(workspace)
        except (FileNotFoundError, NotADirectoryError) as e:
            raise HTTPException(400, f"workspace invalid: {e}")
        token = set_workspace(ws)
        try:
            text = project_summary(root=workspace)
            save_note(text, tags=["project_summary"])
        finally:
            _ws_current.reset(token)
        return {"summary": text, "cached": False}

    @app.get("/api/commands")
    async def commands_list():
        """List user-defined slash commands (sub-project 18).

        Reads ``~/.godbot/commands/*.toml`` (or ``$GODBOT_HOME/commands/``)
        and returns name + description for each. Studio uses this to
        populate a quick-action picker; the agent loop uses the
        ``match_command`` path independently.
        """
        from godbot.core.commands import load_commands

        cmds = load_commands()
        return {
            "commands": [
                {
                    "name": c.name,
                    "description": c.description,
                    "tool_overrides": c.tool_overrides,
                    "source_path": c.source_path,
                }
                for c in cmds.values()
            ],
        }

    @app.post("/api/tasks")
    async def task_start(request: Request):
        """Schedule a background agent task (sub-project 13).

        Body: ``{goal: str, workspace?: str, provider?: str, model?: str,
        max_steps?: int, tool_overrides?: list[str], safe_only?: bool}``.
        Returns the task record's initial state.

        Tasks default to ``safe_only=True`` (non-dangerous tools only) to
        avoid hung gates with no human approver. Override at your own risk.
        """
        from godbot.core.tasks import DEFAULT_RUNNER

        body = await request.json()
        goal = body.get("goal")
        if not goal or not isinstance(goal, str):
            raise HTTPException(400, "goal (str) required")
        workspace = body.get("workspace")
        provider = body.get("provider", "lmstudio")
        model_name = body.get("model_name") or body.get("model") or "auto"
        max_steps = int(body.get("max_steps", 12))
        tool_overrides = body.get("tool_overrides")
        if tool_overrides is not None and not isinstance(tool_overrides, list):
            raise HTTPException(400, "tool_overrides must be a list of names")
        safe_only = bool(body.get("safe_only", True))

        rec = DEFAULT_RUNNER.start_task(
            goal=goal,
            sessions_root=sessions_root,
            workspace=workspace,
            provider=provider,
            model_name=model_name,
            max_steps=max_steps,
            tool_overrides=tool_overrides,
            safe_only=safe_only,
        )
        return rec.to_dict()

    @app.get("/api/tasks")
    async def tasks_list(limit: int = 50):
        from godbot.core.tasks import DEFAULT_RUNNER
        return {"tasks": [t.to_dict() for t in DEFAULT_RUNNER.list_tasks(limit=limit)]}

    @app.get("/api/tasks/{tid}")
    async def task_get(tid: str):
        from godbot.core.tasks import DEFAULT_RUNNER
        rec = DEFAULT_RUNNER.get_task(tid)
        if rec is None:
            raise HTTPException(404, "no such task")
        return rec.to_dict()

    @app.post("/api/tasks/{tid}/cancel")
    async def task_cancel(tid: str):
        from godbot.core.tasks import DEFAULT_RUNNER
        rec = DEFAULT_RUNNER.get_task(tid)
        if rec is None:
            raise HTTPException(404, "no such task")
        sent = DEFAULT_RUNNER.cancel_task(tid)
        return {"ok": True, "sent": sent}

    @app.get("/api/tasks/{tid}/wait")
    async def task_wait(tid: str, timeout: float = 60.0):
        """Long-poll until a task reaches a terminal status (sub-project 45).

        Blocks the request up to ``timeout`` seconds (capped at 300s for
        safety) and returns the task record once it's done/error/cancelled
        /interrupted, or once the timeout elapses (in which case the
        record's status will still be running). Always 200 — clients
        check ``status`` to know whether the wait succeeded or timed out.

        Easier than driving the SSE stream when you just want "did the
        task finish?" — a script can curl this and unblock when done.
        """
        from godbot.core.tasks import DEFAULT_RUNNER, _TERMINAL

        rec = DEFAULT_RUNNER.get_task(tid)
        if rec is None:
            raise HTTPException(404, "no such task")
        if rec.status in _TERMINAL:
            return rec.to_dict()
        timeout = min(max(0.5, float(timeout)), 300.0)
        deadline = _loop_now() + timeout
        # Poll the live record every 250ms. We could subscribe to the
        # event log instead, but a small poll keeps this implementation
        # independent of EventLog wiring quirks (closed log, persisted
        # task, etc.).
        while _loop_now() < deadline:
            await asyncio.sleep(0.25)
            rec = DEFAULT_RUNNER.get_task(tid)
            if rec is None:
                raise HTTPException(404, "task vanished")
            if rec.status in _TERMINAL:
                break
        return rec.to_dict()

    @app.get("/api/tasks/{tid}/stream")
    async def task_stream(request: Request, tid: str, last_event_id: int = 0):
        """SSE stream of live progress for one background task (sub-project 25).

        Reuses the EventLog primitive from the chat stream: each task
        owns a log seeded with a ``status: pending`` event and updated
        with ``status``, ``tool_call``, ``tool_result``, ``done``, and
        ``agent_error`` events as the runner makes progress.
        Reconnecting clients send ``Last-Event-ID`` to resume from their
        cursor; late subscribers replay the buffered events.
        """
        from sse_starlette.sse import EventSourceResponse
        from godbot.core.tasks import DEFAULT_RUNNER

        if DEFAULT_RUNNER.get_task(tid) is None:
            raise HTTPException(404, "no such task")
        log = DEFAULT_RUNNER.get_event_log(tid)
        if log is None:
            # Task exists in the persisted records but has no live log
            # (e.g. loaded from disk after a daemon restart). Surface a
            # one-shot status event with the current record and exit so
            # the client knows the task is terminal.
            rec = DEFAULT_RUNNER.get_task(tid)

            async def gen_static():
                yield {
                    "id": "1",
                    "event": "status",
                    "data": json.dumps({
                        "tid": tid, "status": rec.status,
                        "result": rec.result, "error": rec.error,
                    }),
                }

            return EventSourceResponse(gen_static())

        hdr = request.headers.get("last-event-id")
        if hdr:
            try:
                last_event_id = int(hdr)
            except ValueError:
                pass

        async def gen():
            heartbeat_at = _loop_now() + 15
            cursor = last_event_id
            async for ev in log.follow(since_seq=cursor, idle_timeout=1.0):
                if ev.name == "__idle__":
                    if _loop_now() >= heartbeat_at:
                        yield {"event": "ping", "data": "{}"}
                        heartbeat_at = _loop_now() + 15
                    continue
                yield {
                    "id": str(ev.seq),
                    "event": ev.name,
                    "data": json.dumps(ev.data),
                }
                cursor = ev.seq
                # `done` and the terminal `status` row both signal
                # "agent finished"; close the SSE so the client doesn't
                # hang waiting for keepalives on a closed log.
                if ev.name == "done":
                    return
                if ev.name == "status" and ev.data.get("status") in (
                    "done", "error", "cancelled", "interrupted",
                ):
                    return

        return EventSourceResponse(gen())

    @app.post("/api/complete/stream")
    async def inline_complete_stream(request: Request):
        """Streaming inline completion (sub-project 29).

        Same body shape as POST /api/complete; returns an SSE stream of
        ``chunk`` events whose ``data`` is a JSON object ``{"text": "..."}``
        followed by a final ``done`` event with ``{"model": "..."}``.

        Better UX for ghost text: the caller can render characters as
        they arrive instead of waiting for the full completion. Falls
        back through the same provider-resolution path as the
        non-streaming endpoint; unsupported providers (Anthropic /
        Gemini) yield HTTP 501 before any SSE bytes.
        """
        from sse_starlette.sse import EventSourceResponse
        from godbot.core.completion import stream_completion

        body = await request.json()
        prefix = str(body.get("prefix", ""))
        suffix = str(body.get("suffix", ""))
        language = body.get("language")
        if language is not None and not isinstance(language, str):
            raise HTTPException(400, "language must be a string")
        max_tokens = int(body.get("max_tokens", 64))

        cfg = load_config()
        provider_name = (body.get("provider") or cfg.providers.default or "lmstudio")
        item = cfg.providers.items.get(provider_name)
        if item is None:
            raise HTTPException(
                400, f"provider {provider_name!r} not configured in [providers.{provider_name}]",
            )
        model = body.get("model") or item.default_model or "auto"
        if model == "auto":
            try:
                pcfg = ProviderConfigType(
                    name=provider_name,
                    base_url=item.base_url,
                    api_key=item.api_key
                    or (item.api_key_env and os.environ.get(item.api_key_env, "") or ""),
                    default_model=item.default_model,
                )
                provider_inst = get_provider_factory(provider_name, pcfg)
                resolved = await provider_inst.select_model("auto")
                model = resolved.id
            except Exception as e:
                raise HTTPException(503, f"could not resolve model: {e}")

        api_key = item.api_key
        if not api_key and item.api_key_env:
            api_key = os.environ.get(item.api_key_env, "")

        # Pre-flight the support check so unsupported providers fail before
        # we open the SSE stream (cleaner UX than emitting an error event).
        from godbot.core.completion import _is_openai_compatible
        if not _is_openai_compatible(provider_name):
            raise HTTPException(
                501, f"provider {provider_name!r} not yet supported for streaming completion",
            )

        async def gen():
            try:
                async for chunk in stream_completion(
                    prefix=prefix, suffix=suffix, language=language,
                    provider_name=provider_name, base_url=item.base_url,
                    api_key=api_key, model=model, max_tokens=max_tokens,
                ):
                    yield {"event": "chunk", "data": json.dumps({"text": chunk})}
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"message": f"{type(e).__name__}: {e}"}),
                }
                return
            yield {"event": "done", "data": json.dumps({"model": model})}

        return EventSourceResponse(gen())

    @app.post("/api/complete")
    async def inline_complete(request: Request):
        """Cursor-style inline completion (sub-project 12).

        Body: ``{prefix: str, suffix?: str, language?: str, max_tokens?: int,
        provider?: str, model?: str}``. Returns ``{completion, model, elapsed_ms}``.

        Bypasses the agent loop for keystroke-frequency latency. Uses the
        configured default provider+model unless overridden, falls through
        with HTTP 501 for providers we don't yet support (Anthropic/Gemini).
        """
        from godbot.core.completion import complete_text

        body = await request.json()
        prefix = str(body.get("prefix", ""))
        suffix = str(body.get("suffix", ""))
        language = body.get("language")
        if language is not None and not isinstance(language, str):
            raise HTTPException(400, "language must be a string")
        max_tokens = int(body.get("max_tokens", 64))

        cfg = load_config()
        # Pick provider: explicit override > default. Same for the model.
        provider_name = (body.get("provider") or cfg.providers.default or "lmstudio")
        item = cfg.providers.items.get(provider_name)
        if item is None:
            raise HTTPException(
                400, f"provider {provider_name!r} not configured in [providers.{provider_name}]",
            )
        model = body.get("model") or item.default_model or "auto"
        # 'auto' is meaningless for raw completion — resolve via the provider
        # so we hit a real model id. Reuse the existing factory.
        if model == "auto":
            try:
                pcfg = ProviderConfigType(
                    name=provider_name,
                    base_url=item.base_url,
                    api_key=item.api_key
                    or (item.api_key_env and os.environ.get(item.api_key_env, "") or ""),
                    default_model=item.default_model,
                )
                provider_inst = get_provider_factory(provider_name, pcfg)
                resolved = await provider_inst.select_model("auto")
                model = resolved.id
            except Exception as e:
                raise HTTPException(503, f"could not resolve model: {e}")

        api_key = item.api_key
        if not api_key and item.api_key_env:
            api_key = os.environ.get(item.api_key_env, "")

        try:
            res = await complete_text(
                prefix=prefix,
                suffix=suffix,
                language=language,
                provider_name=provider_name,
                base_url=item.base_url,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
            )
        except ValueError as e:
            # Provider not supported (Anthropic/Gemini for now).
            raise HTTPException(501, str(e))
        except Exception as e:
            raise HTTPException(502, f"upstream provider failed: {type(e).__name__}: {e}")

        return {
            "completion": res.completion,
            "model": res.model,
            "elapsed_ms": res.elapsed_ms,
        }

    @app.post("/api/rag/index_workspace")
    async def rag_index_workspace(request: Request):
        """Build (or rebuild) the RAG index for a workspace.

        Body: ``{workspace: str, force?: bool}``. The collection name is
        derived from the workspace path via SHA-1 so two workspaces with
        the same basename don't collide. Indexing runs synchronously in a
        worker thread; for large repos consider calling this once per
        workspace open (the daemon caches the collection on disk so
        subsequent calls find existing data and bail unless ``force``).

        Returns ``{collection, indexed_chunks, cached: bool}``.
        """
        from godbot.tools.knowledge import workspace_collection_name
        from godbot.index import build_index
        from pathlib import Path as _Path

        body = await request.json()
        workspace = body.get("workspace")
        force = bool(body.get("force", False))
        if not workspace:
            raise HTTPException(400, "workspace required")
        try:
            ws_root = _Path(workspace).resolve()
        except Exception as e:
            raise HTTPException(400, f"workspace invalid: {e}")
        if not ws_root.is_dir():
            raise HTTPException(400, f"workspace not a directory: {ws_root}")

        collection = workspace_collection_name(str(ws_root))
        home = _Path(os.environ.get("GODBOT_HOME", str(_Path.home() / ".godbot")))
        coll_dir = home / "rag" / collection
        if coll_dir.exists() and not force:
            # Already indexed — return without re-running. Caller can pass
            # force=true to rebuild after big code changes.
            return {"collection": collection, "indexed_chunks": 0, "cached": True}

        # The build_index call is blocking (chroma writes, embedding API
        # round-trips). Offload to a thread so the event loop stays free.
        try:
            count = await asyncio.to_thread(
                build_index,
                ws_root,
                collection=collection,
                reindex=force,
                force=force,
            )
        except Exception as e:
            raise HTTPException(502, f"index build failed: {type(e).__name__}: {e}")
        return {"collection": collection, "indexed_chunks": count, "cached": False}

    @app.get("/api/rag/collections")
    async def rag_collections():
        home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
        rag_root = home / "rag"
        if not rag_root.exists():
            return []
        return [d.name for d in rag_root.iterdir() if d.is_dir()]

    @app.post("/api/rag/use")
    async def rag_use(request: Request):
        body = await request.json()
        sid = body["session_id"]
        coll = body.get("collection")
        try:
            session = _sessions_cache.get(sid) or Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        _sessions_cache[sid] = session
        session.set_rag_collection(coll)
        return {"ok": True}


def build_app(*, sessions_root: Optional[Path] = None) -> FastAPI:
    sessions_root = Path(sessions_root or (Path.cwd() / "sessions"))
    sessions_root.mkdir(parents=True, exist_ok=True)

    # Wire persistent task storage so background tasks survive daemon
    # restart. Non-terminal tasks at load time are flipped to
    # `interrupted` so the UI can show them as "killed by restart".
    try:
        from godbot.core.tasks import DEFAULT_RUNNER
        DEFAULT_RUNNER.attach_persistence(sessions_root / ".godbot-tasks")
    except Exception:
        # Persistence is best-effort: a corrupt directory should not stop
        # the daemon from booting.
        import logging as _logging
        _logging.getLogger("godbot.tasks").exception("attach_persistence failed; continuing without it")

    app = FastAPI(title="GodBot")

    # Allow browser surfaces (Tauri WebView, browser-based UIs) to call us.
    # Daemon binds 127.0.0.1 only, so wide-open CORS here is bounded to the
    # local machine.
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/agent/raw_completion")
    async def agent_raw_completion(request: Request):
        """Direct LLM passthrough — no agent loop, no ReAct framing
        (sub-project 73).

        Body:
          {
            messages: [{role, content}, ...],   required
            provider?: str,                     defaults to cfg.providers.default
            model?: str,                        defaults to provider's default
            temperature?: float = 0.7,
            max_tokens?: int = 2048,
          }

        Returns ``{content, finish_reason, usage, model}``.

        Power-user primitive: useful for custom prompt experiments,
        comparing models on the same prompt, or running a vanilla
        completion against the configured provider without going
        through the agent's tool-call machinery.

        OpenAI-compat providers only for now (Anthropic / Gemini have
        their own message shapes — fall through with HTTP 501).
        """
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages (non-empty list) required")
        for m in messages:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise HTTPException(400, "each message needs role + content")

        cfg = load_config()
        provider_name = body.get("provider") or cfg.providers.default or "lmstudio"
        item = cfg.providers.items.get(provider_name)
        if item is None:
            raise HTTPException(
                400, f"provider {provider_name!r} not configured",
            )
        from godbot.core.completion import _is_openai_compatible
        if not _is_openai_compatible(provider_name):
            raise HTTPException(
                501, f"provider {provider_name!r} not yet supported for raw_completion",
            )

        model = body.get("model") or item.default_model or "auto"
        if model == "auto":
            try:
                pcfg = ProviderConfigType(
                    name=provider_name, base_url=item.base_url,
                    api_key=item.api_key
                    or (item.api_key_env and os.environ.get(item.api_key_env, "") or ""),
                    default_model=item.default_model,
                )
                provider_inst = get_provider_factory(provider_name, pcfg)
                resolved = await provider_inst.select_model("auto")
                model = resolved.id
            except Exception as e:
                raise HTTPException(503, f"could not resolve model: {e}")

        api_key = item.api_key
        if not api_key and item.api_key_env:
            api_key = os.environ.get(item.api_key_env, "")
        try:
            temperature = float(body.get("temperature", 0.7))
            max_tokens = int(body.get("max_tokens", 2048))
        except (TypeError, ValueError):
            raise HTTPException(400, "temperature must be a float, max_tokens an int")

        import httpx as _httpx
        url = f"{item.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key or 'lm-studio'}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            async with _httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
        except _httpx.HTTPError as e:
            raise HTTPException(502, f"upstream failed: {type(e).__name__}: {e}")

        choices = data.get("choices") or []
        if not choices:
            raise HTTPException(502, "upstream returned no choices")
        msg = choices[0].get("message") or {}
        content = str(msg.get("content") or "")
        finish_reason = str(choices[0].get("finish_reason") or "stop")
        usage_raw = data.get("usage") or {}
        inp = int(usage_raw.get("input_tokens") or usage_raw.get("prompt_tokens") or 0)
        out = int(usage_raw.get("output_tokens") or usage_raw.get("completion_tokens") or 0)
        total = int(usage_raw.get("total_tokens") or (inp + out))
        return {
            "content": content,
            "finish_reason": finish_reason,
            "usage": {"input_tokens": inp, "output_tokens": out, "total_tokens": total},
            "model": model,
        }

    @app.post("/api/agent/dry_run")
    async def agent_dry_run(request: Request):
        """Render the system prompt + tool catalog WITHOUT invoking the model
        (sub-project 49).

        Body: ``{session_id: str, message?: str}``. The body's ``message``
        is appended to the session log so plan-mode / custom-command
        detection runs against it (the message is then rolled back via
        events.jsonl truncation? No — we instead simulate by checking
        the prefix locally).

        Actually simpler: we don't mutate the session at all. We snapshot
        the registry, run the plan-mode + custom-command checks against
        the supplied message (or the session's last user event), and
        render the prompt the agent loop *would* build.

        Returns:
          {
            system_prompt,
            mode: "plan" | "custom_command" | "default",
            command_name?,
            tool_catalog: [{name, description, schema, dangerous}],
            protocol_pref,
            message_count,
          }
        """
        from godbot.core.commands import match_command
        from godbot.core.registry import DEFAULT
        from godbot.prompts import (
            build_plan_system_prompt,
            build_system_prompt,
            is_plan_request,
        )

        body = await request.json()
        sid = body.get("session_id")
        if not sid:
            raise HTTPException(400, "session_id required")
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")

        # Determine the prompt mode for the hypothetical message. Falls
        # back to the most recent user event if no message supplied.
        msg = body.get("message")
        if msg is None:
            for ev in reversed(list(s._events())):
                if ev.get("type") == "user":
                    msg = ev.get("content")
                    break
        msg = msg or ""

        plan_mode = is_plan_request(msg)
        custom = None if plan_mode else match_command(msg)

        # Tool subset honoured for this run (plan/custom commands narrow it).
        if custom is not None and custom.tool_overrides:
            enabled = DEFAULT.subset(custom.tool_overrides)
        else:
            enabled = DEFAULT.subset(s.tool_overrides)

        if plan_mode:
            sys_prompt = build_plan_system_prompt(enabled)
            mode = "plan"
        else:
            sys_prompt = build_system_prompt(enabled)
            mode = "custom_command" if custom is not None else "default"
            if custom is not None and custom.system_prompt_suffix:
                sys_prompt = sys_prompt + "\n\n" + custom.system_prompt_suffix

        tool_catalog = [
            {
                "name": t.name,
                "description": t.description,
                "dangerous": t.dangerous,
                "schema": t.schema,
            }
            for t in sorted(enabled, key=lambda x: x.name)
        ]

        # Protocol preference: easy when provider is known.
        protocol_pref = s.protocol or "react_json"

        return {
            "system_prompt": sys_prompt,
            "mode": mode,
            "command_name": custom.name if custom is not None else None,
            "tool_catalog": tool_catalog,
            "protocol_pref": protocol_pref,
            "message_count": sum(
                1 for ev in s._events()
                if ev.get("type") in ("user", "assistant_final", "assistant_tool_call")
            ),
        }

    @app.get("/api/health/details")
    async def health_details():
        """Enriched health diagnostics (sub-project 43).

        Returns a structured snapshot the Studio status bar (and any
        external monitor) can consume:

          {
            status: "ok",
            version: <package version>,
            uptime_seconds: <int>,
            providers: {default: name, configured: [name, ...]},
            mcp_servers: [{name, connected, error?, tool_count}],
            tools: {total: int, dangerous: int},
            sessions: {count: int},
            background_tasks: {running: int, total: int},
          }

        Read-only and cheap — no provider probes, no FS walk except to
        count session directories.
        """
        import time
        from godbot.core.registry import DEFAULT
        from godbot.core.tasks import DEFAULT_RUNNER

        cfg = load_config()
        # MCP server status from the registered clients (see godbot.mcp).
        try:
            from godbot.mcp.registry_bridge import _clients as _mcp_clients
        except Exception:
            _mcp_clients = {}
        mcp_status = []
        for name, scfg in cfg.mcp.servers.items():
            client = _mcp_clients.get(name)
            mcp_status.append({
                "name": name,
                "connected": bool(client and client.connected),
                "error": client.connect_error if (client and client.connect_error) else None,
                "tool_count": len(client.tools) if (client and client.connected) else 0,
            })

        all_tools = DEFAULT.all()
        session_count = sum(
            1 for d in sessions_root.iterdir() if d.is_dir() and (d / "meta.json").exists()
        ) if sessions_root.exists() else 0

        tasks = DEFAULT_RUNNER.list_tasks()
        running_tasks = sum(1 for t in tasks if t.status in ("pending", "running"))

        # Read package version from importlib.metadata so we don't have
        # to keep a literal in two places.
        try:
            from importlib.metadata import version as _version
            pkg_version = _version("godbot")
        except Exception:
            pkg_version = "0.0.0"

        return {
            "status": "ok",
            "version": pkg_version,
            "uptime_seconds": int(time.time() - _DAEMON_STARTED_AT),
            "providers": {
                "default": cfg.providers.default,
                "configured": list(cfg.providers.items.keys()),
            },
            "mcp_servers": mcp_status,
            "tools": {
                "total": len(all_tools),
                "dangerous": sum(1 for t in all_tools if t.dangerous),
            },
            "sessions": {"count": session_count},
            "background_tasks": {"running": running_tasks, "total": len(tasks)},
        }

    @app.post("/api/cost/estimate")
    async def cost_estimate(request: Request):
        """Pre-flight cost estimator (sub-project 36).

        Body: ``{provider: str, model: str, input_tokens: int,
        output_tokens: int}``. Returns the same shape as
        ``GET /api/sessions/{sid}/cost`` — but for hypothetical inputs
        rather than a recorded session.

        Useful for "before I send a 50k-token context, what will it cost?"
        and Studio's pre-send price hint. Local providers always return
        $0; unknown (provider, model) returns matched=false with $0.
        """
        from godbot.core.pricing import compute_cost

        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        provider = body.get("provider")
        model = body.get("model")
        if not provider or not isinstance(provider, str):
            raise HTTPException(400, "provider (str) required")
        if not model or not isinstance(model, str):
            raise HTTPException(400, "model (str) required")
        try:
            inp = int(body.get("input_tokens", 0))
            out = int(body.get("output_tokens", 0))
        except (ValueError, TypeError):
            raise HTTPException(400, "input_tokens and output_tokens must be integers")
        if inp < 0 or out < 0:
            raise HTTPException(400, "token counts must be non-negative")
        cost = compute_cost(
            provider=provider, model=model,
            input_tokens=inp, output_tokens=out,
        )
        return cost.to_dict()

    @app.get("/api/audit")
    async def audit_dangerous_calls(limit: int = 100):
        """Log of dangerous-tool invocations across all sessions (sub-project 40).

        Walks every session's events.jsonl, finds ``assistant_tool_call``
        events whose tool name is currently flagged dangerous in the
        registry, and pairs them with the matching ``tool_result`` so
        the caller sees what actually happened. Newest-first; capped at
        ``limit`` (default 100).

        Use case: post-incident review — "did the agent run any
        write_file or run_powershell I don't remember approving?"

        Each entry: ``{session_id, started_at, call_id, name, args,
        result_preview, gated}``. ``gated`` is True iff the daemon's
        gate machinery would have prompted (some are auto-approved by
        sandbox or yolo flags; the audit doesn't differentiate which
        path was taken — it just shows the call happened).
        """
        from godbot.core.registry import DEFAULT

        dangerous_names = {t.name for t in DEFAULT.all() if t.dangerous}
        out: list[dict] = []
        if not sessions_root.exists():
            return {"calls": []}
        for sdir in sorted(sessions_root.iterdir(), reverse=True):
            if not sdir.is_dir():
                continue
            try:
                s = Session.load(sessions_root, sdir.name)
            except Exception:
                continue
            # Walk events sequentially, indexing tool_call by id so we can
            # match a tool_result that follows.
            pending: dict[str, dict] = {}
            for ev in s._events():
                t = ev.get("type")
                if t == "assistant_tool_call":
                    name = ev.get("name")
                    if name in dangerous_names:
                        pending[ev.get("call_id", "")] = {
                            "session_id": s.id,
                            "started_at": s._meta.get("started_at"),
                            "call_id": ev.get("call_id"),
                            "name": name,
                            "args": ev.get("args") or {},
                            "result_preview": "",
                            "gated": True,
                        }
                elif t == "tool_result":
                    cid = ev.get("call_id")
                    if cid in pending:
                        # Stash a preview (LLM-truncated) and finalise.
                        pending[cid]["result_preview"] = (ev.get("content") or "")[:300]
                        out.append(pending.pop(cid))
                if len(out) >= int(limit):
                    break
            # Flush any tool_call entries that had no matching result
            # (e.g. cancelled mid-flight). Mark result_preview accordingly.
            for entry in pending.values():
                entry["result_preview"] = "(no result recorded — cancelled or in-flight)"
                out.append(entry)
                if len(out) >= int(limit):
                    break
            if len(out) >= int(limit):
                break
        return {"calls": out[:limit]}

    @app.get("/api/stats")
    async def stats():
        """Aggregate metrics across all sessions on disk (sub-project 32).

        Walks ``<sessions_root>/*/events.jsonl`` and tallies:

        - session count and turn count
        - tool-call frequency per tool name
        - the top tool by call count
        - cumulative token usage (summed from each session's meta)
        - cumulative estimated cost (best-effort, via the pricing table)
        - background task counts by terminal status

        Read-only; safe to call frequently. The walk reads disk so it's
        bounded by the session count — fine for typical deployments,
        could be cached if a user accumulates 10k+ sessions.
        """
        from collections import Counter
        from godbot.core.pricing import compute_cost
        from godbot.core.tasks import DEFAULT_RUNNER

        session_count = 0
        turn_count = 0
        tool_calls: Counter[str] = Counter()
        total_input = 0
        total_output = 0
        total_usd = 0.0

        if sessions_root.exists():
            for sdir in sorted(sessions_root.iterdir()):
                if not sdir.is_dir():
                    continue
                meta_p = sdir / "meta.json"
                if not meta_p.exists():
                    continue
                session_count += 1
                try:
                    s = Session.load(sessions_root, sdir.name)
                except Exception:
                    continue
                u = s.usage
                turn_count += u["turns"]
                total_input += u["input_tokens"]
                total_output += u["output_tokens"]
                # Per-session cost; ignore unmatched.
                cost = compute_cost(
                    provider=s.provider,
                    model=s.model_name or s.model,
                    input_tokens=u["input_tokens"],
                    output_tokens=u["output_tokens"],
                )
                if cost.matched:
                    total_usd += cost.usd
                # Walk events for tool-call tally.
                for ev in s._events():
                    if ev.get("type") == "assistant_tool_call":
                        name = ev.get("name") or "(unknown)"
                        tool_calls[name] += 1

        # Background tasks: just the in-memory records (persistence stores
        # them on disk too, but DEFAULT_RUNNER reloads them on startup).
        task_recs = DEFAULT_RUNNER.list_tasks()
        task_status: Counter[str] = Counter(r.status for r in task_recs)

        top_tools = [
            {"name": name, "count": count}
            for name, count in tool_calls.most_common(10)
        ]
        return {
            "sessions": session_count,
            "turns": turn_count,
            "tool_calls_total": sum(tool_calls.values()),
            "top_tools": top_tools,
            "tasks": {
                "total": len(task_recs),
                "by_status": dict(task_status),
            },
            "usage": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
            },
            "estimated_cost_usd": round(total_usd, 6),
        }

    @app.post("/api/sessions/new")
    async def session_new(request: Request):
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        # Provider/model selection (sub-project 7). Defaults preserve legacy
        # behaviour: lmstudio + auto + react_json (None on the session
        # field, since None means "use the provider's preference").
        provider = body.get("provider", "lmstudio")
        model_name = body.get("model_name") or body.get("model") or "auto"
        protocol = body.get("protocol")
        s = Session.create(
            sessions_root,
            model=body.get("model", "auto"),
            workspace_root=body.get("workspace"),
            auto_approve_in_sandbox=bool(body.get("auto_approve_in_sandbox", False)),
            provider=provider,
            model_name=model_name,
            protocol=protocol,
        )
        return {
            "session_id": s.id,
            "provider": s.provider,
            "model_name": s.model_name,
            "protocol": s.protocol,
        }

    @app.get("/api/workspaces")
    async def workspaces_list():
        """List distinct workspaces seen across all sessions (sub-project 44).

        Walks every session's meta.json, collects unique
        ``workspace_root`` values, and aggregates per-workspace stats:
        last activity timestamp and session count. Useful for "give me
        the projects I've touched this month" pickers.

        Returns ``{workspaces: [{path, sessions, last_activity}]}``
        ordered by last_activity desc.
        """
        from collections import defaultdict
        agg: dict[str, dict] = {}
        if not sessions_root.exists():
            return {"workspaces": []}
        for sdir in sessions_root.iterdir():
            if not sdir.is_dir():
                continue
            try:
                s = Session.load(sessions_root, sdir.name)
            except Exception:
                continue
            ws = s._meta.get("workspace_root")
            if not ws:
                continue
            entry = agg.setdefault(ws, {"path": ws, "sessions": 0, "last_activity": ""})
            entry["sessions"] += 1
            started = s._meta.get("started_at") or ""
            if started > entry["last_activity"]:
                entry["last_activity"] = started
        out = sorted(
            agg.values(),
            key=lambda e: e["last_activity"], reverse=True,
        )
        return {"workspaces": out}

    @app.get("/api/sessions")
    async def sessions_list(
        workspace: Optional[str] = None,
        limit: Optional[int] = None,
        pinned_only: bool = False,
    ):
        """List sessions with optional workspace + pin filters (sub-project 79).

        Without filters: returns every session's ``{id, model, provider,
        workspace_root, started_at, pinned}`` newest-first. With
        ``workspace=<abs path>`` only sessions whose ``workspace_root``
        matches are returned. ``pinned_only=true`` returns only pinned
        sessions (sub-project 52). ``limit`` caps the count.
        """
        out: list[dict] = []
        try:
            entries = sorted(sessions_root.iterdir(), reverse=True)
        except OSError:
            return []
        for d in entries:
            if not d.is_dir():
                continue
            try:
                s = Session.load(sessions_root, d.name)
            except Exception:
                continue
            if workspace is not None and s._meta.get("workspace_root") != workspace:
                continue
            if pinned_only and not s.pinned:
                continue
            out.append({
                "id": s.id,
                "model": s.model,
                "model_name": s.model_name,
                "provider": s.provider,
                "workspace_root": s._meta.get("workspace_root"),
                "started_at": s._meta.get("started_at"),
                "pinned": s.pinned,
            })
            if limit is not None and len(out) >= int(limit):
                break
        return out

    @app.post("/api/sessions/{sid}/fork")
    async def session_fork(sid: str, request: Request):
        """Branch a session into a new one, preserving history up to an index.

        Body: ``{up_to_index?: int}``. When omitted, the fork copies the
        full event history; otherwise only events at indices < up_to_index
        are carried into the new session. The forked session inherits
        provider/model/workspace from the source.

        Useful for "what if I had said something different at turn 5?"
        without losing the original transcript. The new session starts
        fresh otherwise (no usage carried over, no budget). Returns
        the new session id.
        """
        try:
            src = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        if not isinstance(body, dict):
            body = {}
        events = list(src._events())
        up_to_index = body.get("up_to_index")
        if up_to_index is not None:
            try:
                up_to_index = int(up_to_index)
            except (TypeError, ValueError):
                raise HTTPException(400, "up_to_index must be an int")
            if up_to_index < 0 or up_to_index > len(events):
                raise HTTPException(400, f"up_to_index out of range (0..{len(events)})")
            events = events[:up_to_index]

        new_session = Session.create(
            sessions_root,
            model=src.model,
            workspace_root=src._meta.get("workspace_root"),
            auto_approve_in_sandbox=bool(src._meta.get("auto_approve_in_sandbox", False)),
            provider=src.provider,
            model_name=src.model_name,
            protocol=src.protocol,
        )
        # Copy events.jsonl prefix verbatim. We bypass append_user / etc. so
        # tool_call ↔ tool_result pairs stay correlated.
        if events:
            target_events = new_session.dir / "events.jsonl"
            with open(target_events, "a", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        # Copy any blobs referenced by the carried tool_results.
        import shutil as _shutil
        for ev in events:
            if ev.get("type") == "tool_result" and ev.get("blob"):
                blob_id = ev["blob"]
                src_blob = src.dir / "blobs" / f"{blob_id}.txt"
                if src_blob.exists():
                    dst_blob = new_session.dir / "blobs" / f"{blob_id}.txt"
                    try:
                        _shutil.copy2(src_blob, dst_blob)
                    except OSError:
                        pass  # best-effort; truncated content still in the LLM view
        return {
            "session_id": new_session.id,
            "forked_from": sid,
            "events_copied": len(events),
        }

    @app.post("/api/sessions/{sid}/feedback")
    async def session_post_feedback(sid: str, request: Request):
        """Record thumbs-up/thumbs-down on a turn (sub-project 35).

        Body: ``{target_index: int, rating: "up"|"down", comment?: str}``.
        Multiple feedbacks per index are allowed; the latest is what the
        UI surfaces. Useful for offline analysis of which turns the user
        approved.
        """
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        try:
            target_index = int(body["target_index"])
        except (KeyError, ValueError, TypeError):
            raise HTTPException(400, "target_index (int) required")
        rating = body.get("rating")
        if rating not in {"up", "down"}:
            raise HTTPException(400, "rating must be 'up' or 'down'")
        comment = body.get("comment")
        if comment is not None and not isinstance(comment, str):
            raise HTTPException(400, "comment must be a string")
        try:
            s.append_feedback(target_index, rating, comment=comment)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.get("/api/sessions/{sid}/feedback")
    async def session_get_feedback(sid: str):
        """Return aggregate feedback for one session: ``{up, down, latest_by_index}``."""
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        return s.feedback_summary()

    @app.post("/api/sessions/{sid}/pin")
    async def session_pin(sid: str, request: Request):
        """Pin or unpin a session (sub-project 52).

        Body: ``{pinned: bool}``. Pinned sessions are protected from
        ``POST /api/sessions/cleanup`` bulk-delete; they can still be
        removed explicitly via DELETE. Useful for favorite reference
        transcripts you don't want garbage-collected.
        """
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        if not isinstance(body, dict) or "pinned" not in body:
            raise HTTPException(400, "pinned (bool) required in body")
        s.set_pinned(bool(body["pinned"]))
        return {"ok": True, "pinned": s.pinned}

    @app.delete("/api/sessions/{sid}")
    async def session_delete(sid: str):
        """Permanently delete a session (sub-project 34).

        Removes the entire ``<sessions_root>/<sid>/`` directory: meta,
        events.jsonl, blobs/. Also drops the in-process cache entry so
        subsequent /api/chat for the same id will 404. Returns
        ``{ok: true}`` on success, 404 if the session never existed.

        Irreversible — no soft-delete in v1. The user can re-export to
        markdown via /api/sessions/{sid}/export before deleting.
        """
        sdir = sessions_root / sid
        if not sdir.exists() or not sdir.is_dir():
            raise HTTPException(404, "no such session")
        # Remove from caches + cancel any in-flight stream first.
        _sessions_cache.pop(sid, None)
        log = _logs.pop(sid, None)
        if log is not None and not log.closed:
            log.close()
        cancel = _cancels.pop(sid, None)
        if cancel is not None:
            cancel.set()
        import shutil as _shutil
        try:
            _shutil.rmtree(sdir)
        except Exception as e:
            raise HTTPException(500, f"delete failed: {type(e).__name__}: {e}")
        return {"ok": True, "deleted": sid}

    @app.post("/api/sessions/cleanup")
    async def sessions_cleanup(request: Request):
        """Bulk-delete sessions older than a cutoff (sub-project 34).

        Body: ``{older_than_days?: int = 30, dry_run?: bool = false}``.
        ``dry_run=true`` reports which sessions WOULD be deleted without
        actually removing anything; the default is destructive. Active
        sessions (started today, regardless of cutoff) are always kept.

        Returns ``{deleted: [sid, ...], kept: int, dry_run: bool}``.
        """
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        days = int(body.get("older_than_days", 30))
        if days < 1:
            raise HTTPException(400, "older_than_days must be >= 1")
        dry_run = bool(body.get("dry_run", False))

        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=days)
        to_delete: list[str] = []
        kept = 0
        if not sessions_root.exists():
            return {"deleted": [], "kept": 0, "dry_run": dry_run}
        for sdir in sessions_root.iterdir():
            if not sdir.is_dir():
                continue
            try:
                s = Session.load(sessions_root, sdir.name)
            except Exception:
                # Bad meta — count as kept rather than blow it away.
                kept += 1
                continue
            # Sub-project 52: pinned sessions are exempt from bulk cleanup.
            if s.pinned:
                kept += 1
                continue
            started = s._meta.get("started_at")
            try:
                ts = datetime.fromisoformat(started) if started else None
            except Exception:
                ts = None
            if ts is None or ts >= cutoff:
                kept += 1
                continue
            to_delete.append(s.id)
        if not dry_run:
            import shutil as _shutil
            for sid in to_delete:
                try:
                    _sessions_cache.pop(sid, None)
                    log = _logs.pop(sid, None)
                    if log is not None and not log.closed:
                        log.close()
                    _cancels.pop(sid, None)
                    _shutil.rmtree(sessions_root / sid)
                except Exception:
                    pass  # best-effort; report-as-attempted
        return {"deleted": to_delete, "kept": kept, "dry_run": dry_run}

    @app.get("/api/sessions/search")
    async def sessions_search(q: str, workspace: Optional[str] = None, limit: int = 20):
        """Substring search across session transcripts (sub-project 33).

        Walks every session's events.jsonl, finds messages containing
        the query (case-insensitive), and returns the first match per
        session as ``{id, started_at, model, snippet, role}``. Results
        are ordered by started_at desc.

        ``workspace`` (optional) limits the search to sessions whose
        ``workspace_root`` matches that absolute path.

        Linear-walk impl — fine for hundreds of sessions; users with
        thousands should reach for the RAG search instead.
        """
        if not q or not isinstance(q, str):
            raise HTTPException(400, "q (query string) required")
        q_lower = q.lower()
        out: list[dict] = []
        if not sessions_root.exists():
            return {"matches": []}
        for sdir in sorted(sessions_root.iterdir(), reverse=True):
            if not sdir.is_dir():
                continue
            try:
                s = Session.load(sessions_root, sdir.name)
            except Exception:
                continue
            if workspace is not None:
                ws_root = s._meta.get("workspace_root")
                if ws_root != workspace:
                    continue
            match: Optional[dict] = None
            for ev in s._events():
                t = ev.get("type")
                # We only search human-readable content, not raw JSON envelopes.
                content = ""
                role = ""
                if t == "user":
                    content = ev.get("content", "")
                    role = "user"
                elif t == "assistant_final":
                    content = ev.get("content", "")
                    role = "assistant"
                elif t == "tool_result":
                    content = ev.get("content", "")
                    role = "tool"
                else:
                    continue
                if q_lower in content.lower():
                    # Build a snippet centred on the first hit (capped).
                    idx = content.lower().find(q_lower)
                    start = max(0, idx - 40)
                    end = min(len(content), idx + len(q) + 40)
                    snippet = ("…" if start > 0 else "") + content[start:end] + ("…" if end < len(content) else "")
                    match = {
                        "id": s.id,
                        "started_at": s._meta.get("started_at"),
                        "model": s.model_name or s.model,
                        "snippet": snippet,
                        "role": role,
                    }
                    break
            if match:
                out.append(match)
            if len(out) >= int(limit):
                break
        return {"matches": out}

    @app.get("/api/sessions/{sid}")
    async def session_get(sid: str):
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        return {
            "id": s.id,
            "model": s.model,
            "provider": s.provider,
            "model_name": s.model_name,
            "protocol": s.protocol,
            "messages": s.messages_for_llm(max_context=0),
            "yolo": s.yolo,
            "auto_approved_tools": list(s._meta.get("auto_approved_tools", [])),
            "tool_overrides": s.tool_overrides,
            "rag_collection": s.rag_collection,
            "workspace_root": s._meta.get("workspace_root"),
            "auto_approve_in_sandbox": bool(s._meta.get("auto_approve_in_sandbox", False)),
            "usage": s.usage,
        }

    @app.get("/api/sessions/{sid}/usage")
    async def session_usage(sid: str):
        """Cumulative token usage for one session (sub-project 20).

        Always returns the normalized shape ``{input_tokens, output_tokens,
        total_tokens, turns}`` regardless of which provider drove the
        session. Zero everything for sessions that ran on legacy LLMClient
        (no usage exposed).
        """
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        return s.usage

    @app.post("/api/sessions/import")
    async def session_import(request: Request):
        """Import a session from a previously-exported JSON blob (sub-project 51).

        Body is the same shape ``GET /api/sessions/{sid}/export`` returns
        in JSON mode (id is replaced with a fresh one — the export's id
        is recorded in the new session's meta as ``imported_from``).
        Provider/model/workspace/usage/budget are all carried over;
        events are written into events.jsonl in order.

        Returns ``{session_id, imported_from, events_imported}``. Useful
        for restoring from backup, sharing a repro between machines, or
        seeding a new daemon with a known starting state.
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")

        events = body.get("events")
        if not isinstance(events, list):
            raise HTTPException(400, "events array required")

        # Build a new session with the imported provider/model/workspace.
        new_session = Session.create(
            sessions_root,
            model=str(body.get("model") or "auto"),
            workspace_root=body.get("workspace_root"),
            auto_approve_in_sandbox=False,
            provider=str(body.get("provider") or "lmstudio"),
            model_name=body.get("model_name"),
            protocol=body.get("protocol"),
        )
        # Mark the lineage in meta so an audit can trace the import chain.
        original_id = body.get("id")
        if original_id:
            new_session._meta["imported_from"] = str(original_id)
        # Carry over usage + budget if present so cost tracking continues.
        usage = body.get("usage") or {}
        if isinstance(usage, dict):
            new_session._meta["usage"] = {
                k: int(usage.get(k, 0))
                for k in ("input_tokens", "output_tokens", "total_tokens", "turns")
            }
        budget = body.get("budget") or {}
        if isinstance(budget, dict):
            new_session._meta["budget"] = {
                k: budget.get(k) for k in ("max_total_tokens", "max_usd")
                if budget.get(k) is not None
            }
        new_session._save_meta()

        # Append events verbatim. Skip non-dict entries defensively.
        target = new_session.dir / "events.jsonl"
        kept = 0
        with open(target, "a", encoding="utf-8") as f:
            for ev in events:
                if not isinstance(ev, dict) or "type" not in ev:
                    continue
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                kept += 1
        return {
            "session_id": new_session.id,
            "imported_from": original_id,
            "events_imported": kept,
        }

    @app.get("/api/sessions/{sid}/messages")
    async def session_messages(sid: str, offset: int = 0, limit: int = 50, role: Optional[str] = None):
        """Paginated access to a session's message log (sub-project 72).

        Returns ``{messages: [...], total: int, has_more: bool}``.
        Each message: ``{role, content, index}`` where index is the
        0-based position in the LLM-format log.

        ``role`` filters to one of ``"user"`` / ``"assistant"`` /
        ``"system"``. ``offset``/``limit`` control pagination
        (limit clamped to [1, 200]). Use this instead of GET
        /api/sessions/{sid} when you have a big session and only need
        recent messages.
        """
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        msgs = s.messages_for_llm(max_context=0)
        indexed = [
            {"role": m["role"], "content": m["content"], "index": i}
            for i, m in enumerate(msgs)
        ]
        if role is not None:
            if role not in {"user", "assistant", "system"}:
                raise HTTPException(400, "role must be 'user', 'assistant', or 'system'")
            indexed = [m for m in indexed if m["role"] == role]
        total = len(indexed)
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 200))
        page = indexed[offset:offset + limit]
        return {
            "messages": page,
            "total": total,
            "has_more": offset + limit < total,
        }

    @app.get("/api/sessions/{sid}/transcript")
    async def session_transcript(sid: str, include_tools: bool = False):
        """Plain-text conversation transcript (sub-project 80).

        Returns ``text/plain`` with simple ``Role: content`` lines, blank
        line separated. Strictly conversational — by default skips
        ``assistant_tool_call`` / ``tool_result`` events; pass
        ``include_tools=true`` to interleave them.

        Cleaner than the full markdown export when you just want the
        gist for sharing or pasting into another chat.
        """
        from fastapi.responses import PlainTextResponse
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        lines: list[str] = []
        for ev in s._events():
            t = ev.get("type")
            if t == "user":
                lines.append(f"User: {ev.get('content', '')}")
                lines.append("")
            elif t == "assistant_final":
                lines.append(f"Assistant: {ev.get('content', '')}")
                lines.append("")
            elif include_tools and t == "assistant_tool_call":
                args = ev.get("args") or {}
                lines.append(f"Tool call: {ev.get('name', '?')}({args})")
                lines.append("")
            elif include_tools and t == "tool_result":
                lines.append(f"Tool result: {(ev.get('content') or '')[:500]}")
                lines.append("")
        text = "\n".join(lines).rstrip() + "\n"
        return PlainTextResponse(text, media_type="text/plain; charset=utf-8")

    @app.get("/api/sessions/{sid}/explain")
    async def session_explain(sid: str):
        """Concise auto-label for a session (sub-project 61).

        Useful for SessionList UIs that want readable labels instead of
        opaque session ids. Returns:

          {
            label: <first 80 chars of the first user message>,
            tool_calls: <int>,
            messages: <int>,        # user + assistant_final
            errors: <int>,          # synthetic_tool_result counts
            tools_used: [name, ...] # distinct tools called, top 10
          }

        Read-only walk of events.jsonl; safe to call frequently.
        """
        from collections import Counter

        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")

        first_user = ""
        tool_calls = 0
        messages = 0
        errors = 0
        tools_used: Counter[str] = Counter()
        for ev in s._events():
            t = ev.get("type")
            if t == "user":
                if not first_user:
                    first_user = (ev.get("content") or "").strip()
                messages += 1
            elif t == "assistant_final":
                messages += 1
            elif t == "assistant_tool_call":
                tool_calls += 1
                name = ev.get("name") or "(unknown)"
                tools_used[name] += 1
            elif t == "synthetic_tool_result":
                errors += 1
        label = first_user[:80].replace("\n", " ") or "(no user message)"
        return {
            "label": label,
            "tool_calls": tool_calls,
            "messages": messages,
            "errors": errors,
            "tools_used": [n for n, _ in tools_used.most_common(10)],
        }

    @app.get("/api/sessions/{sid}/replay")
    async def session_replay(sid: str, delay_ms: int = 0):
        """Replay a past session's events as SSE (sub-project 58).

        Reads ``events.jsonl`` and re-emits each entry as an SSE event
        whose name is the event ``type`` and whose data is the full
        JSON record. ``delay_ms`` (0..2000) optionally throttles between
        events for animated UI playback. A final ``replay_done`` event
        signals the end of the stream.

        Read-only; doesn't touch the live agent runner. Useful for
        timeline UIs ("re-watch this turn step by step") and debug
        tools that want to scrub through a past session.
        """
        from sse_starlette.sse import EventSourceResponse
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        delay = max(0, min(int(delay_ms), 2000))

        async def gen():
            seq = 0
            for ev in s._events():
                seq += 1
                ev_type = ev.get("type", "unknown")
                yield {
                    "id": str(seq),
                    "event": ev_type,
                    "data": json.dumps(ev),
                }
                if delay > 0:
                    await asyncio.sleep(delay / 1000.0)
            yield {
                "event": "replay_done",
                "data": json.dumps({"events_replayed": seq}),
            }

        return EventSourceResponse(gen())

    @app.get("/api/sessions/{sid}/export")
    async def session_export(sid: str, format: str = "json"):
        """Export a full session as a self-contained record (sub-project 31).

        Bundles meta, every event from events.jsonl, usage, budget, and
        the cost breakdown. Useful for archival, sharing a repro,
        post-mortem analysis, or feeding a session into another tool.

        ``format=json`` (default) returns a structured JSON object.
        ``format=markdown`` returns a human-readable transcript with
        sections for user/assistant/tool turns.
        """
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        events = list(s._events())
        from godbot.core.pricing import compute_cost
        cost = compute_cost(
            provider=s.provider,
            model=s.model_name or s.model,
            input_tokens=s.usage["input_tokens"],
            output_tokens=s.usage["output_tokens"],
        )
        if format == "markdown":
            from fastapi.responses import PlainTextResponse
            md = _render_session_markdown(s, events, cost.to_dict())
            return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")
        return {
            "id": s.id,
            "model": s.model,
            "provider": s.provider,
            "model_name": s.model_name,
            "protocol": s.protocol,
            "workspace_root": s._meta.get("workspace_root"),
            "started_at": s._meta.get("started_at"),
            "ended_at": s._meta.get("ended_at"),
            "usage": s.usage,
            "budget": s.budget,
            "cost": cost.to_dict(),
            "events": events,
        }

    @app.post("/api/sessions/{sid}/budget")
    async def session_set_budget(sid: str, request: Request):
        """Set or clear soft spending caps on a session (sub-project 28).

        Body: ``{max_total_tokens?: int, max_usd?: float}``. Pass an empty
        body or omit a field to leave that cap unchanged. Pass ``null`` to
        clear that cap entirely. Both caps apply additively — the agent
        loop bails on whichever fires first.
        """
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        # Build the new cap set: start from existing, overlay incoming, drop nulls.
        cur = dict(s.budget)
        for k in ("max_total_tokens", "max_usd"):
            if k in body:
                cur[k] = body[k]  # may be None to clear
        s.set_budget(
            max_total_tokens=cur.get("max_total_tokens"),
            max_usd=cur.get("max_usd"),
        )
        return s.budget

    @app.get("/api/sessions/{sid}/budget")
    async def session_get_budget(sid: str):
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        return s.budget

    @app.get("/api/sessions/{sid}/blobs/{call_id}")
    async def session_blob(sid: str, call_id: str, as_text: bool = False):
        """Return the full tool-result blob for a given call (sub-project 26).

        Tool results larger than 8 KB are persisted under
        ``<session>/blobs/<call_id>.txt`` and the LLM view is truncated.
        This endpoint surfaces the complete content — for the agent to
        re-read its own elided output, or for the UI to render a "Show
        full result" disclosure.

        ``as_text=true`` switches to ``text/plain`` for raw downloads
        (large logs); default is JSON ``{call_id, content, length}``.
        """
        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        try:
            content = s.read_blob(call_id)
        except FileNotFoundError:
            raise HTTPException(404, "no such blob")
        if as_text:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content)
        return {
            "call_id": call_id,
            "length": len(content),
            "content": content,
        }

    @app.get("/api/sessions/{sid}/cost")
    async def session_cost(sid: str):
        """Estimated USD cost for one session (sub-project 23).

        Multiplies session usage by the rate from the configured pricing
        table (builtin defaults overlaid with ``~/.godbot/pricing.toml``).
        Returns ``matched=False`` with ``usd=0`` when the (provider, model)
        pair has no rate row — the UI should render "(rate unknown)" in
        that case rather than a misleading zero. Local providers (LM
        Studio, Ollama, vLLM) are always $0 with ``matched=True``.
        """
        from godbot.core.pricing import compute_cost

        try:
            s = Session.load(sessions_root, sid)
        except FileNotFoundError:
            raise HTTPException(404, "no such session")
        usage = s.usage
        breakdown = compute_cost(
            provider=s.provider,
            model=s.model_name or s.model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )
        out = breakdown.to_dict()
        out["usage"] = usage
        return out

    _register_endpoints(app, sessions_root)

    # OpenAI-compatible /v1/chat/completions shim (sub-project 8). Lets
    # any OpenAI-SDK client (Cursor, Continue, Aider, LangChain, vanilla
    # ``openai`` SDK) route through GodBot's agent loop without code
    # changes. Mounted here so the static catch-all below doesn't
    # shadow the route.
    from godbot.interfaces.openai_shim import build_router as build_openai_router
    app.include_router(build_openai_router(sessions_root))

    # Static frontend mounted in 9.5.
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


def main() -> int:
    import argparse
    import uvicorn
    from godbot.config import load_config
    parser = argparse.ArgumentParser(prog="godbot-web")
    parser.add_argument("--sessions-root", default=None,
                        help="Where to store sessions (default: ./sessions in cwd)")
    parser.add_argument("--port", type=int, default=None,
                        help="Override the configured web port")
    args = parser.parse_args()
    cfg = load_config()
    sessions_root = Path(args.sessions_root) if args.sessions_root else None
    app = build_app(sessions_root=sessions_root)
    port = args.port if args.port is not None else cfg.ui.web_port
    uvicorn.run(app, host="127.0.0.1", port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
