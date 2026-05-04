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


# Shared per-process state.
_streams: dict[str, asyncio.Queue] = {}
_cancels: dict[str, asyncio.Event] = {}
_sessions_cache: dict[str, Session] = {}
# Sessions whose SSE stream currently has an attached consumer.
# True resumability would require buffering all events for replay; for v0.1
# we fail-fast with HTTP 409 instead of hanging the second connection.
_active_streams: set[str] = set()


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


def _ensure_queue(sid: str) -> asyncio.Queue:
    q = _streams.get(sid)
    if q is None:
        q = asyncio.Queue()
        _streams[sid] = q
    return q


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
        queue = _ensure_queue(sid)

        async def emit(ev):
            await queue.put(ev)

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
                await queue.put(None)  # sentinel

        asyncio.create_task(runner())
        return {"session_id": sid}

    @app.get("/api/chat/stream")
    async def chat_stream(session_id: str):
        from sse_starlette.sse import EventSourceResponse
        # Single-consumer guard. The Queue can only be drained once; a second
        # consumer would silently hang (or worse, race the first). Fail fast.
        if session_id in _active_streams:
            raise HTTPException(
                409,
                "stream already attached; wait for current run or POST /api/stop",
            )
        queue = _ensure_queue(session_id)
        _active_streams.add(session_id)

        async def gen():
            try:
                heartbeat_at = _loop_now() + 15
                while True:
                    try:
                        timeout = max(0.1, heartbeat_at - _loop_now())
                        ev = await asyncio.wait_for(queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": "{}"}
                        heartbeat_at = _loop_now() + 15
                        continue
                    if ev is None:
                        return
                    d = event_to_dict(ev)
                    yield {"event": d["type"], "data": json.dumps(d)}
            finally:
                _active_streams.discard(session_id)

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

    @app.get("/api/sessions")
    async def sessions_list():
        out = []
        for d in sorted(sessions_root.iterdir()):
            if not d.is_dir():
                continue
            try:
                s = Session.load(sessions_root, d.name)
                out.append({"id": s.id, "model": s.model})
            except Exception:
                continue
        return out

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
