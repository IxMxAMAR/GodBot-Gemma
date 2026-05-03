from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from godbot.config import load_config
from godbot.core.agent import run_turn
from godbot.core.events import event_to_dict
from godbot.core.llm import LLMClient
from godbot.core.registry import DEFAULT
from godbot.core.session import Session
from godbot.prompts import build_system_prompt


# Shared per-process state.
_streams: dict[str, asyncio.Queue] = {}
_cancels: dict[str, asyncio.Event] = {}
_sessions_cache: dict[str, Session] = {}


def get_llm() -> LLMClient:
    cfg = load_config()
    client = LLMClient(base_url=cfg.llm.base_url, model=cfg.llm.model)
    client.probe()
    return client


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
                await run_turn(
                    llm=llm, session=session, registry=DEFAULT, emit=emit,
                    cancel=cancel, max_steps=cfg.agent.max_steps,
                    max_context=cfg.llm.max_context, system_prompt="",
                    system_prompt_builder=build_system_prompt,
                )
            finally:
                await queue.put(None)  # sentinel

        asyncio.create_task(runner())
        return {"session_id": sid}

    @app.get("/api/chat/stream")
    async def chat_stream(session_id: str):
        from sse_starlette.sse import EventSourceResponse
        queue = _ensure_queue(session_id)

        async def gen():
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

        return EventSourceResponse(gen())

    @app.post("/api/gate/{call_id}")
    async def gate_resolve(call_id: str, request: Request):
        body = await request.json()
        sid = body["session_id"]
        decision = body["decision"]
        session = _sessions_cache.get(sid)
        if session is None:
            raise HTTPException(404, "no active session")
        ok = session.resolve_gate(call_id, decision)
        return {"ok": ok}

    @app.post("/api/stop")
    async def stop(request: Request):
        body = await request.json()
        sid = body["session_id"]
        ev = _cancels.get(sid)
        if ev is not None:
            ev.set()
        return {"stopped": True}

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

    app = FastAPI(title="GodBot")

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
        s = Session.create(sessions_root, model=body.get("model", "auto"))
        return {"session_id": s.id}

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
            "messages": s.messages_for_llm(max_context=0),
            "yolo": s.yolo,
            "auto_approved_tools": list(s._meta.get("auto_approved_tools", [])),
            "tool_overrides": s.tool_overrides,
            "rag_collection": s.rag_collection,
        }

    _register_endpoints(app, sessions_root)

    # Static frontend mounted in 9.5.
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


def main() -> int:
    import uvicorn
    from godbot.config import load_config
    cfg = load_config()
    app = build_app()
    uvicorn.run(app, host="127.0.0.1", port=cfg.ui.web_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
