from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from godbot.core.session import Session


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
        }

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
