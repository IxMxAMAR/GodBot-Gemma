"""OpenAI Chat Completions API shim.

Maps ``POST /v1/chat/completions`` requests onto GodBot's internal agent
loop, hiding tool orchestration from the caller. Compatible with OpenAI's
SDK and every tool that already speaks that wire format (Cursor,
Continue, Aider, LangChain, vanilla ``openai`` SDK).

Two operating modes:

  * **Stateless** (default): every request creates an ephemeral session,
    runs one turn, then leaves the session on disk (no auto-delete in
    v1 — useful for forensic debugging). No memory across calls.
  * **Stateful** (opt-in via ``X-Godbot-Session: <sid>`` header): request
    maps onto a persistent session by id. Memory persists across calls.

The ``model`` field in the request is interpreted as a GodBot
provider+model spec — see :func:`parse_model_spec` for the grammar.

Deviations from sub-project 8 spec section 4.1
----------------------------------------------
The spec calls ``run_turn(llm=None, ...)`` with no provider — but the
agent loop (post-sub-project 7) requires *exactly one* of ``llm`` or
``provider`` to be supplied (XOR check at the top of ``run_turn``). We
therefore always resolve a :class:`Provider` from the session's
provider/model selection and pass ``provider=`` instead. Net effect is
identical: the agent loop runs with the right backend.

We also default stateless requests to ``auto_approve_in_sandbox=False``
(the conservative default). External clients have no UI to click gate
buttons, so dangerous tools will hang for the gate's 5-minute timeout
then deny — at which point the agent recovers with a "can't do that
without approval" message. That's acceptable for v1; we accept the risk
of slightly slow rejection over the risk of silent dangerous-tool
execution. Users who *want* unattended dangerous execution should set
``yolo`` on a stateful session.
"""
from __future__ import annotations
import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from godbot.config import load_config, providers_to_configs
from godbot.core.agent import run_turn
from godbot.core.events import DoneEvent, ErrorEvent, TokenEvent
from godbot.core.providers import (
    ProviderConfig,
    get_provider as get_provider_factory,
    load_providers_from_config,
)
from godbot.core.registry import DEFAULT
from godbot.core.session import Session
from godbot.prompts import build_system_prompt


def parse_model_spec(model: str) -> tuple[Optional[str], str]:
    """Parse ``godbot`` / ``provider:model`` / ``model`` into (provider, model).

    Examples:

      * ``"godbot"`` or ``""`` -> ``(None, "auto")`` (use daemon defaults)
      * ``"lmstudio:auto"`` -> ``("lmstudio", "auto")``
      * ``"anthropic:claude-3-5-sonnet-latest"`` -> cross-provider routing
      * ``"gpt-4o"`` (no colon) -> ``(None, "gpt-4o")`` (default provider, this model)
    """
    if model in ("godbot", ""):
        return (None, "auto")
    if ":" in model:
        provider, mdl = model.split(":", 1)
        return (provider.strip() or None, mdl.strip() or "auto")
    return (None, model)


def _resolve_provider_for_session(session: Session):
    """Build a :class:`Provider` for ``session`` from current config.

    Mirrors :func:`godbot.interfaces.web._provider_for_session` but always
    returns a real provider (never ``None``) — the openai-shim path has
    no legacy ``LLMClient`` fallback to fall through to. lmstudio is
    handled as a vanilla OpenAI-compatible provider, which has the same
    on-the-wire behaviour as the legacy shim.
    """
    name = session.provider or "lmstudio"
    cfg = load_config()
    raw = providers_to_configs(cfg.providers)
    parsed = load_providers_from_config(raw)
    pcfg = parsed.get(name) or ProviderConfig(name=name)
    return get_provider_factory(name, pcfg)


def build_router(sessions_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        x_godbot_session: Optional[str] = Header(default=None, alias="X-Godbot-Session"),
    ):
        body = await request.json()
        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(400, "messages required")
        model_spec = body.get("model", "godbot")
        provider_name, model_name = parse_model_spec(model_spec)
        stream = bool(body.get("stream", False))

        # Resolve session: stateful via header, otherwise ephemeral.
        if x_godbot_session:
            try:
                session = Session.load(sessions_root, x_godbot_session)
            except FileNotFoundError:
                raise HTTPException(404, f"no such session {x_godbot_session!r}")
            ephemeral = False
        else:
            session = Session.create(
                sessions_root,
                model=model_name,
                provider=provider_name or "lmstudio",
                model_name=model_name,
            )
            ephemeral = True

        # Append messages from the request. In stateless mode we replay
        # all of them onto the fresh session so the agent has the same
        # history shape an OpenAI caller expects. In stateful mode we
        # only consume the *last* user message — the session already has
        # its own history and we don't want to double-write.
        if ephemeral:
            for m in messages:
                role, content = m.get("role"), m.get("content", "")
                if role == "system" and isinstance(content, str):
                    # System messages are honoured implicitly: GodBot's
                    # own system prompt subsumes most use cases. Future
                    # work could splice the caller's system into the
                    # agent prompt; for v1 we silently drop it.
                    pass
                elif role == "user" and isinstance(content, str):
                    session.append_user(content)
                elif role == "assistant" and isinstance(content, str):
                    session.append_assistant_final(content)
        else:
            # Stateful: only the last user message is appended.
            last = messages[-1]
            if last.get("role") != "user":
                raise HTTPException(400, "last message must be role=user in stateful mode")
            session.append_user(str(last.get("content", "")))

        cfg = load_config()
        cancel = asyncio.Event()
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:20]
        created = int(time.time())

        if stream:
            return _streaming_response(
                completion_id, created, model_spec, session, cancel, cfg, ephemeral,
            )
        return await _blocking_response(
            completion_id, created, model_spec, session, cancel, cfg, ephemeral,
        )

    return router


def _final_answer_so_far(buf: str) -> Optional[str]:
    """Best-effort extract ``final_answer`` from a partial JSON buffer.

    Returns ``None`` until the field is parseable, then returns the
    partial value (with common JSON escapes unescaped). The buffer may
    still be growing when this is called — we deliberately tolerate an
    unterminated quoted string.
    """
    m = re.search(r'"final_answer"\s*:\s*"((?:\\.|[^"\\])*)', buf)
    if not m:
        return None
    raw = m.group(1)
    return (raw
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\"))


def _full_final_answer(buf: str) -> str:
    """After the agent is done: parse JSON and extract ``final_answer``.

    Falls back to the raw buffer if the model emitted something that
    isn't a valid ReAct JSON object (e.g. a native-tools provider that
    streams plain text). Callers downstream still get something sensible
    rather than an empty string.
    """
    try:
        obj = json.loads(buf)
        if isinstance(obj, dict) and "final_answer" in obj:
            return str(obj["final_answer"])
    except Exception:
        pass
    return buf


async def _run_agent(session: Session, emit, cancel: asyncio.Event, cfg) -> None:
    """Resolve a provider for the session and run one agent turn.

    Centralised here so streaming + blocking paths stay in sync. The
    provider resolution uses the live config so config edits take
    effect without a daemon restart (same contract as
    ``/api/chat``).
    """
    provider = _resolve_provider_for_session(session)
    await run_turn(
        provider=provider,
        session=session,
        registry=DEFAULT,
        emit=emit,
        cancel=cancel,
        max_steps=cfg.agent.max_steps,
        max_context=cfg.llm.max_context,
        system_prompt="",
        system_prompt_builder=build_system_prompt,
    )


def _streaming_response(
    completion_id: str, created: int, model_spec: str,
    session: Session, cancel: asyncio.Event, cfg, ephemeral: bool,
):
    """Stream OpenAI-format SSE chunks while the agent runs.

    Tool-call events are intentionally NOT forwarded — the shim hides
    tool orchestration from the caller. Only the agent's final answer
    text reaches the wire, streamed as ``delta.content`` chunks.
    """
    queue: asyncio.Queue = asyncio.Queue()
    raw_buf = {"text": ""}
    last_emitted = {"text": ""}

    async def emit(ev):
        if isinstance(ev, TokenEvent):
            raw_buf["text"] += ev.text
            partial = _final_answer_so_far(raw_buf["text"])
            if partial is not None and len(partial) > len(last_emitted["text"]):
                new_text = partial[len(last_emitted["text"]):]
                last_emitted["text"] = partial
                await queue.put({"delta": {"content": new_text}})
        elif isinstance(ev, DoneEvent):
            final = _full_final_answer(raw_buf["text"])
            # Emit any remaining text we hadn't streamed yet (e.g. when
            # a native-tools provider returns the whole answer in one
            # final chunk without a partial JSON window).
            if len(final) > len(last_emitted["text"]):
                tail = final[len(last_emitted["text"]):]
                await queue.put({"delta": {"content": tail}})
            await queue.put({"done": True})
        elif isinstance(ev, ErrorEvent):
            await queue.put({"error": ev.message})
        # ToolCallEvent / ToolResultEvent / GateEvent are intentionally
        # NOT forwarded — see docstring above.

    async def runner():
        try:
            await _run_agent(session, emit, cancel, cfg)
        except Exception as e:
            await queue.put({"error": f"agent crash: {e}"})
        finally:
            await queue.put({"done": True})

    async def gen():
        asyncio.create_task(runner())
        while True:
            ev = await queue.get()
            if ev.get("done"):
                final_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_spec,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield {"data": json.dumps(final_chunk)}
                yield {"data": "[DONE]"}
                return
            if "error" in ev:
                err_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_spec,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"\n\n[error] {ev['error']}"},
                        "finish_reason": "stop",
                    }],
                }
                yield {"data": json.dumps(err_chunk)}
                yield {"data": "[DONE]"}
                return
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model_spec,
                "choices": [{"index": 0, "delta": ev["delta"], "finish_reason": None}],
            }
            yield {"data": json.dumps(chunk)}

    return EventSourceResponse(gen())


async def _blocking_response(
    completion_id: str, created: int, model_spec: str,
    session: Session, cancel: asyncio.Event, cfg, ephemeral: bool,
):
    """Run the agent to completion and return one OpenAI-format JSON body."""
    raw_buf = {"text": ""}
    error = {"msg": None}

    async def emit(ev):
        if isinstance(ev, TokenEvent):
            raw_buf["text"] += ev.text
        elif isinstance(ev, ErrorEvent):
            error["msg"] = ev.message

    try:
        await _run_agent(session, emit, cancel, cfg)
    except Exception as e:
        error["msg"] = f"agent crash: {e}"

    final = _full_final_answer(raw_buf["text"])
    if error["msg"]:
        # Surface the error inline so the OpenAI client doesn't see a
        # silent empty completion. We still return 200 because partial
        # output may be useful — same contract as the streaming path.
        final = f"{final}\n\n[error] {error['msg']}" if final else f"[error] {error['msg']}"
    return {
        "id": completion_id, "object": "chat.completion",
        "created": created, "model": model_spec,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final},
            "finish_reason": "stop",
        }],
        # Token accounting is provider-specific and we don't have a
        # unified counter yet; report zeros so the response shape is
        # valid OpenAI JSON. Clients that key off "usage" being present
        # (rather than nonzero) keep working.
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
