from __future__ import annotations
import asyncio
import json
import time
import uuid
from typing import Awaitable, Callable

from godbot.core.events import (
    DoneEvent, ErrorEvent, Event, GateEvent,
    TokenEvent, ToolCallEvent, ToolResultEvent,
)
from godbot.core.registry import Registry
from godbot.core.session import Session
from godbot.core.schema import build_react_schema, validate_react_response


EmitFn = Callable[[Event], Awaitable[None]]


def _new_call_id() -> str:
    return "c" + uuid.uuid4().hex[:10]


async def run_turn(
    *,
    llm,
    session: Session,
    registry: Registry,
    emit: EmitFn,
    cancel: asyncio.Event,
    max_steps: int,
    max_context: int,
    system_prompt: str,
) -> None:
    enabled = registry.subset(session.tool_overrides)
    tool_schemas = {t.name: t.schema for t in enabled}
    react_schema = build_react_schema(tool_schemas)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "react", "schema": react_schema, "strict": True},
    }

    for step in range(max_steps):
        if cancel.is_set():
            await emit(ErrorEvent(message="cancelled", recoverable=False))
            return

        messages = [{"role": "system", "content": system_prompt}] + session.messages_for_llm(max_context=max_context)

        async def _on_delta(t: str) -> None:
            await emit(TokenEvent(text=t))

        full_text = await llm.complete_streaming(
            messages=messages,
            on_delta=_on_delta,
            response_format=response_format,
            cancel=cancel,
        )

        try:
            parsed = json.loads(full_text)
        except json.JSONDecodeError as e:
            session.append_synthetic_tool_result(
                f"Your last reply was not valid JSON: {e}. Reply ONLY with the JSON object."
            )
            continue

        err = validate_react_response(parsed, react_schema)
        if err:
            session.append_synthetic_tool_result(
                f"Your last reply did not match the schema: {err}. Try again."
            )
            continue

        if "final_answer" in parsed:
            session.append_assistant_final(parsed["final_answer"])
            await emit(DoneEvent(step_count=step + 1))
            return

        await _handle_action(parsed, session, registry, enabled, emit, cancel)

    await emit(ErrorEvent(message="max_steps exceeded", recoverable=False))


async def _handle_action(
    parsed: dict,
    session: Session,
    registry: Registry,
    enabled,
    emit: EmitFn,
    cancel: asyncio.Event,
) -> None:
    name = parsed["action"]
    args = parsed.get("args") or {}
    call_id = _new_call_id()

    session.append_assistant_tool_call(call_id, name, args, raw=json.dumps(parsed))
    await emit(ToolCallEvent(id=call_id, name=name, args=args))

    err = registry.validate_args(name, args)
    if err is not None:
        msg = f"args invalid: {err}"
        session.append_tool_result(call_id, msg)
        await emit(ToolResultEvent(id=call_id, preview=msg, blob=None, duration_ms=0))
        return

    if registry.is_dangerous(name) and not session.is_auto_approved(name) and not session.yolo:
        decision = await session.await_gate(call_id, name, args, emit, timeout=300)
        if decision == "deny":
            msg = "User denied this tool call."
            session.append_tool_result(call_id, msg)
            await emit(ToolResultEvent(id=call_id, preview=msg, blob=None, duration_ms=0))
            return
        if decision == "always":
            session.mark_auto_approved(name)

    started = time.time()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(registry.execute, name, args),
            timeout=registry.timeout_for(name),
        )
    except asyncio.TimeoutError:
        result = f"tool {name!r} timed out after {registry.timeout_for(name)}s"
    except Exception as e:
        result = f"tool {name!r} raised {type(e).__name__}: {e}"
    duration_ms = int((time.time() - started) * 1000)

    llm_view = session.record_tool_result(call_id, str(result))
    blob = call_id if len(str(result)) > 8 * 1024 else None
    await emit(ToolResultEvent(id=call_id, preview=llm_view[:400], blob=blob, duration_ms=duration_ms))
