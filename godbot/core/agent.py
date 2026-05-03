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
    raise NotImplementedError  # filled in 6.2
