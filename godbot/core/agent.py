from __future__ import annotations
import asyncio
import json
import time
import uuid
from typing import Awaitable, Callable, Optional

from godbot.core.events import (
    DoneEvent, ErrorEvent, Event, GateEvent,
    TokenEvent, ToolCallEvent, ToolResultEvent,
)
from godbot.core.registry import Registry, ToolSpec
from godbot.core.session import Session
from godbot.core.schema import build_react_schema, validate_react_response
from godbot.core.workspace import set_workspace, _current as _workspace_current


EmitFn = Callable[[Event], Awaitable[None]]
SystemPromptBuilder = Callable[[list[ToolSpec]], str]


# Dangerous tools that are OK to skip the gate for when a workspace is active
# AND the user has opted into sandbox auto-approval. FS-confined tools only;
# shell tools stay gated because the soft sandbox can't actually contain them.
SANDBOX_SAFE_DANGEROUS_TOOLS = {"write_file", "edit_file"}


def _new_call_id() -> str:
    return "c" + uuid.uuid4().hex[:10]


def _compute_fs_diff(name: str, args: dict) -> Optional[dict]:
    """Build a {path, before, after} dict for FS-write gates.

    For ``write_file`` the proposed ``after`` is ``args["content"]``.
    For ``edit_file`` we simulate the find/replace locally so the UI can
    show the diff without the tool actually running. Returns ``None`` for
    any other tool.

    All filesystem errors collapse to ``before=None`` so a missing/unreadable
    file just renders as a "creating new file" diff in the UI.
    """
    if name not in {"write_file", "edit_file"}:
        return None
    from pathlib import Path
    path_str = str(args.get("path", ""))
    before: Optional[str]
    try:
        p = Path(path_str)
        if p.exists() and p.is_file():
            before = p.read_text(encoding="utf-8", errors="replace")
        else:
            before = None
    except Exception:
        before = None
    if name == "write_file":
        after = str(args.get("content", ""))
    else:  # edit_file
        old_str = str(args.get("old", ""))
        new_str = str(args.get("new", ""))
        if before is not None and old_str and old_str in before:
            after = before.replace(old_str, new_str, 1)
        else:
            after = before or ""
    return {"path": path_str, "before": before, "after": after}


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
    system_prompt_builder: Optional[SystemPromptBuilder] = None,
) -> None:
    enabled = registry.subset(session.tool_overrides)
    tool_schemas = {t.name: t.schema for t in enabled}
    react_schema = build_react_schema(tool_schemas)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "react", "schema": react_schema, "strict": True},
    }

    # Build the system prompt from the filtered tool subset so the model never
    # sees catalog entries that the schema enum would reject anyway. Falls
    # back to the static `system_prompt` string for back-compat with callers
    # that don't pass a builder.
    sys_prompt = system_prompt_builder(enabled) if system_prompt_builder is not None else system_prompt

    # Activate the workspace contextvar for the duration of the turn so
    # tools running in worker threads (asyncio.to_thread) inherit it.
    ws = session.workspace
    ws_token = set_workspace(ws) if ws is not None else None

    # Inject prior-session context for this workspace so the model has continuity
    # across launches. Only on the first turn (when the session has no prior
    # assistant messages) — otherwise we'd duplicate context every turn.
    if ws is not None and not any(
        ev.get("type") in ("assistant_final", "assistant_tool_call")
        for ev in session._events()
    ):
        try:
            from godbot.tools.memory import load_recent_workspace_notes
            mem = load_recent_workspace_notes(str(ws.root), limit=5)
            if mem:
                sys_prompt = sys_prompt + "\n\n" + mem
        except Exception:
            pass  # memory injection is best-effort
    try:
        for step in range(max_steps):
            if cancel.is_set():
                await emit(ErrorEvent(message="cancelled", recoverable=False))
                return

            messages = [{"role": "system", "content": sys_prompt}] + session.messages_for_llm(max_context=max_context)

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
    finally:
        if ws_token is not None:
            _workspace_current.reset(ws_token)


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

    # Compute fs_diff for FS-write tools so the UI can render an inline diff.
    fs_diff = _compute_fs_diff(name, args)

    # Workspace auto-approval and gate logic.
    ws = session.workspace
    args_override: Optional[dict] = None
    if ws is not None and ws.auto_approve_in_sandbox:
        # In sandbox YOLO mode, FS-safe dangerous tools auto-approve;
        # all other dangerous tools STILL gate (regardless of any prior
        # session-level "always" approval, which we deliberately ignore here
        # because the sandbox is a stronger guarantee than the per-session flag).
        if registry.is_dangerous(name) and name not in SANDBOX_SAFE_DANGEROUS_TOOLS:
            decision, args_override = await session.await_gate(
                call_id, name, args, emit, fs_diff=fs_diff, timeout=300,
            )
            if decision == "deny":
                msg = "User denied this tool call."
                session.append_tool_result(call_id, msg)
                await emit(ToolResultEvent(id=call_id, preview=msg, blob=None, duration_ms=0))
                return
            if decision == "always":
                session.mark_auto_approved(name)
        # else: FS-safe and dangerous, or non-dangerous — fall through to execute.
    elif (
        registry.is_dangerous(name)
        and not session.is_auto_approved(name)
        and not session.yolo
    ):
        decision, args_override = await session.await_gate(
            call_id, name, args, emit, fs_diff=fs_diff, timeout=300,
        )
        if decision == "deny":
            msg = "User denied this tool call."
            session.append_tool_result(call_id, msg)
            await emit(ToolResultEvent(id=call_id, preview=msg, blob=None, duration_ms=0))
            return
        if decision == "always":
            session.mark_auto_approved(name)
    # else: non-dangerous, OR sandbox-FS-safe, OR plain auto_approved/yolo — proceed.

    # Apply user-provided args override (e.g. user-edited diff content).
    # The merged dict is re-validated against the tool's JSON schema so a
    # malicious override can't smuggle invalid args past the validator.
    if args_override:
        merged = {**args, **args_override}
        err = registry.validate_args(name, merged)
        if err is not None:
            msg = f"args invalid after override: {err}"
            session.append_tool_result(call_id, msg)
            await emit(ToolResultEvent(id=call_id, preview=msg, blob=None, duration_ms=0))
            return
        args = merged

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
