from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from godbot.core.events import (
    ErrorEvent, GateEvent, ToolCallEvent, ToolResultEvent,
)


GROUP_COLORS: dict[str, int] = {
    "fs": 0x60A5FA,        # blue
    "shell": 0xF87171,     # red
    "web": 0x4ADE80,       # green
    "python": 0xC084FC,    # purple
    "memory": 0x2DD4BF,    # teal
    "task": 0xFBBF24,      # amber
    "rag": 0xA78BFA,       # violet
    "default": 0x9CA3AF,   # gray
}

_FS_TOOLS = {"read_file", "write_file", "edit_file", "glob", "grep", "list_dir", "read_blob"}
_SHELL_TOOLS = {"run_powershell", "run_bash"}
_WEB_TOOLS = {"web_fetch", "web_search"}
_MEMORY_TOOLS = {"save_note", "recall_notes"}
_TASK_TOOLS = {"todo_set", "todo_check"}


def group_for(name: str) -> str:
    if name in _FS_TOOLS:
        return "fs"
    if name in _SHELL_TOOLS:
        return "shell"
    if name in _WEB_TOOLS:
        return "web"
    if name == "run_python":
        return "python"
    if name in _MEMORY_TOOLS:
        return "memory"
    if name in _TASK_TOOLS:
        return "task"
    if name == "search_knowledge":
        return "rag"
    return "default"


@dataclass
class TokenAccumulator:
    """Accumulates streamed tokens and decides when to flush to Discord.

    Discord rate-limits message edits. We throttle by min interval AND by
    char threshold (whichever first).
    """
    throttle_ms: int = 750
    char_threshold: int = 400
    now_ms_fn: Callable[[], int] = field(default_factory=lambda: lambda: int(time.monotonic() * 1000))
    _buf: str = ""
    _last_emit_ms: int = -10**9
    _last_emit_len: int = 0

    def feed(self, chunk: str) -> Tuple[str, bool]:
        """Add chunk to buffer, return (full_text, should_emit_now)."""
        self._buf += chunk
        now = self.now_ms_fn()
        elapsed = now - self._last_emit_ms
        new_chars = len(self._buf) - self._last_emit_len
        if elapsed >= self.throttle_ms or new_chars >= self.char_threshold:
            self._last_emit_ms = now
            self._last_emit_len = len(self._buf)
            return (self._buf, True)
        return (self._buf, False)

    def flush(self) -> str:
        """Return the full accumulated text (used at DoneEvent)."""
        return self._buf


def extract_final_answer(raw: str) -> str:
    """Try to parse raw as ReAct JSON; return final_answer if present, else raw."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict) and "final_answer" in parsed:
        return str(parsed["final_answer"])
    return raw


def _trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def build_tool_call_embed(ev: ToolCallEvent) -> dict[str, Any]:
    args_str = ", ".join(f"{k}={v!r}" for k, v in list(ev.args.items())[:5])
    return {
        "title": f"▸ {ev.name}",
        "description": _trim(args_str, 1900),
        "color": GROUP_COLORS[group_for(ev.name)],
        "footer": {"text": "Running…"},
    }


def build_tool_result_update(ev: ToolResultEvent) -> dict[str, Any]:
    footer_text = f"[{ev.duration_ms}ms]"
    if ev.blob:
        footer_text += f" (blob:{ev.blob})"
    return {
        "fields": [{"name": "Result", "value": _trim(ev.preview or "(empty)", 800), "inline": False}],
        "footer": {"text": footer_text},
    }


def build_gate_embed(ev: GateEvent) -> dict[str, Any]:
    args_dump = json.dumps(ev.args, indent=2)
    return {
        "title": f"⚠ Approve {ev.name}?",
        "description": f"```json\n{_trim(args_dump, 1800)}\n```",
        "color": 0xFFA500,  # orange
    }


def build_error_embed(ev: ErrorEvent) -> dict[str, Any]:
    return {
        "title": "⛔ Error",
        "description": _trim(ev.message, 2000),
        "color": 0xEF4444,
    }
