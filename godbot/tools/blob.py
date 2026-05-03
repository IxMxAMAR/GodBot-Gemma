from __future__ import annotations
import os
from pathlib import Path
from godbot.core.registry import tool


@tool()
def read_blob(call_id: str, start: int = 0, lines: int = 200) -> str:
    """Read a slice of a tool-result blob from the active session.

    Reads `sessions/<id>/blobs/<call_id>.txt`. Active session dir is in
    GODBOT_ACTIVE_SESSION env var (set by the runtime).
    """
    sdir = os.environ.get("GODBOT_ACTIVE_SESSION")
    if not sdir:
        return "[error] no active session in env"
    p = Path(sdir) / "blobs" / f"{call_id}.txt"
    if not p.exists():
        return f"[error] blob not found: {call_id}"
    text = p.read_text(encoding="utf-8")
    all_lines = text.splitlines()
    end = min(len(all_lines), start + lines)
    return "\n".join(all_lines[start:end])
