from __future__ import annotations
import os
from pathlib import Path
from godbot.core.registry import tool


@tool()
def read_blob(call_id: str, start: int = 0, lines: int = 200) -> str:
    """Read a slice of a previous tool-result blob from THIS session.

    A "blob" is the full text of a prior tool result that was too large
    to inline (>8 KB). The truncated tool_result the agent saw will
    have ended with ``... [N chars elided; full result in blob <id>]``
    — that ``<id>`` is the ``call_id`` to pass here. ``call_id`` values
    are NOT free-form: they must come from your own prior tool_result
    events in the current session.

    If you're trying to peek at a regular file on disk, use
    ``read_file`` / ``head`` / ``tail`` instead. Blobs are agent-internal.

    Reads ``sessions/<sid>/blobs/<call_id>.txt``.
    """
    sdir = os.environ.get("GODBOT_ACTIVE_SESSION")
    if not sdir:
        return "[error] no active session in env"
    p = Path(sdir) / "blobs" / f"{call_id}.txt"
    if not p.exists():
        # List a couple of recent blobs so the agent sees what's actually
        # available — much more steerable than just "not found".
        blob_dir = p.parent
        existing = sorted(blob_dir.glob("*.txt"))[-5:] if blob_dir.exists() else []
        if existing:
            avail = ", ".join(f.stem for f in existing)
            return (
                f"[error] no blob with call_id={call_id!r}. "
                f"call_ids must reference your own prior >8KB tool_result. "
                f"Available blob ids in this session: {avail}. "
                f"For regular files, use read_file/head/tail instead."
            )
        return (
            f"[error] no blob with call_id={call_id!r}. "
            f"This session has no blobs yet — blobs are created automatically "
            f"when a tool_result exceeds 8KB. For regular files, use "
            f"read_file/head/tail instead."
        )
    text = p.read_text(encoding="utf-8")
    all_lines = text.splitlines()
    end = min(len(all_lines), start + lines)
    return "\n".join(all_lines[start:end])
