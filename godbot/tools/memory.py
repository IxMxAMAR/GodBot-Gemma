from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path

from godbot.core.registry import tool


def _notes_dir() -> Path:
    home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
    d = home / "notes"
    d.mkdir(parents=True, exist_ok=True)
    return d


@tool()
def save_note(content: str, tags: list[str] | None = None) -> str:
    """Save a note to ~/.godbot/notes/ with optional tags."""
    if tags is None:
        tags = []
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    # Ensure uniqueness even when two saves land in the same microsecond
    # (Windows time resolution can collapse %f).
    base = _notes_dir()
    p = base / f"{ts}.json"
    n = 0
    while p.exists():
        n += 1
        p = base / f"{ts}_{n}.json"
    p.write_text(
        json.dumps({"timestamp": ts, "content": content, "tags": tags}, indent=2),
        encoding="utf-8",
    )
    return f"ok: saved note {p.name}"


@tool()
def recall_notes(query: str = "", top_k: int = 10) -> str:
    """Substring-search saved notes and return matches (semantic search added in Phase 10)."""
    notes = []
    for p in sorted(_notes_dir().glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if query == "" or query.lower() in data["content"].lower():
            notes.append(data)
    notes = notes[-top_k:]
    if not notes:
        return "(no notes match)"
    return "\n\n".join(
        f"[{n['timestamp']}] tags={n['tags']}\n{n['content']}" for n in notes
    )
