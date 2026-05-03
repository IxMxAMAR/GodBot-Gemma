from __future__ import annotations
import json
import os
from pathlib import Path
from godbot.core.registry import tool


def _todo_path() -> Path:
    home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
    home.mkdir(parents=True, exist_ok=True)
    return home / "todos.json"


def _load() -> list[dict]:
    p = _todo_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(items: list[dict]) -> None:
    _todo_path().write_text(json.dumps(items, indent=2), encoding="utf-8")


@tool()
def todo_set(items: list[str]) -> str:
    """Replace the current todo list with a new set of items (all unchecked)."""
    state = [{"text": t, "done": False} for t in items]
    _save(state)
    return _render(state)


@tool()
def todo_check(index: int) -> str:
    """Toggle the checkbox at zero-based index."""
    state = _load()
    if not (0 <= index < len(state)):
        return f"[error] index out of range (0..{len(state)-1})"
    state[index]["done"] = not state[index]["done"]
    _save(state)
    return _render(state)


def _render(state: list[dict]) -> str:
    return "\n".join(
        f"{i}. [{'x' if it['done'] else ' '}] {it['text']}" for i, it in enumerate(state)
    ) or "(empty)"
