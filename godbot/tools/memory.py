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


def _active_workspace() -> str | None:
    """Read workspace from contextvar (set by agent loop) or env fallback."""
    try:
        from godbot.core.workspace import current_workspace
        ws = current_workspace()
        if ws is not None:
            return str(ws.root)
    except Exception:
        pass
    return os.environ.get("WORKSPACE_ROOT")


@tool()
def save_note(content: str, tags: list[str] | None = None) -> str:
    """Save a note. Auto-tags with the active workspace path.

    Notes saved while a workspace is active are scoped to that workspace and
    will be returned by recall_notes for that workspace by default.
    """
    if tags is None:
        tags = []
    workspace = _active_workspace()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    # Ensure uniqueness even when two saves land in the same microsecond
    # (Windows time resolution can collapse %f).
    base = _notes_dir()
    p = base / f"{ts}.json"
    n = 0
    uniq_ts = ts
    while p.exists():
        n += 1
        uniq_ts = f"{ts}_{n}"
        p = base / f"{uniq_ts}.json"
    record = {
        "timestamp": ts,
        "content": content,
        "tags": tags,
        "workspace": workspace,
    }
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        from godbot.core.rag import Embedder, RagStore

        home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
        emb = Embedder.create(prefer="lmstudio")
        store = RagStore(root=home / "rag", collection="_notes")
        store.add(
            ids=[uniq_ts],
            embeddings=emb.embed([content]),
            documents=[content],
            metadatas=[{
                "path": str(p),
                "lines": "1-1",
                "tags": ",".join(tags),
                "workspace": workspace or "",
            }],
        )
    except Exception as e:
        return f"ok: saved note {p.name} (RAG embed failed: {e})"
    return f"ok: saved note {p.name}"


@tool()
def auto_journal(summary: str) -> str:
    """Save a short end-of-task summary to the active workspace's journal.

    Use this when finishing a multi-turn task. One paragraph, ~3-5 sentences:
    what the user wanted, what got done, what's still pending. The next session
    in this workspace will see recent journals automatically.
    """
    return save_note(summary, tags=["journal:auto"])


def _filter_by_workspace(notes: list[dict], workspace: str | None) -> list[dict]:
    """Keep only notes matching `workspace`. Notes without a workspace tag are
    treated as global and always included."""
    if workspace is None:
        return notes
    out = []
    for n in notes:
        nw = n.get("workspace")
        if nw is None or nw == workspace:
            out.append(n)
    return out


@tool()
def recall_notes(query: str = "", top_k: int = 10, all_workspaces: bool = False) -> str:
    """Semantic search saved notes. Defaults to scoping by the active workspace.

    Set all_workspaces=True to ignore the workspace filter (search the whole
    note pool, including notes from other projects).
    """
    workspace = None if all_workspaces else _active_workspace()
    if not query:
        # Just list latest, filtered by workspace.
        all_notes = []
        for p in sorted(_notes_dir().glob("*.json")):
            try:
                all_notes.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        all_notes = _filter_by_workspace(all_notes, workspace)
        notes = all_notes[-top_k:]
        return (
            "\n\n".join(
                f"[{nt['timestamp']}] tags={nt['tags']}\n{nt['content']}" for nt in notes
            )
            or "(no notes)"
        )
    # Only attempt RAG if the _notes collection dir already exists; otherwise
    # constructing RagStore would create the dir as a side-effect, which then
    # masks the "fallback to substring" behavior expected when RAG is unset.
    home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
    notes_coll_dir = home / "rag" / "_notes"
    if notes_coll_dir.exists():
        try:
            from godbot.core.rag import Embedder, RagStore

            emb = Embedder.create(prefer="lmstudio")
            store = RagStore(root=home / "rag", collection="_notes")
            hits = store.search(emb.embed([query])[0], top_k=top_k)
            if hits:
                return "\n\n".join(
                    f"[{h['id']}] score={h['score']:.3f}\n{h['content']}" for h in hits
                )
            # Fall through to substring search if RAG returned no hits.
        except Exception:
            pass
    # Substring fallback (Phase 7 behavior).
    notes = []
    for p in sorted(_notes_dir().glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if query.lower() in data["content"].lower():
            notes.append(data)
    notes = _filter_by_workspace(notes, workspace)
    notes = notes[-top_k:]
    if not notes:
        return "(no notes match)"
    return "\n\n".join(
        f"[{nt['timestamp']}] tags={nt['tags']}\n{nt['content']}" for nt in notes
    )


def _notes_for_workspace(workspace: str) -> list[dict]:
    """Read every note matching `workspace`, sorted by timestamp ascending."""
    out: list[dict] = []
    for p in sorted(_notes_dir().glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("workspace") == workspace:
            out.append(data)
    return out


def load_recent_workspace_notes(workspace: str, limit: int = 5) -> str:
    """Return a formatted block of recent notes for `workspace`.

    All notes tagged ``pinned`` are ALWAYS included (sticky-to-top, capped at
    5 to keep the prompt bounded), followed by the most-recent unpinned notes
    until the total reaches ``limit``. Used by the agent loop to inject
    prior-session context into the system prompt at the start of a new
    session. Returns an empty string when no relevant notes exist.
    """
    workspace_notes = _notes_for_workspace(workspace)
    if not workspace_notes:
        return ""

    pinned = [n for n in workspace_notes if "pinned" in (n.get("tags") or [])]
    unpinned = [n for n in workspace_notes if "pinned" not in (n.get("tags") or [])]

    # Cap pinned to 5 to bound prompt size on workspaces that accumulate many
    # pins — matches the Risk row in the spec.
    pinned = pinned[-5:]

    remaining = max(0, limit - len(pinned))
    tail = unpinned[-remaining:] if remaining else []

    notes: list[dict] = list(pinned) + list(tail)
    if not notes:
        return ""

    def _fmt(nt: dict) -> str:
        marker = " [pinned]" if "pinned" in (nt.get("tags") or []) else ""
        return f"- [{nt['timestamp']}]{marker} {nt['content']}"

    body = "\n\n".join(_fmt(nt) for nt in notes)
    return f"<workspace_memory>\nRecent notes from this workspace ({workspace}):\n{body}\n</workspace_memory>"


def _find_note_path_by_timestamp(timestamp: str) -> Path | None:
    """Locate the on-disk file for a note whose record ``timestamp`` matches.

    The file name uses the unique form (``ts`` or ``ts_<n>`` for collisions),
    which differs from the ``timestamp`` field in the record body. We scan
    files because the suffix-disambiguator means filename != timestamp.
    """
    for p in sorted(_notes_dir().glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("timestamp") == timestamp:
            return p
    return None


def set_pin(timestamp: str, pinned: bool) -> bool:
    """Add or remove the ``pinned`` tag on a note. Returns True on success.

    RAG metadata is best-effort updated to keep the pin tag visible to
    similarity search; failures are silently swallowed because the note's
    on-disk truth is the source of authority.
    """
    p = _find_note_path_by_timestamp(timestamp)
    if p is None:
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    tags = list(data.get("tags") or [])
    if pinned and "pinned" not in tags:
        tags.append("pinned")
    elif not pinned and "pinned" in tags:
        tags = [t for t in tags if t != "pinned"]
    data["tags"] = tags
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Best-effort RAG sync — never let a RAG hiccup break a UI pin click.
    try:
        from godbot.core.rag import RagStore

        home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
        notes_coll_dir = home / "rag" / "_notes"
        if notes_coll_dir.exists():
            store = RagStore(root=home / "rag", collection="_notes")
            # File stem matches save_note's `uniq_ts` record id. update() on
            # the underlying chroma collection is the right primitive — patch
            # only the metadata.tags field.
            store._coll.update(
                ids=[p.stem],
                metadatas=[{"tags": ",".join(tags)}],
            )
    except Exception:
        pass
    return True


def delete_note(timestamp: str) -> bool:
    """Delete a note file by its timestamp. RAG entry removed best-effort."""
    p = _find_note_path_by_timestamp(timestamp)
    if p is None:
        return False
    stem = p.stem
    try:
        p.unlink()
    except Exception:
        return False
    try:
        from godbot.core.rag import RagStore

        home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
        notes_coll_dir = home / "rag" / "_notes"
        if notes_coll_dir.exists():
            store = RagStore(root=home / "rag", collection="_notes")
            store._coll.delete(ids=[stem])
    except Exception:
        pass
    return True


def list_notes(workspace: str | None = None, query: str = "") -> list[dict]:
    """Return notes (newest first), optionally scoped to ``workspace`` and
    filtered by case-insensitive substring against the content."""
    out: list[dict] = []
    for p in sorted(_notes_dir().glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if workspace is not None and data.get("workspace") != workspace:
            continue
        if query and query.lower() not in (data.get("content") or "").lower():
            continue
        out.append(data)
    # Newest first.
    out.reverse()
    return out
