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
    """Save a note + embed it into the _notes RAG collection."""
    if tags is None:
        tags = []
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
    p.write_text(
        json.dumps({"timestamp": ts, "content": content, "tags": tags}, indent=2),
        encoding="utf-8",
    )
    try:
        from godbot.core.rag import Embedder, RagStore

        home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
        emb = Embedder.create(prefer="lmstudio")
        store = RagStore(root=home / "rag", collection="_notes")
        store.add(
            ids=[uniq_ts],
            embeddings=emb.embed([content]),
            documents=[content],
            metadatas=[{"path": str(p), "lines": "1-1", "tags": ",".join(tags)}],
        )
    except Exception as e:
        return f"ok: saved note {p.name} (RAG embed failed: {e})"
    return f"ok: saved note {p.name}"


@tool()
def recall_notes(query: str = "", top_k: int = 10) -> str:
    """Semantic search saved notes (substring fallback if RAG unavailable)."""
    if not query:
        # Just list latest.
        notes = []
        for p in sorted(_notes_dir().glob("*.json"))[-top_k:]:
            try:
                notes.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
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
    notes = notes[-top_k:]
    if not notes:
        return "(no notes match)"
    return "\n\n".join(
        f"[{nt['timestamp']}] tags={nt['tags']}\n{nt['content']}" for nt in notes
    )
