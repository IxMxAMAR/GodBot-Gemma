from __future__ import annotations
import os
from pathlib import Path

from godbot.core.registry import tool


def _rag_root() -> Path:
    home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
    return home / "rag"


@tool()
def search_knowledge(query: str, top_k: int = 5, collection: str = "default") -> str:
    """Semantic search over indexed files. Returns chunks with path + line range."""
    from godbot.core.rag import Embedder, RagStore

    root = _rag_root()
    coll_dir = root / collection
    if not coll_dir.exists():
        return f"(collection {collection!r} not indexed; run `godbot-index <path>`)"
    try:
        e = Embedder.create(prefer="lmstudio")
        qv = e.embed([query])[0]
        store = RagStore(root=root, collection=collection)
        hits = store.search(qv, top_k=top_k)
    except Exception as exc:
        return f"(search_knowledge failed: {exc})"
    if not hits:
        return "(no matches)"
    return "\n\n".join(
        f"[{h['path']}:{h['lines']}] score={h['score']:.3f}\n{h['content']}" for h in hits
    )
