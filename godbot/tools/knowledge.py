from __future__ import annotations
import hashlib
import os
from pathlib import Path

from godbot.core.registry import tool
from godbot.core.workspace import current_workspace


def _rag_root() -> Path:
    home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
    return home / "rag"


def workspace_collection_name(workspace_root: str) -> str:
    """Stable, collision-free collection name for a workspace path.

    Chroma collection names must match ``[a-zA-Z0-9_-]+`` and start/end with
    an alphanumeric, length 3-512. We hash the absolute path so two
    workspaces with the same basename don't share an index, and prefix
    ``ws_`` so the listing UI groups them cleanly.
    """
    abs_path = str(Path(workspace_root).resolve())
    h = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:16]
    return f"ws_{h}"


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


@tool()
def update_workspace_index(path: str) -> str:
    """Re-embed a single file's chunks into the active workspace's RAG index.

    Use this after editing a file (write_file / edit_file) so subsequent
    search_workspace calls see the new content. Pass either a relative
    path (resolved against the workspace) or an absolute path that
    actually lives inside the workspace. Returns a one-line status
    saying how many chunks were upserted.

    The workspace must already have an index — call POST
    /api/rag/index_workspace once first if not.
    """
    ws = current_workspace()
    if ws is None:
        return "(no active workspace; open a folder first)"
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.is_absolute():
        p = ws.root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"[error] cannot resolve {path!r}: {e}"
    # Require the file to be inside the workspace — keep RAG scoped.
    try:
        p.relative_to(ws.root)
    except ValueError:
        return f"[error] {p} is outside the active workspace"

    collection = workspace_collection_name(str(ws.root))
    coll_dir = _rag_root() / collection
    if not coll_dir.exists():
        return (
            "(workspace not indexed yet; POST /api/rag/index_workspace "
            "first to seed the collection)"
        )

    from godbot.index import update_file

    if not p.is_file():
        return f"[error] not a regular file: {p}"
    try:
        n = update_file(p, collection=collection, workspace_root=ws.root)
    except Exception as e:
        return f"(update_workspace_index failed: {type(e).__name__}: {e})"
    rel = p.relative_to(ws.root)
    return f"ok: re-embedded {n} chunks for {rel} into {collection}"


@tool()
def search_workspace(query: str, top_k: int = 5) -> str:
    """Semantic search over the active workspace's indexed files.

    Resolves the current workspace from the session contextvar, derives its
    collection name via SHA-1 of the absolute path (so two workspaces with
    the same basename never collide), and runs the same query path as
    `search_knowledge`. Returns a hint to run /api/rag/index_workspace if
    the collection isn't built yet.
    """
    ws = current_workspace()
    if ws is None:
        return "(no active workspace; open a folder before searching)"
    collection = workspace_collection_name(str(ws.root))
    root = _rag_root()
    coll_dir = root / collection
    if not coll_dir.exists():
        return (
            f"(workspace not indexed yet; POST /api/rag/index_workspace "
            f"with workspace={ws.root!s} or run `godbot-index {ws.root!s} "
            f"--collection {collection}`)"
        )
    from godbot.core.rag import Embedder, RagStore

    try:
        e = Embedder.create(prefer="lmstudio")
        qv = e.embed([query])[0]
        store = RagStore(root=root, collection=collection)
        hits = store.search(qv, top_k=top_k)
    except Exception as exc:
        return f"(search_workspace failed: {exc})"
    if not hits:
        return "(no matches)"
    return "\n\n".join(
        f"[{h['path']}:{h['lines']}] score={h['score']:.3f}\n{h['content']}" for h in hits
    )
