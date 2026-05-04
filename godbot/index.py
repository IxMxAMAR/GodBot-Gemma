from __future__ import annotations
import argparse
import fnmatch
import mimetypes
import os
import sys
from pathlib import Path
from typing import Iterator

DEFAULT_SKIPS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
}


def _read_gitignore(root: Path) -> list[str]:
    p = root / ".gitignore"
    if not p.exists():
        return []
    return [
        line.strip().rstrip("/")
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _is_text(p: Path) -> bool:
    mt, _ = mimetypes.guess_type(str(p))
    if mt and mt.startswith("text/"):
        return True
    if p.suffix in {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".md",
        ".rst",
        ".toml",
        ".yml",
        ".yaml",
        ".json",
        ".html",
        ".css",
    }:
        return True
    return False


def walk_files(root: Path) -> Iterator[Path]:
    patterns = _read_gitignore(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        rel_dir = dp.relative_to(root)
        # prune dirs in-place
        dirnames[:] = [
            d
            for d in dirnames
            if d not in DEFAULT_SKIPS
            and not any(
                fnmatch.fnmatch(str(rel_dir / d), pat) or fnmatch.fnmatch(d, pat)
                for pat in patterns
            )
        ]
        for f in filenames:
            p = dp / f
            rel = p.relative_to(root)
            if any(
                fnmatch.fnmatch(str(rel), pat) or fnmatch.fnmatch(f, pat)
                for pat in patterns
            ):
                continue
            if not _is_text(p):
                continue
            yield p


def _lock_path(coll_dir: Path) -> Path:
    coll_dir.mkdir(parents=True, exist_ok=True)
    return coll_dir / ".lock"


def _chunks_for(path: Path):
    """Return the chunk list for ``path`` using extension-aware chunking."""
    from godbot.core.rag import chunk_text, chunk_markdown

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return chunk_markdown(text) if path.suffix == ".md" else chunk_text(text)


def _embed_and_upsert(path: Path, root: Path, store, emb) -> int:
    """Embed every chunk of ``path`` and upsert into ``store``.

    Used by both the full ``build_index`` walk and the incremental
    ``update_file`` path. Path ids are scoped by absolute path + line
    range so re-embedding the same file overwrites the prior entries
    cleanly via chroma's upsert semantics.
    """
    chunks = _chunks_for(path)
    if not chunks:
        return 0
    ids = [f"{path}:{c['start_line']}-{c['end_line']}" for c in chunks]
    docs = [c["content"] for c in chunks]
    metas = [
        {
            "path": str(path.relative_to(root)) if root in path.parents or path == root else str(path),
            "lines": f"{c['start_line']}-{c['end_line']}",
        }
        for c in chunks
    ]
    embs = emb.embed(docs)
    store.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    return len(chunks)


def update_file(
    file_path: Path,
    *,
    collection: str,
    workspace_root: Path,
) -> int:
    """Re-embed a single file's chunks into ``collection``.

    Used by the ``update_workspace_index`` tool — lets the agent keep
    its own search index fresh after editing without paying for a full
    workspace rewalk. Returns the chunk count written.
    """
    from godbot.core.rag import Embedder, RagStore

    home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    emb = Embedder.create(prefer="lmstudio")
    store = RagStore(root=home / "rag", collection=collection)
    return _embed_and_upsert(file_path, workspace_root, store, emb)


def build_index(
    root: Path, collection: str, reindex: bool = False, force: bool = False
) -> int:
    from godbot.core.rag import Embedder, RagStore

    home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
    coll_dir = home / "rag" / collection
    lock = _lock_path(coll_dir)
    if lock.exists() and not force:
        raise RuntimeError(f"collection {collection!r} is locked: {lock}")
    lock.write_text(str(os.getpid()))
    try:
        emb = Embedder.create(prefer="lmstudio")
        store = RagStore(root=home / "rag", collection=collection)
        n = 0
        for p in walk_files(root):
            n += _embed_and_upsert(p, root, store, emb)
        return n
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="directory to index")
    ap.add_argument("--collection")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore stale .lock file from a previous crashed run",
    )
    args = ap.parse_args()
    root = Path(args.path).resolve()
    coll = args.collection or root.name
    n = build_index(root, collection=coll, reindex=args.reindex, force=args.force)
    print(f"indexed {n} chunks into collection {coll!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
