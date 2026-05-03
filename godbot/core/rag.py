from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable
import httpx

log = logging.getLogger("godbot.rag")


class Embedder:
    def __init__(self, kind: str, model_id: str = "", base_url: str = "") -> None:
        self.kind = kind
        self.model_id = model_id
        self.base_url = base_url.rstrip("/") if base_url else ""
        self._st = None  # sentence-transformers lazy

    @classmethod
    def create(cls, prefer: str, base_url: str = "http://localhost:1234/v1") -> "Embedder":
        if prefer == "lmstudio":
            try:
                with httpx.Client(timeout=10.0) as c:
                    r = c.get(f"{base_url.rstrip('/')}/models")
                    r.raise_for_status()
                    for m in r.json().get("data", []):
                        if "embed" in m.get("id", "").lower():
                            # Probe.
                            probe = c.post(
                                f"{base_url.rstrip('/')}/embeddings",
                                json={"model": m["id"], "input": "ping"},
                            )
                            if probe.status_code == 200:
                                return cls("lmstudio", m["id"], base_url)
            except Exception as e:
                log.warning(
                    "LM Studio embed probe failed: %s; falling back to sentence-transformers",
                    e,
                )
        return cls("sentence-transformers")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.kind == "lmstudio":
            with httpx.Client(timeout=60.0) as c:
                r = c.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model_id, "input": texts},
                )
                r.raise_for_status()
                return [d["embedding"] for d in r.json()["data"]]
        # sentence-transformers
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._st.encode(texts, normalize_embeddings=True).tolist()


def chunk_text(text: str, window: int = 40, overlap: int = 5) -> list[dict]:
    """Line-window chunking. Returns chunks with content + start/end line numbers."""
    lines = text.splitlines()
    chunks: list[dict] = []
    if not lines:
        return chunks
    step = max(1, window - overlap)
    i = 0
    while i < len(lines):
        end = min(len(lines), i + window)
        chunks.append(
            {
                "content": "\n".join(lines[i:end]),
                "start_line": i + 1,
                "end_line": end,
            }
        )
        if end == len(lines):
            break
        i += step
    return chunks


def chunk_markdown(text: str) -> list[dict]:
    """Heading-based markdown chunking."""
    chunks: list[dict] = []
    cur_heading = ""
    cur_buf: list[str] = []
    cur_start = 1
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("#"):
            if cur_buf:
                chunks.append(
                    {
                        "content": "\n".join(cur_buf),
                        "heading": cur_heading,
                        "start_line": cur_start,
                        "end_line": lineno - 1,
                    }
                )
                cur_buf = []
            cur_heading = line.lstrip("# ").strip()
            cur_start = lineno
            cur_buf.append(line)
        else:
            cur_buf.append(line)
    if cur_buf:
        chunks.append(
            {
                "content": "\n".join(cur_buf),
                "heading": cur_heading,
                "start_line": cur_start,
                "end_line": cur_start + len(cur_buf) - 1,
            }
        )
    return chunks


class RagStore:
    def __init__(self, root: Path, collection: str, embed_dim: int = 384) -> None:
        import chromadb

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.root / collection))
        self._coll = self.client.get_or_create_collection(name=collection)
        self.embed_dim = embed_dim

    def add(self, ids, embeddings, documents, metadatas) -> None:
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(metadatas, dict):
            metadatas = [metadatas]
        self._coll.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def search(self, embedding, top_k: int = 5) -> list[dict]:
        res = self._coll.query(query_embeddings=[embedding], n_results=top_k)
        out = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        scores = res.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "path": metas[i].get("path", ""),
                    "lines": metas[i].get("lines", ""),
                    "score": float(scores[i]) if i < len(scores) else 0.0,
                    "content": docs[i],
                }
            )
        return out

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]
