from __future__ import annotations
import logging
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
