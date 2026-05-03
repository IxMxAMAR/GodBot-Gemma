import os
import respx
import httpx
import pytest
from godbot.core.rag import Embedder


@respx.mock
def test_lmstudio_embedder_used_if_probe_passes():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "nomic-embed-text"}]})
    )
    respx.post("http://localhost:1234/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )
    e = Embedder.create(prefer="lmstudio", base_url="http://localhost:1234/v1")
    assert e.kind == "lmstudio"
    assert e.embed(["hi"])[0] == [0.1, 0.2, 0.3]


@pytest.mark.skipif(
    os.environ.get("RAG_BGE_DOWNLOAD") != "1",
    reason="set RAG_BGE_DOWNLOAD=1 to run; downloads BGE-small (~135MB)",
)
@respx.mock
def test_fallback_to_sentence_transformers_when_no_embed_model():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "gemma-3n-e4b-it"}]})
    )
    e = Embedder.create(prefer="lmstudio", base_url="http://localhost:1234/v1")
    assert e.kind == "sentence-transformers"
    vecs = e.embed(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) > 16
