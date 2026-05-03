from godbot.core.rag import RagStore


def test_store_add_and_search(tmp_path):
    store = RagStore(root=tmp_path, collection="tt1", embed_dim=4)
    store.add(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        documents=["alpha", "beta"],
        metadatas=[{"path": "a.py", "lines": "1-5"}, {"path": "b.py", "lines": "1-5"}],
    )
    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert hits[0]["path"] == "a.py"
    assert "alpha" in hits[0]["content"]
