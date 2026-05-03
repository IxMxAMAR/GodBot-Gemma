import os
from pathlib import Path
import httpx
import pytest
import respx
from godbot.index import build_index, walk_files


def _mock_lmstudio_embedder():
    """Helper: mock LM Studio /models + /embeddings so Embedder.create returns lmstudio kind."""
    import json as _json

    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "nomic-embed-text"}]}
        )
    )

    def _embed_response(req):
        body = _json.loads(req.content or b"{}")
        inp = body.get("input", "")
        if isinstance(inp, str):
            inp = [inp]
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]} for _ in inp]},
        )

    respx.post("http://localhost:1234/v1/embeddings").mock(side_effect=_embed_response)


def test_walk_skips_gitignored(tmp_path, monkeypatch):
    (tmp_path / ".gitignore").write_text("ignored/\n*.bin\n")
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "z.py").write_text("y")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")
    files = list(walk_files(tmp_path))
    rel = {str(Path(f).relative_to(tmp_path)) for f in files}
    assert "a.py" in rel
    assert "ignored\\z.py" not in rel and "ignored/z.py" not in rel
    assert "data.bin" not in rel


@respx.mock
def test_build_index_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    _mock_lmstudio_embedder()
    (tmp_path / "x.py").write_text("def hello():\n    return 'hi'\n")
    n = build_index(tmp_path, collection="tt1", reindex=True)
    assert n >= 1


@respx.mock
def test_force_clears_stale_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    _mock_lmstudio_embedder()
    coll_dir = tmp_path / ".godbot" / "rag" / "tt2"
    coll_dir.mkdir(parents=True)
    (coll_dir / ".lock").write_text("99999")  # stale pid
    (tmp_path / "x.py").write_text("def hi(): pass\n")
    # Without --force this raises:
    with pytest.raises(RuntimeError, match="locked"):
        build_index(tmp_path, collection="tt2")
    # With --force it proceeds:
    n = build_index(tmp_path, collection="tt2", force=True)
    assert n >= 0
