import respx
import httpx
import pytest
from godbot.core.llm import LLMClient


@respx.mock
def test_pick_first_gemma_model():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [
            {"id": "nomic-embed", "loaded_context_length": 2048},
            {"id": "gemma-3n-e4b-it", "loaded_context_length": 32768},
            {"id": "qwen-1.5b", "loaded_context_length": 4096},
        ]})
    )
    client = LLMClient(base_url="http://localhost:1234/v1", model="auto")
    info = client.probe()
    assert info.id == "gemma-3n-e4b-it"
    assert info.context_length == 32768


@respx.mock
def test_pinned_model_used():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [
            {"id": "qwen", "loaded_context_length": 4096},
            {"id": "gemma-3n", "loaded_context_length": 32768},
        ]})
    )
    client = LLMClient(base_url="http://localhost:1234/v1", model="qwen")
    info = client.probe()
    assert info.id == "qwen"
    assert info.context_length == 4096


@respx.mock
def test_no_match_raises():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "qwen-7b"}]})
    )
    client = LLMClient(base_url="http://localhost:1234/v1", model="auto")
    with pytest.raises(RuntimeError, match="no.*gemma"):
        client.probe()
