import json
import pytest
import respx
import httpx
from godbot.client import Session


@respx.mock
@pytest.mark.asyncio
async def test_toggle_tool_in_daemon_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    respx.get("http://127.0.0.1:7878/api/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.post("http://127.0.0.1:7878/api/sessions/new").mock(
        return_value=httpx.Response(200, json={"session_id": "s1"})
    )
    respx.post("http://127.0.0.1:7878/api/tools/toggle").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    async with Session() as s:
        await s.toggle_tool("run_powershell", False)


@respx.mock
@pytest.mark.asyncio
async def test_use_rag_collection_in_daemon_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    respx.get("http://127.0.0.1:7878/api/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.post("http://127.0.0.1:7878/api/sessions/new").mock(
        return_value=httpx.Response(200, json={"session_id": "s1"})
    )
    respx.post("http://127.0.0.1:7878/api/rag/use").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    async with Session() as s:
        await s.use_rag_collection("godbot")


@pytest.mark.asyncio
async def test_toggle_tool_in_embedded_mode_is_noop(tmp_path, monkeypatch):
    """In embedded mode, tool overrides update the local session metadata."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    from godbot.client.embedded import EmbeddedRunner
    from godbot.core.registry import Registry
    from tests._mock_llm import MockLLM

    reg = Registry()

    @reg.tool()
    def echo(text: str) -> str:
        """."""
        return text

    llm = MockLLM([])  # never called in this test

    def factory():
        return EmbeddedRunner.create(
            sessions_root=tmp_path / "sessions", registry=reg, llm=llm,
        )

    async with Session(base_url="http://127.0.0.1:9999", embedded_factory=factory) as s:
        await s.toggle_tool("echo", False)
        # Verify the underlying core_session has the override.
        assert "echo" not in (s._runner._core_session.tool_overrides or [])
