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


# --- list_tools ---

@respx.mock
@pytest.mark.asyncio
async def test_list_tools_in_daemon_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    fake_tools = [{"name": "echo", "description": "echo", "dangerous": False}]
    respx.get("http://127.0.0.1:7878/api/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.post("http://127.0.0.1:7878/api/sessions/new").mock(
        return_value=httpx.Response(200, json={"session_id": "s1"})
    )
    respx.get("http://127.0.0.1:7878/api/tools").mock(
        return_value=httpx.Response(200, json=fake_tools)
    )
    async with Session() as s:
        out = await s.list_tools()
    assert out == fake_tools


@pytest.mark.asyncio
async def test_list_tools_in_embedded_mode(tmp_path, monkeypatch):
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

    llm = MockLLM([])

    def factory():
        return EmbeddedRunner.create(
            sessions_root=tmp_path / "sessions", registry=reg, llm=llm,
        )

    async with Session(base_url="http://127.0.0.1:9999", embedded_factory=factory) as s:
        out = await s.list_tools()
    # Embedded reads from DEFAULT registry, not the passed one — so just
    # assert the shape: list of dicts with the expected keys.
    assert isinstance(out, list)
    for entry in out:
        assert set(entry.keys()) >= {"name", "description", "dangerous"}


# --- list_sessions ---

@respx.mock
@pytest.mark.asyncio
async def test_list_sessions_in_daemon_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    fake_sessions = [{"id": "s1", "model": "auto"}, {"id": "s2", "model": "auto"}]
    respx.get("http://127.0.0.1:7878/api/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.post("http://127.0.0.1:7878/api/sessions/new").mock(
        return_value=httpx.Response(200, json={"session_id": "s1"})
    )
    respx.get("http://127.0.0.1:7878/api/sessions").mock(
        return_value=httpx.Response(200, json=fake_sessions)
    )
    async with Session() as s:
        out = await s.list_sessions()
    assert out == fake_sessions


@pytest.mark.asyncio
async def test_list_sessions_in_embedded_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    from godbot.client.embedded import EmbeddedRunner
    from godbot.core.registry import Registry
    from tests._mock_llm import MockLLM

    reg = Registry()
    llm = MockLLM([])
    sessions_root = tmp_path / "sessions"

    def factory():
        return EmbeddedRunner.create(
            sessions_root=sessions_root, registry=reg, llm=llm,
        )

    async with Session(base_url="http://127.0.0.1:9999", embedded_factory=factory) as s:
        out = await s.list_sessions()
    # Should contain at least the active session's id.
    ids = {entry["id"] for entry in out}
    assert ids  # non-empty
    for entry in out:
        assert "id" in entry and "model" in entry


# --- list_rag_collections ---

@respx.mock
@pytest.mark.asyncio
async def test_list_rag_collections_in_daemon_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    respx.get("http://127.0.0.1:7878/api/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.post("http://127.0.0.1:7878/api/sessions/new").mock(
        return_value=httpx.Response(200, json={"session_id": "s1"})
    )
    respx.get("http://127.0.0.1:7878/api/rag/collections").mock(
        return_value=httpx.Response(200, json=["godbot", "notes"])
    )
    async with Session() as s:
        out = await s.list_rag_collections()
    assert out == ["godbot", "notes"]


@pytest.mark.asyncio
async def test_list_rag_collections_in_embedded_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    # Create some RAG collection dirs.
    rag_root = tmp_path / ".godbot" / "rag"
    (rag_root / "alpha").mkdir(parents=True)
    (rag_root / "beta").mkdir(parents=True)
    # A stray file should be ignored.
    (rag_root / "stray.txt").write_text("ignore me")

    from godbot.client.embedded import EmbeddedRunner
    from godbot.core.registry import Registry
    from tests._mock_llm import MockLLM

    reg = Registry()
    llm = MockLLM([])

    def factory():
        return EmbeddedRunner.create(
            sessions_root=tmp_path / "sessions", registry=reg, llm=llm,
        )

    async with Session(base_url="http://127.0.0.1:9999", embedded_factory=factory) as s:
        out = await s.list_rag_collections()
    assert out == ["alpha", "beta"]
