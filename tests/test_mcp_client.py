"""End-to-end MCP client tests against a tiny in-process Python stub server.

These tests spawn a real subprocess running an MCP stdio server, so they're
the closest we can get to "the bridge actually talks to a server" without
depending on `npx` / Node being on PATH. They're skipped if the `mcp` SDK
isn't importable (covered by ``pytest.importorskip("mcp")``).

The stub server is generated as a one-off script in ``tmp_path`` so each
test gets a clean process; we never try to share a server across tests
because the SDK's ClientSession owns a long-lived stream.
"""

from __future__ import annotations
import sys
import textwrap
import pytest


# Skip the whole module if the SDK isn't importable.
mcp = pytest.importorskip("mcp")


from godbot.mcp.client import MCPClient, MCPServerConfig


STUB_SERVER_SRC = textwrap.dedent('''
    """Tiny MCP stdio server with one echo tool."""
    import asyncio
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp import types

    server = Server("stub")

    @server.list_tools()
    async def _list_tools():
        return [
            types.Tool(
                name="echo",
                description="Echo the input back.",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ]

    @server.call_tool()
    async def _call_tool(name, args):
        if name == "echo":
            return [types.TextContent(type="text",
                                      text=f"echoed: {args.get('text', '')}")]
        return [types.TextContent(type="text", text="unknown tool")]

    async def main():
        async with stdio_server() as (read, write):
            await server.run(
                read,
                write,
                InitializationOptions(
                    server_name="stub",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        NotificationOptions(), {}
                    ),
                ),
            )

    asyncio.run(main())
''').lstrip()


@pytest.fixture
def stub_server_cfg(tmp_path):
    server_script = tmp_path / "stub_server.py"
    server_script.write_text(STUB_SERVER_SRC, encoding="utf-8")
    return MCPServerConfig(
        name="stub",
        command=sys.executable,
        args=[str(server_script)],
    )


def test_mcp_client_connects_and_lists_tools(stub_server_cfg):
    client = MCPClient(stub_server_cfg)
    client.connect()
    try:
        assert client.connected, f"connect failed: {client.connect_error}"
        names = [t["name"] for t in client.tools]
        assert "echo" in names
        echo = next(t for t in client.tools if t["name"] == "echo")
        assert echo["description"] == "Echo the input back."
        assert echo["inputSchema"]["type"] == "object"
    finally:
        client.close()


def test_mcp_client_call_tool_echo(stub_server_cfg):
    client = MCPClient(stub_server_cfg)
    client.connect()
    try:
        assert client.connected
        result = client.call_tool("echo", {"text": "hi"})
        assert "echoed: hi" in result
    finally:
        client.close()


def test_mcp_client_failed_connect_marks_state(tmp_path):
    """Bad command must fail cleanly, not raise."""
    cfg = MCPServerConfig(
        name="nope",
        command=str(tmp_path / "definitely_does_not_exist.exe"),
    )
    client = MCPClient(cfg)
    client.connect(timeout=5.0)
    assert client.connected is False
    assert client.connect_error  # populated
    # call_tool on a disconnected client must not raise.
    out = client.call_tool("anything", {})
    assert "not connected" in out
    client.close()  # idempotent


def test_mcp_client_bridge_round_trip(stub_server_cfg):
    """Full path: live MCP server → bridge → DEFAULT.execute."""
    import godbot.tools  # noqa: F401 — discovery
    from godbot.core.registry import DEFAULT
    from godbot.mcp.registry_bridge import register_mcp_server, _clients

    client = MCPClient(stub_server_cfg)
    client.connect()
    try:
        assert client.connected
        n = register_mcp_server(client)
        assert n == 1
        out = DEFAULT.execute("mcp_stub_echo", {"text": "world"})
        assert "echoed: world" in out
    finally:
        # Clean up registry + client map.
        DEFAULT._tools.pop("mcp_stub_echo", None)
        _clients.pop("stub", None)
        client.close()
