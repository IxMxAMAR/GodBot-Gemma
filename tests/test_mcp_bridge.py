"""Bridge tests: MCP tool descriptors → ToolSpec entries in DEFAULT registry.

These tests deliberately avoid spawning a subprocess. They use a tiny
``_FakeClient`` that mimics :class:`godbot.mcp.client.MCPClient` well enough
for the bridge: ``cfg.name``, ``connected``, ``connect_error``, ``tools``,
``call_tool``. That keeps these tests fast and CI-portable on Windows where
spawning ``npx`` for a real MCP server is brittle.

Each test cleans up its own registry entries so tests stay independent — the
bridge writes into ``DEFAULT._tools`` directly, so we pop those keys back out.
"""

from __future__ import annotations
import pytest

import godbot.tools  # noqa: F401 — triggers @tool discovery so mcp_status exists
from godbot.core.registry import DEFAULT
from godbot.mcp.client import MCPServerConfig
from godbot.mcp.registry_bridge import (
    register_mcp_server,
    shutdown_all,
    _clients,
)


class _FakeClient:
    """Stand-in for MCPClient — no subprocess, no asyncio."""

    def __init__(self, name: str, tools: list[dict], connected: bool = True):
        self.cfg = MCPServerConfig(name=name, command="x")
        self.tools = tools
        self.connected = connected
        self.connect_error = None if connected else "fake disconnect"
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def call_tool(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return f"called {name}({args})"

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _clean_bridge_state():
    """Snapshot registry + bridge state and restore after each test."""
    pre_tools = dict(DEFAULT._tools)
    pre_clients = dict(_clients)
    yield
    # Wipe any tools the test added.
    for k in list(DEFAULT._tools.keys()):
        if k not in pre_tools:
            DEFAULT._tools.pop(k, None)
    # Restore clients map.
    _clients.clear()
    _clients.update(pre_clients)


def test_register_namespaces_tools_and_marks_dangerous():
    fc = _FakeClient("fs", [
        {"name": "read", "description": "read a file",
         "inputSchema": {"type": "object",
                         "properties": {"path": {"type": "string"}}}},
    ])
    n = register_mcp_server(fc)
    assert n == 1

    spec = DEFAULT.spec("mcp_fs_read")
    assert spec is not None
    assert spec.dangerous is True
    assert spec.timeout == 120
    assert spec.description == "read a file"
    assert spec.schema["type"] == "object"
    # Bridge forces additionalProperties=True so MCP servers can evolve.
    assert spec.schema.get("additionalProperties") is True


def test_register_dispatches_to_underlying_client():
    fc = _FakeClient("github", [
        {"name": "list_repos", "description": "list user repos",
         "inputSchema": {"type": "object", "properties": {}}},
    ])
    register_mcp_server(fc)

    out = DEFAULT.execute("mcp_github_list_repos", {})
    assert "list_repos" in out
    assert fc.calls == [("list_repos", {})]


def test_register_passes_kwargs_through():
    fc = _FakeClient("brave", [
        {"name": "search",
         "description": "web search",
         "inputSchema": {"type": "object",
                         "properties": {"query": {"type": "string"}}}},
    ])
    register_mcp_server(fc)
    out = DEFAULT.execute("mcp_brave_search", {"query": "godbot"})
    assert fc.calls == [("search", {"query": "godbot"})]
    assert "search" in out


def test_disconnected_client_registers_nothing():
    fc = _FakeClient("oops", [], connected=False)
    n = register_mcp_server(fc)
    assert n == 0
    # And no tools leaked into the registry under that prefix.
    assert not any(k.startswith("mcp_oops_") for k in DEFAULT.names())


def test_empty_description_falls_back_to_friendly_default():
    fc = _FakeClient("svc", [
        {"name": "noop", "description": "",
         "inputSchema": {"type": "object", "properties": {}}},
    ])
    register_mcp_server(fc)
    spec = DEFAULT.spec("mcp_svc_noop")
    assert spec is not None
    assert spec.description  # non-empty
    assert "noop" in spec.description


def test_missing_input_schema_normalized_to_object():
    fc = _FakeClient("legacy", [
        {"name": "blank", "description": "no schema"},  # no inputSchema key
    ])
    register_mcp_server(fc)
    spec = DEFAULT.spec("mcp_legacy_blank")
    assert spec is not None
    assert spec.schema["type"] == "object"
    assert spec.schema["properties"] == {}


def test_non_dict_input_schema_normalized():
    fc = _FakeClient("weird", [
        {"name": "x", "description": "weird",
         "inputSchema": "not-a-dict"},
    ])
    register_mcp_server(fc)
    spec = DEFAULT.spec("mcp_weird_x")
    assert spec is not None
    assert spec.schema["type"] == "object"


def test_re_registration_replaces_prior_tools():
    fc1 = _FakeClient("svc", [
        {"name": "old", "description": "old tool",
         "inputSchema": {"type": "object", "properties": {}}},
    ])
    register_mcp_server(fc1)
    assert "mcp_svc_old" in DEFAULT.names()

    fc2 = _FakeClient("svc", [
        {"name": "new", "description": "new tool",
         "inputSchema": {"type": "object", "properties": {}}},
    ])
    register_mcp_server(fc2)
    assert "mcp_svc_new" in DEFAULT.names()
    # Old tool is gone — re-boot of the same server replaces its tools.
    assert "mcp_svc_old" not in DEFAULT.names()


def test_two_servers_with_same_tool_name_dont_collide():
    fc_a = _FakeClient("alpha", [
        {"name": "ping", "description": "ping a",
         "inputSchema": {"type": "object", "properties": {}}},
    ])
    fc_b = _FakeClient("beta", [
        {"name": "ping", "description": "ping b",
         "inputSchema": {"type": "object", "properties": {}}},
    ])
    register_mcp_server(fc_a)
    register_mcp_server(fc_b)
    assert "mcp_alpha_ping" in DEFAULT.names()
    assert "mcp_beta_ping" in DEFAULT.names()

    DEFAULT.execute("mcp_alpha_ping", {})
    DEFAULT.execute("mcp_beta_ping", {})
    assert fc_a.calls == [("ping", {})]
    assert fc_b.calls == [("ping", {})]


def test_validate_args_does_not_choke_on_permissive_schema():
    """Bridge sets additionalProperties=True — extra args must NOT error."""
    fc = _FakeClient("svc", [
        {"name": "echo",
         "description": "echo",
         "inputSchema": {"type": "object",
                         "properties": {"text": {"type": "string"}}}},
    ])
    register_mcp_server(fc)
    err = DEFAULT.validate_args("mcp_svc_echo", {"text": "hi", "extra": 1})
    assert err is None  # permissive: extra fields are accepted


def test_shutdown_all_closes_every_registered_client():
    fc1 = _FakeClient("a", [{"name": "t",
                             "inputSchema": {"type": "object", "properties": {}}}])
    fc2 = _FakeClient("b", [{"name": "t",
                             "inputSchema": {"type": "object", "properties": {}}}])
    register_mcp_server(fc1)
    register_mcp_server(fc2)
    shutdown_all()
    assert fc1.closed is True
    assert fc2.closed is True
    assert _clients == {}


def test_mcp_status_with_no_servers_does_not_raise(tmp_path, monkeypatch):
    """mcp_status must always return a friendly string, never raise."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    out = DEFAULT.execute("mcp_status", {})
    assert isinstance(out, str)
    assert "no MCP servers configured" in out


def test_mcp_status_lists_connected_server(tmp_path, monkeypatch):
    """When a server is registered, status reports its tool list."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[mcp.servers.fakefs]\n'
        'command = "x"\n'
        'args = []\n',
        encoding="utf-8",
    )
    fc = _FakeClient("fakefs", [
        {"name": "read", "description": "read",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "write", "description": "write",
         "inputSchema": {"type": "object", "properties": {}}},
    ])
    register_mcp_server(fc)
    out = DEFAULT.execute("mcp_status", {})
    assert "fakefs" in out
    assert "connected" in out
    assert "read" in out and "write" in out


def test_mcp_status_shows_failed_connect(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[mcp.servers.broken]\n'
        'command = "noexist"\n',
        encoding="utf-8",
    )
    # Simulate a registered-but-failed client by injecting one with
    # connected=False directly into _clients.
    fc = _FakeClient("broken", [], connected=False)
    _clients["broken"] = fc
    out = DEFAULT.execute("mcp_status", {})
    assert "broken" in out
    assert "failed" in out.lower() or "fake disconnect" in out
