"""Tests for sub-project 83 — mcp_describe individual server detail."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.mcp_meta import mcp_describe


def test_mcp_describe_empty_server_name_errors():
    out = mcp_describe(server="")
    assert out.startswith("[error]")
    assert "server name required" in out


def test_mcp_describe_unknown_server(monkeypatch, tmp_path):
    """A server name that's not in the config returns a clear error."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    out = mcp_describe(server="totally-not-configured-server")
    assert out.startswith("[error]")
    assert "no MCP server" in out


def test_mcp_describe_not_booted(monkeypatch, tmp_path):
    """Configured but not booted server reports status: not-booted."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        "[mcp.servers.demo]\n"
        'command = "true"\n',
        encoding="utf-8",
    )
    # Don't boot it. Expected status: not-booted.
    out = mcp_describe(server="demo")
    assert "server: demo" in out
    assert "not-booted" in out


def test_mcp_describe_with_fake_connected_client(monkeypatch, tmp_path):
    """Inject a fake client into the registry_bridge cache."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        "[mcp.servers.fake]\n"
        'command = "true"\n',
        encoding="utf-8",
    )

    class _FakeCfg:
        name = "fake"
        command = "true"

    class _FakeClient:
        cfg = _FakeCfg()
        connected = True
        connect_error = None
        tools = [
            {"name": "echo", "description": "Echo input back."},
            {"name": "ping", "description": "Ping the server."},
        ]

    from godbot.mcp.registry_bridge import _clients
    _clients["fake"] = _FakeClient()  # type: ignore[assignment]
    try:
        out = mcp_describe(server="fake")
        assert "server: fake" in out
        assert "status: connected" in out
        assert "tools (2):" in out
        assert "- echo: Echo input back." in out
        assert "- ping: Ping the server." in out
    finally:
        _clients.pop("fake", None)


def test_mcp_describe_connection_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        "[mcp.servers.broken]\n"
        'command = "true"\n',
        encoding="utf-8",
    )

    class _FakeBroken:
        connected = False
        connect_error = "command not found"
        tools = []

    from godbot.mcp.registry_bridge import _clients
    _clients["broken"] = _FakeBroken()  # type: ignore[assignment]
    try:
        out = mcp_describe(server="broken")
        assert "status: disconnected" in out
        assert "command not found" in out
    finally:
        _clients.pop("broken", None)


def test_mcp_describe_is_registered_non_dangerous():
    spec = DEFAULT.spec("mcp_describe")
    assert spec is not None
    assert spec.dangerous is False
