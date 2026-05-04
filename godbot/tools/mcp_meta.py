"""Meta-tool: introspect the MCP subsystem from inside the agent.

This tool *never raises*. It always returns a friendly string — even when no
MCP servers are configured, when the config is malformed, or when the MCP
package isn't importable. The agent (and the user) need a stable surface for
"what external capabilities do I have right now?" without crash-coupling to
optional infrastructure.
"""

from __future__ import annotations
from godbot.core.registry import tool


@tool()
def mcp_status() -> str:
    """List configured MCP servers, connection state, and advertised tools.

    Useful when the agent (or the user) wants to know what external tool
    surface is available right now. Returns a friendly summary string; never
    raises, even if the MCP subsystem is broken or no servers are configured.
    """
    # All imports + lookups inside the function body so any failure mode
    # (missing config, missing dep, partial init) collapses to a string.
    try:
        from godbot.config import load_config
    except Exception as e:
        return f"(mcp_status: failed to load config — {type(e).__name__}: {e})"

    try:
        cfg = load_config()
    except Exception as e:
        return f"(mcp_status: load_config raised — {type(e).__name__}: {e})"

    if not cfg.mcp.servers:
        return (
            "(no MCP servers configured; add a "
            "[mcp.servers.<name>] section to ~/.godbot/config.toml)"
        )

    try:
        from godbot.mcp.registry_bridge import _clients
    except Exception:
        _clients = {}  # type: ignore[assignment]

    lines: list[str] = []
    for name, scfg in cfg.mcp.servers.items():
        client = _clients.get(name)
        if client is None:
            lines.append(
                f"{name}: not booted (cmd: {scfg.command or '(empty)'})"
            )
            continue
        if not client.connected:
            err = client.connect_error or "unknown error"
            lines.append(f"{name}: failed to connect — {err}")
            continue
        tool_names = [t["name"] for t in client.tools]
        if tool_names:
            lines.append(
                f"{name}: connected — {len(tool_names)} tools: "
                + ", ".join(tool_names)
            )
        else:
            lines.append(f"{name}: connected — no tools advertised")
    return "\n".join(lines)
