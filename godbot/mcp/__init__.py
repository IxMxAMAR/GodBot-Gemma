"""MCP (Model Context Protocol) client integration for GodBot.

This subpackage lets GodBot act as an *MCP client*, connecting to third-party
MCP servers (filesystem, github, postgres, brave-search, puppeteer, ...) over
stdio and surfacing their tools through GodBot's existing ``DEFAULT`` tool
registry under namespaced names ``mcp_<server>_<tool>``.

Public entry points:

* :func:`boot_mcp` — read the user's TOML config, spawn each configured
  server, and register its advertised tools. Best-effort: failures are logged
  and skipped, never raised. Safe to call at module-import time.
* :func:`shutdown_mcp` — close every connected MCP client cleanly. Safe to
  call from shutdown hooks; never raises.

The actual machinery lives in :mod:`godbot.mcp.client` (subprocess + asyncio
loop) and :mod:`godbot.mcp.registry_bridge` (JSON-Schema → ToolSpec).
"""

from __future__ import annotations
import logging

log = logging.getLogger("godbot.mcp")


def boot_mcp() -> int:
    """Spawn every MCP server in the user's config and register its tools.

    Returns the count of MCP tools registered into ``DEFAULT``. Always
    succeeds — an unreachable server, missing dep, or malformed config is
    logged and skipped, never raised. This keeps the daemon's startup robust
    against whatever the user typed into ``~/.godbot/config.toml``.
    """
    try:
        from godbot.config import load_config
        from godbot.mcp.client import MCPClient, MCPServerConfig
        from godbot.mcp.registry_bridge import register_mcp_server
    except Exception:
        log.exception("MCP imports failed; skipping MCP boot")
        return 0

    try:
        cfg = load_config()
    except Exception:
        log.exception("load_config() failed; skipping MCP boot")
        return 0

    total = 0
    for name, server_cfg in cfg.mcp.servers.items():
        if not server_cfg.command:
            log.warning("MCP server %r has empty command; skipping", name)
            continue
        try:
            client = MCPClient(MCPServerConfig(
                name=name,
                command=server_cfg.command,
                args=list(server_cfg.args),
                env=dict(server_cfg.env),
            ))
            client.connect()
            if client.connected:
                total += register_mcp_server(client)
            else:
                log.warning(
                    "MCP server %r failed to connect: %s",
                    name, client.connect_error,
                )
        except Exception:
            log.exception("MCP server %r booted with exception", name)
    return total


def shutdown_mcp() -> None:
    """Close every connected MCP client. Never raises."""
    try:
        from godbot.mcp.registry_bridge import shutdown_all
        shutdown_all()
    except Exception:
        log.exception("MCP shutdown raised; ignoring")
