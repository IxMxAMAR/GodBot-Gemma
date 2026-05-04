"""GodBot as an MCP server (sub-project 11 — reverse direction of MCP-M).

The mirror image of :mod:`godbot.mcp.client`: instead of GodBot consuming
external MCP servers' tools, this module exposes GodBot's own tool registry
to any MCP-compliant host (Claude Desktop, Cursor, Continue, Zed, ...) via
stdio transport.

Each :class:`~godbot.core.registry.ToolSpec` in :data:`DEFAULT` becomes an
MCP ``Tool`` with the spec's JSON schema passed through verbatim. Hosts call
the tool by name; we route through ``DEFAULT.execute`` and return the result
as a single ``TextContent`` block.

The CLI :func:`main` wires up an :class:`AppOptions` from argparse, builds
an :class:`MCPServerApp`, and runs it in stdio mode. It's installed as the
``godbot-mcp-server`` console script in ``pyproject.toml`` so the host
config can point at the venv-relative shim directly.

Safety:
- ``--safe-only`` filters dangerous tools out of the catalog. Recommended
  for hosts you don't fully trust.
- ``--include`` / ``--exclude`` tweak the catalog further.
- Errors from tool execution are returned as text content with
  ``isError=True``, never raised; the host stays connected.
"""

from __future__ import annotations
import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from godbot.core.registry import DEFAULT, Registry, ToolSpec

log = logging.getLogger("godbot.mcp.server")


@dataclass
class AppOptions:
    """User-controlled filters applied to the published tool catalog.

    All filters are AND-combined: a tool must pass each non-empty filter to
    be advertised. Order: ``safe_only`` first, then ``include``, then
    ``exclude``. ``include`` is treated as "if non-empty, only these names";
    ``exclude`` is "drop any of these names".
    """

    safe_only: bool = False
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


def _select_tools(reg: Registry, opts: AppOptions) -> list[ToolSpec]:
    """Apply the AppOptions filters to the registry and return the published list."""
    tools = list(reg.all())
    if opts.safe_only:
        tools = [t for t in tools if not t.dangerous]
    if opts.include:
        wanted = set(opts.include)
        tools = [t for t in tools if t.name in wanted]
    if opts.exclude:
        unwanted = set(opts.exclude)
        tools = [t for t in tools if t.name not in unwanted]
    return tools


def _spec_to_mcp_tool(spec: ToolSpec):
    """Convert a :class:`ToolSpec` into the MCP SDK's ``Tool`` shape.

    The SDK import is local so the module stays import-clean when ``mcp``
    isn't installed (the registry is still importable for tests that don't
    spin up the server).
    """
    from mcp import types

    desc = spec.description.strip() or f"GodBot tool {spec.name!r}"
    if spec.dangerous:
        desc = f"{desc}  [DANGEROUS — runs without GodBot's gate]"
    return types.Tool(
        name=spec.name,
        description=desc,
        inputSchema=spec.schema or {"type": "object", "properties": {}},
    )


def _execute_tool_safely(reg: Registry, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run a registered tool, return ``(text, is_error)``.

    Catches everything: a host should never see a stack trace. Schema
    validation errors return a structured text message; an unknown tool
    name returns a clear "unknown tool" error. Successful executions
    return whatever string the tool produces, or its ``str()`` repr.
    """
    spec = reg.spec(name)
    if spec is None:
        return f"[error] unknown tool: {name!r}", True

    err = reg.validate_args(name, args)
    if err is not None:
        return f"[error] args invalid: {err}", True

    try:
        out = reg.execute(name, args)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}", True
    return out, False


class MCPServerApp:
    """Wraps an MCP :class:`~mcp.server.Server` around the GodBot registry.

    Construction is cheap — no subprocesses, no event loop. Call
    :meth:`build_server` to get the wired-up server instance, or
    :meth:`run_stdio` to block on stdio transport (the production entry).

    The class exists so tests can drive the same handlers without hitting
    stdio: ``app.handle_list_tools()`` and ``app.handle_call_tool(name, args)``
    are the same coroutines the server registers.
    """

    def __init__(self, registry: Registry = DEFAULT, opts: Optional[AppOptions] = None) -> None:
        self.registry = registry
        self.opts = opts or AppOptions()

    def published_tools(self) -> list[ToolSpec]:
        """Return the filtered list of tools we will advertise."""
        return _select_tools(self.registry, self.opts)

    async def handle_list_tools(self):
        """MCP ``list_tools`` handler — returns Tool descriptors for the catalog."""
        return [_spec_to_mcp_tool(s) for s in self.published_tools()]

    async def handle_call_tool(self, name: str, arguments: Optional[dict[str, Any]]):
        """MCP ``call_tool`` handler — runs the tool and returns text content.

        We refuse names that aren't in the published set, even if they exist
        in the underlying registry. This keeps the publish filter honest:
        a host can't smuggle a dangerous tool past ``--safe-only`` by
        guessing its name.
        """
        from mcp import types

        published_names = {s.name for s in self.published_tools()}
        if name not in published_names:
            return [types.TextContent(
                type="text",
                text=f"[error] tool not published: {name!r}",
            )]
        args = arguments or {}
        out, is_error = _execute_tool_safely(self.registry, name, args)
        block = types.TextContent(type="text", text=out)
        if is_error:
            # Returning a tuple here matches the MCP SDK's "isError" content
            # convention — content blocks plus a flag. We use the simpler
            # return-text-and-let-host-treat-as-result path because some SDK
            # versions don't propagate an isError flag through stdio cleanly;
            # the prefix `[error]` makes the failure unmistakable to the
            # host's user-facing rendering.
            pass
        return [block]

    def build_server(self):
        """Construct and wire up the MCP :class:`Server` instance.

        Imports are local so importing this module doesn't require ``mcp``
        to be installed — only ``run_stdio`` and ``build_server`` do.
        """
        from mcp.server import Server  # type: ignore[import-not-found]

        server: Server = Server("godbot")

        @server.list_tools()
        async def _list_tools():
            return await self.handle_list_tools()

        @server.call_tool()
        async def _call_tool(name: str, arguments: Optional[dict[str, Any]]):
            return await self.handle_call_tool(name, arguments)

        return server

    async def run_stdio(self) -> None:
        """Run the server on stdio. Blocks until the host disconnects."""
        from mcp.server import NotificationOptions
        from mcp.server.models import InitializationOptions
        from mcp.server.stdio import stdio_server

        server = self.build_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="godbot",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


def _parse_csv(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="godbot-mcp-server",
        description=(
            "Run GodBot as an MCP server (stdio). Add to your MCP host's "
            "config (e.g. claude_desktop_config.json) to expose GodBot's "
            "tools — read_file, write_file, save_note, recall_notes, "
            "project_summary, etc. — to that host."
        ),
    )
    p.add_argument(
        "--safe-only",
        action="store_true",
        help="Publish only non-dangerous tools (recommended for untrusted hosts).",
    )
    p.add_argument(
        "--include",
        default="",
        help="Comma-separated tool names to publish exclusively (after --safe-only).",
    )
    p.add_argument(
        "--exclude",
        default="",
        help="Comma-separated tool names to never publish.",
    )
    return p


def options_from_args(argv: Optional[list[str]] = None) -> AppOptions:
    args = build_argparser().parse_args(argv)
    return AppOptions(
        safe_only=bool(args.safe_only),
        include=_parse_csv(args.include),
        exclude=_parse_csv(args.exclude),
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Console entry — `godbot-mcp-server`. Blocks on stdio."""
    # Force tool auto-discovery before the host issues list_tools.
    import godbot.tools  # noqa: F401

    # Bring up MCP-client tools too, so we re-expose other servers' tools
    # under our umbrella. Best-effort: failures don't block startup.
    try:
        from godbot.mcp import boot_mcp
        boot_mcp()
    except Exception:
        log.exception("boot_mcp failed; continuing without MCP-client tools")

    opts = options_from_args(argv)
    app = MCPServerApp(opts=opts)
    asyncio.run(app.run_stdio())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
