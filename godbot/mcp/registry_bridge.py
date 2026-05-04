"""Bridge MCP-discovered tools into GodBot's ``DEFAULT`` registry.

For each connected :class:`~godbot.mcp.client.MCPClient`, we synthesise a
``ToolSpec`` per tool the server advertises and stash it directly into the
registry's internal ``_tools`` map. This deliberately bypasses the public
:meth:`Registry.tool` decorator because:

* That decorator derives the JSON Schema from a Python function signature
  via ``pydantic.create_model``. We already *have* the canonical JSON Schema
  the MCP server published; we don't want to round-trip it through pydantic.
* The decorator forces a non-empty docstring; some MCP servers ship empty
  descriptions. We tolerate that with a sensible default.

Direct ``_tools[name] = spec`` is a documented private-attr poke (Registry
exposes no public ``register_raw`` today). If/when one is added we should
migrate. See ``godbot/core/registry.py``.

Tool names are namespaced ``mcp_<server>_<tool>`` so two servers exposing
the same tool name do not collide. All MCP tools are marked
``dangerous=True`` because they're third-party — the user gates each call.
"""

from __future__ import annotations
import logging
from typing import Any

from godbot.core.registry import DEFAULT, ToolSpec
from godbot.mcp.client import MCPClient

log = logging.getLogger("godbot.mcp")


# Track registered MCP clients so that:
#   - ``mcp_status`` can introspect connection state without re-listing tools,
#   - ``shutdown_all`` can close every subprocess on daemon teardown,
#   - the bridge can detect & log re-registration of the same server name.
_clients: dict[str, MCPClient] = {}


def _normalize_schema(raw: Any) -> dict[str, Any]:
    """Coerce whatever the MCP server returned into a usable JSON Schema.

    The SDK normally hands back a ``dict``, but defensively we accept None /
    non-dict and fall back to a minimal object schema. ``additionalProperties``
    is forced to ``True`` so jsonschema validation passes through unknown
    fields rather than rejecting the call — MCP servers occasionally evolve
    schemas faster than tool catalogs do.
    """
    if not isinstance(raw, dict):
        raw = {}
    schema = dict(raw)  # shallow copy so we don't mutate the SDK's object
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    # MCP servers may accept extra args — be permissive in what we send.
    schema.setdefault("additionalProperties", True)
    return schema


def _short_description(raw: str, fallback: str) -> str:
    """First line of the description, trimmed; fallback if empty."""
    if not raw or not raw.strip():
        return fallback
    return raw.strip().splitlines()[0]


def register_mcp_server(client: MCPClient) -> int:
    """Register all tools advertised by ``client`` into ``DEFAULT``.

    Returns the count of tools registered. Returns 0 (silently) if the client
    failed to connect — the caller is expected to check ``client.connected``
    or reconcile via ``mcp_status``. Idempotent on re-call: re-registering
    the same server replaces its tools.
    """
    if not client.connected:
        return 0

    name = client.cfg.name
    if name in _clients:
        # Tear down previous registration's tool entries so a re-boot doesn't
        # leave stale tools pointing at a dead client.
        prefix = f"mcp_{name}_"
        for k in list(DEFAULT._tools.keys()):
            if k.startswith(prefix):
                DEFAULT._tools.pop(k, None)

    _clients[name] = client
    count = 0
    for tool_info in client.tools:
        mcp_name = tool_info["name"]
        registered_name = f"mcp_{name}_{mcp_name}"
        description = _short_description(
            tool_info.get("description", ""),
            f"(MCP tool {mcp_name!r} from {name!r})",
        )

        # Make a closure per tool so the loop variables don't all alias the
        # final iteration. (`bound_client` and `bound_name` are captured.)
        def _make_fn(_client: MCPClient, _mcp_name: str, _doc: str):
            def _call(**kwargs: Any) -> str:
                return _client.call_tool(_mcp_name, kwargs)
            _call.__doc__ = _doc
            _call.__name__ = f"mcp_{_client.cfg.name}_{_mcp_name}"
            return _call

        fn = _make_fn(client, mcp_name, description)
        schema = _normalize_schema(tool_info.get("inputSchema"))

        spec = ToolSpec(
            name=registered_name,
            description=description,
            fn=fn,
            schema=schema,
            dangerous=True,  # third-party — gate by default in v1
            timeout=120,
        )
        # NOTE: writing to ``_tools`` directly is a deliberate private-attr
        # poke; the public ``@tool`` decorator builds the schema from a Python
        # signature and we already have the schema from the MCP handshake.
        DEFAULT._tools[registered_name] = spec
        count += 1

    log.info("registered %d MCP tools from %r", count, name)
    return count


def shutdown_all() -> None:
    """Close every registered MCP client cleanly. Never raises."""
    for c in list(_clients.values()):
        try:
            c.close()
        except Exception:
            log.exception("MCP client %r close raised; ignoring", c.cfg.name)
    _clients.clear()
