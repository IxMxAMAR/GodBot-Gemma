"""MCP stdio client wrapper.

The MCP Python SDK is async-first, but GodBot's tool registry is sync. We
bridge by:

1. Running a *single* asyncio event loop in a worker thread (:class:`MCPLoopRunner`).
2. Each :class:`MCPClient` owns one stdio subprocess + one ``ClientSession``.
   Connect / call / close all dispatch their coroutines onto the shared loop
   via ``run_coroutine_threadsafe`` and block on the resulting Future.

The shared-loop model keeps the thread count bounded (one thread total, no
matter how many servers the user configures) and avoids per-server thread
juggling. The trade-off is that all MCP traffic is serialised through that
loop — fine for v1, where MCP calls happen inside the agent's already-serial
turn loop.

A connect-time wall-clock guard is enforced (default 15s). MCP servers that
fail to initialise within the window are marked as failed and skipped during
registry bridging. The daemon stays up.
"""

from __future__ import annotations
import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("godbot.mcp")


# Bound on connect() so a hung MCP server doesn't pin daemon startup forever.
DEFAULT_CONNECT_TIMEOUT = 15.0
# Bound on call_tool() so a runaway server doesn't pin the agent loop.
DEFAULT_CALL_TIMEOUT = 120.0


@dataclass
class MCPServerConfig:
    """Connection-time config for one MCP server (stdio transport)."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


class MCPLoopRunner:
    """Singleton: runs one asyncio loop in a daemon thread for all MCP traffic.

    On daemon process exit the worker thread dies with the process — no
    explicit shutdown is required for correctness. We expose :meth:`shutdown`
    anyway so tests can stop the loop deterministically.
    """

    _instance: Optional["MCPLoopRunner"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, daemon=True, name="mcp-loop",
        )
        self.thread.start()

    @classmethod
    def get(cls) -> "MCPLoopRunner":
        with cls._lock:
            if cls._instance is None or not cls._instance.thread.is_alive():
                cls._instance = cls()
            return cls._instance

    def call(self, coro, timeout: Optional[float] = None):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass


class MCPClient:
    """Connects to one MCP server via stdio. Wraps the official ``mcp`` SDK."""

    def __init__(self, cfg: MCPServerConfig) -> None:
        self.cfg = cfg
        self._session = None
        self._exit_stack = None
        self.connected = False
        self.connect_error: Optional[str] = None
        # Cached tool descriptors {name, description, inputSchema}.
        self.tools: list[dict[str, Any]] = []

    async def _aconnect(self) -> None:
        # Imports kept inside the coroutine so that an SDK ImportError on
        # boot only fails this single server, not the whole package import.
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # MCP server processes commonly need PATH, APPDATA, etc. to find npx.
        # Merging on top of the parent env keeps Windows-style spawns working.
        import os as _os
        merged_env = {**_os.environ, **self.cfg.env}

        params = StdioServerParameters(
            command=self.cfg.command,
            args=list(self.cfg.args),
            env=merged_env,
        )
        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        result = await self._session.list_tools()
        self.tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            for t in result.tools
        ]
        self.connected = True

    def connect(self, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> None:
        """Spawn subprocess, initialise, list tools. Best-effort.

        On any failure (timeout, missing command, protocol error) the error
        is captured on ``self.connect_error`` and ``self.connected`` stays
        ``False``. The caller should check ``self.connected`` before doing
        anything else with this client.
        """
        runner = MCPLoopRunner.get()
        try:
            runner.call(
                asyncio.wait_for(self._aconnect(), timeout=timeout),
                timeout=timeout + 5.0,
            )
        except Exception as e:
            self.connect_error = f"{type(e).__name__}: {e}"
            log.warning(
                "MCP server %r failed to connect: %s",
                self.cfg.name, self.connect_error,
            )
            # Best-effort cleanup of any half-built async resources.
            try:
                if self._exit_stack is not None:
                    runner.call(self._exit_stack.aclose(), timeout=5.0)
            except Exception:
                pass
            self._exit_stack = None
            self._session = None
            self.connected = False

    async def _acall_tool(self, name: str, args: dict[str, Any]) -> str:
        result = await self._session.call_tool(name, args)
        # MCP returns a list of content blocks; flatten text blocks. Non-text
        # content (images, resources) is repr()'d for visibility — the agent
        # rarely surfaces these and we don't have a binary channel here.
        chunks: list[str] = []
        for c in result.content:
            text = getattr(c, "text", None)
            if text:
                chunks.append(text)
            else:
                chunks.append(repr(c))
        out = "\n".join(chunks)
        # MCP exposes an isError flag on the response; surface it inline so
        # the agent can see and react.
        is_err = getattr(result, "isError", False)
        if is_err:
            return f"[mcp tool error] {out}"
        return out

    def call_tool(
        self, name: str, args: dict[str, Any],
        timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> str:
        """Sync entry point used by registry-bridged tool wrappers."""
        if not self.connected:
            return f"[error] MCP server {self.cfg.name!r} not connected"
        try:
            return MCPLoopRunner.get().call(
                asyncio.wait_for(self._acall_tool(name, args), timeout=timeout),
                timeout=timeout + 5.0,
            )
        except Exception as e:
            return f"[error] MCP call_tool({name!r}) failed: {type(e).__name__}: {e}"

    async def _aclose(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                # The SDK occasionally raises during teardown if the server
                # already exited; that's noise, not a real error.
                pass
        self._exit_stack = None
        self._session = None

    def close(self) -> None:
        """Tear down the MCP session + subprocess. Idempotent, never raises."""
        if self._exit_stack is not None:
            try:
                MCPLoopRunner.get().call(self._aclose(), timeout=5.0)
            except Exception:
                pass
        self.connected = False
