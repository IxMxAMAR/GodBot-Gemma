from __future__ import annotations
from typing import AsyncIterator, Callable, Literal, Optional

from godbot.client.daemon import launch_daemon
from godbot.client.http import Client
from godbot.core.events import Event


class Session:
    """Mode-transparent session over either the daemon (HTTP/SSE) or an embedded runner.

    On `connect()` (called automatically by `__aenter__`), probes the daemon's
    /api/health endpoint and routes to the matching mode.
    """

    def __init__(
        self,
        sid: Optional[str] = None,
        *,
        base_url: str = "http://127.0.0.1:7878",
        auto_launch: bool = False,
        embedded_factory: Optional[Callable[[], "object"]] = None,
    ) -> None:
        self._sid = sid
        self._base_url = base_url
        self._auto_launch = auto_launch
        self._embedded_factory = embedded_factory
        self._mode: Optional[Literal["daemon", "embedded"]] = None
        self._client: Optional[Client] = None
        self._runner = None  # EmbeddedRunner

    async def __aenter__(self) -> "Session":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    @property
    def mode(self) -> Literal["daemon", "embedded"]:
        if self._mode is None:
            raise RuntimeError("Session not connected")
        return self._mode

    @property
    def sid(self) -> str:
        if self._sid is None:
            raise RuntimeError("Session has no sid yet (call run() first or pass sid=)")
        return self._sid

    async def _enter_daemon_mode(self) -> None:
        """Open Client and acquire/use sid; clean up on failure to avoid leaks."""
        self._client = Client(base_url=self._base_url)
        await self._client.__aenter__()
        try:
            if self._sid is None:
                self._sid = await self._client.new_session()
        except BaseException:
            # If sid acquisition fails, do not leak the entered Client.
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._client = None
            raise
        self._mode = "daemon"

    async def connect(self) -> None:
        # Probe.
        async with Client(base_url=self._base_url, timeout=2.0) as probe:
            alive = await probe.health()
        if alive:
            await self._enter_daemon_mode()
            return
        if self._auto_launch:
            await launch_daemon(base_url=self._base_url)
            await self._enter_daemon_mode()
            return
        if self._embedded_factory is None:
            raise RuntimeError(
                "no daemon at %s and no embedded_factory provided" % self._base_url
            )
        self._mode = "embedded"
        self._runner = self._embedded_factory()
        self._sid = self._runner.sid

    async def run(self, message: str) -> AsyncIterator[Event]:
        if self._mode == "daemon":
            await self._client.send(self._sid, message)
            async for ev in self._client.stream(self._sid):
                yield ev
        elif self._mode == "embedded":
            async for ev in self._runner.run(message):
                yield ev
        else:
            raise RuntimeError("Session not connected")

    async def resolve_gate(
        self, call_id: str, decision: Literal["allow", "deny", "always"]
    ) -> None:
        if self._mode == "daemon":
            await self._client.resolve_gate(self._sid, call_id, decision)
        else:
            await self._runner.resolve_gate(call_id, decision)

    async def stop(self) -> None:
        if self._mode == "daemon":
            await self._client.stop(self._sid)
        elif self._mode == "embedded":
            await self._runner.stop()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
        if self._runner is not None:
            await self._runner.close()
            self._runner = None
