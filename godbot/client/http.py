from __future__ import annotations
import json
from typing import Any, AsyncIterator, Literal, Optional
import httpx

from godbot.core.events import Event, dict_to_event


class ClientError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.message = message


class Client:
    def __init__(self, base_url: str = "http://127.0.0.1:7878", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "Client":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def _c(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not entered (use 'async with Client() as c')")
        return self._client

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            r = await self._c.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise ClientError(0, f"transport: {e}") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise ClientError(r.status_code, str(detail))
        return r

    # --- Health ---
    async def health(self) -> bool:
        try:
            r = await self._c.get("/api/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # --- Sessions ---
    async def new_session(self, model: str = "auto") -> str:
        r = await self._request("POST", "/api/sessions/new", json={"model": model})
        return r.json()["session_id"]

    async def list_sessions(self) -> list[dict]:
        r = await self._request("GET", "/api/sessions")
        return r.json()

    async def get_session(self, sid: str) -> dict:
        r = await self._request("GET", f"/api/sessions/{sid}")
        return r.json()

    # --- Chat ---
    async def send(self, sid: str, message: str) -> None:
        await self._request("POST", "/api/chat", json={"session_id": sid, "message": message})

    async def stream(self, sid: str) -> AsyncIterator[Event]:
        """Open the SSE stream for a session and yield parsed Event objects.

        Drops `ping` heartbeat events. Stops after `DoneEvent` or stream close.
        """
        try:
            async with self._c.stream(
                "GET",
                "/api/chat/stream",
                params={"session_id": sid},
                timeout=httpx.Timeout(None, connect=10.0),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ClientError(resp.status_code, body.decode("utf-8", "replace"))
                event_name: Optional[str] = None
                data_buf: list[str] = []
                async for raw in resp.aiter_lines():
                    if raw == "":
                        # End of one SSE event — dispatch.
                        if event_name and data_buf:
                            data_str = "\n".join(data_buf)
                            try:
                                payload = json.loads(data_str)
                            except json.JSONDecodeError:
                                event_name, data_buf = None, []
                                continue
                            if event_name != "ping":
                                ev = dict_to_event(payload)
                                yield ev
                                if ev.__class__.__name__ == "DoneEvent":
                                    return
                        event_name, data_buf = None, []
                        continue
                    if raw.startswith(":"):
                        # SSE comment line.
                        continue
                    if raw.startswith("event:"):
                        event_name = raw[6:].strip()
                    elif raw.startswith("data:"):
                        data_buf.append(raw[5:].lstrip())
        except httpx.HTTPError as e:
            raise ClientError(0, f"stream transport: {e}") from e

    # --- Gates / control ---
    async def resolve_gate(
        self, sid: str, call_id: str, decision: Literal["allow", "deny", "always"]
    ) -> None:
        await self._request(
            "POST", f"/api/gate/{call_id}",
            json={"session_id": sid, "decision": decision},
        )

    async def stop(self, sid: str) -> None:
        await self._request("POST", "/api/stop", json={"session_id": sid})

    # --- Tools / RAG ---
    async def list_tools(self) -> list[dict]:
        r = await self._request("GET", "/api/tools")
        return r.json()

    async def toggle_tool(self, sid: str, name: str, enabled: bool) -> None:
        await self._request(
            "POST", "/api/tools/toggle",
            json={"session_id": sid, "name": name, "enabled": enabled},
        )

    async def list_rag_collections(self) -> list[str]:
        r = await self._request("GET", "/api/rag/collections")
        return r.json()

    async def use_rag_collection(self, sid: str, collection: Optional[str]) -> None:
        await self._request(
            "POST", "/api/rag/use",
            json={"session_id": sid, "collection": collection},
        )
