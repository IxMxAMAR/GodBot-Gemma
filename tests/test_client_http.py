import pytest
import respx
import httpx
from godbot.client import Client, ClientError


@respx.mock
@pytest.mark.asyncio
async def test_health_returns_true_on_200():
    respx.get("http://127.0.0.1:7878/api/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    async with Client() as c:
        assert await c.health() is True


@respx.mock
@pytest.mark.asyncio
async def test_health_returns_false_on_connection_error():
    respx.get("http://127.0.0.1:7878/api/health").mock(side_effect=httpx.ConnectError("nope"))
    async with Client() as c:
        assert await c.health() is False


@respx.mock
@pytest.mark.asyncio
async def test_new_session_returns_id():
    respx.post("http://127.0.0.1:7878/api/sessions/new").mock(
        return_value=httpx.Response(200, json={"session_id": "s1"})
    )
    async with Client() as c:
        sid = await c.new_session()
        assert sid == "s1"


@respx.mock
@pytest.mark.asyncio
async def test_list_sessions():
    respx.get("http://127.0.0.1:7878/api/sessions").mock(
        return_value=httpx.Response(200, json=[{"id": "s1", "model": "g"}])
    )
    async with Client() as c:
        sessions = await c.list_sessions()
        assert sessions[0]["id"] == "s1"


@respx.mock
@pytest.mark.asyncio
async def test_non_2xx_raises_client_error():
    respx.get("http://127.0.0.1:7878/api/sessions/missing").mock(
        return_value=httpx.Response(404, json={"detail": "no such session"})
    )
    async with Client() as c:
        with pytest.raises(ClientError) as exc:
            await c.get_session("missing")
        assert exc.value.status_code == 404


import json as _json
from godbot.core.events import TokenEvent, DoneEvent


def _sse_body(events: list[tuple[str, dict]]) -> str:
    out = ""
    for type_name, data in events:
        out += f"event: {type_name}\ndata: {_json.dumps(data)}\n\n"
    return out


@respx.mock
@pytest.mark.asyncio
async def test_stream_yields_events_until_done():
    body = _sse_body([
        ("token", {"type": "token", "text": "hi"}),
        ("token", {"type": "token", "text": " there"}),
        ("done", {"type": "done", "step_count": 1}),
    ])
    respx.get("http://127.0.0.1:7878/api/chat/stream").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )
    )
    async with Client() as c:
        events = []
        async for ev in c.stream("s1"):
            events.append(ev)
    assert events == [TokenEvent(text="hi"), TokenEvent(text=" there"), DoneEvent(step_count=1)]


@respx.mock
@pytest.mark.asyncio
async def test_stream_drops_ping_heartbeats():
    body = _sse_body([
        ("ping", {}),
        ("token", {"type": "token", "text": "ok"}),
        ("ping", {}),
        ("done", {"type": "done", "step_count": 1}),
    ])
    respx.get("http://127.0.0.1:7878/api/chat/stream").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )
    )
    async with Client() as c:
        events = [ev async for ev in c.stream("s1")]
    types = [type(e).__name__ for e in events]
    assert types == ["TokenEvent", "DoneEvent"]


@respx.mock
@pytest.mark.asyncio
async def test_stream_raises_on_error_status():
    respx.get("http://127.0.0.1:7878/api/chat/stream").mock(
        return_value=httpx.Response(409, content=b"")
    )
    async with Client() as c:
        with pytest.raises(ClientError) as exc:
            async for _ in c.stream("s1"):
                pass
        assert exc.value.status_code == 409
