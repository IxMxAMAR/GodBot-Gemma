import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Cost protection: every test that hits a cloud API uses respx mocks. This
# autouse fixture patches httpx so a stray real request to a known cloud
# endpoint fails loudly instead of silently spending money. respx-mocked
# requests are unaffected because respx replaces the transport BEFORE this
# guard checks the URL — by the time the request reaches us here, respx has
# already short-circuited matched routes.
_BLOCKED_HOSTS = (
    "anthropic.com",
    "openai.com",
    "googleapis.com",
    "groq.com",
    "together.xyz",
    "openrouter.ai",
    "mistral.ai",
    "fireworks.ai",
    "cerebras.ai",
)


def _host_is_blocked(url: str) -> bool:
    lowered = url.lower()
    return any(host in lowered for host in _BLOCKED_HOSTS)


@pytest.fixture(autouse=True)
def _block_live_cloud_requests(monkeypatch, request):
    """Fail any test that issues a real HTTP request to a known cloud host.

    Tests that need to mock cloud endpoints decorate themselves with
    ``@respx.mock`` (or use the respx fixture); respx intercepts at the
    transport layer, so matched calls never trip this guard.
    """
    real_send = httpx.HTTPTransport.handle_request
    real_async_send = httpx.AsyncHTTPTransport.handle_async_request

    def guarded(self, req, *a, **k):
        if _host_is_blocked(str(req.url)):
            raise AssertionError(
                f"live cloud request blocked: {req.method} {req.url} "
                f"(test {request.node.nodeid}). Use @respx.mock."
            )
        return real_send(self, req, *a, **k)

    async def guarded_async(self, req, *a, **k):
        if _host_is_blocked(str(req.url)):
            raise AssertionError(
                f"live cloud request blocked: {req.method} {req.url} "
                f"(test {request.node.nodeid}). Use @respx.mock."
            )
        return await real_async_send(self, req, *a, **k)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", guarded_async)
    yield


@pytest.fixture
def tmp_godbot_home(tmp_path, monkeypatch):
    """Redirect ~/.godbot to a tmpdir so tests don't pollute real home."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    return tmp_path / ".godbot"
