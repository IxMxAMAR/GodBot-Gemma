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


def _respx_is_active() -> bool:
    """Return True if respx is currently patching the HTTP stack.

    respx subclasses ``Mocker`` push ``unittest.mock.patch`` objects onto a
    class-level ``_patches`` list while active (specifically on
    ``HTTPCoreMocker``, which is the deeper subclass that actually
    intercepts ``httpcore``). Walk the subclass tree so we catch it
    regardless of where in the hierarchy patches landed.
    """
    try:
        from respx.mocks import Mocker
        stack = list(Mocker.__subclasses__())
        while stack:
            cls = stack.pop()
            if getattr(cls, "_patches", None):
                return True
            stack.extend(cls.__subclasses__())
    except Exception:
        pass
    return False


@pytest.fixture(autouse=True)
def _block_live_cloud_requests(monkeypatch, request):
    """Fail any test that issues a real HTTP request to a known cloud host.

    Tests that need to mock cloud endpoints decorate themselves with
    ``@respx.mock`` (or use the respx fixture); when respx is active we
    let the request through so respx can match it. Only un-mocked calls
    to a blocked host trip the guard.
    """
    real_send = httpx.HTTPTransport.handle_request
    real_async_send = httpx.AsyncHTTPTransport.handle_async_request

    def guarded(self, req, *a, **k):
        if _host_is_blocked(str(req.url)) and not _respx_is_active():
            raise AssertionError(
                f"live cloud request blocked: {req.method} {req.url} "
                f"(test {request.node.nodeid}). Use @respx.mock."
            )
        return real_send(self, req, *a, **k)

    async def guarded_async(self, req, *a, **k):
        if _host_is_blocked(str(req.url)) and not _respx_is_active():
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
