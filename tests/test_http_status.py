"""Tests for sub-project 66 — http_status tool."""
from __future__ import annotations

import httpx
import pytest
import respx

from godbot.core.registry import DEFAULT
from godbot.tools.web import http_status


@respx.mock
def test_http_status_basic_head_success():
    respx.head("https://example.com/").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/html", "Content-Length": "1234"},
        )
    )
    out = http_status(url="https://example.com/")
    assert "status: 200" in out
    assert "content-type: text/html" in out
    assert "content-length: 1234" in out


@respx.mock
def test_http_status_404():
    respx.head("https://example.com/missing").mock(return_value=httpx.Response(404))
    out = http_status(url="https://example.com/missing")
    assert "status: 404" in out


@respx.mock
def test_http_status_falls_back_to_get_on_405():
    """Servers that 405 on HEAD should fall through to a Range GET."""
    respx.head("https://example.com/no-head").mock(return_value=httpx.Response(405))
    respx.get("https://example.com/no-head").mock(
        return_value=httpx.Response(200, headers={"Content-Type": "application/json"})
    )
    out = http_status(url="https://example.com/no-head")
    assert "status: 200" in out
    assert "application/json" in out


@respx.mock
def test_http_status_handles_transport_error():
    respx.head("https://example.com/bad").mock(side_effect=httpx.ConnectError("boom"))
    respx.get("https://example.com/bad").mock(side_effect=httpx.ConnectError("boom"))
    out = http_status(url="https://example.com/bad")
    assert out.startswith("[error]")


def test_http_status_is_registered_non_dangerous():
    spec = DEFAULT.spec("http_status")
    assert spec is not None
    assert spec.dangerous is False


@respx.mock
def test_http_status_redirect_reports_final_url():
    respx.head("https://short.example/x").mock(
        return_value=httpx.Response(
            301, headers={"Location": "https://final.example/x"},
        )
    )
    respx.head("https://final.example/x").mock(return_value=httpx.Response(200))
    out = http_status(url="https://short.example/x")
    assert "status: 200" in out
    assert "final.example" in out
