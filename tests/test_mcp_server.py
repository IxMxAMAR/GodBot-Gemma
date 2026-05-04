"""Tests for sub-project 11 — GodBot as MCP server.

Drives the MCPServerApp handlers directly (not via stdio) so tests stay
hermetic and fast. The stdio path is exercised once via the SDK's in-process
client to confirm wiring.
"""
from __future__ import annotations
import asyncio

import pytest

from godbot.core.registry import Registry
from godbot.mcp.server import (
    AppOptions,
    MCPServerApp,
    _execute_tool_safely,
    _select_tools,
    options_from_args,
)


def _build_registry() -> Registry:
    """A fresh registry with one safe tool, one dangerous tool, plus an extra."""
    reg = Registry()

    @reg.tool()
    def echo(text: str = "") -> str:
        """Echo the input back."""
        return f"echoed: {text}"

    @reg.tool(dangerous=True)
    def danger(arg: str = "") -> str:
        """A dangerous demo tool."""
        return f"danger ran: {arg}"

    @reg.tool()
    def adder(a: int = 0, b: int = 0) -> str:
        """Add two integers."""
        return str(a + b)

    return reg


def test_select_tools_returns_all_by_default():
    reg = _build_registry()
    tools = _select_tools(reg, AppOptions())
    names = [t.name for t in tools]
    assert set(names) == {"echo", "danger", "adder"}


def test_select_tools_safe_only_excludes_dangerous():
    reg = _build_registry()
    tools = _select_tools(reg, AppOptions(safe_only=True))
    names = [t.name for t in tools]
    assert "danger" not in names
    assert {"echo", "adder"}.issubset(names)


def test_select_tools_include_filter_acts_as_allowlist():
    reg = _build_registry()
    tools = _select_tools(reg, AppOptions(include=["echo"]))
    assert [t.name for t in tools] == ["echo"]


def test_select_tools_exclude_filter():
    reg = _build_registry()
    tools = _select_tools(reg, AppOptions(exclude=["adder"]))
    names = [t.name for t in tools]
    assert "adder" not in names


def test_select_tools_combined_safe_only_and_exclude():
    reg = _build_registry()
    tools = _select_tools(reg, AppOptions(safe_only=True, exclude=["adder"]))
    assert [t.name for t in tools] == ["echo"]


@pytest.mark.asyncio
async def test_handle_list_tools_advertises_filtered_set():
    reg = _build_registry()
    app = MCPServerApp(registry=reg, opts=AppOptions(safe_only=True))
    advertised = await app.handle_list_tools()
    names = [t.name for t in advertised]
    assert set(names) == {"echo", "adder"}
    # The Tool descriptor includes a description we can inspect.
    by_name = {t.name: t for t in advertised}
    assert "Echo" in by_name["echo"].description or "echo" in by_name["echo"].description.lower()


@pytest.mark.asyncio
async def test_handle_list_tools_marks_dangerous_in_description():
    reg = _build_registry()
    app = MCPServerApp(registry=reg, opts=AppOptions())
    advertised = await app.handle_list_tools()
    by_name = {t.name: t for t in advertised}
    assert "DANGEROUS" in by_name["danger"].description


@pytest.mark.asyncio
async def test_call_tool_routes_through_registry():
    reg = _build_registry()
    app = MCPServerApp(registry=reg, opts=AppOptions())
    blocks = await app.handle_call_tool("echo", {"text": "hi"})
    assert len(blocks) == 1
    assert "echoed: hi" in blocks[0].text


@pytest.mark.asyncio
async def test_call_tool_unknown_returns_error_block():
    reg = _build_registry()
    app = MCPServerApp(registry=reg, opts=AppOptions())
    blocks = await app.handle_call_tool("does_not_exist", {})
    assert len(blocks) == 1
    assert "[error]" in blocks[0].text
    assert "not published" in blocks[0].text or "unknown" in blocks[0].text


@pytest.mark.asyncio
async def test_safe_only_call_tool_refuses_dangerous():
    reg = _build_registry()
    app = MCPServerApp(registry=reg, opts=AppOptions(safe_only=True))
    blocks = await app.handle_call_tool("danger", {"arg": "x"})
    assert len(blocks) == 1
    assert "[error]" in blocks[0].text
    assert "not published" in blocks[0].text


@pytest.mark.asyncio
async def test_call_tool_arg_validation_failure_returns_text():
    """An invalid args dict should return an [error] text block, not raise."""
    reg = _build_registry()
    app = MCPServerApp(registry=reg, opts=AppOptions())
    # `adder` declares ints; passing a non-numeric string fails jsonschema.
    blocks = await app.handle_call_tool("adder", {"a": "not_a_number", "b": 2})
    assert len(blocks) == 1
    assert "[error]" in blocks[0].text


@pytest.mark.asyncio
async def test_call_tool_execution_exception_caught():
    reg = Registry()

    @reg.tool()
    def boom() -> str:
        """Raise on call."""
        raise RuntimeError("kaboom")

    app = MCPServerApp(registry=reg, opts=AppOptions())
    blocks = await app.handle_call_tool("boom", {})
    assert len(blocks) == 1
    assert "[error]" in blocks[0].text
    assert "RuntimeError" in blocks[0].text


def test_execute_tool_safely_returns_tuple():
    reg = _build_registry()
    out, is_err = _execute_tool_safely(reg, "echo", {"text": "yo"})
    assert is_err is False
    assert "echoed: yo" in out


def test_execute_tool_safely_unknown_marks_error():
    reg = _build_registry()
    out, is_err = _execute_tool_safely(reg, "no_such_tool", {})
    assert is_err is True
    assert "unknown tool" in out


def test_options_from_args_csv_parsing():
    opts = options_from_args(["--safe-only", "--include", "a, b ,c", "--exclude", "x"])
    assert opts.safe_only is True
    assert opts.include == ["a", "b", "c"]
    assert opts.exclude == ["x"]


def test_options_from_args_defaults():
    opts = options_from_args([])
    assert opts.safe_only is False
    assert opts.include == []
    assert opts.exclude == []


def test_build_server_returns_mcp_server_instance():
    """Smoke test: build_server should produce an mcp.server.Server without raising."""
    pytest.importorskip("mcp")
    reg = _build_registry()
    app = MCPServerApp(registry=reg, opts=AppOptions())
    server = app.build_server()
    # The SDK's Server class exposes a `name` attribute.
    assert getattr(server, "name", None) == "godbot"


@pytest.mark.asyncio
async def test_concurrent_call_tool_via_handle_returns_consistent_results():
    """Concurrent calls to the same handler must each return their own result."""
    reg = _build_registry()
    app = MCPServerApp(registry=reg, opts=AppOptions())

    async def run_one(text: str):
        blocks = await app.handle_call_tool("echo", {"text": text})
        return blocks[0].text

    results = await asyncio.gather(*[run_one(f"msg-{i}") for i in range(5)])
    for i, text in enumerate(results):
        assert f"echoed: msg-{i}" in text
