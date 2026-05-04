"""Tests for sub-project 37 — tool_help introspection tool."""
from __future__ import annotations

import json

import godbot.tools  # noqa: F401
from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import tool_help


def test_tool_help_lists_all_when_name_empty():
    out = tool_help(name="")
    assert "Available tools:" in out
    assert "read_file" in out
    assert "write_file" in out
    # write_file is dangerous so it should be flagged.
    for line in out.splitlines():
        if line.strip().startswith("write_file"):
            assert "[DANGEROUS]" in line
            break


def test_tool_help_specific_returns_description():
    out = tool_help(name="read_file")
    assert "read_file" in out
    assert "description:" in out
    assert "args schema:" in out


def test_tool_help_specific_returns_schema_json():
    out = tool_help(name="read_file")
    schema_str = out.split("args schema:\n", 1)[1]
    parsed = json.loads(schema_str)
    assert parsed.get("type") == "object"


def test_tool_help_unknown_returns_error():
    out = tool_help(name="no_such_tool")
    assert out.startswith("[error]")
    assert "no_such_tool" in out


def test_tool_help_marks_dangerous_in_specific_view():
    out = tool_help(name="write_file")
    assert "[DANGEROUS]" in out


def test_tool_help_is_registered_and_non_dangerous():
    spec = DEFAULT.spec("tool_help")
    assert spec is not None
    assert spec.dangerous is False


def test_tool_help_self_lookup():
    """tool_help can describe itself — the agent's reflective bottom."""
    out = tool_help(name="tool_help")
    assert "tool_help" in out
    assert "description:" in out
