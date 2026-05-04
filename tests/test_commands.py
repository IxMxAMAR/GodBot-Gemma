"""Tests for sub-project 18 — custom slash commands."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.commands import (
    CommandSpec,
    _load_one,
    load_commands,
    match_command,
)
from godbot.interfaces.web import build_app


def _write_command(commands_dir: Path, name: str, body: str) -> Path:
    commands_dir.mkdir(parents=True, exist_ok=True)
    p = commands_dir / f"{name}.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_one_minimal_file(tmp_path):
    p = _write_command(tmp_path, "refactor", '''
name = "refactor"
description = "Refactor code"
system_prompt_suffix = "Be careful."
tool_overrides = ["read_file", "edit_file"]
''')
    spec = _load_one(p)
    assert spec is not None
    assert spec.name == "refactor"
    assert spec.description == "Refactor code"
    assert spec.system_prompt_suffix == "Be careful."
    assert spec.tool_overrides == ["read_file", "edit_file"]


def test_load_one_uses_filename_as_default_name(tmp_path):
    p = _write_command(tmp_path, "doc", "description = \"Add docstrings\"\n")
    spec = _load_one(p)
    assert spec is not None
    assert spec.name == "doc"


def test_load_one_rejects_invalid_name(tmp_path):
    p = _write_command(tmp_path, "weird", 'name = "Has Spaces"\n')
    spec = _load_one(p)
    assert spec is None


def test_load_one_corrupt_toml_returns_none(tmp_path):
    p = tmp_path / "broken.toml"
    p.write_text("not = valid toml [[[", encoding="utf-8")
    assert _load_one(p) is None


def test_load_commands_empty_dir(tmp_path):
    out = load_commands(tmp_path)
    assert out == {}


def test_load_commands_missing_dir_returns_empty(tmp_path):
    out = load_commands(tmp_path / "no_such")
    assert out == {}


def test_load_commands_picks_up_multiple(tmp_path):
    _write_command(tmp_path, "refactor", 'description = "x"\n')
    _write_command(tmp_path, "doc", 'description = "y"\n')
    out = load_commands(tmp_path)
    assert set(out.keys()) == {"refactor", "doc"}


def test_match_command_strict_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    cmd_dir = tmp_path / "commands"
    _write_command(cmd_dir, "refactor", 'description = "x"\n')
    cmds = load_commands(cmd_dir)

    assert match_command("/refactor make this DRY", cmds).name == "refactor"
    assert match_command("/refactor", cmds).name == "refactor"
    assert match_command("refactor this", cmds) is None
    assert match_command("/refactorize this", cmds) is None  # name boundary
    assert match_command("Could you /refactor", cmds) is None  # not at start
    assert match_command("", cmds) is None
    assert match_command("/", cmds) is None


def test_match_command_skips_plan(tmp_path, monkeypatch):
    """`/plan` is reserved for built-in plan mode and must NOT match a custom
    command even if a malicious user dropped a plan.toml."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    cmd_dir = tmp_path / "commands"
    _write_command(cmd_dir, "plan", 'description = "imposter"\n')
    cmds = load_commands(cmd_dir)
    assert match_command("/plan rewrite", cmds) is None


def test_match_command_unknown_returns_none(tmp_path):
    cmds = {"refactor": CommandSpec(name="refactor")}
    assert match_command("/notacommand args", cmds) is None


def test_endpoint_returns_commands_list(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    cmd_dir = tmp_path / "commands"
    _write_command(cmd_dir, "refactor", 'description = "Refactor"\ntool_overrides = ["read_file"]\n')

    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/commands")
    assert r.status_code == 200
    body = r.json()
    assert "commands" in body
    by_name = {x["name"]: x for x in body["commands"]}
    assert "refactor" in by_name
    assert by_name["refactor"]["description"] == "Refactor"
    assert by_name["refactor"]["tool_overrides"] == ["read_file"]


def test_endpoint_empty_when_no_commands_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.get("/api/commands")
    assert r.status_code == 200
    assert r.json()["commands"] == []


@pytest.mark.asyncio
async def test_agent_loop_appends_custom_suffix_to_system_prompt(tmp_path, monkeypatch):
    """End-to-end through run_turn: a /<command> message swaps the prompt."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    cmd_dir = tmp_path / "commands"
    _write_command(cmd_dir, "doc", '''
description = "Add docstrings"
system_prompt_suffix = "DOCSTRING-MODE-MARKER"
''')

    from godbot.core.agent import run_turn
    from godbot.core.events import DoneEvent
    from godbot.core.registry import Registry
    from godbot.core.session import Session
    from tests._mock_llm import MockLLM

    reg = Registry()
    s = Session.create(root=tmp_path / "sessions", model="m")
    s.append_user("/doc add a docstring to add(a, b)")

    llm = MockLLM([json.dumps({"thought": "ok", "final_answer": "added"})])
    events = []
    async def emit(ev):
        events.append(ev)

    cancel = asyncio.Event()
    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit, cancel=cancel,
        max_steps=3, max_context=10000, system_prompt="SYS",
    )
    assert any(isinstance(e, DoneEvent) for e in events)
    sys_msg = llm.calls[0]["messages"][0]
    assert sys_msg["role"] == "system"
    assert "DOCSTRING-MODE-MARKER" in sys_msg["content"]


@pytest.mark.asyncio
async def test_agent_loop_narrows_tools_to_command_overrides(tmp_path, monkeypatch):
    """When a command declares tool_overrides, the LLM only sees those tools."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    cmd_dir = tmp_path / "commands"
    _write_command(cmd_dir, "review", '''
description = "Review code only — no writes"
tool_overrides = ["only_safe_one"]
''')

    from godbot.core.agent import run_turn
    from godbot.core.events import DoneEvent
    from godbot.core.registry import Registry
    from godbot.core.session import Session
    from tests._mock_llm import MockLLM

    reg = Registry()

    @reg.tool()
    def only_safe_one(arg: str = "") -> str:
        """Allowed under /review."""
        return "ok"

    @reg.tool()
    def dangerous_writer(path: str = "") -> str:
        """Should not appear under /review."""
        return "wrote"

    s = Session.create(root=tmp_path / "sessions", model="m")
    s.append_user("/review the auth module")

    llm = MockLLM([json.dumps({"thought": "ok", "final_answer": "looks fine"})])
    events = []
    async def emit(ev):
        events.append(ev)

    from godbot.prompts import build_system_prompt
    cancel = asyncio.Event()
    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit, cancel=cancel,
        max_steps=3, max_context=10000, system_prompt="SYS",
        system_prompt_builder=build_system_prompt,
    )
    assert any(isinstance(e, DoneEvent) for e in events)
    sys_msg = llm.calls[0]["messages"][0]
    # The catalog block in the system prompt must include only the allowed tool.
    assert "only_safe_one" in sys_msg["content"]
    assert "dangerous_writer" not in sys_msg["content"]
