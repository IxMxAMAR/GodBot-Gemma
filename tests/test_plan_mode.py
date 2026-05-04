"""Tests for sub-project 10.4 — plan mode.

Verifies:
- `is_plan_request` detects strict `/plan ` prefix.
- `build_plan_system_prompt` produces a prompt distinct from the regular one
  and contains the plan-mode directives.
- The agent loop swaps to the plan prompt when the latest user message starts
  with `/plan ` and the LLM's final_answer round-trips through the schema.
"""
from __future__ import annotations
import asyncio
import json

import pytest

from godbot.core.agent import run_turn
from godbot.core.events import DoneEvent
from godbot.core.registry import Registry
from godbot.core.session import Session
from godbot.prompts import (
    PLAN_PREFIX,
    build_plan_system_prompt,
    build_system_prompt,
    is_plan_request,
)
from tests._mock_llm import MockLLM


def test_is_plan_request_strict_prefix():
    assert is_plan_request("/plan rewrite the README")
    assert is_plan_request("/plan ")  # empty goal is still plan mode
    # Non-matches:
    assert not is_plan_request("plan the work")
    assert not is_plan_request("/planner please help")  # no trailing space
    assert not is_plan_request("Let's /plan together")  # not at start
    assert not is_plan_request("/PLAN something")  # case-sensitive
    assert not is_plan_request("")


def test_build_plan_system_prompt_differs_from_regular():
    plan_p = build_plan_system_prompt([])
    regular_p = build_system_prompt([])
    assert plan_p != regular_p
    assert "PLAN MODE" in plan_p
    assert "checklist" in plan_p.lower() or "tasks" in plan_p.lower()
    assert "do not invoke any tools" in plan_p.lower()


def test_plan_prefix_constant():
    """Sentinel: the public prefix must remain `/plan ` (with trailing space).
    Studio relies on this for its plan-mode UI badge."""
    assert PLAN_PREFIX == "/plan "


@pytest.mark.asyncio
async def test_agent_uses_plan_prompt_when_user_says_slash_plan(tmp_path):
    """The agent's first call should pass the plan-mode system prompt."""
    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    s.append_user("/plan rewrite the README")

    plan_payload = json.dumps({
        "plan": {
            "goal": "Rewrite README",
            "tasks": [
                {"id": 1, "title": "Read README.md", "status": "pending"},
                {"id": 2, "title": "Draft new sections", "status": "pending"},
                {"id": 3, "title": "Write README.md", "status": "pending"},
            ],
        }
    })
    llm = MockLLM([json.dumps({"thought": "planning", "final_answer": plan_payload})])
    events = []

    async def emit(ev):
        events.append(ev)

    cancel = asyncio.Event()
    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit, cancel=cancel,
        max_steps=5, max_context=10000, system_prompt="SYS",
    )
    # The DoneEvent fires.
    assert any(isinstance(e, DoneEvent) for e in events)
    # The system prompt that went to the LLM was the plan variant.
    sys_msg = llm.calls[0]["messages"][0]
    assert sys_msg["role"] == "system"
    assert "PLAN MODE" in sys_msg["content"]


@pytest.mark.asyncio
async def test_agent_uses_regular_prompt_for_normal_message(tmp_path):
    """Sanity: a non-`/plan` message goes through the regular system prompt."""
    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi there")

    llm = MockLLM([json.dumps({"thought": "easy", "final_answer": "hello!"})])
    events = []

    async def emit(ev):
        events.append(ev)

    cancel = asyncio.Event()
    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit, cancel=cancel,
        max_steps=5, max_context=10000, system_prompt="SYS",
        system_prompt_builder=build_system_prompt,
    )
    sys_msg = llm.calls[0]["messages"][0]
    assert "PLAN MODE" not in sys_msg["content"]


@pytest.mark.asyncio
async def test_plan_final_answer_is_parseable_json(tmp_path):
    """The final_answer string emitted in plan mode parses back to a plan dict."""
    reg = Registry()
    s = Session.create(root=tmp_path, model="m")
    s.append_user("/plan add unit tests for utils")

    plan_payload = json.dumps({
        "plan": {
            "goal": "Add unit tests for utils",
            "tasks": [
                {"id": 1, "title": "Read utils.py", "status": "pending"},
                {"id": 2, "title": "Write test_utils.py", "status": "pending"},
                {"id": 3, "title": "Run pytest", "status": "pending"},
            ],
        }
    })
    llm = MockLLM([json.dumps({"thought": "planning", "final_answer": plan_payload})])

    async def emit(ev):
        return None

    cancel = asyncio.Event()
    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit, cancel=cancel,
        max_steps=5, max_context=10000, system_prompt="SYS",
    )

    # The session's last assistant message should be the plan JSON, parseable.
    msgs = s.messages_for_llm()
    last = msgs[-1]
    assert last["role"] == "assistant"
    parsed = json.loads(last["content"])
    assert "plan" in parsed
    assert parsed["plan"]["goal"] == "Add unit tests for utils"
    assert len(parsed["plan"]["tasks"]) == 3
    assert all(t["status"] == "pending" for t in parsed["plan"]["tasks"])
