"""Tests for workspace-scoped memory: save_note auto-tags, recall_notes filters,
and the agent loop injects prior workspace notes into the system prompt."""
from __future__ import annotations
import asyncio
import json
import pytest

from godbot.core.registry import DEFAULT
from godbot.core.workspace import Workspace, set_workspace, _current
from godbot.tools.memory import load_recent_workspace_notes
import godbot.tools  # noqa: F401 — trigger discovery


@pytest.fixture
def workspace_token(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")
    ws = Workspace.of(str(tmp_path))
    token = _current.set(ws)
    yield ws
    _current.reset(token)


def test_save_note_tags_workspace(workspace_token, tmp_path):
    DEFAULT.execute("save_note", {"content": "alpha", "tags": []})
    notes_dir = tmp_path / ".godbot" / "notes"
    files = list(notes_dir.glob("*.json"))
    assert files, "save_note didn't write a file"
    data = json.loads(files[0].read_text())
    assert data["workspace"] == str(workspace_token.root)


def test_save_note_no_workspace_leaves_workspace_null(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    DEFAULT.execute("save_note", {"content": "global note", "tags": []})
    notes_dir = tmp_path / ".godbot" / "notes"
    files = list(notes_dir.glob("*.json"))
    data = json.loads(files[0].read_text())
    assert data["workspace"] is None


def test_recall_notes_filters_by_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ws_a = Workspace.of(str(tmp_path / "a"))
    ws_b = Workspace.of(str(tmp_path / "b"))

    token_a = _current.set(ws_a)
    DEFAULT.execute("save_note", {"content": "alpha-a", "tags": []})
    _current.reset(token_a)

    token_b = _current.set(ws_b)
    DEFAULT.execute("save_note", {"content": "alpha-b", "tags": []})
    out = DEFAULT.execute("recall_notes", {"query": "alpha"})
    _current.reset(token_b)
    assert "alpha-b" in out
    assert "alpha-a" not in out


def test_recall_notes_all_workspaces_includes_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ws_a = Workspace.of(str(tmp_path / "a"))
    ws_b = Workspace.of(str(tmp_path / "b"))

    token_a = _current.set(ws_a)
    DEFAULT.execute("save_note", {"content": "alpha-a", "tags": []})
    _current.reset(token_a)

    token_b = _current.set(ws_b)
    DEFAULT.execute("save_note", {"content": "alpha-b", "tags": []})
    out = DEFAULT.execute("recall_notes", {"query": "alpha", "all_workspaces": True})
    _current.reset(token_b)
    assert "alpha-a" in out
    assert "alpha-b" in out


def test_load_recent_workspace_notes(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")

    (tmp_path / "ws1").mkdir()
    ws = Workspace.of(str(tmp_path / "ws1"))
    token = _current.set(ws)
    DEFAULT.execute("save_note", {"content": "first thing", "tags": []})
    DEFAULT.execute("save_note", {"content": "second thing", "tags": ["journal:auto"]})
    _current.reset(token)

    block = load_recent_workspace_notes(str(ws.root))
    assert "<workspace_memory>" in block
    assert "first thing" in block
    assert "second thing" in block
    assert str(ws.root) in block


def test_load_recent_workspace_notes_filters_other_workspaces(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")

    (tmp_path / "wsA").mkdir()
    (tmp_path / "wsB").mkdir()
    ws_a = Workspace.of(str(tmp_path / "wsA"))
    ws_b = Workspace.of(str(tmp_path / "wsB"))

    t = _current.set(ws_a)
    DEFAULT.execute("save_note", {"content": "from-A", "tags": []})
    _current.reset(t)
    t = _current.set(ws_b)
    DEFAULT.execute("save_note", {"content": "from-B", "tags": []})
    _current.reset(t)

    block = load_recent_workspace_notes(str(ws_a.root))
    assert "from-A" in block
    assert "from-B" not in block


def test_auto_journal_tool(workspace_token, tmp_path):
    DEFAULT.execute("auto_journal", {"summary": "User asked X. I did Y. Z is pending."})
    notes_dir = tmp_path / ".godbot" / "notes"
    files = list(notes_dir.glob("*.json"))
    data = json.loads(files[0].read_text())
    assert data["content"] == "User asked X. I did Y. Z is pending."
    assert "journal:auto" in data["tags"]
    assert data["workspace"] == str(workspace_token.root)


@pytest.mark.asyncio
async def test_agent_injects_workspace_memory_into_system_prompt(tmp_path, monkeypatch):
    """Agent loop on first turn appends recent workspace notes to the system prompt."""
    import json as _json
    from godbot.core.agent import run_turn
    from godbot.core.registry import Registry
    from godbot.core.session import Session
    from tests._mock_llm import MockLLM

    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    monkeypatch.setenv("GODBOT_ACTIVE_SESSION", "")

    # Pre-seed a note in the workspace.
    (tmp_path / "ws").mkdir()
    ws = Workspace.of(str(tmp_path / "ws"))
    token = _current.set(ws)
    DEFAULT.execute("save_note", {"content": "prior progress: implemented foo", "tags": []})
    _current.reset(token)

    reg = Registry()
    s = Session.create(
        root=tmp_path / "sessions",
        model="m",
        workspace_root=str(ws.root),
    )
    s.append_user("hi")

    captured_prompt = {"text": ""}

    class CapturingLLM(MockLLM):
        async def complete_streaming(self, messages, on_delta, response_format=None,
                                     cancel=None, temperature=0.7, max_tokens=4096):
            for m in messages:
                if m.get("role") == "system":
                    captured_prompt["text"] = m.get("content", "")
                    break
            return await super().complete_streaming(
                messages, on_delta, response_format, cancel, temperature, max_tokens,
            )

    llm = CapturingLLM([_json.dumps({"thought": "ok", "final_answer": "hi back"})])

    async def emit(_ev):
        pass

    await run_turn(
        llm=llm, session=s, registry=reg, emit=emit,
        cancel=asyncio.Event(), max_steps=3, max_context=10000,
        system_prompt="BASE_PROMPT",
    )
    assert "BASE_PROMPT" in captured_prompt["text"]
    assert "<workspace_memory>" in captured_prompt["text"]
    assert "prior progress: implemented foo" in captured_prompt["text"]
