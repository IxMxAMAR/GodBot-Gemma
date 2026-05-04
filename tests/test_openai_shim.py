"""Tests for the OpenAI-compatible /v1/chat/completions shim.

The agent loop is monkeypatched out via ``shim.run_turn`` so these tests
exercise the shim's request/response shaping in isolation — no real
LLM, no provider HTTP calls.
"""
import json

from fastapi.testclient import TestClient

from godbot.interfaces.web import build_app


def test_blocking_chat_completion(tmp_path, monkeypatch):
    """A non-streaming request returns one OpenAI-format JSON body
    whose ``choices[0].message.content`` is the agent's final answer."""
    import godbot.interfaces.openai_shim as shim

    async def fake_run_turn(*, emit, **kwargs):
        from godbot.core.events import DoneEvent, TokenEvent
        await emit(TokenEvent(text='{"thought":"x","final_answer":"hi back"}'))
        await emit(DoneEvent(step_count=1))

    monkeypatch.setattr(shim, "run_turn", fake_run_turn)

    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "godbot",
            "messages": [{"role": "user", "content": "say hi"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "hi back"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_streaming_chat_completion(tmp_path, monkeypatch):
    """Streaming request emits OpenAI-format SSE chunks; ``delta.content``
    pieces concatenate to the full final answer; terminator is ``[DONE]``."""
    import godbot.interfaces.openai_shim as shim

    async def fake_run_turn(*, emit, **kwargs):
        from godbot.core.events import DoneEvent, TokenEvent
        # Simulate partial JSON streaming — the shim must extract the
        # ``final_answer`` substring as it grows.
        for chunk in ['{"thought":"x","final_answer":"', "hello", " world", '"}']:
            await emit(TokenEvent(text=chunk))
        await emit(DoneEvent(step_count=1))

    monkeypatch.setattr(shim, "run_turn", fake_run_turn)

    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    chunks_received = []
    saw_done = False
    with c.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "godbot",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                chunks_received.append(json.loads(data))
    assert saw_done, "missing [DONE] terminator"
    contents = "".join(
        ch["choices"][0]["delta"].get("content", "") for ch in chunks_received
    )
    assert "hello world" in contents


def test_model_spec_parsing():
    """``parse_model_spec`` covers the four documented forms."""
    from godbot.interfaces.openai_shim import parse_model_spec

    assert parse_model_spec("godbot") == (None, "auto")
    assert parse_model_spec("") == (None, "auto")
    assert parse_model_spec("anthropic:claude-3-5-sonnet-latest") == (
        "anthropic",
        "claude-3-5-sonnet-latest",
    )
    assert parse_model_spec("gpt-4o") == (None, "gpt-4o")


def test_stateful_session_via_header(tmp_path, monkeypatch):
    """``X-Godbot-Session`` reuses an existing session — the agent sees
    the persisted session id, not a fresh ephemeral one."""
    import godbot.interfaces.openai_shim as shim

    async def fake_run_turn(*, emit, session, **kwargs):
        from godbot.core.events import DoneEvent, TokenEvent
        await emit(
            TokenEvent(
                text=f'{{"thought":"x","final_answer":"sid={session.id}"}}'
            )
        )
        await emit(DoneEvent(step_count=1))

    monkeypatch.setattr(shim, "run_turn", fake_run_turn)

    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    sessions_root = tmp_path / "sessions"
    app = build_app(sessions_root=sessions_root)
    c = TestClient(app)

    # Pre-create a session on disk.
    from godbot.core.session import Session

    s = Session.create(sessions_root, model="auto")
    sid = s.id

    r = c.post(
        "/v1/chat/completions",
        headers={"X-Godbot-Session": sid},
        json={
            "model": "godbot",
            "messages": [{"role": "user", "content": "what's my sid?"}],
        },
    )
    assert r.status_code == 200, r.text
    assert sid in r.json()["choices"][0]["message"]["content"]


def test_missing_messages_returns_400(tmp_path, monkeypatch):
    """An empty ``messages`` list is a client error, not a silent no-op."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.post(
        "/v1/chat/completions", json={"model": "godbot", "messages": []}
    )
    assert r.status_code == 400


def test_unknown_session_returns_404(tmp_path, monkeypatch):
    """A bogus ``X-Godbot-Session`` header yields 404 rather than
    silently creating a new session at that id."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.post(
        "/v1/chat/completions",
        headers={"X-Godbot-Session": "definitely-not-a-real-session"},
        json={
            "model": "godbot",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 404
