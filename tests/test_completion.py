"""Unit tests for godbot.core.completion (sub-project 12)."""
from __future__ import annotations

import httpx
import pytest
import respx

from godbot.core.completion import (
    DEFAULT_MAX_TOKENS,
    MAX_PREFIX_CHARS,
    MAX_SUFFIX_CHARS,
    _build_user_message,
    _is_openai_compatible,
    _strip_completion,
    _trim_context,
    complete_text,
)


def test_trim_context_caps_prefix_and_suffix():
    long_prefix = "a" * (MAX_PREFIX_CHARS + 50)
    long_suffix = "b" * (MAX_SUFFIX_CHARS + 50)
    p, s = _trim_context(long_prefix, long_suffix)
    assert len(p) == MAX_PREFIX_CHARS
    # Prefix tail-truncates.
    assert p == long_prefix[-MAX_PREFIX_CHARS:]
    assert len(s) == MAX_SUFFIX_CHARS
    # Suffix head-truncates.
    assert s == long_suffix[:MAX_SUFFIX_CHARS]


def test_trim_context_short_inputs_pass_through():
    p, s = _trim_context("hi", "yo")
    assert p == "hi"
    assert s == "yo"


def test_build_user_message_includes_language():
    msg = _build_user_message("def foo():", "    return 1", "python")
    assert "LANGUAGE: python" in msg
    assert "PREFIX:" in msg
    assert "SUFFIX:" in msg
    assert "COMPLETION:" in msg


def test_build_user_message_omits_suffix_block_if_empty():
    msg = _build_user_message("def foo():", "", "python")
    assert "SUFFIX:" not in msg
    assert "COMPLETION:" in msg


def test_strip_completion_removes_markdown_fences():
    raw = "```python\ndef foo():\n    pass\n```"
    out = _strip_completion(raw, prefix="")
    assert "```" not in out
    assert "def foo():" in out


def test_strip_completion_drops_repeated_prefix_last_line():
    prefix = "def add(a, b):\n    return"
    raw = "    return a + b\n"
    out = _strip_completion(raw, prefix)
    # The last prefix line is `    return` (with leading spaces), and the
    # model echoed it — the helper should drop it.
    assert out.startswith(" a + b") or out.startswith("a + b")


def test_strip_completion_handles_no_prefix_overlap():
    prefix = "x = "
    raw = "42"
    out = _strip_completion(raw, prefix)
    assert out == "42"


def test_is_openai_compatible_excludes_anthropic_gemini():
    assert _is_openai_compatible("openai") is True
    assert _is_openai_compatible("lmstudio") is True
    assert _is_openai_compatible("groq") is True
    assert _is_openai_compatible("anthropic") is False
    assert _is_openai_compatible("gemini") is False


@respx.mock
@pytest.mark.asyncio
async def test_complete_text_happy_path():
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": " a + b"}}],
                "model": "gemma-3",
            },
        )
    )
    res = await complete_text(
        prefix="def add(a, b):\n    return",
        provider_name="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key="",
        model="gemma-3",
    )
    assert res.completion == "a + b" or res.completion == " a + b"
    assert res.model == "gemma-3"
    assert res.elapsed_ms >= 0


@respx.mock
@pytest.mark.asyncio
async def test_complete_text_passes_max_tokens_and_temperature():
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "x"}}]},
        )
    )
    await complete_text(
        prefix="hi",
        provider_name="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key="",
        model="m",
        max_tokens=8,
    )
    # Inspect the recorded request body.
    import json as _json
    body = _json.loads(route.calls.last.request.content)
    assert body["max_tokens"] == 8
    assert body["temperature"] == 0.1
    assert body["stream"] is False
    assert body["model"] == "m"
    # Two messages: system + user.
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_complete_text_unsupported_provider_raises():
    with pytest.raises(ValueError):
        await complete_text(
            prefix="x",
            provider_name="anthropic",
            base_url="https://api.anthropic.com",
            api_key="k",
            model="claude-haiku",
        )


@pytest.mark.asyncio
async def test_complete_text_empty_input_returns_empty():
    res = await complete_text(
        prefix="",
        suffix="",
        provider_name="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key="",
        model="m",
    )
    assert res.completion == ""


@respx.mock
@pytest.mark.asyncio
async def test_complete_text_strips_fences_from_response():
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "```python\nprint(x)\n```"}}],
            },
        )
    )
    res = await complete_text(
        prefix="def foo():\n    ",
        provider_name="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key="",
        model="m",
    )
    assert "```" not in res.completion
    assert "print(x)" in res.completion


@respx.mock
@pytest.mark.asyncio
async def test_complete_text_default_max_tokens():
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]}),
    )
    await complete_text(
        prefix="x",
        provider_name="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key="",
        model="m",
    )
    import json as _json
    body = _json.loads(route.calls.last.request.content)
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS
