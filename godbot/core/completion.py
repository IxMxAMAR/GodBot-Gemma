"""Inline code-completion service (sub-project 12).

A separate, agent-loop-free path for Cursor-style ghost-text suggestions.
The agent loop is too expensive for keystroke-frequency calls — it loads
the full tool catalog, applies ReAct/native-tools framing, and goes
through the whole turn machinery. Inline completions need a single
short prompt → short response, sub-200ms when possible.

This module talks to OpenAI-compatible Chat Completions endpoints
directly (which covers ~12 of our 13 supported providers). Anthropic
and Gemini have their own message shapes; for v1 they fall through to
"unsupported" — Studio should route around them. Future v2 can add
provider-specific completion paths.

The public API is :func:`complete_text`. It is async because httpx is,
and the daemon's web layer is async-native.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

log = logging.getLogger("godbot.completion")


# Default cap so a buggy client typing into a textarea can't burn a model
# call's worth of tokens on every keystroke.
DEFAULT_MAX_TOKENS = 64

# We trim the prefix and suffix so the prompt stays bounded even on huge
# files. 4000 chars on each side is roughly 1k tokens — plenty of context
# for completions, well within any model's input budget.
MAX_PREFIX_CHARS = 4000
MAX_SUFFIX_CHARS = 2000


_SYSTEM_PROMPT = """\
You are a code completion engine, not a chat assistant.

The user will provide PREFIX (text before their cursor) and optionally
SUFFIX (text after their cursor). Your job is to predict the most likely
text that should appear AT THE CURSOR — the smallest plausible completion
that makes the surrounding code coherent.

Rules:
- Output ONLY the completion text. No markdown fences, no commentary,
  no quotes around it, no apology, no "here's the completion".
- Do NOT repeat the prefix. The cursor is at the end of the prefix.
- Do NOT include the suffix. Stop at the point where the user's existing
  text takes over.
- Prefer short completions (1 line) unless the natural completion clearly
  spans multiple lines (e.g. function body).
- If the prefix ends mid-token, complete that token first.
- If you genuinely cannot predict anything useful, emit an empty response.
"""


@dataclass
class CompletionResult:
    """Returned by :func:`complete_text`."""

    completion: str
    model: str
    elapsed_ms: int


def _trim_context(prefix: str, suffix: str) -> tuple[str, str]:
    """Cap prefix/suffix at known limits, keeping the side near the cursor.

    For the prefix we keep the *tail* (most recent code); for the suffix we
    keep the *head*. Truncation preserves exactly the context the model
    needs to continue from the cursor.
    """
    if len(prefix) > MAX_PREFIX_CHARS:
        prefix = prefix[-MAX_PREFIX_CHARS:]
    if len(suffix) > MAX_SUFFIX_CHARS:
        suffix = suffix[:MAX_SUFFIX_CHARS]
    return prefix, suffix


def _build_user_message(prefix: str, suffix: str, language: Optional[str]) -> str:
    """Compose the single user message that frames the FIM-style request."""
    parts: list[str] = []
    if language:
        parts.append(f"LANGUAGE: {language}")
    parts.append("PREFIX:")
    parts.append(prefix)
    if suffix:
        parts.append("")
        parts.append("SUFFIX:")
        parts.append(suffix)
    parts.append("")
    parts.append("COMPLETION:")
    return "\n".join(parts)


def _strip_completion(text: str, prefix: str) -> str:
    """Clean common LLM artefacts from the completion.

    Strips opening/closing markdown fences, an accidental repeat of the
    prefix's last line, and surrounding whitespace at the boundaries
    (but preserves internal whitespace). Trims trailing newlines so the
    insertion point is predictable.
    """
    s = text

    # Drop markdown fences if the model wrapped the answer in them.
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.rstrip()

    # If the model echoed the prefix's last line, drop it.
    last_prefix_line = prefix.rsplit("\n", 1)[-1]
    if last_prefix_line and s.startswith(last_prefix_line):
        s = s[len(last_prefix_line):]

    return s


def _is_openai_compatible(provider_name: str) -> bool:
    """True iff we can hit ``{base_url}/chat/completions`` directly.

    Anthropic and Gemini ship their own request shapes; everyone else in
    our provider set speaks OpenAI Chat Completions verbatim.
    """
    return provider_name not in {"anthropic", "gemini"}


async def _post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
) -> tuple[str, int]:
    """One non-streaming POST. Returns (assistant_text, http_status)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key or 'lm-studio'}"}
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,  # deterministic completions
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return "", r.status_code
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    return str(content), r.status_code


async def complete_text(
    *,
    prefix: str,
    suffix: str = "",
    language: Optional[str] = None,
    provider_name: str,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 30.0,
) -> CompletionResult:
    """Run a single inline-completion turn against an OpenAI-compatible provider.

    Raises :class:`ValueError` if the provider is one we don't support
    (Anthropic/Gemini for now). Bubbles ``httpx.HTTPError`` on transport
    failures so the caller can decide what to surface to the user.
    """
    if not _is_openai_compatible(provider_name):
        raise ValueError(
            f"provider {provider_name!r} not yet supported for inline completion"
        )
    if not prefix and not suffix:
        return CompletionResult(completion="", model=model, elapsed_ms=0)

    prefix_t, suffix_t = _trim_context(prefix, suffix)
    user_msg = _build_user_message(prefix_t, suffix_t, language)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    import time
    started = time.monotonic()
    raw, _ = await _post_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    cleaned = _strip_completion(raw, prefix_t)
    return CompletionResult(completion=cleaned, model=model, elapsed_ms=elapsed_ms)
