"""Anthropic provider — native tool_use via the Messages API.

Talks to ``POST /v1/messages`` with ``tools=[...]`` and parses streamed
content blocks. Uses raw ``httpx`` (no SDK on the hot path) so respx
mocks plug in cleanly and we stay in control of the wire format. The
``anthropic`` SDK is still a soft dep declared in ``[providers]`` for
users who want the full SDK; this module imports it lazily inside
:meth:`AnthropicProvider.list_models` only.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

import httpx

from godbot.core.providers.base import (
    NATIVE_TOOLS,
    REACT_JSON,
    ModelInfo,
    ParsedToolCall,
    Provider,
    ProviderConfig,
    TurnResult,
)

log = logging.getLogger("godbot.providers.anthropic")


_ANTHROPIC_BASE = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"

# Curated list — Anthropic doesn't expose a /models endpoint with context
# lengths in the same shape OpenAI does, so we ship a known set. Models
# not in this list still work — ``select_model`` synthesises a
# ModelInfo for unknown ids.
_KNOWN_MODELS: tuple[tuple[str, int], ...] = (
    ("claude-opus-4-5", 200_000),
    ("claude-sonnet-4-5", 200_000),
    ("claude-haiku-4-5", 200_000),
    ("claude-3-5-sonnet-latest", 200_000),
    ("claude-3-5-haiku-latest", 200_000),
    ("claude-3-opus-latest", 200_000),
)


class AnthropicProvider(Provider):
    """Anthropic Messages API provider with native tool_use."""

    def _base(self) -> str:
        return (self.config.base_url or _ANTHROPIC_BASE).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=mid,
                context_length=ctx,
                supports_native_tools=True,
                provider=self.config.name,
            )
            for mid, ctx in _KNOWN_MODELS
        ]

    async def select_model(self, requested: str) -> ModelInfo:
        requested = (requested or "auto").strip()
        if requested == "auto":
            target = self.config.default_model
            if not target or target == "auto":
                target = "claude-3-5-sonnet-latest"
            requested = target
        for mid, ctx in _KNOWN_MODELS:
            if mid == requested:
                return ModelInfo(
                    id=mid,
                    context_length=ctx,
                    supports_native_tools=True,
                    provider=self.config.name,
                )
        # Unknown but plausibly valid — synth a record.
        return ModelInfo(
            id=requested,
            context_length=200_000,
            supports_native_tools=True,
            provider=self.config.name,
        )

    def preferred_protocol(self, model: ModelInfo) -> str:
        if self.config.protocol in (NATIVE_TOOLS, REACT_JSON):
            return self.config.protocol  # type: ignore[return-value]
        # Anthropic's strength is native tool_use; default there.
        return NATIVE_TOOLS

    async def complete_streaming(
        self,
        model: ModelInfo,
        messages: list[dict[str, Any]],
        on_delta: Callable[[str], Any],
        protocol: str,
        tool_schemas: Optional[list[dict]] = None,
        react_schema: Optional[dict] = None,
        cancel: Optional[asyncio.Event] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> TurnResult:
        if cancel is not None and cancel.is_set():
            return TurnResult(finish_reason="cancelled")

        # Anthropic's wire format separates ``system`` from ``messages``
        # and forbids consecutive same-role turns. Translate the OpenAI-
        # shaped ``messages`` we get from the agent loop.
        system_text, anth_messages = _translate_messages(messages)

        body: dict[str, Any] = {
            "model": model.id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "messages": anth_messages,
        }
        if system_text:
            body["system"] = system_text

        if protocol == NATIVE_TOOLS and tool_schemas:
            body["tools"] = _openai_tools_to_anthropic(tool_schemas)

        url = f"{self._base()}/messages"

        text_parts: list[str] = []
        # Anthropic streams as a sequence of named SSE events (message_start,
        # content_block_start, content_block_delta, content_block_stop,
        # message_delta, message_stop). Tool calls arrive as
        # content_block(type=tool_use) with ``input`` filled out via
        # input_json_delta partial-JSON deltas.
        tool_blocks: list[dict[str, Any]] = []
        current_block: Optional[dict[str, Any]] = None
        finish_reason = ""
        usage: dict[str, int] = {}

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=self.config.timeout)
        ) as client:
            async with client.stream("POST", url, json=body, headers=self._headers()) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    if cancel is not None and cancel.is_set():
                        finish_reason = "cancelled"
                        break
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if not data:
                        continue
                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    etype = evt.get("type", "")
                    if etype == "message_start":
                        msg = evt.get("message") or {}
                        u = msg.get("usage") or {}
                        if u:
                            usage = _anthropic_usage(u)
                    elif etype == "content_block_start":
                        block = evt.get("content_block") or {}
                        bt = block.get("type")
                        if bt == "tool_use":
                            current_block = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "args_str": "",
                            }
                            tool_blocks.append(current_block)
                        else:
                            current_block = None
                    elif etype == "content_block_delta":
                        delta = evt.get("delta") or {}
                        dt = delta.get("type")
                        if dt == "text_delta":
                            chunk = delta.get("text", "")
                            if chunk:
                                text_parts.append(chunk)
                                r = on_delta(chunk)
                                if asyncio.iscoroutine(r):
                                    await r
                        elif dt == "input_json_delta" and current_block is not None:
                            current_block["args_str"] += delta.get("partial_json", "")
                    elif etype == "content_block_stop":
                        current_block = None
                    elif etype == "message_delta":
                        d = evt.get("delta") or {}
                        sr = d.get("stop_reason")
                        if sr:
                            finish_reason = sr
                        u = evt.get("usage") or {}
                        if u:
                            # output_tokens lives on message_delta.usage in
                            # Anthropic's stream; merge with the input_tokens
                            # we captured at message_start. Only update the
                            # token fields actually present in this delta —
                            # message_delta.usage typically only carries
                            # output_tokens, so a naive merge would zero out
                            # the input_tokens we got from message_start.
                            if "input_tokens" in u:
                                usage["input_tokens"] = int(u.get("input_tokens") or 0)
                            if "output_tokens" in u:
                                usage["output_tokens"] = int(u.get("output_tokens") or 0)
                            usage["total_tokens"] = (
                                usage.get("input_tokens", 0)
                                + usage.get("output_tokens", 0)
                            )
                    elif etype == "message_stop":
                        break

        full_text = "".join(text_parts)

        parsed_calls: list[ParsedToolCall] = []
        for tb in tool_blocks:
            args: dict[str, Any] = {}
            if tb["args_str"]:
                try:
                    args = json.loads(tb["args_str"])
                except json.JSONDecodeError:
                    args = {"_raw": tb["args_str"]}
            if not isinstance(args, dict):
                args = {"_raw": tb["args_str"]}
            parsed_calls.append(
                ParsedToolCall(id=tb["id"], name=tb["name"], args=args)
            )

        if protocol == REACT_JSON:
            from godbot.core.providers.openai_compat import _parse_react_envelope
            return _parse_react_envelope(full_text, finish_reason or "stop", usage)

        return TurnResult(
            final_answer=(full_text if not parsed_calls else None),
            tool_calls=parsed_calls,
            raw_text=full_text,
            finish_reason=finish_reason or ("tool_use" if parsed_calls else "end_turn"),
            usage=usage,
        )


def _translate_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Split OpenAI-shaped messages into (system_text, anthropic_messages).

    Anthropic requires alternating ``user`` / ``assistant`` roles and
    holds ``system`` separately. The agent loop hands us a leading
    ``system`` message followed by alternating ``user``/``assistant``
    turns; collapse adjacent same-role turns by joining their content
    strings on a newline (Anthropic rejects two consecutive ``user``
    turns).
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        if role not in ("user", "assistant"):
            continue
        if out and out[-1]["role"] == role:
            # Collapse consecutive same-role turns.
            prev = out[-1]
            prev["content"] = (prev["content"] + "\n\n" + str(content)).strip()
            continue
        out.append({"role": role, "content": str(content)})
    return ("\n\n".join(system_parts), out)


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI's ``[{type:"function", function:{name,description,parameters}}]``
    shape into Anthropic's ``[{name, description, input_schema}]`` shape."""
    out: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") or {}
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object"},
            }
        )
    return out


def _anthropic_usage(u: dict[str, Any]) -> dict[str, int]:
    inp = int(u.get("input_tokens") or 0)
    out = int(u.get("output_tokens") or 0)
    total = inp + out
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}
