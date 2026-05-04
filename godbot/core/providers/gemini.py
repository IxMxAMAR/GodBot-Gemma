"""Gemini provider — Google Generative Language API via raw httpx.

We deliberately don't depend on ``google-generativeai``: it pulls a large
transitive set (grpcio, protobuf, googleapis-common-protos…) for what's
ultimately a JSON-over-HTTPS API. Going direct keeps cold start fast,
the wire format under our control, and respx mocks straightforward.

Endpoint: ``POST {base_url}/v1beta/models/{model}:streamGenerateContent``
with ``?key=<API_KEY>&alt=sse``. Tool calls arrive as
``parts[].functionCall``; tool results are submitted by the caller as
``parts[].functionResponse``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional
from urllib.parse import quote

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

log = logging.getLogger("godbot.providers.gemini")


_GEMINI_BASE = "https://generativelanguage.googleapis.com"

_KNOWN_MODELS: tuple[tuple[str, int], ...] = (
    ("gemini-2.5-pro", 2_000_000),
    ("gemini-2.5-flash", 1_000_000),
    ("gemini-2.0-flash", 1_000_000),
    ("gemini-2.0-flash-exp", 1_000_000),
    ("gemini-1.5-pro", 2_000_000),
    ("gemini-1.5-flash", 1_000_000),
)


class GeminiProvider(Provider):
    """Google Gemini provider with native function-calling."""

    def _base(self) -> str:
        return (self.config.base_url or _GEMINI_BASE).rstrip("/")

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
                target = "gemini-2.0-flash"
            requested = target
        for mid, ctx in _KNOWN_MODELS:
            if mid == requested:
                return ModelInfo(
                    id=mid, context_length=ctx,
                    supports_native_tools=True, provider=self.config.name,
                )
        return ModelInfo(
            id=requested, context_length=1_000_000,
            supports_native_tools=True, provider=self.config.name,
        )

    def preferred_protocol(self, model: ModelInfo) -> str:
        if self.config.protocol in (NATIVE_TOOLS, REACT_JSON):
            return self.config.protocol  # type: ignore[return-value]
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

        system_text, contents = _translate_messages_to_contents(messages)

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        if protocol == NATIVE_TOOLS and tool_schemas:
            body["tools"] = [
                {"functionDeclarations": _openai_tools_to_gemini(tool_schemas)}
            ]

        url = (
            f"{self._base()}/v1beta/models/{quote(model.id, safe='')}"
            f":streamGenerateContent"
        )
        params: dict[str, str] = {"alt": "sse"}
        if self.config.api_key:
            params["key"] = self.config.api_key

        text_parts: list[str] = []
        parsed_calls: list[ParsedToolCall] = []
        finish_reason = ""
        usage: dict[str, int] = {}
        seen_call_ids: set[str] = set()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=self.config.timeout)
        ) as client:
            async with client.stream("POST", url, json=body, params=params) as resp:
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

                    candidates = evt.get("candidates") or []
                    if candidates:
                        cand0 = candidates[0]
                        fr = cand0.get("finishReason")
                        if fr:
                            finish_reason = fr
                        content = cand0.get("content") or {}
                        for part in content.get("parts") or []:
                            txt = part.get("text")
                            if txt:
                                text_parts.append(txt)
                                r = on_delta(txt)
                                if asyncio.iscoroutine(r):
                                    await r
                            fc = part.get("functionCall")
                            if fc:
                                name = fc.get("name", "")
                                args = fc.get("args") or {}
                                if not isinstance(args, dict):
                                    args = {}
                                # Gemini doesn't always assign call ids;
                                # synthesise a stable one to avoid collisions.
                                cid = fc.get("id") or f"call_{name}_{len(parsed_calls)}"
                                if cid in seen_call_ids:
                                    cid = f"{cid}_{len(parsed_calls)}"
                                seen_call_ids.add(cid)
                                parsed_calls.append(
                                    ParsedToolCall(id=cid, name=name, args=args)
                                )

                    um = evt.get("usageMetadata") or {}
                    if um:
                        usage = _gemini_usage(um)

        full_text = "".join(text_parts)

        if protocol == REACT_JSON:
            from godbot.core.providers.openai_compat import _parse_react_envelope
            return _parse_react_envelope(full_text, finish_reason or "STOP", usage)

        return TurnResult(
            final_answer=(full_text if not parsed_calls else None),
            tool_calls=parsed_calls,
            raw_text=full_text,
            finish_reason=finish_reason or ("tool_calls" if parsed_calls else "stop"),
            usage=usage,
        )


def _translate_messages_to_contents(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split into (system_text, gemini_contents).

    Gemini uses ``role: "user"`` and ``role: "model"`` (note: not
    "assistant"). System prompts go into ``systemInstruction`` separately.
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
        gem_role = "user" if role == "user" else "model" if role == "assistant" else None
        if gem_role is None:
            continue
        if out and out[-1]["role"] == gem_role:
            # Collapse consecutive same-role turns by appending a part.
            out[-1]["parts"].append({"text": str(content)})
            continue
        out.append({"role": gem_role, "parts": [{"text": str(content)}]})
    return ("\n\n".join(system_parts), out)


def _openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI ``tools=[...]`` shape into Gemini's
    ``functionDeclarations`` shape: a list of ``{name, description,
    parameters}``. Keys mostly line up, so this is a flatten."""
    out: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") or {}
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object"},
            }
        )
    return out


def _gemini_usage(um: dict[str, Any]) -> dict[str, int]:
    inp = int(um.get("promptTokenCount") or 0)
    out = int(um.get("candidatesTokenCount") or 0)
    total = int(um.get("totalTokenCount") or (inp + out))
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}
