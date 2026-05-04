"""Generic OpenAI-compatible provider.

Covers every backend that exposes the OpenAI Chat Completions shape:

  - LM Studio (``localhost:1234/v1``, no key)
  - Ollama (``localhost:11434/v1``, no key)
  - vLLM
  - OpenAI itself
  - Together / Groq / Cerebras / OpenRouter / Mistral / Fireworks
  - Custom user-provided base URL

The class supports both tool-call protocols:

  - ``REACT_JSON`` → forwards ``response_format={"type": "json_schema", ...}``
    (LM Studio's grammar-constrained mode); same behaviour as the legacy
    :class:`godbot.core.llm.LLMClient`.
  - ``NATIVE_TOOLS`` → sends ``tools=[...]`` and parses ``tool_calls`` out
    of the streamed response, accumulating partial JSON arg strings until
    the model's stream is complete.
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

log = logging.getLogger("godbot.providers.openai_compat")


# Substring heuristic: if a model id contains any of these, default to ReAct
# instead of native tool_calls. Small/old/general-purpose models tend to lack
# reliable native tool-call support.
_REACT_HINTS = (
    "gemma",
    "tinyllama",
    "phi-2",
    "phi2",
    "qwen-0.5b",
    "qwen2-0.5b",
    "qwen2.5-0.5b",
    "stablelm",
)


class GenericOpenAICompatProvider(Provider):
    """OpenAI Chat Completions-compatible provider."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._models_cache: Optional[list[ModelInfo]] = None

    # -- helpers -------------------------------------------------------

    def _base(self) -> str:
        return self.config.base_url.rstrip("/") if self.config.base_url else ""

    def _auth_header(self) -> dict[str, str]:
        # LM Studio accepts (and ignores) any bearer; OpenAI etc. require it.
        # If no key configured, send a placeholder so LM Studio is happy.
        key = self.config.api_key or "lm-studio"
        return {"Authorization": f"Bearer {key}"}

    def _likely_native_tool_capable(self, model_id: str) -> bool:
        lid = model_id.lower()
        if any(hint in lid for hint in _REACT_HINTS):
            return False
        return True

    # -- introspection -------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        if self._models_cache is not None:
            return self._models_cache
        url = f"{self._base()}/models"
        async with httpx.AsyncClient(timeout=self.config.timeout) as c:
            r = await c.get(url, headers=self._auth_header())
            r.raise_for_status()
            data = r.json().get("data", [])
        out: list[ModelInfo] = []
        for m in data:
            mid = str(m.get("id", ""))
            if not mid:
                continue
            ctx = m.get("loaded_context_length") or m.get("context_length")
            out.append(
                ModelInfo(
                    id=mid,
                    context_length=int(ctx) if isinstance(ctx, int) else None,
                    supports_native_tools=self._likely_native_tool_capable(mid),
                    provider=self.config.name,
                )
            )
        self._models_cache = out
        return out

    async def select_model(self, requested: str) -> ModelInfo:
        requested = (requested or "auto").strip()
        models = await self.list_models()
        if not models:
            raise RuntimeError(
                f"provider {self.config.name!r} returned no models from {self._base()}/models"
            )
        if requested == "auto":
            # Legacy LM Studio behaviour: prefer Gemma if available; otherwise
            # fall back to the first model. Only LM Studio's heuristic is
            # this aggressive — for cloud providers the user almost always
            # pins ``default_model``.
            if self.config.name == "lmstudio":
                for m in models:
                    if "gemma" in m.id.lower():
                        return m
                # No gemma loaded — surface the original error so existing
                # tests / behaviour stay identical.
                raise RuntimeError(
                    "no model with 'gemma' in id; pin one in config.toml"
                )
            # Other providers: use default_model from config if not "auto", else the first.
            target = self.config.default_model
            if target and target != "auto":
                for m in models:
                    if m.id == target:
                        return m
                # Don't fail — return a synthesised ModelInfo so cloud APIs
                # that don't list models (or that list a different shape)
                # still work.
                return ModelInfo(
                    id=target,
                    supports_native_tools=self._likely_native_tool_capable(target),
                    provider=self.config.name,
                )
            return models[0]
        for m in models:
            if m.id == requested:
                return m
        # Pinned model not in /models — synthesise so the user can still try it.
        return ModelInfo(
            id=requested,
            supports_native_tools=self._likely_native_tool_capable(requested),
            provider=self.config.name,
        )

    def preferred_protocol(self, model: ModelInfo) -> str:
        if self.config.protocol in (NATIVE_TOOLS, REACT_JSON):
            return self.config.protocol  # type: ignore[return-value]
        return NATIVE_TOOLS if model.supports_native_tools else REACT_JSON

    # -- streaming -----------------------------------------------------

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

        body: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if protocol == REACT_JSON and react_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "react", "schema": react_schema, "strict": True},
            }
        elif protocol == NATIVE_TOOLS and tool_schemas:
            body["tools"] = tool_schemas
            body["tool_choice"] = "auto"

        url = f"{self._base()}/chat/completions"
        headers = self._auth_header()

        text_parts: list[str] = []
        # OpenAI streams tool calls as a sparse list keyed by ``index``;
        # each delta may carry partial JSON for ``arguments``. Accumulate
        # into a per-index slot, finalising at stream end.
        tool_acc: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        usage: dict[str, int] = {}

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=self.config.timeout)
        ) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    if cancel is not None and cancel.is_set():
                        finish_reason = "cancelled"
                        break
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        # Some providers (Groq, OpenRouter) trail a usage-only
                        # chunk with no choices.
                        u = chunk.get("usage") or {}
                        if u:
                            usage = _normalize_usage(u)
                        continue
                    ch0 = choices[0]
                    delta = ch0.get("delta") or {}
                    fr = ch0.get("finish_reason")
                    if fr:
                        finish_reason = fr
                    content = delta.get("content")
                    if content:
                        text_parts.append(content)
                        result = on_delta(content)
                        if asyncio.iscoroutine(result):
                            await result
                    tcs = delta.get("tool_calls")
                    if tcs:
                        for tc in tcs:
                            idx = int(tc.get("index", 0))
                            slot = tool_acc.setdefault(
                                idx, {"id": "", "name": "", "args_str": ""}
                            )
                            if "id" in tc and tc["id"]:
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["args_str"] += fn["arguments"]
                    # Some providers also include usage on the terminal chunk.
                    u = chunk.get("usage") or {}
                    if u:
                        usage = _normalize_usage(u)

        full_text = "".join(text_parts)

        # Build tool_calls from accumulators, parsing arg-string JSON.
        parsed_calls: list[ParsedToolCall] = []
        for idx in sorted(tool_acc.keys()):
            slot = tool_acc[idx]
            args: dict[str, Any] = {}
            if slot["args_str"]:
                try:
                    args = json.loads(slot["args_str"])
                except json.JSONDecodeError:
                    args = {"_raw": slot["args_str"]}
            parsed_calls.append(
                ParsedToolCall(
                    id=slot["id"] or f"call_{idx}",
                    name=slot["name"],
                    args=args if isinstance(args, dict) else {"_raw": slot["args_str"]},
                )
            )

        # ReAct path: parse the JSON envelope out of full_text and surface as
        # final_answer / tool_calls so the agent loop can branch uniformly.
        if protocol == REACT_JSON:
            return _parse_react_envelope(full_text, finish_reason, usage)

        # Native-tools path.
        return TurnResult(
            final_answer=(full_text if not parsed_calls else None),
            tool_calls=parsed_calls,
            raw_text=full_text,
            finish_reason=finish_reason or ("tool_calls" if parsed_calls else "stop"),
            usage=usage,
        )


def _normalize_usage(u: dict[str, Any]) -> dict[str, int]:
    """Map provider usage fields to a uniform shape.

    OpenAI uses ``prompt_tokens`` / ``completion_tokens``; some others use
    ``input_tokens`` / ``output_tokens``. Coerce both to the latter so
    downstream code only ever reads one set of names.
    """
    inp = int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
    out = int(u.get("output_tokens") or u.get("completion_tokens") or 0)
    total = int(u.get("total_tokens") or (inp + out))
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}


def _parse_react_envelope(
    full_text: str, finish_reason: str, usage: dict[str, int]
) -> TurnResult:
    """Decode a ReAct JSON envelope into a TurnResult.

    The envelope is one of::

        {"thought": "...", "final_answer": "..."}
        {"thought": "...", "action": "tool_name", "args": {...}}

    On JSON parse failure we surface ``raw_text`` so the agent loop's
    existing schema-validation branch still runs and synthesises a retry
    nudge.
    """
    res = TurnResult(raw_text=full_text, finish_reason=finish_reason or "stop", usage=usage)
    try:
        parsed = json.loads(full_text)
    except json.JSONDecodeError:
        return res
    if not isinstance(parsed, dict):
        return res
    thought = parsed.get("thought") or ""
    res.thought = str(thought) if isinstance(thought, str) else ""
    if "final_answer" in parsed:
        res.final_answer = str(parsed.get("final_answer") or "")
        return res
    action = parsed.get("action")
    if isinstance(action, str) and action:
        args = parsed.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        res.tool_calls = [ParsedToolCall(id="", name=action, args=args)]
    return res
