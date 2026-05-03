from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
import httpx

log = logging.getLogger("godbot.llm")


@dataclass
class ModelInfo:
    id: str
    context_length: Optional[int]


class LLMClient:
    def __init__(self, base_url: str, model: str = "auto", api_key: str = "lm-studio") -> None:
        self.base_url = base_url.rstrip("/")
        self.model_pref = model
        self.api_key = api_key
        self._info: Optional[ModelInfo] = None

    def probe(self) -> ModelInfo:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
            data = r.json().get("data", [])

        if self.model_pref == "auto":
            for m in data:
                if "gemma" in m.get("id", "").lower():
                    self._info = ModelInfo(id=m["id"], context_length=m.get("loaded_context_length"))
                    return self._info
            raise RuntimeError("no model with 'gemma' in id; pin one in config.toml")
        for m in data:
            if m.get("id") == self.model_pref:
                self._info = ModelInfo(id=m["id"], context_length=m.get("loaded_context_length"))
                return self._info
        raise RuntimeError(f"pinned model {self.model_pref!r} not loaded in LM Studio")

    @property
    def info(self) -> ModelInfo:
        if self._info is None:
            return self.probe()
        return self._info

    async def complete_streaming(
        self,
        messages: list[dict[str, Any]],
        on_delta: Callable[[str], Any],
        response_format: Optional[dict[str, Any]] = None,
        cancel: Optional[asyncio.Event] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """POST /v1/chat/completions with stream=true.

        Accumulates content deltas, awaits on_delta for each. Returns full
        assistant content. If cancel is set during streaming, closes the HTTP
        response and returns the partial content received so far.
        """
        if cancel is not None and cancel.is_set():
            return ""

        body: dict[str, Any] = {
            "model": self.info.id,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format

        full_parts: list[str] = []
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0)) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    if cancel is not None and cancel.is_set():
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        full_parts.append(delta)
                        result = on_delta(delta)
                        if asyncio.iscoroutine(result):
                            await result
        return "".join(full_parts)
