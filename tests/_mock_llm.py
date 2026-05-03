"""A scripted async LLM stand-in for agent loop tests."""
from __future__ import annotations
import asyncio
import json
from typing import Any, Callable, Optional


class MockLLM:
    def __init__(self, scripted_responses: list[str]) -> None:
        self._responses = list(scripted_responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_streaming(
        self,
        messages,
        on_delta: Callable[[str], Any],
        response_format: Optional[dict[str, Any]] = None,
        cancel: Optional[asyncio.Event] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        self.calls.append({"messages": messages, "response_format": response_format})
        if not self._responses:
            raise AssertionError("MockLLM: no more scripted responses")
        full = self._responses.pop(0)
        for ch in full:
            if cancel is not None and cancel.is_set():
                return ""
            r = on_delta(ch)
            if asyncio.iscoroutine(r):
                await r
        return full
