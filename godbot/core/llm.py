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
