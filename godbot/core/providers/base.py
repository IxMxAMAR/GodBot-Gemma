"""Provider abstract base class and shared types.

The provider layer abstracts every supported model backend (LM Studio,
Ollama, OpenAI, Anthropic, Gemini, etc.) behind a single async interface
the agent loop can talk to. Two tool-call protocols are supported and
selected per-model: ``REACT_JSON`` (the legacy Gemma path) and
``NATIVE_TOOLS`` (Claude/OpenAI/Gemini native function calling).

Concrete providers live in sibling modules and are wired up via
``godbot.core.providers.__init__``'s registry.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# Tool-call protocol identifiers.
NATIVE_TOOLS = "native"
REACT_JSON = "react_json"


@dataclass
class ProviderConfig:
    """Generic per-provider configuration.

    ``base_url`` and ``api_key`` are populated for any HTTP-based provider;
    concrete subclasses may ignore them (e.g. an in-process provider) or
    consume ``extra`` for provider-specific knobs.
    """

    name: str
    base_url: str = ""
    api_key: str = ""
    default_model: str = "auto"
    timeout: float = 60.0
    protocol: Optional[str] = None  # if set, overrides the per-model heuristic
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Subset of model metadata the agent loop cares about.

    ``supports_native_tools`` drives the protocol heuristic in
    :meth:`Provider.preferred_protocol`. ``supports_vision`` is forward-
    looking and currently unused; carried to keep the abstraction stable
    when the multimodal sub-project lands.
    """

    id: str
    context_length: Optional[int] = None
    supports_native_tools: bool = False
    supports_vision: bool = False
    provider: str = ""


@dataclass
class ParsedToolCall:
    """A single tool invocation extracted from a model turn."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class TurnResult:
    """Unified result of one model turn across protocols.

    Exactly one of ``final_answer`` and ``tool_calls`` is populated when the
    turn produced a usable response. ``raw_text`` always holds the model's
    full text output for logging / replay (in REACT_JSON it's the JSON
    envelope; in NATIVE_TOOLS it's the assistant's visible content).
    """

    final_answer: Optional[str] = None
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    raw_text: str = ""
    thought: str = ""
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class Provider(ABC):
    """Abstract base class for all model providers."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    # -- introspection -------------------------------------------------

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return models available from this provider."""

    @abstractmethod
    async def select_model(self, requested: str) -> ModelInfo:
        """Resolve ``requested`` (which may be ``"auto"`` or a specific id)
        to a concrete :class:`ModelInfo`.

        Raises ``RuntimeError`` if no matching model is available.
        """

    @abstractmethod
    def preferred_protocol(self, model: ModelInfo) -> str:
        """Return the tool-call protocol the agent loop should use with
        ``model``: either :data:`NATIVE_TOOLS` or :data:`REACT_JSON`.
        """

    # -- the one entry-point the agent loop calls ----------------------

    @abstractmethod
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
        """Run one model turn and return a :class:`TurnResult`.

        ``on_delta`` is called with each visible content chunk for UI
        streaming. In ``NATIVE_TOOLS`` mode, tool-call argument deltas
        (which are partial JSON) are NOT forwarded — only the assistant's
        visible text content is streamed.
        """
