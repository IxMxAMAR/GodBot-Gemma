"""Provider registry and factory.

Importing this package exposes the public API:

  - :class:`Provider`, :class:`ProviderConfig`, :class:`ModelInfo`,
    :class:`TurnResult`, :class:`ParsedToolCall`
  - :data:`NATIVE_TOOLS`, :data:`REACT_JSON`
  - :func:`register_provider` / :func:`get_provider`
  - :func:`load_providers_from_config`

Concrete provider classes are imported lazily inside the registry so a
missing optional dep (e.g. the ``anthropic`` SDK) doesn't tank module
import for the common LM Studio path.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from godbot.core.providers.base import (
    NATIVE_TOOLS,
    REACT_JSON,
    ModelInfo,
    ParsedToolCall,
    Provider,
    ProviderConfig,
    TurnResult,
)


# name -> factory(ProviderConfig) -> Provider
_FACTORIES: dict[str, Callable[[ProviderConfig], Provider]] = {}
# name -> instantiated singleton
_INSTANCES: dict[str, Provider] = {}


def register_provider(name: str, factory: Callable[[ProviderConfig], Provider]) -> None:
    """Register a provider factory under ``name``.

    Re-registration replaces the prior entry (useful for tests).
    """
    _FACTORIES[name] = factory
    # Drop any cached instance so the new factory takes effect immediately.
    _INSTANCES.pop(name, None)


def unregister_provider(name: str) -> None:
    """Remove a provider from the registry (test helper)."""
    _FACTORIES.pop(name, None)
    _INSTANCES.pop(name, None)


def clear_providers() -> None:
    """Wipe all registrations and cached instances (test helper)."""
    _FACTORIES.clear()
    _INSTANCES.clear()


def list_registered() -> list[str]:
    return sorted(_FACTORIES.keys())


def get_provider(name: str, config: Optional[ProviderConfig] = None) -> Provider:
    """Return a provider instance for ``name``.

    The first call constructs the instance from ``config`` (or a minimal
    default config if none is supplied) and caches it; subsequent calls
    return the same instance unless ``config`` is provided, in which case
    the cache is refreshed.
    """
    if config is not None:
        factory = _FACTORIES.get(name)
        if factory is None:
            raise KeyError(f"unknown provider {name!r}")
        inst = factory(config)
        _INSTANCES[name] = inst
        return inst

    inst = _INSTANCES.get(name)
    if inst is not None:
        return inst
    factory = _FACTORIES.get(name)
    if factory is None:
        raise KeyError(f"unknown provider {name!r}")
    inst = factory(ProviderConfig(name=name))
    _INSTANCES[name] = inst
    return inst


def _resolve_api_key(entry: dict[str, Any]) -> str:
    """Read an API key from an explicit ``api_key`` field or, failing
    that, from the env var named in ``api_key_env``. Empty string if
    neither is set so cloud-providers fail loudly (rather than silently
    sending blank Authorization headers)."""
    direct = entry.get("api_key")
    if isinstance(direct, str) and direct:
        return direct
    env_name = entry.get("api_key_env")
    if isinstance(env_name, str) and env_name:
        return os.environ.get(env_name, "")
    return ""


def load_providers_from_config(providers_section: dict[str, Any]) -> dict[str, ProviderConfig]:
    """Translate a parsed ``[providers.<name>]`` TOML section into a
    name -> :class:`ProviderConfig` map.

    Unknown entries are skipped silently (matching the rest of the config
    loader's "be liberal in what you accept" stance).
    """
    out: dict[str, ProviderConfig] = {}
    if not isinstance(providers_section, dict):
        return out
    for name, entry in providers_section.items():
        if not isinstance(entry, dict):
            continue
        cfg = ProviderConfig(
            name=str(name),
            base_url=str(entry.get("base_url", "")),
            api_key=_resolve_api_key(entry),
            default_model=str(entry.get("default_model", "auto")),
            timeout=float(entry.get("timeout", 60.0)),
            protocol=entry.get("protocol") or None,
            extra={
                k: v
                for k, v in entry.items()
                if k
                not in {
                    "base_url",
                    "api_key",
                    "api_key_env",
                    "default_model",
                    "timeout",
                    "protocol",
                }
            },
        )
        out[str(name)] = cfg
    return out


def _register_builtin_providers() -> None:
    """Wire up the built-in providers. Called at module import time.

    Import the concrete classes lazily so a missing optional dep (e.g.
    ``anthropic``) doesn't break the import of this package.
    """
    def _openai_compat(cfg: ProviderConfig) -> Provider:
        from godbot.core.providers.openai_compat import GenericOpenAICompatProvider
        return GenericOpenAICompatProvider(cfg)

    # Every OpenAI-compatible flavour shares the same class.
    for name in (
        "lmstudio",
        "ollama",
        "openai",
        "groq",
        "together",
        "cerebras",
        "openrouter",
        "mistral",
        "fireworks",
        "vllm",
        "openai_compat",
    ):
        register_provider(name, _openai_compat)

    # Anthropic — optional dep, lazy-import inside the factory.
    def _anthropic(cfg: ProviderConfig) -> Provider:
        from godbot.core.providers.anthropic import AnthropicProvider
        return AnthropicProvider(cfg)

    register_provider("anthropic", _anthropic)

    # Gemini — uses raw httpx, no SDK dep.
    def _gemini(cfg: ProviderConfig) -> Provider:
        from godbot.core.providers.gemini import GeminiProvider
        return GeminiProvider(cfg)

    register_provider("gemini", _gemini)


_register_builtin_providers()


__all__ = [
    "NATIVE_TOOLS",
    "REACT_JSON",
    "ModelInfo",
    "ParsedToolCall",
    "Provider",
    "ProviderConfig",
    "TurnResult",
    "clear_providers",
    "get_provider",
    "list_registered",
    "load_providers_from_config",
    "register_provider",
    "unregister_provider",
]
