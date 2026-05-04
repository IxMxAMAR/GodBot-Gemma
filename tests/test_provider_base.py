"""Tests for the Provider ABC, dataclasses, and registry plumbing.

No HTTP, no concrete providers. The OpenAI-compat / Anthropic / Gemini
classes get exercised in their own files with respx.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import pytest

from godbot.core.providers import (
    NATIVE_TOOLS,
    REACT_JSON,
    ModelInfo,
    ParsedToolCall,
    Provider,
    ProviderConfig,
    TurnResult,
    clear_providers,
    get_provider,
    list_registered,
    load_providers_from_config,
    register_provider,
    unregister_provider,
)
import godbot.core.providers as providers_pkg


def _restore_builtins():
    """Re-register the built-in providers after a test wipes the registry."""
    providers_pkg._register_builtin_providers()


# ---- ProviderConfig / ModelInfo / TurnResult ----------------------------


def test_provider_config_defaults():
    cfg = ProviderConfig(name="lmstudio")
    assert cfg.name == "lmstudio"
    assert cfg.base_url == ""
    assert cfg.api_key == ""
    assert cfg.default_model == "auto"
    assert cfg.timeout == 60.0
    assert cfg.protocol is None
    assert cfg.extra == {}


def test_model_info_defaults():
    m = ModelInfo(id="x")
    assert m.id == "x"
    assert m.context_length is None
    assert m.supports_native_tools is False
    assert m.supports_vision is False
    assert m.provider == ""


def test_turn_result_default_lists_are_independent():
    a = TurnResult()
    b = TurnResult()
    a.tool_calls.append(ParsedToolCall(id="1", name="x", args={}))
    assert b.tool_calls == []  # no shared default mutable state


def test_protocol_constants_distinct():
    assert NATIVE_TOOLS != REACT_JSON


# ---- ABC contract --------------------------------------------------------


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        # Cannot instantiate the ABC directly.
        Provider(ProviderConfig(name="x"))  # type: ignore[abstract]


class _Stub(Provider):
    async def list_models(self):
        return [ModelInfo(id="m", provider=self.config.name)]

    async def select_model(self, requested):
        return ModelInfo(id=requested or "m", provider=self.config.name)

    def preferred_protocol(self, model):
        return NATIVE_TOOLS if model.supports_native_tools else REACT_JSON

    async def complete_streaming(
        self, model, messages, on_delta, protocol,
        tool_schemas=None, react_schema=None, cancel=None,
        temperature=0.7, max_tokens=4096,
    ):
        return TurnResult(final_answer="ok", raw_text="ok", finish_reason="stop")


def test_concrete_provider_satisfies_abc():
    p = _Stub(ProviderConfig(name="stub"))
    assert isinstance(p, Provider)


@pytest.mark.asyncio
async def test_concrete_provider_methods_callable():
    p = _Stub(ProviderConfig(name="stub"))
    models = await p.list_models()
    assert models[0].id == "m"
    chosen = await p.select_model("auto")
    assert chosen.id == "auto"
    assert p.preferred_protocol(chosen) == REACT_JSON
    assert p.preferred_protocol(ModelInfo(id="x", supports_native_tools=True)) == NATIVE_TOOLS

    res = await p.complete_streaming(
        chosen, [], lambda t: None, REACT_JSON,
    )
    assert res.final_answer == "ok"


# ---- Registry ------------------------------------------------------------


def test_builtin_providers_registered():
    names = list_registered()
    for n in ("lmstudio", "ollama", "openai", "anthropic", "gemini",
              "groq", "together", "openrouter"):
        assert n in names, f"{n!r} not registered"


def test_register_and_get_provider():
    try:
        clear_providers()
        register_provider("stub", lambda cfg: _Stub(cfg))
        p = get_provider("stub")
        assert isinstance(p, _Stub)
        assert p.config.name == "stub"
        # cached singleton
        assert get_provider("stub") is p
    finally:
        clear_providers()
        _restore_builtins()


def test_get_provider_with_explicit_config_refreshes_instance():
    try:
        clear_providers()
        register_provider("stub", lambda cfg: _Stub(cfg))
        p1 = get_provider("stub")
        p2 = get_provider("stub", ProviderConfig(name="stub", base_url="http://x"))
        assert p1 is not p2
        assert p2.config.base_url == "http://x"
    finally:
        clear_providers()
        _restore_builtins()


def test_get_unknown_provider_raises():
    try:
        clear_providers()
        with pytest.raises(KeyError):
            get_provider("does-not-exist")
    finally:
        clear_providers()
        _restore_builtins()


def test_unregister_provider():
    try:
        clear_providers()
        register_provider("stub", lambda cfg: _Stub(cfg))
        get_provider("stub")  # cache it
        unregister_provider("stub")
        with pytest.raises(KeyError):
            get_provider("stub")
    finally:
        clear_providers()
        _restore_builtins()


# ---- load_providers_from_config -----------------------------------------


def test_load_providers_basic():
    section = {
        "lmstudio": {"base_url": "http://localhost:1234/v1", "default_model": "auto"},
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "TEST_OPENAI_KEY_DOES_NOT_EXIST",
            "default_model": "gpt-4o",
        },
    }
    cfgs = load_providers_from_config(section)
    assert set(cfgs.keys()) == {"lmstudio", "openai"}
    assert cfgs["lmstudio"].base_url == "http://localhost:1234/v1"
    assert cfgs["openai"].default_model == "gpt-4o"
    # missing env var -> empty key
    assert cfgs["openai"].api_key == ""


def test_load_providers_reads_env_for_api_key(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-fake")
    section = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "TEST_PROVIDER_KEY",
        },
    }
    cfgs = load_providers_from_config(section)
    assert cfgs["openai"].api_key == "sk-fake"


def test_load_providers_explicit_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("OTHER_KEY", "from-env")
    section = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "from-config",
            "api_key_env": "OTHER_KEY",
        },
    }
    cfgs = load_providers_from_config(section)
    assert cfgs["openai"].api_key == "from-config"


def test_load_providers_skips_non_dict_entries():
    section = {"good": {"base_url": "x"}, "bad": "not-a-dict", "other": 42}
    cfgs = load_providers_from_config(section)
    assert set(cfgs.keys()) == {"good"}


def test_load_providers_handles_non_dict_root():
    assert load_providers_from_config(None) == {}  # type: ignore[arg-type]
    assert load_providers_from_config("nope") == {}  # type: ignore[arg-type]


def test_load_providers_extra_fields_preserved():
    section = {"custom": {"base_url": "x", "weird_knob": True}}
    cfgs = load_providers_from_config(section)
    assert cfgs["custom"].extra == {"weird_knob": True}


def test_load_providers_protocol_pin():
    section = {"groq": {"base_url": "x", "protocol": "react_json"}}
    cfgs = load_providers_from_config(section)
    assert cfgs["groq"].protocol == "react_json"
