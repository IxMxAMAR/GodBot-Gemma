"""Provider/model pricing table (sub-project 23).

Lookups dollar cost from accumulated token usage. Bundled defaults
reflect publicly-listed prices as of early 2026 for the major cloud
providers; users can override or extend via
``~/.godbot/pricing.toml``::

    [openai."gpt-4o"]
    input_per_1m = 2.50
    output_per_1m = 10.00

    [anthropic."claude-haiku-4-5-20251001"]
    input_per_1m = 1.00
    output_per_1m = 5.00

The defaults are best-effort and may drift; do NOT use this as billing
truth — it's a "running cost so far" hint for the UI, nothing more.
Local providers (LM Studio, Ollama, vLLM) are pinned at $0.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("godbot.pricing")


@dataclass
class ModelRate:
    """Dollar cost per 1M input/output tokens for one model."""

    input_per_1m: float
    output_per_1m: float


# Known cloud-provider pricing snapshot (USD per 1M tokens). The table
# is intentionally small; users with bespoke models or contracts add
# rows via ~/.godbot/pricing.toml. Local providers stay at $0.
_BUILTIN_RATES: dict[str, dict[str, ModelRate]] = {
    "lmstudio": {},  # local — always free
    "ollama": {},    # local — always free
    "vllm": {},      # self-hosted — always free
    "openai": {
        "gpt-4o": ModelRate(2.50, 10.00),
        "gpt-4o-mini": ModelRate(0.15, 0.60),
        "gpt-4.1": ModelRate(2.00, 8.00),
        "gpt-4.1-mini": ModelRate(0.40, 1.60),
        "o1-mini": ModelRate(1.10, 4.40),
        "o1": ModelRate(15.00, 60.00),
    },
    "anthropic": {
        "claude-3-5-haiku-latest": ModelRate(1.00, 5.00),
        "claude-3-5-sonnet-latest": ModelRate(3.00, 15.00),
        "claude-haiku-4-5-20251001": ModelRate(1.00, 5.00),
        "claude-sonnet-4-6": ModelRate(3.00, 15.00),
        "claude-opus-4-7": ModelRate(15.00, 75.00),
    },
    "gemini": {
        "gemini-2.5-flash": ModelRate(0.30, 2.50),
        "gemini-2.5-pro": ModelRate(1.25, 10.00),
    },
    "groq": {
        "llama-3.3-70b-versatile": ModelRate(0.59, 0.79),
        "llama-3.1-8b-instant": ModelRate(0.05, 0.08),
        "llama-3.1-70b-versatile": ModelRate(0.59, 0.79),
        "deepseek-r1-distill-llama-70b": ModelRate(0.75, 0.99),
        "qwen-2.5-32b": ModelRate(0.79, 0.79),
        "mixtral-8x7b-32768": ModelRate(0.24, 0.24),
    },
    "together": {
        # Together's published per-1M rates as of early 2026 (best-effort
        # snapshot — users should override via ~/.godbot/pricing.toml).
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelRate(0.88, 0.88),
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelRate(0.88, 0.88),
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelRate(0.18, 0.18),
        "Qwen/Qwen2.5-Coder-32B-Instruct": ModelRate(0.80, 0.80),
        "deepseek-ai/DeepSeek-V3": ModelRate(1.25, 1.25),
        "deepseek-ai/DeepSeek-R1": ModelRate(3.00, 7.00),
    },
    "cerebras": {
        "llama-3.3-70b": ModelRate(0.85, 1.20),
        "llama-3.1-8b": ModelRate(0.10, 0.10),
        "llama3.1-70b": ModelRate(0.85, 1.20),
        "qwen-3-32b": ModelRate(0.40, 0.80),
    },
    "openrouter": {
        # OpenRouter passes through provider rates with a small markup.
        # These mirror the underlying-provider snapshot for the most
        # commonly-routed models.
        "anthropic/claude-3.5-sonnet": ModelRate(3.00, 15.00),
        "openai/gpt-4o": ModelRate(2.50, 10.00),
        "openai/gpt-4o-mini": ModelRate(0.15, 0.60),
        "meta-llama/llama-3.3-70b-instruct": ModelRate(0.45, 0.45),
        "google/gemini-2.5-flash": ModelRate(0.30, 2.50),
    },
    "mistral": {
        "mistral-large-latest": ModelRate(2.00, 6.00),
        "mistral-small-latest": ModelRate(0.20, 0.60),
        "open-mistral-nemo": ModelRate(0.15, 0.15),
        "codestral-latest": ModelRate(0.30, 0.90),
    },
    "fireworks": {
        # Fireworks uses /per-token billing; converted to per-1M.
        "accounts/fireworks/models/llama-v3p3-70b-instruct": ModelRate(0.90, 0.90),
        "accounts/fireworks/models/qwen2p5-coder-32b-instruct": ModelRate(0.90, 0.90),
        "accounts/fireworks/models/deepseek-v3": ModelRate(0.90, 0.90),
        "accounts/fireworks/models/deepseek-r1": ModelRate(3.00, 8.00),
    },
}


# Local providers are cost-free regardless of model id.
_FREE_PROVIDERS = {"lmstudio", "ollama", "vllm"}


@dataclass
class CostBreakdown:
    """Result of :func:`compute_cost`."""

    usd: float
    input_usd: float
    output_usd: float
    rate: Optional[ModelRate]
    matched: bool          # True iff we found a real rate; False = $0 fallback
    provider: str
    model: str

    def to_dict(self) -> dict:
        return {
            "usd": round(self.usd, 6),
            "input_usd": round(self.input_usd, 6),
            "output_usd": round(self.output_usd, 6),
            "rate": (
                {
                    "input_per_1m": self.rate.input_per_1m,
                    "output_per_1m": self.rate.output_per_1m,
                }
                if self.rate else None
            ),
            "matched": self.matched,
            "provider": self.provider,
            "model": self.model,
        }


def _user_overrides_path() -> Path:
    home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
    return home / "pricing.toml"


def load_pricing() -> dict[str, dict[str, ModelRate]]:
    """Return the merged pricing table: builtin defaults overlaid with user overrides.

    User TOML at ``~/.godbot/pricing.toml`` (or ``$GODBOT_HOME/pricing.toml``)
    is merged shallow — per-model entries replace builtin entries; new
    providers/models add cleanly. Missing or malformed file → builtins only.
    """
    table: dict[str, dict[str, ModelRate]] = {
        prov: dict(models) for prov, models in _BUILTIN_RATES.items()
    }
    p = _user_overrides_path()
    if not p.exists():
        return table
    try:
        import tomllib
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.exception("failed to load %s; falling back to builtin rates", p)
        return table
    if not isinstance(data, dict):
        return table
    for prov, models in data.items():
        if not isinstance(models, dict):
            continue
        slot = table.setdefault(prov, {})
        for model_id, rate_dict in models.items():
            if not isinstance(rate_dict, dict):
                continue
            try:
                slot[model_id] = ModelRate(
                    input_per_1m=float(rate_dict.get("input_per_1m", 0.0)),
                    output_per_1m=float(rate_dict.get("output_per_1m", 0.0)),
                )
            except (TypeError, ValueError):
                continue
    return table


def _lookup_rate(
    table: dict[str, dict[str, ModelRate]], provider: str, model: str,
) -> Optional[ModelRate]:
    """Find a rate for (provider, model). Tolerates loose model id shapes
    by trying a couple of obvious normalisations."""
    prov_models = table.get(provider) or {}
    if model in prov_models:
        return prov_models[model]
    # Loose match: try lowercase + strip date suffix (e.g.
    # "claude-haiku-4-5-20251001" -> "claude-haiku-4-5").
    lc = model.lower()
    if lc in prov_models:
        return prov_models[lc]
    # Trim a trailing -YYYYMMDD if present.
    trimmed = lc
    if len(trimmed) >= 9 and trimmed[-9] == "-" and trimmed[-8:].isdigit():
        trimmed = trimmed[:-9]
        if trimmed in prov_models:
            return prov_models[trimmed]
    return None


def compute_cost(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    table: Optional[dict[str, dict[str, ModelRate]]] = None,
) -> CostBreakdown:
    """Compute USD cost for one usage block.

    Local providers are pinned at $0 (matched=True, rate=$0/$0 implicit).
    Unknown provider/model returns $0 with ``matched=False`` so the UI
    can render "(rate unknown)" instead of a misleading dollar figure.
    """
    if table is None:
        table = load_pricing()
    if provider in _FREE_PROVIDERS:
        return CostBreakdown(
            usd=0.0, input_usd=0.0, output_usd=0.0,
            rate=ModelRate(0.0, 0.0), matched=True,
            provider=provider, model=model,
        )
    rate = _lookup_rate(table, provider, model)
    if rate is None:
        return CostBreakdown(
            usd=0.0, input_usd=0.0, output_usd=0.0,
            rate=None, matched=False,
            provider=provider, model=model,
        )
    inp_usd = (input_tokens / 1_000_000.0) * rate.input_per_1m
    out_usd = (output_tokens / 1_000_000.0) * rate.output_per_1m
    return CostBreakdown(
        usd=inp_usd + out_usd,
        input_usd=inp_usd,
        output_usd=out_usd,
        rate=rate,
        matched=True,
        provider=provider,
        model=model,
    )
