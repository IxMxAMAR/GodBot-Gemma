from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_context: int = 28000


@dataclass
class AgentConfig:
    """Agent loop limits.

    ``max_steps``: hard cap on LLM round-trips per chat turn. 100 is a
    reasonable headroom for non-trivial multi-tool work; bump higher
    for marathon refactors. **Set to 0 (or any value <= 0) to disable
    the brake entirely** — the loop runs until the agent emits a
    final_answer or the user cancels via /api/stop. Disabling is
    safer for local models than for paid providers, where a runaway
    can rack up real cost.

    ``default_tool_timeout``: per-tool timeout in seconds, applied
    when a tool doesn't override via its own ``@tool(timeout=…)``.
    """

    max_steps: int = 100
    default_tool_timeout: int = 60


@dataclass
class ToolsConfig:
    enabled: Union[str, list[str]] = "*"
    yolo: bool = False


@dataclass
class RAGConfig:
    embedder: str = "lmstudio"
    embed_model: str = "auto"
    auto_inject: bool = False
    top_k_default: int = 5


@dataclass
class UIConfig:
    web_port: int = 7878
    theme: str = "dark"


@dataclass
class DiscordConfig:
    token: str = ""
    owner_id: int = 0


@dataclass
class AuthConfig:
    """Optional API token authentication (sub-project 88).

    When ``token`` is non-empty (either in config or via the
    ``GODBOT_API_TOKEN`` env var, which wins), every ``/api/*`` request
    must carry ``Authorization: Bearer <token>``. ``/api/health``
    stays open so liveness probes don't need credentials.

    Empty token = auth disabled (the default — back-compat with every
    existing local deployment).
    """

    token: str = ""


@dataclass
class MCPServerConfigItem:
    """One MCP server entry from ``[mcp.servers.<name>]`` in config.toml.

    Stdio transport only in v1: ``command`` is the executable, ``args`` is
    the arg vector, ``env`` is merged on top of the parent process env when
    spawning. An empty ``command`` means "skip this server".
    """

    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPConfig:
    """Parsed ``[mcp]`` section. ``servers`` keys are user-chosen names and
    become the tool-namespace prefix (``mcp_<name>_<tool>``)."""

    servers: dict[str, MCPServerConfigItem] = field(default_factory=dict)


@dataclass
class ProviderConfigItem:
    """One ``[providers.<name>]`` entry from config.toml.

    ``api_key_env`` reads the key from the env var of that name; ``api_key``
    is a literal string (less secure — use only when the env-var path
    isn't workable). ``protocol`` (optional) pins the tool-call protocol
    for this provider regardless of the per-model heuristic.
    """

    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    default_model: str = "auto"
    timeout: float = 60.0
    protocol: str = ""


@dataclass
class ProvidersSection:
    """Parsed ``[providers]`` and ``[providers.<name>]`` sections."""

    default: str = "lmstudio"
    items: dict[str, ProviderConfigItem] = field(default_factory=dict)


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    providers: ProvidersSection = field(default_factory=ProvidersSection)
    auth: AuthConfig = field(default_factory=AuthConfig)


def godbot_home() -> Path:
    """Resolve the godbot home dir each call so tests can monkeypatch GODBOT_HOME."""
    return Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))


def default_config_text() -> str:
    return (
        "[llm]\n"
        'base_url = "http://localhost:1234/v1"\n'
        'model = "auto"\n'
        "temperature = 0.7\n"
        "max_tokens = 4096\n"
        "max_context = 28000\n"
        "\n[agent]\n"
        "# Hard cap on LLM round-trips per chat turn. Set to 0 to disable.\n"
        "# 100 is generous for normal multi-tool work; bump for marathon runs.\n"
        "# Disabling is fine for local models; on paid providers a runaway\n"
        "# can rack up real charges, so consider pairing with a session\n"
        "# budget cap (POST /api/sessions/{sid}/budget {max_usd: 1.0}).\n"
        "max_steps = 100\n"
        "default_tool_timeout = 60\n"
        "\n[tools]\n"
        'enabled = "*"\n'
        "yolo = false\n"
        "\n[rag]\n"
        'embedder = "lmstudio"\n'
        'embed_model = "auto"\n'
        "auto_inject = false\n"
        "top_k_default = 5\n"
        "\n[ui]\n"
        "web_port = 7878\n"
        'theme = "dark"\n'
        "\n[discord]\n"
        'token = ""\n'
        "owner_id = 0\n"
        "\n# Provider abstraction (sub-project 7). Each [providers.<name>]\n"
        "# entry can be selected per-session via POST /api/sessions/new\n"
        '# {"provider":"<name>", "model":"<id>"}.\n'
        "[providers]\n"
        'default = "lmstudio"\n'
        "\n[providers.lmstudio]\n"
        'base_url = "http://localhost:1234/v1"\n'
        'default_model = "auto"\n'
        "\n[providers.ollama]\n"
        'base_url = "http://localhost:11434/v1"\n'
        'default_model = "auto"\n'
        "\n[providers.openai]\n"
        'base_url = "https://api.openai.com/v1"\n'
        'api_key_env = "OPENAI_API_KEY"\n'
        'default_model = "gpt-4o"\n'
        "\n[providers.anthropic]\n"
        'api_key_env = "ANTHROPIC_API_KEY"\n'
        'default_model = "claude-3-5-sonnet-latest"\n'
        "\n[providers.gemini]\n"
        'api_key_env = "GOOGLE_API_KEY"\n'
        'default_model = "gemini-2.0-flash"\n'
        "\n[providers.groq]\n"
        'base_url = "https://api.groq.com/openai/v1"\n'
        'api_key_env = "GROQ_API_KEY"\n'
        'default_model = "llama-3.3-70b-versatile"\n'
        "\n[providers.together]\n"
        'base_url = "https://api.together.xyz/v1"\n'
        'api_key_env = "TOGETHER_API_KEY"\n'
        'default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"\n'
        "\n[providers.openrouter]\n"
        'base_url = "https://openrouter.ai/api/v1"\n'
        'api_key_env = "OPENROUTER_API_KEY"\n'
        'default_model = "anthropic/claude-3-5-sonnet"\n'
    )


def _filter_known(raw: dict, cls) -> dict:
    """Drop keys not present in the dataclass (unknown sections in TOML are tolerated)."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in raw.items() if k in known}


def _parse_mcp(raw_mcp: dict) -> MCPConfig:
    """Parse ``[mcp]`` from TOML.

    The ``servers`` field is a *dict of dataclasses keyed by user-chosen name*,
    which ``_filter_known`` can't handle (it's not a fixed-shape dataclass).
    We walk the sub-table by hand, skipping entries that aren't dict-shaped
    or contain unknown keys (rather than crashing the whole config load).
    """
    servers_raw = (raw_mcp or {}).get("servers", {})
    if not isinstance(servers_raw, dict):
        return MCPConfig()
    parsed: dict[str, MCPServerConfigItem] = {}
    for name, entry in servers_raw.items():
        if not isinstance(entry, dict):
            continue
        try:
            parsed[name] = MCPServerConfigItem(
                **_filter_known(entry, MCPServerConfigItem)
            )
        except Exception:
            # Malformed entry — skip rather than tank the whole config load.
            continue
    return MCPConfig(servers=parsed)


def _parse_providers(raw_providers: dict) -> ProvidersSection:
    """Parse the ``[providers]`` block.

    The block has a scalar ``default`` plus one sub-table per provider
    name, e.g. ``[providers.openai]``. tomllib renders these as nested
    dicts so we can walk the keys. Unknown / malformed entries are
    skipped silently (matching the rest of the config loader's
    "be liberal" stance).
    """
    if not isinstance(raw_providers, dict):
        return ProvidersSection()
    default = str(raw_providers.get("default", "lmstudio") or "lmstudio")
    items: dict[str, ProviderConfigItem] = {}
    for name, entry in raw_providers.items():
        if name == "default":
            continue
        if not isinstance(entry, dict):
            continue
        try:
            items[str(name)] = ProviderConfigItem(
                **_filter_known(entry, ProviderConfigItem)
            )
        except Exception:
            continue
    return ProvidersSection(default=default, items=items)


def load_config() -> Config:
    home = godbot_home()
    home.mkdir(parents=True, exist_ok=True)
    cfg_path = home / "config.toml"
    if not cfg_path.exists():
        cfg_path.write_text(default_config_text(), encoding="utf-8")
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    return Config(
        llm=LLMConfig(**_filter_known(raw.get("llm") or {}, LLMConfig)),
        agent=AgentConfig(**_filter_known(raw.get("agent") or {}, AgentConfig)),
        tools=ToolsConfig(**_filter_known(raw.get("tools") or {}, ToolsConfig)),
        rag=RAGConfig(**_filter_known(raw.get("rag") or {}, RAGConfig)),
        ui=UIConfig(**_filter_known(raw.get("ui") or {}, UIConfig)),
        discord=DiscordConfig(**_filter_known(raw.get("discord") or {}, DiscordConfig)),
        mcp=_parse_mcp(raw.get("mcp") or {}),
        providers=_parse_providers(raw.get("providers") or {}),
        auth=AuthConfig(**_filter_known(raw.get("auth") or {}, AuthConfig)),
    )


def providers_to_configs(providers: ProvidersSection) -> dict[str, Any]:
    """Convert a parsed :class:`ProvidersSection` into the shape that
    :func:`godbot.core.providers.load_providers_from_config` expects:
    ``{name: {base_url, api_key, api_key_env, default_model, timeout, protocol}}``.
    Used by the web layer to wire config-driven providers at runtime.
    """
    out: dict[str, Any] = {}
    for name, item in providers.items.items():
        entry: dict[str, Any] = {
            "base_url": item.base_url,
            "default_model": item.default_model,
            "timeout": item.timeout,
        }
        if item.api_key:
            entry["api_key"] = item.api_key
        if item.api_key_env:
            entry["api_key_env"] = item.api_key_env
        if item.protocol:
            entry["protocol"] = item.protocol
        out[name] = entry
    return out
