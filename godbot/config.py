from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

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
    max_steps: int = 25
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
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)


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
        "max_steps = 25\n"
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
    )


def _filter_known(raw: dict, cls) -> dict:
    """Drop keys not present in the dataclass (unknown sections in TOML are tolerated)."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in raw.items() if k in known}


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
    )
