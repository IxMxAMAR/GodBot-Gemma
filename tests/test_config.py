import os
from pathlib import Path
import pytest
from godbot.config import Config, load_config, default_config_text


def test_defaults_load(tmp_godbot_home):
    cfg = load_config()
    assert cfg.llm.base_url == "http://localhost:1234/v1"
    assert cfg.llm.model == "auto"
    assert cfg.llm.max_context == 28000
    assert cfg.agent.max_steps == 100  # bumped from 25 in SP110
    assert cfg.tools.enabled == "*"
    assert cfg.tools.yolo is False
    assert cfg.rag.embedder == "lmstudio"
    assert cfg.ui.web_port == 7878


def test_creates_default_file_on_first_run(tmp_godbot_home):
    load_config()
    cfg_path = tmp_godbot_home / "config.toml"
    assert cfg_path.exists()
    assert "[llm]" in cfg_path.read_text()


def test_user_override(tmp_godbot_home):
    tmp_godbot_home.mkdir(parents=True, exist_ok=True)
    (tmp_godbot_home / "config.toml").write_text(
        '[llm]\nmodel = "gemma-special"\n[ui]\nweb_port = 9000\n'
    )
    cfg = load_config()
    assert cfg.llm.model == "gemma-special"
    assert cfg.ui.web_port == 9000
    assert cfg.agent.max_steps == 100  # unspecified -> default (SP110 bump)


def test_default_config_text_is_parseable(tmp_godbot_home):
    tmp_godbot_home.mkdir(parents=True, exist_ok=True)
    (tmp_godbot_home / "config.toml").write_text(default_config_text())
    cfg = load_config()
    assert cfg.tools.enabled == "*"


def test_unknown_keys_in_toml_tolerated(tmp_godbot_home):
    tmp_godbot_home.mkdir(parents=True, exist_ok=True)
    (tmp_godbot_home / "config.toml").write_text(
        '[llm]\nmodel = "x"\nfuture_key = "ignore me"\n'
    )
    cfg = load_config()  # must not raise
    assert cfg.llm.model == "x"


def test_discord_config_defaults(tmp_godbot_home):
    cfg = load_config()
    assert cfg.discord.token == ""
    assert cfg.discord.owner_id == 0


def test_discord_config_overrides(tmp_godbot_home):
    tmp_godbot_home.mkdir(parents=True, exist_ok=True)
    (tmp_godbot_home / "config.toml").write_text(
        '[discord]\ntoken = "abc"\nowner_id = 12345\n'
    )
    cfg = load_config()
    assert cfg.discord.token == "abc"
    assert cfg.discord.owner_id == 12345


def test_mcp_config_default_empty(tmp_godbot_home):
    """No [mcp] section => empty servers dict, no breakage."""
    cfg = load_config()
    assert cfg.mcp.servers == {}


def test_mcp_config_parses_single_server(tmp_godbot_home):
    tmp_godbot_home.mkdir(parents=True, exist_ok=True)
    (tmp_godbot_home / "config.toml").write_text(
        '[mcp.servers.filesystem]\n'
        'command = "npx"\n'
        'args = ["-y", "@modelcontextprotocol/server-filesystem", "/some/path"]\n'
    )
    cfg = load_config()
    assert "filesystem" in cfg.mcp.servers
    fs = cfg.mcp.servers["filesystem"]
    assert fs.command == "npx"
    assert fs.args == ["-y", "@modelcontextprotocol/server-filesystem", "/some/path"]
    assert fs.env == {}


def test_mcp_config_parses_env_subtable(tmp_godbot_home):
    tmp_godbot_home.mkdir(parents=True, exist_ok=True)
    (tmp_godbot_home / "config.toml").write_text(
        '[mcp.servers.github]\n'
        'command = "npx"\n'
        'args = ["-y", "@modelcontextprotocol/server-github"]\n'
        '\n'
        '[mcp.servers.github.env]\n'
        'GITHUB_PERSONAL_ACCESS_TOKEN = "secret123"\n'
    )
    cfg = load_config()
    gh = cfg.mcp.servers["github"]
    assert gh.env == {"GITHUB_PERSONAL_ACCESS_TOKEN": "secret123"}


def test_mcp_config_multiple_servers(tmp_godbot_home):
    tmp_godbot_home.mkdir(parents=True, exist_ok=True)
    (tmp_godbot_home / "config.toml").write_text(
        '[mcp.servers.fs]\ncommand = "a"\n'
        '[mcp.servers.gh]\ncommand = "b"\n'
        '[mcp.servers.db]\ncommand = "c"\n'
    )
    cfg = load_config()
    assert set(cfg.mcp.servers.keys()) == {"fs", "gh", "db"}


def test_mcp_config_unknown_keys_tolerated(tmp_godbot_home):
    """Unknown keys inside an mcp.server entry must not crash load_config."""
    tmp_godbot_home.mkdir(parents=True, exist_ok=True)
    (tmp_godbot_home / "config.toml").write_text(
        '[mcp.servers.svc]\n'
        'command = "x"\n'
        'future_key = "ignored"\n'
    )
    cfg = load_config()  # must not raise
    assert cfg.mcp.servers["svc"].command == "x"


def test_mcp_config_existing_users_not_broken(tmp_godbot_home):
    """An existing config.toml without [mcp.*] must continue to load."""
    tmp_godbot_home.mkdir(parents=True, exist_ok=True)
    # Pre-existing user config — note: no [mcp] section.
    (tmp_godbot_home / "config.toml").write_text(default_config_text())
    cfg = load_config()
    assert cfg.mcp.servers == {}
    # And all the normal values are unchanged.
    assert cfg.ui.web_port == 7878
