import os
from pathlib import Path
import pytest
from godbot.config import Config, load_config, default_config_text


def test_defaults_load(tmp_godbot_home):
    cfg = load_config()
    assert cfg.llm.base_url == "http://localhost:1234/v1"
    assert cfg.llm.model == "auto"
    assert cfg.llm.max_context == 28000
    assert cfg.agent.max_steps == 25
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
    assert cfg.agent.max_steps == 25  # unspecified -> default


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
