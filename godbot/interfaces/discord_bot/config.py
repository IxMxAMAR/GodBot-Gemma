from __future__ import annotations
import os
from godbot.config import load_config, DiscordConfig


def load_discord_config() -> DiscordConfig:
    """Load DiscordConfig from godbot config + env overrides.

    GODBOT_DISCORD_TOKEN overrides token.
    GODBOT_DISCORD_OWNER_ID overrides owner_id.
    """
    cfg = load_config().discord
    if env_token := os.environ.get("GODBOT_DISCORD_TOKEN"):
        cfg = DiscordConfig(token=env_token, owner_id=cfg.owner_id)
    if env_owner := os.environ.get("GODBOT_DISCORD_OWNER_ID"):
        try:
            cfg = DiscordConfig(token=cfg.token, owner_id=int(env_owner))
        except ValueError:
            pass
    return cfg
