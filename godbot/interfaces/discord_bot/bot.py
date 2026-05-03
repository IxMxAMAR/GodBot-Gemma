from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import discord

from godbot.client import Session
from godbot.client.embedded import EmbeddedRunner
from godbot.core.events import (
    DoneEvent, ErrorEvent, GateEvent, TokenEvent, ToolCallEvent, ToolResultEvent,
)
from godbot.interfaces.discord_bot.config import load_discord_config
from godbot.interfaces.discord_bot.rendering import (
    TokenAccumulator, build_error_embed, build_gate_embed,
    build_tool_call_embed, build_tool_result_update, extract_final_answer,
)
from godbot.interfaces.discord_bot.session_map import ChannelSessionMap
from godbot.interfaces.discord_bot.views import GateView


log = logging.getLogger("godbot.discord")


def _godbot_home() -> Path:
    return Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))


class GodBotClient(discord.Client):
    """The Discord bot. One per process."""

    def __init__(
        self,
        *,
        owner_id: int,
        base_url: str = "http://127.0.0.1:7878",
        auto_launch: bool = True,
        sessions_root: Optional[Path] = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.owner_id = owner_id
        self.base_url = base_url
        self.auto_launch = auto_launch
        self.sessions_root = sessions_root or (Path.cwd() / "sessions")
        self._channel_sessions = ChannelSessionMap(_godbot_home() / "discord-sessions.json")
        self._channel_tasks: dict[int, asyncio.Task] = {}
        self._global_lock = asyncio.Lock()  # used in --no-daemon mode
        self._embedded_warning_emitted = False

    async def on_ready(self) -> None:
        log.info("Bot ready as %s", self.user)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if message.author.id != self.owner_id:
            return
        # In a server channel, require a mention.
        if message.guild is not None and self.user not in message.mentions:
            return
        text = message.content
        if message.guild is not None:
            # Strip the leading mention.
            text = text.replace(f"<@{self.user.id}>", "").strip()
        if not text:
            return
        await self._dispatch_user_message(message.channel, text)

    async def _dispatch_user_message(self, channel: discord.abc.Messageable, text: str) -> None:
        channel_id = getattr(channel, "id", 0)
        existing = self._channel_tasks.get(channel_id)
        if existing and not existing.done():
            await channel.send("⏳ already processing your previous message — queueing this one")
            await existing
        task = asyncio.create_task(self._process_message(channel, text))
        self._channel_tasks[channel_id] = task

    def _get_session_factory(self, channel_id: int):
        sid = self._channel_sessions.get(channel_id)
        workspace = self._channel_sessions.get_workspace(channel_id)
        def factory():
            if sid:
                return EmbeddedRunner.create(sessions_root=self.sessions_root, sid=sid)
            # Embedded creation also accepts workspace via core_session.set_workspace.
            runner = EmbeddedRunner.create(sessions_root=self.sessions_root)
            if workspace:
                runner._core_session.set_workspace(workspace)
            return runner
        return factory

    async def _process_message(self, channel: discord.abc.Messageable, text: str) -> None:
        channel_id = getattr(channel, "id", 0)
        sid = self._channel_sessions.get(channel_id)
        # Build / connect the Session for this channel.
        try:
            session = Session(
                sid=sid, base_url=self.base_url, auto_launch=self.auto_launch,
                embedded_factory=self._get_session_factory(channel_id),
            )
            await session.connect()
        except Exception as e:
            await channel.send(embed=discord.Embed.from_dict(build_error_embed(ErrorEvent(message=str(e)))))
            return

        # Embedded-mode warning: only one channel can be active at a time.
        if session.mode == "embedded" and not self._embedded_warning_emitted:
            self._embedded_warning_emitted = True
            log.warning(
                "Discord bot running in EMBEDDED mode; only one channel can run at a time. "
                "Start godbot-web to enable parallel channels."
            )

        # Persist session id (and workspace if known) for next time.
        workspace = self._channel_sessions.get_workspace(channel_id)
        self._channel_sessions.set(channel_id, session.sid, workspace=workspace)

        # Stream events.
        try:
            if session.mode == "embedded":
                async with self._global_lock:
                    await self._stream_to_channel(channel, session, text)
            else:
                await self._stream_to_channel(channel, session, text)
        finally:
            await session.close()

    async def _stream_to_channel(
        self, channel: discord.abc.Messageable, session: Session, text: str,
    ) -> None:
        placeholder = await channel.send("⏳ thinking…")
        accumulator = TokenAccumulator()
        tool_msgs: dict[str, discord.Message] = {}
        tool_embeds: dict[str, dict] = {}

        async def on_decision(call_id: str, decision: str) -> None:
            await session.resolve_gate(call_id, decision)

        try:
            async for ev in session.run(text):
                if isinstance(ev, TokenEvent):
                    full, should_emit = accumulator.feed(ev.text)
                    if should_emit:
                        try:
                            await placeholder.edit(content=f"```\n{full[-1900:]}\n```")
                        except discord.HTTPException:
                            pass
                elif isinstance(ev, ToolCallEvent):
                    embed_dict = build_tool_call_embed(ev)
                    tool_embeds[ev.id] = embed_dict
                    msg = await channel.send(embed=discord.Embed.from_dict(embed_dict))
                    tool_msgs[ev.id] = msg
                elif isinstance(ev, ToolResultEvent):
                    msg = tool_msgs.get(ev.id)
                    if msg is None:
                        continue
                    update = build_tool_result_update(ev)
                    base = tool_embeds.get(ev.id, {}).copy()
                    base["footer"] = update["footer"]
                    base["fields"] = update["fields"]
                    try:
                        await msg.edit(embed=discord.Embed.from_dict(base))
                    except discord.HTTPException:
                        pass
                elif isinstance(ev, GateEvent):
                    view = GateView(
                        call_id=ev.id, owner_id=self.owner_id, on_decision=on_decision,
                    )
                    await channel.send(embed=discord.Embed.from_dict(build_gate_embed(ev)), view=view)
                elif isinstance(ev, ErrorEvent):
                    await channel.send(embed=discord.Embed.from_dict(build_error_embed(ev)))
                elif isinstance(ev, DoneEvent):
                    final = extract_final_answer(accumulator.flush())
                    try:
                        await placeholder.edit(content=final[:2000] or "(empty reply)")
                    except discord.HTTPException:
                        pass
        except Exception as e:
            await channel.send(
                embed=discord.Embed.from_dict(build_error_embed(ErrorEvent(message=str(e))))
            )


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="godbot-discord")
    parser.add_argument("--base-url", default="http://127.0.0.1:7878")
    parser.add_argument("--no-daemon", action="store_true",
                        help="Skip daemon auto-launch; embedded mode only (single channel at a time)")
    args = parser.parse_args()

    cfg = load_discord_config()
    if not cfg.token or not cfg.owner_id:
        log.error("discord token / owner_id missing — set [discord] in ~/.godbot/config.toml or env vars")
        return 2

    client = GodBotClient(
        owner_id=cfg.owner_id,
        base_url=args.base_url,
        auto_launch=not args.no_daemon,
    )
    client.run(cfg.token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
