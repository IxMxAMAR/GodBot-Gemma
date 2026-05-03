import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from godbot.interfaces.discord_bot.bot import GodBotClient


@pytest.fixture
def bot(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    return GodBotClient(owner_id=123, sessions_root=tmp_path / "sessions")


def test_status_shows_session_id(bot):
    bot._channel_sessions.set(42, "sid-xyz")
    assert bot._channel_sessions.get(42) == "sid-xyz"


def test_reset_drops_session(bot):
    bot._channel_sessions.set(42, "sid-xyz")
    bot._channel_sessions.pop(42)
    assert bot._channel_sessions.get(42) is None


@pytest.mark.asyncio
async def test_on_message_ignores_other_users(bot):
    msg = MagicMock()
    msg.author.id = 999  # not owner
    msg.author.bot = False
    msg.content = "hello"
    msg.guild = None
    # Should silently return without dispatching.
    bot._dispatch_user_message = AsyncMock()
    await bot.on_message(msg)
    bot._dispatch_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_message_ignores_bots(bot):
    msg = MagicMock()
    msg.author.bot = True
    msg.content = "hi"
    bot._dispatch_user_message = AsyncMock()
    await bot.on_message(msg)
    bot._dispatch_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_message_in_dm_processes_owner(bot):
    msg = MagicMock()
    msg.author.id = 123
    msg.author.bot = False
    msg.content = "hello"
    msg.guild = None  # DM
    msg.channel = MagicMock()
    bot._dispatch_user_message = AsyncMock()
    await bot.on_message(msg)
    bot._dispatch_user_message.assert_awaited_once()
    args = bot._dispatch_user_message.await_args.args
    assert args[1] == "hello"


@pytest.mark.asyncio
async def test_on_message_in_guild_requires_mention(bot):
    msg = MagicMock()
    msg.author.id = 123
    msg.author.bot = False
    msg.content = "hello (no mention)"
    msg.guild = MagicMock()
    msg.mentions = []  # bot not mentioned
    bot._dispatch_user_message = AsyncMock()
    await bot.on_message(msg)
    bot._dispatch_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_message_in_guild_with_mention_strips_it(bot):
    # discord.Client.user is a read-only property — patch it on the class.
    fake_user = MagicMock()
    fake_user.id = 555
    with patch.object(type(bot), "user", new=fake_user):
        msg = MagicMock()
        msg.author.id = 123
        msg.author.bot = False
        msg.guild = MagicMock()
        msg.mentions = [fake_user]
        msg.content = f"<@{fake_user.id}> ask something"
        msg.channel = MagicMock()
        bot._dispatch_user_message = AsyncMock()
        await bot.on_message(msg)
        bot._dispatch_user_message.assert_awaited_once()
        text = bot._dispatch_user_message.await_args.args[1]
        assert text == "ask something"
