import httpx
import pytest
import respx
from unittest.mock import AsyncMock, MagicMock, patch
from godbot.interfaces.discord_bot.bot import GodBotClient
from godbot.interfaces.discord_bot.rendering import (
    build_abort_embed, build_cost_embed, build_stats_embed,
)


@pytest.fixture
def bot(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    return GodBotClient(owner_id=123, sessions_root=tmp_path / "sessions")


def _make_interaction(*, user_id: int = 123, channel_id: int = 42) -> MagicMock:
    """Build a MagicMock standing in for discord.Interaction.

    `response.is_done()` reflects whether `defer`/`send_message` was called,
    matching real discord.py semantics so `_send_command_error` routes via
    `followup.send` after `defer` and via `response.send_message` otherwise.
    """
    interaction = MagicMock()
    interaction.user.id = user_id
    interaction.channel_id = channel_id
    interaction._done = False

    def _is_done() -> bool:
        return interaction._done

    async def _defer(**_kwargs):
        interaction._done = True

    async def _send_message(*_args, **_kwargs):
        interaction._done = True

    interaction.response.is_done.side_effect = _is_done
    interaction.response.defer = AsyncMock(side_effect=_defer)
    interaction.response.send_message = AsyncMock(side_effect=_send_message)
    interaction.followup.send = AsyncMock()
    return interaction


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


# ---------------------------------------------------------------------------
# Slash commands: /stats, /cost, /abort (sub-projects 32, 23, 98).
# ---------------------------------------------------------------------------


def test_slash_commands_registered(bot):
    names = {cmd.name for cmd in bot.tree.get_commands()}
    assert {"stats", "cost", "abort"}.issubset(names)


# --- /stats -----------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_cmd_stats_renders_embed_from_api(bot):
    payload = {
        "sessions": 5,
        "turns": 42,
        "tool_calls_total": 100,
        "top_tools": [
            {"name": "read_file", "count": 50},
            {"name": "grep", "count": 30},
        ],
        "tasks": {"total": 3, "by_status": {"done": 2, "running": 1}},
        "usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
        "estimated_cost_usd": 0.1234,
    }
    respx.get(f"{bot.base_url}/api/stats").mock(
        return_value=httpx.Response(200, json=payload)
    )
    interaction = _make_interaction()
    await bot._cmd_stats(interaction)
    interaction.response.defer.assert_awaited_once()
    interaction.followup.send.assert_awaited_once()
    embed = interaction.followup.send.await_args.kwargs["embed"]
    assert embed.title == "GodBot stats"
    field_text = " ".join(f.value for f in embed.fields)
    assert "5 / 42" in field_text  # sessions / turns
    assert "100" in field_text  # tool_calls_total
    assert "$0.1234" in field_text  # cost
    assert "read_file" in field_text  # top tool


@respx.mock
@pytest.mark.asyncio
async def test_cmd_stats_reports_http_error(bot):
    respx.get(f"{bot.base_url}/api/stats").mock(
        return_value=httpx.Response(500, json={"detail": "boom"})
    )
    interaction = _make_interaction()
    await bot._cmd_stats(interaction)
    # defer succeeded, error goes to followup.
    interaction.followup.send.assert_awaited_once()
    embed = interaction.followup.send.await_args.kwargs["embed"]
    assert "Error" in embed.title


def test_build_stats_embed_handles_missing_fields():
    embed = build_stats_embed({})
    # Should not raise and should still produce a renderable dict.
    assert embed["title"] == "GodBot stats"
    assert any("0 / 0" in f["value"] for f in embed["fields"])
    assert any("(none yet)" in f["value"] for f in embed["fields"])


# --- /cost ------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_cmd_cost_renders_embed_for_bound_session(bot):
    bot._channel_sessions.set(42, "sid-abc12345")
    respx.get(f"{bot.base_url}/api/sessions/sid-abc12345/cost").mock(
        return_value=httpx.Response(200, json={
            "matched": True,
            "usd": 0.005678,
            "provider": "anthropic",
            "model": "claude-opus-4",
            "usage": {"input_tokens": 500, "output_tokens": 200, "turns": 4},
        })
    )
    interaction = _make_interaction(channel_id=42)
    await bot._cmd_cost(interaction)
    interaction.response.defer.assert_awaited_once()
    interaction.followup.send.assert_awaited_once()
    embed = interaction.followup.send.await_args.kwargs["embed"]
    assert "sid-abc1" in embed.title
    assert "$0.005678" in embed.description
    assert "rate unknown" not in embed.description
    field_text = " ".join(f.value for f in embed.fields)
    assert "anthropic" in field_text
    assert "claude-opus-4" in field_text


@pytest.mark.asyncio
async def test_cmd_cost_with_no_session_replies_ephemeral(bot):
    # No session bound to channel 42.
    interaction = _make_interaction(channel_id=42)
    await bot._cmd_cost(interaction)
    interaction.response.send_message.assert_awaited_once()
    kwargs = interaction.response.send_message.await_args.kwargs
    assert kwargs.get("ephemeral") is True
    interaction.followup.send.assert_not_awaited()


def test_build_cost_embed_unmatched_rate():
    embed = build_cost_embed("sid-xyz", {
        "matched": False, "usd": 0.0,
        "provider": "lmstudio", "model": "qwen",
        "usage": {"input_tokens": 0, "output_tokens": 0, "turns": 0},
    })
    assert "rate unknown" in embed["description"]


# --- /abort -----------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_cmd_abort_owner_signals_and_renders(bot):
    respx.post(f"{bot.base_url}/api/agent/abort_all").mock(
        return_value=httpx.Response(200, json={
            "ok": True, "sessions_signaled": 2, "tasks_signaled": 5,
        })
    )
    interaction = _make_interaction(user_id=123)  # matches bot.owner_id
    await bot._cmd_abort(interaction)
    interaction.response.defer.assert_awaited_once()
    interaction.followup.send.assert_awaited_once()
    embed = interaction.followup.send.await_args.kwargs["embed"]
    assert "Abort" in embed.title
    assert "2" in embed.description and "5" in embed.description


@pytest.mark.asyncio
async def test_cmd_abort_non_owner_blocked(bot):
    interaction = _make_interaction(user_id=999)  # not owner
    await bot._cmd_abort(interaction)
    interaction.response.send_message.assert_awaited_once()
    kwargs = interaction.response.send_message.await_args.kwargs
    assert kwargs.get("ephemeral") is True
    interaction.response.defer.assert_not_awaited()
    interaction.followup.send.assert_not_awaited()


def test_build_abort_embed_zero_counts():
    embed = build_abort_embed({"sessions_signaled": 0, "tasks_signaled": 0})
    assert "0" in embed["description"]
