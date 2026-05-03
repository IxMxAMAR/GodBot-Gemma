import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from godbot.interfaces.discord_bot.views import GateView


@pytest.mark.asyncio
async def test_owner_allow_resolves_with_allow():
    on_decision = AsyncMock()
    view = GateView(call_id="c1", owner_id=123, on_decision=on_decision, timeout=60)

    interaction = MagicMock()
    interaction.user.id = 123
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()

    # Find the allow button.
    allow_btn = next(c for c in view.children if getattr(c, "custom_id", None) == "allow")
    await allow_btn.callback(interaction)
    on_decision.assert_awaited_once_with("c1", "allow")


@pytest.mark.asyncio
async def test_non_owner_blocked_by_interaction_check():
    on_decision = AsyncMock()
    view = GateView(call_id="c1", owner_id=123, on_decision=on_decision, timeout=60)

    interaction = MagicMock()
    interaction.user.id = 999  # not the owner
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()

    ok = await view.interaction_check(interaction)
    assert ok is False
    interaction.response.send_message.assert_awaited_once()
    on_decision.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_resolves_with_deny():
    on_decision = AsyncMock()
    view = GateView(call_id="c1", owner_id=123, on_decision=on_decision, timeout=0.05)
    # Give the timeout time to fire.
    await asyncio.sleep(0.15)
    on_decision.assert_awaited_once_with("c1", "deny")


@pytest.mark.asyncio
async def test_always_button_resolves_with_always():
    on_decision = AsyncMock()
    view = GateView(call_id="c1", owner_id=123, on_decision=on_decision, timeout=60)
    interaction = MagicMock()
    interaction.user.id = 123
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()

    btn = next(c for c in view.children if getattr(c, "custom_id", None) == "always")
    await btn.callback(interaction)
    on_decision.assert_awaited_once_with("c1", "always")


@pytest.mark.asyncio
async def test_deny_button_resolves_with_deny():
    on_decision = AsyncMock()
    view = GateView(call_id="c1", owner_id=123, on_decision=on_decision, timeout=60)
    interaction = MagicMock()
    interaction.user.id = 123
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()

    btn = next(c for c in view.children if getattr(c, "custom_id", None) == "deny")
    await btn.callback(interaction)
    on_decision.assert_awaited_once_with("c1", "deny")


@pytest.mark.asyncio
async def test_second_click_after_resolved_is_noop():
    on_decision = AsyncMock()
    view = GateView(call_id="c1", owner_id=123, on_decision=on_decision, timeout=60)
    interaction = MagicMock()
    interaction.user.id = 123
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()

    btn = next(c for c in view.children if getattr(c, "custom_id", None) == "allow")
    await btn.callback(interaction)
    await btn.callback(interaction)
    assert on_decision.await_count == 1
