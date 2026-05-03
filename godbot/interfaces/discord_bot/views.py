from __future__ import annotations
import asyncio
from typing import Awaitable, Callable, Optional

import discord
from discord.ui import View, Button, button


class GateView(View):
    """Button view rendered alongside a GateEvent embed.

    Allow / Always / Deny. Owner-only via interaction_check. Auto-deny on timeout.

    Note: discord.py only starts its built-in timeout task once the view is
    attached to a message dispatched through the gateway. Tests construct the
    view standalone, so we start an independent local timer in __init__ that
    invokes ``on_timeout`` after ``timeout`` seconds. The local timer is a
    no-op once the view is resolved (button click or explicit stop).
    """

    def __init__(
        self,
        call_id: str,
        owner_id: int,
        on_decision: Callable[[str, str], Awaitable[None]],
        timeout: float = 300.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self._call_id = call_id
        self._owner_id = owner_id
        self._on_decision = on_decision
        self._resolved = False
        self._local_timeout_task: Optional[asyncio.Task] = None
        if timeout is not None and timeout > 0:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                self._local_timeout_task = loop.create_task(self._local_timeout(timeout))

    async def _local_timeout(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if not self._resolved:
            await self.on_timeout()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self._owner_id:
            await interaction.response.send_message(
                "This gate is for the bot owner only.", ephemeral=True
            )
            return False
        return True

    async def on_timeout(self) -> None:
        if not self._resolved:
            self._resolved = True
            await self._on_decision(self._call_id, "deny")

    async def _resolve(self, interaction: discord.Interaction, decision: str) -> None:
        if self._resolved:
            return
        self._resolved = True
        if self._local_timeout_task is not None and not self._local_timeout_task.done():
            self._local_timeout_task.cancel()
        await interaction.response.defer()
        for child in self.children:
            child.disabled = True
        self.stop()
        await self._on_decision(self._call_id, decision)

    @button(label="Allow", style=discord.ButtonStyle.success, custom_id="allow")
    async def _allow(self, interaction: discord.Interaction, _btn: Button) -> None:
        await self._resolve(interaction, "allow")

    @button(label="Always", style=discord.ButtonStyle.primary, custom_id="always")
    async def _always(self, interaction: discord.Interaction, _btn: Button) -> None:
        await self._resolve(interaction, "always")

    @button(label="Deny", style=discord.ButtonStyle.danger, custom_id="deny")
    async def _deny(self, interaction: discord.Interaction, _btn: Button) -> None:
        await self._resolve(interaction, "deny")
