from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import AsyncIterator, Optional

from godbot.config import load_config, Config
from godbot.core.agent import run_turn
from godbot.core.events import DoneEvent, ErrorEvent, Event
from godbot.core.registry import DEFAULT, Registry
from godbot.core.session import Session as CoreSession
from godbot.prompts import build_system_prompt


class EmbeddedRunner:
    """Run the agent loop in-process. Used by Session helper in embedded mode."""

    def __init__(
        self,
        *,
        core_session: CoreSession,
        registry: Registry,
        llm,
        config: Config,
    ) -> None:
        self._core_session = core_session
        self._registry = registry
        self._llm = llm
        self._config = config
        self._cancel = asyncio.Event()

    @classmethod
    def create(
        cls,
        *,
        sessions_root: Path,
        registry: Optional[Registry] = None,
        llm=None,
        config: Optional[Config] = None,
        sid: Optional[str] = None,
    ) -> "EmbeddedRunner":
        """Build a runner with sensible defaults; tests inject MockLLM."""
        cfg = config or load_config()
        reg = registry or DEFAULT
        sessions_root = Path(sessions_root)
        sessions_root.mkdir(parents=True, exist_ok=True)
        if sid:
            cs = CoreSession.load(sessions_root, sid)
        else:
            cs = CoreSession.create(sessions_root, model=cfg.llm.model)
        os.environ["GODBOT_ACTIVE_SESSION"] = str(cs.dir)

        if llm is None:
            from godbot.core.llm import LLMClient
            llm = LLMClient(base_url=cfg.llm.base_url, model=cfg.llm.model)
            llm.probe()

        return cls(core_session=cs, registry=reg, llm=llm, config=cfg)

    @property
    def sid(self) -> str:
        return self._core_session.id

    async def run(self, message: str) -> AsyncIterator[Event]:
        """Run one user turn. Yields events until DoneEvent or ErrorEvent."""
        self._core_session.append_user(message)
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(ev: Event) -> None:
            await queue.put(ev)

        async def runner_task():
            try:
                await run_turn(
                    llm=self._llm,
                    session=self._core_session,
                    registry=self._registry,
                    emit=emit,
                    cancel=self._cancel,
                    max_steps=self._config.agent.max_steps,
                    max_context=self._config.llm.max_context,
                    system_prompt="",
                    system_prompt_builder=build_system_prompt,
                )
            except Exception as e:
                await queue.put(ErrorEvent(message=f"runner crash: {e}", recoverable=False))
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(runner_task())
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    return
                yield ev
                if isinstance(ev, (DoneEvent, ErrorEvent)):
                    # Drain any remaining events but don't yield them.
                    return
        finally:
            if not task.done():
                self._cancel.set()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.TimeoutError:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

    async def resolve_gate(self, call_id: str, decision: str) -> None:
        self._core_session.resolve_gate(call_id, decision)

    async def stop(self) -> None:
        self._cancel.set()

    async def close(self) -> None:
        self._cancel.set()
        self._core_session.end()
