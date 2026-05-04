"""Background agent tasks (sub-project 13 + 17 persistence).

A "background task" is a goal the user hands off to the agent to work on
without supervision. The runner spins up a Session, runs the agent loop
to completion, and stores the result so the user can come back later
and check status.

Design:

- Tasks default to *safe-only* tools (non-dangerous in the registry's
  sense). This avoids the dead-gate problem: there's no human to approve
  a write_file or run_powershell, and gating with no consumer would hang
  the runner forever. Callers can opt into a wider tool surface via
  ``tool_overrides``, accepting that any dangerous-tool invocation will
  hit the gate machinery and likely time out.
- Tasks are identified by short ids (``t-<8hex>``) independent of the
  underlying session id. Sessions are still recorded under
  ``<sessions_root>`` so the user can inspect the trace.
- Statuses: ``pending`` → ``running`` → (``done`` | ``error`` | ``cancelled``).
- **Persistent**: every state change writes the record to disk under
  ``<sessions_root>/.godbot-tasks/<tid>.json``. On daemon startup the
  runner reloads them and marks any non-terminal task as ``interrupted``
  (a terminal status that signals "restart killed me"). This lets the UI
  show prior results even after a daemon restart.

The web-layer wires this up in :mod:`godbot.interfaces.web`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from godbot.core.events import (
    DoneEvent,
    ErrorEvent,
    Event,
    ToolCallEvent,
    ToolResultEvent,
)

log = logging.getLogger("godbot.tasks")


@dataclass
class TaskRecord:
    """User-facing snapshot of one background task."""

    id: str
    goal: str
    workspace: Optional[str]
    session_id: Optional[str]
    status: str  # "pending" | "running" | "done" | "error" | "cancelled"
    started_at: float
    ended_at: Optional[float] = None
    result: str = ""
    error: str = ""
    step_count: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _new_task_id() -> str:
    return "t-" + uuid.uuid4().hex[:8]


_TERMINAL = {"done", "error", "cancelled", "interrupted"}


class TaskRunner:
    """Process-local registry of background agent tasks (with optional disk persistence)."""

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        self._records: dict[str, TaskRecord] = {}
        self._cancels: dict[str, asyncio.Event] = {}
        self._asyncio_tasks: dict[str, asyncio.Task] = {}
        self._persist_dir = persist_dir

    # --- persistence --------------------------------------------------

    def attach_persistence(self, persist_dir: Path) -> int:
        """Enable disk persistence and load any existing records.

        Returns the count of records loaded. Non-terminal records are
        flipped to ``interrupted`` (terminal) so the UI can distinguish
        them from a fresh task. Subsequent state changes write through
        to ``persist_dir/<tid>.json``.
        """
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        loaded = 0
        for p in sorted(self._persist_dir.glob("t-*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                rec = TaskRecord(**data)
            except TypeError:
                # Schema drift — skip rather than crash startup.
                continue
            if rec.status not in _TERMINAL:
                rec.status = "interrupted"
                rec.error = rec.error or "daemon restart killed this task"
                rec.ended_at = rec.ended_at or time.time()
            self._records[rec.id] = rec
            loaded += 1
        # Persist the flipped records back so the next startup is consistent.
        for rec in self._records.values():
            self._persist(rec)
        return loaded

    def _persist(self, rec: TaskRecord) -> None:
        if self._persist_dir is None:
            return
        try:
            p = self._persist_dir / f"{rec.id}.json"
            p.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
        except Exception:
            log.exception("failed to persist task %s", rec.id)

    # --- public API ---------------------------------------------------

    def start_task(
        self,
        *,
        goal: str,
        sessions_root: Path,
        workspace: Optional[str] = None,
        provider: str = "lmstudio",
        model_name: str = "auto",
        max_steps: int = 12,
        max_context: int = 32000,
        tool_overrides: Optional[list[str]] = None,
        safe_only: bool = True,
    ) -> TaskRecord:
        """Schedule a new background task. Returns its initial record.

        The actual agent loop runs in an :class:`asyncio.Task` spawned on
        the current running loop; this method is sync-callable from web
        request handlers because ``asyncio.get_running_loop`` will succeed
        inside FastAPI's request scope.
        """
        from godbot.core.registry import DEFAULT
        from godbot.core.session import Session
        from godbot.config import load_config, providers_to_configs
        from godbot.core.providers import (
            get_provider as get_provider_factory,
            load_providers_from_config,
        )
        from godbot.core.providers.base import ProviderConfig as PCT
        from godbot.core.agent import run_turn
        from godbot.core.llm import LLMClient
        from godbot.prompts import build_system_prompt

        tid = _new_task_id()
        rec = TaskRecord(
            id=tid,
            goal=goal,
            workspace=workspace,
            session_id=None,
            status="pending",
            started_at=time.time(),
            provider=provider,
            model=model_name,
        )
        self._records[tid] = rec
        self._persist(rec)
        cancel = asyncio.Event()
        self._cancels[tid] = cancel

        # Session is created in the worker task to keep start_task fast and
        # to ensure the session lives on the runner thread's event loop.

        # Compute the tool subset honoured by the agent. Safe-only filter
        # drops every tool flagged dangerous so nothing tries to gate.
        if tool_overrides is None and safe_only:
            tool_overrides = [t.name for t in DEFAULT.all() if not t.dangerous]

        async def _runner() -> None:
            rec.status = "running"
            self._persist(rec)
            try:
                session = Session.create(
                    sessions_root,
                    model=model_name,
                    workspace_root=workspace,
                    auto_approve_in_sandbox=False,
                    provider=provider,
                    model_name=model_name,
                )
                if tool_overrides is not None:
                    session.set_tool_overrides(sorted(tool_overrides))
                rec.session_id = session.id

                session.append_user(goal)

                # Resolve provider — fall back to legacy LLMClient for lmstudio.
                cfg = load_config()
                if provider == "lmstudio":
                    llm_kwargs: dict[str, Any] = {"llm": LLMClient(
                        base_url=cfg.llm.base_url, model=cfg.llm.model,
                    )}
                else:
                    raw = providers_to_configs(cfg.providers)
                    parsed = load_providers_from_config(raw)
                    pcfg = parsed.get(provider) or PCT(name=provider)
                    prov = get_provider_factory(provider, pcfg)
                    llm_kwargs = {"provider": prov}

                async def _emit(ev: Event) -> None:
                    if isinstance(ev, ToolCallEvent):
                        rec.tool_calls.append({
                            "id": ev.id, "name": ev.name, "args": ev.args,
                        })
                    elif isinstance(ev, ToolResultEvent):
                        # Patch the matching tool-call entry with the preview.
                        for tc in rec.tool_calls:
                            if tc.get("id") == ev.id:
                                tc["result_preview"] = ev.preview
                                tc["duration_ms"] = ev.duration_ms
                                break
                    elif isinstance(ev, DoneEvent):
                        rec.step_count = ev.step_count
                    elif isinstance(ev, ErrorEvent):
                        rec.error = ev.message

                await run_turn(
                    session=session,
                    registry=DEFAULT,
                    emit=_emit,
                    cancel=cancel,
                    max_steps=max_steps,
                    max_context=max_context,
                    system_prompt="",
                    system_prompt_builder=build_system_prompt,
                    **llm_kwargs,
                )

                # Capture the final answer from the session's last assistant message.
                msgs = session.messages_for_llm()
                final = ""
                for m in reversed(msgs):
                    if m.get("role") == "assistant" and m.get("content"):
                        final = m["content"]
                        break
                rec.result = final

                # If cancel was set, that wins over success.
                if cancel.is_set():
                    rec.status = "cancelled"
                elif rec.error:
                    rec.status = "error"
                else:
                    rec.status = "done"
            except Exception as e:
                rec.error = f"{type(e).__name__}: {e}"
                rec.status = "error"
                log.exception("background task %r raised", tid)
            finally:
                rec.ended_at = time.time()
                self._cancels.pop(tid, None)
                self._asyncio_tasks.pop(tid, None)
                self._persist(rec)

        t = asyncio.get_running_loop().create_task(_runner(), name=f"task-{tid}")
        self._asyncio_tasks[tid] = t
        return rec

    def get_task(self, tid: str) -> Optional[TaskRecord]:
        return self._records.get(tid)

    def list_tasks(self, *, limit: Optional[int] = None) -> list[TaskRecord]:
        """Newest first, optionally capped."""
        items = sorted(self._records.values(), key=lambda r: r.started_at, reverse=True)
        if limit is not None:
            items = items[:limit]
        return items

    def cancel_task(self, tid: str) -> bool:
        """Signal cancel on a running task. Returns True if a cancel was sent."""
        ev = self._cancels.get(tid)
        if ev is None:
            return False
        ev.set()
        # The runner picks this up on the next agent step and emits ErrorEvent;
        # finalisation in `_runner` flips status to 'cancelled'.
        return True

    def clear_finished(self) -> int:
        """Remove records in a terminal state. Returns count removed.

        Persisted records are deleted from disk too so the next startup
        doesn't reload them.
        """
        kill = [tid for tid, r in self._records.items() if r.status in _TERMINAL]
        for tid in kill:
            self._records.pop(tid, None)
            if self._persist_dir is not None:
                try:
                    (self._persist_dir / f"{tid}.json").unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    log.exception("failed to delete persisted task %s", tid)
        return len(kill)


# Module-level default instance, shared by the web layer.
DEFAULT_RUNNER = TaskRunner()
