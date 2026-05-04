from __future__ import annotations
import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from godbot.core.events import GateEvent


# Module constants used in 3.3:
BLOB_INLINE_LIMIT = 8 * 1024
LLM_RESULT_LIMIT = 4000
LLM_RESULT_HEAD = 2000
LLM_RESULT_TAIL = 1500


def _now_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:6]


def _truncate_for_llm(text: str, call_id: str) -> str:
    if len(text) <= LLM_RESULT_LIMIT:
        return text
    head = text[:LLM_RESULT_HEAD]
    tail = text[-LLM_RESULT_TAIL:]
    elided = len(text) - LLM_RESULT_HEAD - LLM_RESULT_TAIL
    return f"{head}\n... [{elided} chars elided; full result in blob {call_id}] ...\n{tail}"


def estimate_tokens(text: str) -> int:
    return len(text) // 4


class Session:
    def __init__(self, root: Path, session_id: str, meta: dict[str, Any]) -> None:
        self._root = Path(root)
        self.id = session_id
        self.dir = self._root / session_id
        self._meta = meta
        self._gate_events: dict[str, asyncio.Event] = {}
        self._gate_decisions: dict[str, str] = {}
        # Optional per-gate args overrides (set by resolve_gate, consumed by
        # await_gate). Caller-edited content for write_file diffs lives here.
        self._gate_args_overrides: dict[str, Optional[dict[str, Any]]] = {}

    @property
    def model(self) -> str:
        # Legacy ``model`` field — kept for back-compat with callers that
        # use ``session.model`` to display "the model in use" without
        # caring which provider hosts it. New code should prefer
        # :attr:`provider` + :attr:`model_name`.
        return self._meta.get("model", "")

    @property
    def provider(self) -> str:
        """Provider id for this session.

        Defaults to ``"lmstudio"`` for legacy meta.json files that pre-date
        the provider abstraction (sub-project 7) so older sessions keep
        working unchanged.
        """
        return self._meta.get("provider", "lmstudio")

    @property
    def model_name(self) -> str:
        """Provider-scoped model id (e.g. ``"claude-3-5-sonnet-latest"``,
        ``"auto"``). Falls back to legacy ``model`` for back-compat."""
        v = self._meta.get("model_name")
        if isinstance(v, str) and v:
            return v
        return self._meta.get("model", "auto") or "auto"

    @property
    def protocol(self) -> Optional[str]:
        """Per-session tool-call protocol override.

        ``None`` means "use the provider's preference for this model".
        Concrete values are :data:`godbot.core.providers.NATIVE_TOOLS` or
        :data:`godbot.core.providers.REACT_JSON`.
        """
        v = self._meta.get("protocol")
        return v if isinstance(v, str) and v else None

    @property
    def yolo(self) -> bool:
        return bool(self._meta.get("yolo", False))

    @property
    def tool_overrides(self) -> Optional[list[str]]:
        return self._meta.get("tool_overrides")

    @property
    def rag_collection(self) -> Optional[str]:
        return self._meta.get("rag_collection")

    @classmethod
    def create(
        cls, root: Path, model: str,
        rag_collection: Optional[str] = None,
        workspace_root: Optional[str] = None,
        auto_approve_in_sandbox: bool = False,
        provider: str = "lmstudio",
        model_name: Optional[str] = None,
        protocol: Optional[str] = None,
    ) -> "Session":
        sid = _now_id()
        sdir = Path(root) / sid
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "blobs").mkdir(exist_ok=True)
        ws_root_resolved = None
        if workspace_root:
            ws_root_resolved = str(Path(workspace_root).resolve())
        meta = {
            "id": sid,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "ended_at": None,
            "model": model,
            # New (sub-project 7) — provider/model_name/protocol let one
            # session pick a backend. Legacy meta without these keys reads
            # back as ``provider="lmstudio"`` via the property accessors.
            "provider": provider,
            "model_name": model_name if model_name is not None else model,
            "protocol": protocol,
            "tool_overrides": None,
            "auto_approved_tools": [],
            "yolo": False,
            "rag_collection": rag_collection,
            "workspace_root": ws_root_resolved,
            "auto_approve_in_sandbox": bool(auto_approve_in_sandbox),
        }
        cls._write_meta(sdir, meta)
        (sdir / "events.jsonl").touch()
        return cls(Path(root), sid, meta)

    def set_provider(
        self,
        provider: str,
        model_name: Optional[str] = None,
        protocol: Optional[str] = None,
    ) -> None:
        """Update the session's provider selection. ``model_name`` and
        ``protocol`` are independently optional — pass ``None`` to leave
        them unchanged."""
        self._meta["provider"] = provider
        if model_name is not None:
            self._meta["model_name"] = model_name
        if protocol is not None:
            self._meta["protocol"] = protocol
        self._save_meta()

    @property
    def workspace(self):
        from godbot.core.workspace import Workspace
        root = self._meta.get("workspace_root")
        if not root:
            return None
        try:
            return Workspace.of(
                root,
                auto_approve=bool(self._meta.get("auto_approve_in_sandbox", False)),
            )
        except (FileNotFoundError, NotADirectoryError):
            return None

    def set_workspace(self, root: Optional[str], auto_approve: bool = False) -> None:
        if root:
            self._meta["workspace_root"] = str(Path(root).resolve())
        else:
            self._meta["workspace_root"] = None
        self._meta["auto_approve_in_sandbox"] = bool(auto_approve)
        self._save_meta()

    @classmethod
    def load(cls, root: Path, session_id: str) -> "Session":
        sdir = Path(root) / session_id
        if not sdir.exists():
            raise FileNotFoundError(sdir)
        meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
        return cls(Path(root), session_id, meta)

    def set_yolo(self, value: bool) -> None:
        self._meta["yolo"] = bool(value)
        self._save_meta()

    def mark_auto_approved(self, tool_name: str) -> None:
        approved = set(self._meta.get("auto_approved_tools", []))
        approved.add(tool_name)
        self._meta["auto_approved_tools"] = sorted(approved)
        self._save_meta()

    def is_auto_approved(self, tool_name: str) -> bool:
        return tool_name in self._meta.get("auto_approved_tools", [])

    def set_tool_overrides(self, overrides: Optional[list[str]]) -> None:
        self._meta["tool_overrides"] = overrides
        self._save_meta()

    def set_rag_collection(self, name: Optional[str]) -> None:
        self._meta["rag_collection"] = name
        self._save_meta()

    def end(self) -> None:
        self._meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
        self._save_meta()

    @property
    def usage(self) -> dict[str, int]:
        """Cumulative token usage for this session.

        Keys: ``input_tokens``, ``output_tokens``, ``total_tokens``,
        ``turns``. Always returns the full set, with zeros for missing
        fields, so callers can render a consistent label even on a brand
        new session.
        """
        u = self._meta.get("usage") or {}
        return {
            "input_tokens": int(u.get("input_tokens", 0)),
            "output_tokens": int(u.get("output_tokens", 0)),
            "total_tokens": int(u.get("total_tokens", 0)),
            "turns": int(u.get("turns", 0)),
        }

    def add_usage(self, turn_usage: dict[str, int]) -> None:
        """Accumulate one turn's usage onto the session total (sub-project 20).

        Provider classes return a normalized ``{input_tokens,
        output_tokens, total_tokens}`` dict on every TurnResult. We sum
        these and bump a turn counter so the UI can show "10 turns,
        45,000 tokens" without computing it every read.

        No-op when ``turn_usage`` is empty (e.g. a legacy provider that
        doesn't expose token counts), and never raises.
        """
        if not turn_usage:
            return
        try:
            inp = int(turn_usage.get("input_tokens", 0))
            out = int(turn_usage.get("output_tokens", 0))
            total = int(turn_usage.get("total_tokens", 0))
        except Exception:
            return
        if inp == 0 and out == 0 and total == 0:
            return
        cur = dict(self._meta.get("usage") or {})
        cur["input_tokens"] = int(cur.get("input_tokens", 0)) + inp
        cur["output_tokens"] = int(cur.get("output_tokens", 0)) + out
        cur["total_tokens"] = int(cur.get("total_tokens", 0)) + total
        cur["turns"] = int(cur.get("turns", 0)) + 1
        self._meta["usage"] = cur
        self._save_meta()

    @property
    def budget(self) -> dict[str, float | None]:
        """Soft spending caps for this session (sub-project 28).

        Keys: ``max_total_tokens`` and ``max_usd`` — either may be ``None``
        meaning "no cap on this dimension". Both apply additively: the
        session is over-budget if EITHER cap is exceeded.
        """
        b = self._meta.get("budget") or {}
        return {
            "max_total_tokens": b.get("max_total_tokens"),
            "max_usd": b.get("max_usd"),
        }

    def set_budget(
        self,
        *,
        max_total_tokens: Optional[int] = None,
        max_usd: Optional[float] = None,
    ) -> None:
        """Set/clear soft spending caps. Pass None to remove a cap.

        The agent loop checks :meth:`is_over_budget` before each turn and
        bails out with an ErrorEvent when any cap is exceeded — so the
        session always finishes the in-flight turn but won't start a
        new one once the cap is hit.
        """
        b: dict[str, Any] = {}
        if max_total_tokens is not None:
            b["max_total_tokens"] = int(max_total_tokens)
        if max_usd is not None:
            b["max_usd"] = float(max_usd)
        self._meta["budget"] = b
        self._save_meta()

    def is_over_budget(self) -> tuple[bool, str]:
        """Return ``(over, reason)``.

        Reason is empty string when ``over`` is False; otherwise a short
        human-readable explanation that the agent loop can surface as an
        ErrorEvent message.
        """
        b = self.budget
        u = self.usage
        cap_tokens = b.get("max_total_tokens")
        if cap_tokens is not None and u["total_tokens"] >= int(cap_tokens):
            return True, f"token budget exceeded ({u['total_tokens']} >= {cap_tokens})"
        cap_usd = b.get("max_usd")
        if cap_usd is not None:
            try:
                from godbot.core.pricing import compute_cost
                cost = compute_cost(
                    provider=self.provider,
                    model=self.model_name or self.model,
                    input_tokens=u["input_tokens"],
                    output_tokens=u["output_tokens"],
                )
                if cost.matched and cost.usd >= float(cap_usd):
                    return True, f"cost budget exceeded (${cost.usd:.4f} >= ${cap_usd})"
            except Exception:
                pass  # pricing failures don't block the session
        return False, ""

    def _save_meta(self) -> None:
        self._write_meta(self.dir, self._meta)

    @staticmethod
    def _write_meta(sdir: Path, meta: dict[str, Any]) -> None:
        path = sdir / "meta.json"
        tmp = sdir / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _append_event(self, event: dict[str, Any]) -> None:
        path = self.dir / "events.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    def append_user(self, content: str) -> None:
        self._append_event({"type": "user", "content": content})

    def append_assistant_final(self, content: str) -> None:
        self._append_event({"type": "assistant_final", "content": content})

    def append_assistant_tool_call(self, call_id: str, name: str, args: dict[str, Any], raw: str) -> None:
        self._append_event({
            "type": "assistant_tool_call",
            "call_id": call_id, "name": name, "args": args, "raw": raw,
        })

    def append_tool_result(self, call_id: str, content: str, blob: Optional[str] = None) -> None:
        self._append_event({"type": "tool_result", "call_id": call_id, "content": content, "blob": blob})

    def append_synthetic_tool_result(self, content: str) -> None:
        self._append_event({"type": "synthetic_tool_result", "content": content})

    def append_meta_event(self, kind: str, payload: dict[str, Any]) -> None:
        self._append_event({"type": kind, **payload})

    def _events(self) -> Iterable[dict[str, Any]]:
        path = self.dir / "events.jsonl"
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def messages_for_llm(self, max_context: Optional[int] = None) -> list[dict[str, Any]]:
        msgs = self._build_messages()
        if max_context is None or max_context <= 0:
            return msgs
        return _trim_to_context(msgs, max_context)

    def _build_messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ev in self._events():
            t = ev["type"]
            if t == "user":
                out.append({"role": "user", "content": ev["content"]})
            elif t == "assistant_final":
                out.append({"role": "assistant", "content": ev["content"]})
            elif t == "assistant_tool_call":
                # Send the model's own ReAct JSON back as the assistant turn so it sees what it last decided.
                out.append({"role": "assistant", "content": ev["raw"]})
            elif t == "tool_result":
                # Synthesized as a user-role 'tool_result(call_id): ...' message — matches §4.8 system prompt contract.
                out.append({"role": "user", "content": f"tool_result({ev['call_id']}): {ev['content']}"})
            elif t == "synthetic_tool_result":
                out.append({"role": "user", "content": f"<system>{ev['content']}</system>"})
        return out

    def record_tool_result(self, call_id: str, full_result: str) -> str:
        blob_id: Optional[str] = None
        if len(full_result) > BLOB_INLINE_LIMIT:
            blob_id = call_id
            (self.dir / "blobs" / f"{call_id}.txt").write_text(full_result, encoding="utf-8")
        llm_view = _truncate_for_llm(full_result, call_id)
        self.append_tool_result(call_id, llm_view, blob=blob_id)
        return llm_view

    def read_blob(self, call_id: str) -> str:
        path = self.dir / "blobs" / f"{call_id}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8")

    def has_pending_gate(self, call_id: str) -> bool:
        return call_id in self._gate_events and call_id not in self._gate_decisions

    def resolve_gate(
        self,
        call_id: str,
        decision: str,
        args_override: Optional[dict[str, Any]] = None,
    ) -> bool:
        if decision not in {"allow", "deny", "always"}:
            raise ValueError(f"bad decision {decision!r}")
        if call_id not in self._gate_events:
            return False
        self._gate_decisions[call_id] = decision
        if args_override is not None:
            if not isinstance(args_override, dict):
                raise ValueError("args_override must be a dict")
            self._gate_args_overrides[call_id] = dict(args_override)
        self._gate_events[call_id].set()
        meta_payload: dict[str, Any] = {"call_id": call_id, "decision": decision}
        if args_override is not None:
            meta_payload["args_override_keys"] = sorted(args_override.keys())
        self.append_meta_event("gate_decision", meta_payload)
        return True

    async def await_gate(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any],
        emit,
        *,
        fs_diff: Optional[dict[str, Any]] = None,
        timeout: float = 300.0,
    ) -> tuple[str, Optional[dict[str, Any]]]:
        ev = asyncio.Event()
        self._gate_events[call_id] = ev
        await emit(GateEvent(id=call_id, name=name, args=args, fs_diff=fs_diff))
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            decision = self._gate_decisions.get(call_id, "deny")
            override = self._gate_args_overrides.pop(call_id, None)
            return decision, override
        except asyncio.TimeoutError:
            self._gate_args_overrides.pop(call_id, None)
            return "deny", None
        finally:
            self._gate_events.pop(call_id, None)


def _msg_tokens(m: dict[str, Any]) -> int:
    if "content" in m and isinstance(m["content"], str):
        return estimate_tokens(m["content"])
    if m.get("tool_calls"):
        return sum(estimate_tokens(json.dumps(tc)) for tc in m["tool_calls"])
    return 0


def _trim_to_context(msgs: list[dict[str, Any]], max_context: int) -> list[dict[str, Any]]:
    soft = int(max_context * 0.8)
    target = int(max_context * 0.7)
    total = sum(_msg_tokens(m) for m in msgs)
    if total <= soft:
        return msgs
    keep_tail = 2
    head: list[dict[str, Any]] = []
    droppable = msgs[:-keep_tail] if len(msgs) > keep_tail else []
    tail = msgs[-keep_tail:] if len(msgs) > keep_tail else msgs
    dropped = 0
    while droppable and (sum(_msg_tokens(m) for m in head + droppable + tail) > target):
        droppable.pop(0)
        dropped += 1
    if dropped:
        head = [{"role": "system", "content": f"<elided {dropped} earlier messages>"}]
    return head + droppable + tail
