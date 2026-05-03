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

    @property
    def model(self) -> str:
        return self._meta.get("model", "")

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
    def create(cls, root: Path, model: str, rag_collection: Optional[str] = None) -> "Session":
        sid = _now_id()
        sdir = Path(root) / sid
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "blobs").mkdir(exist_ok=True)
        meta = {
            "id": sid,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "ended_at": None,
            "model": model,
            "tool_overrides": None,
            "auto_approved_tools": [],
            "yolo": False,
            "rag_collection": rag_collection,
        }
        cls._write_meta(sdir, meta)
        (sdir / "events.jsonl").touch()
        return cls(Path(root), sid, meta)

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

    def resolve_gate(self, call_id: str, decision: str) -> bool:
        if decision not in {"allow", "deny", "always"}:
            raise ValueError(f"bad decision {decision!r}")
        if call_id not in self._gate_events:
            return False
        self._gate_decisions[call_id] = decision
        self._gate_events[call_id].set()
        self.append_meta_event("gate_decision", {"call_id": call_id, "decision": decision})
        return True

    async def await_gate(
        self, call_id: str, name: str, args: dict[str, Any], emit, timeout: float = 300.0,
    ) -> str:
        ev = asyncio.Event()
        self._gate_events[call_id] = ev
        await emit(GateEvent(id=call_id, name=name, args=args))
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return self._gate_decisions.get(call_id, "deny")
        except asyncio.TimeoutError:
            return "deny"
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
