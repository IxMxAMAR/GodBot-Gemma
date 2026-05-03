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
