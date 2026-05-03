from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional


class ChannelSessionMap:
    """Persisted channel_id -> godbot session_id map.

    Backing store is a small JSON file. Atomic writes via .tmp + os.replace.
    Corrupt files are silently treated as empty (next write overwrites).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(self._data, dict):
                self._data = {}
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def get(self, channel_id: int) -> Optional[str]:
        return self._data.get(str(channel_id))

    def set(self, channel_id: int, session_id: str) -> None:
        self._data[str(channel_id)] = session_id
        self._save()

    def pop(self, channel_id: int) -> Optional[str]:
        sid = self._data.pop(str(channel_id), None)
        if sid is not None:
            self._save()
        return sid
