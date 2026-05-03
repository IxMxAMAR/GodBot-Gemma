from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, Union


class ChannelSessionMap:
    """Persisted channel_id -> godbot session_id (or {sid, workspace}) map.

    Backing store is a small JSON file. Atomic writes via .tmp + os.replace.
    Corrupt files are silently treated as empty (next write overwrites).

    Schema is dual-format for forward/backward compatibility:
      - Legacy: ``{"<channel_id>": "<sid>"}`` (bare string)
      - New:    ``{"<channel_id>": {"sid": "<sid>", "workspace": "<path>"}}``
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Union[str, dict]] = {}
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
        """Returns the session_id for this channel."""
        entry = self._data.get(str(channel_id))
        if isinstance(entry, dict):
            return entry.get("sid")
        return entry  # legacy: bare string (or None if missing)

    def set(self, channel_id: int, session_id: str, workspace: Optional[str] = None) -> None:
        if workspace:
            self._data[str(channel_id)] = {"sid": session_id, "workspace": workspace}
        else:
            self._data[str(channel_id)] = session_id
        self._save()

    def get_workspace(self, channel_id: int) -> Optional[str]:
        entry = self._data.get(str(channel_id))
        if isinstance(entry, dict):
            return entry.get("workspace")
        return None

    def pop(self, channel_id: int) -> Optional[str]:
        entry = self._data.pop(str(channel_id), None)
        if entry is not None:
            self._save()
        if isinstance(entry, dict):
            return entry.get("sid")
        return entry
