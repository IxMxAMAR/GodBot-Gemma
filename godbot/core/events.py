from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Optional, Union


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResultEvent:
    id: str
    preview: str
    blob: Optional[str]
    duration_ms: int


@dataclass(frozen=True)
class GateEvent:
    id: str
    name: str
    args: dict[str, Any]
    # Optional FS-write diff payload: {path, before, after}. Populated for
    # write_file/edit_file gates so UIs can render an inline diff. None for
    # all other gated tools (run_powershell, etc.) — UIs fall through to the
    # standard gate card in that case.
    fs_diff: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ErrorEvent:
    """Loop-level error event. ``recoverable=False`` terminates the turn."""
    message: str
    recoverable: bool = False


@dataclass(frozen=True)
class DoneEvent:
    step_count: int


Event = Union[TokenEvent, ToolCallEvent, ToolResultEvent, GateEvent, ErrorEvent, DoneEvent]


_TYPE_NAMES = {
    TokenEvent: "token",
    ToolCallEvent: "tool_call",
    ToolResultEvent: "tool_result",
    GateEvent: "gate",
    # Renamed from "error" to avoid colliding with the EventSource builtin
    # `error` event (fired on connection drop with no `data` payload, which
    # would crash a JSON.parse-based handler on the frontend).
    ErrorEvent: "agent_error",
    DoneEvent: "done",
}


def event_to_dict(e: Event) -> dict[str, Any]:
    d = asdict(e)
    d["type"] = _TYPE_NAMES[type(e)]
    return d


_NAME_TO_CLASS: dict[str, type] = {v: k for k, v in _TYPE_NAMES.items()}


def dict_to_event(d: dict[str, Any]) -> Event:
    """Reconstruct an Event from its serialized dict.

    Does not mutate the input. Tolerates unknown fields by filtering against
    the dataclass's declared fields.
    """
    payload = dict(d)
    type_name = payload.pop("type")
    cls = _NAME_TO_CLASS[type_name]
    known = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in payload.items() if k in known}
    return cls(**filtered)


# Sanity: every Event Union member must be registered in _TYPE_NAMES.
# Fails at module load (not at first use) if a new event type is added without registration.
_registered = set(_TYPE_NAMES.keys())
_declared = set(Event.__args__)
assert _registered == _declared, (
    f"_TYPE_NAMES out of sync with Event Union: "
    f"missing={_declared - _registered}, extra={_registered - _declared}"
)
