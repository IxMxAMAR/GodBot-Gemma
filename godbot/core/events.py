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


# Sanity: every Event Union member must be registered in _TYPE_NAMES.
# Fails at module load (not at first use) if a new event type is added without registration.
_registered = set(_TYPE_NAMES.keys())
_declared = set(Event.__args__)
assert _registered == _declared, (
    f"_TYPE_NAMES out of sync with Event Union: "
    f"missing={_declared - _registered}, extra={_registered - _declared}"
)
