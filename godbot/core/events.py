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
    ErrorEvent: "error",
    DoneEvent: "done",
}


def event_to_dict(e: Event) -> dict[str, Any]:
    d = asdict(e)
    d["type"] = _TYPE_NAMES[type(e)]
    return d
