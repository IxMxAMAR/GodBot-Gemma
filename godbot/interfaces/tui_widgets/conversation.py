"""ConversationModel + Textual widgets for the TUI conversation pane.

The model is a pure-Python class (no Textual deps). The widgets that wrap it
are defined later in the same file; they delegate event handling to the model
and re-render reactively. This split keeps the dispatch logic unit-testable
without spinning up a Textual app.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from godbot.core.events import (
    DoneEvent, ErrorEvent, Event, GateEvent,
    TokenEvent, ToolCallEvent, ToolResultEvent,
)


@dataclass
class ToolCard:
    id: str
    name: str
    args: dict
    result: Optional[str] = None
    duration_ms: int = 0
    blob: Optional[str] = None


@dataclass
class GateState:
    id: str
    name: str
    args: dict


@dataclass
class AssistantSegment:
    parts: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "".join(self.parts)


class ConversationModel:
    """Pure model that interprets the streaming Event types.

    Tracks per-turn assistant text segments, tool call cards, pending gates,
    and error banners. Widget code reads this model to render.
    """

    def __init__(self) -> None:
        self._assistants: list[AssistantSegment] = []
        self._users: list[str] = []
        self._tool_cards: dict[str, ToolCard] = {}
        self._tool_card_order: list[str] = []
        self._gates: dict[str, GateState] = {}
        self._errors: list[str] = []
        self._current_assistant: Optional[AssistantSegment] = None

    def handle(self, ev: Event) -> None:
        if isinstance(ev, TokenEvent):
            if self._current_assistant is None:
                self._current_assistant = AssistantSegment()
                self._assistants.append(self._current_assistant)
            self._current_assistant.parts.append(ev.text)
        elif isinstance(ev, ToolCallEvent):
            self._current_assistant = None
            card = ToolCard(id=ev.id, name=ev.name, args=ev.args)
            self._tool_cards[ev.id] = card
            self._tool_card_order.append(ev.id)
        elif isinstance(ev, ToolResultEvent):
            card = self._tool_cards.get(ev.id)
            if card is not None:
                card.result = ev.preview
                card.duration_ms = ev.duration_ms
                card.blob = ev.blob
        elif isinstance(ev, GateEvent):
            self._gates[ev.id] = GateState(id=ev.id, name=ev.name, args=ev.args)
        elif isinstance(ev, ErrorEvent):
            self._errors.append(ev.message)
        elif isinstance(ev, DoneEvent):
            self._current_assistant = None

    def append_user(self, text: str) -> None:
        self._users.append(text)
        self._current_assistant = None

    def resolve_gate(self, call_id: str) -> Optional[GateState]:
        return self._gates.pop(call_id, None)

    # --- Read accessors used by tests + widgets ---
    def last_assistant_text(self) -> str:
        return self._assistants[-1].text() if self._assistants else ""

    def assistant_count(self) -> int:
        return len(self._assistants)

    def has_tool_card(self, call_id: str) -> bool:
        return call_id in self._tool_cards

    def tool_card(self, call_id: str) -> ToolCard:
        return self._tool_cards[call_id]

    def pending_gates(self) -> list[GateState]:
        return list(self._gates.values())

    def has_error(self, fragment: str) -> bool:
        return any(fragment in e for e in self._errors)
