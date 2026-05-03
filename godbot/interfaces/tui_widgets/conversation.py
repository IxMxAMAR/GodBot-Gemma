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


# ---------------------------------------------------------------------------
# Textual widgets (kept in same module so the model + view live together).
# ---------------------------------------------------------------------------
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Button, Collapsible
from textual.reactive import reactive
from textual.message import Message


_GROUP_BORDERS = {
    "fs": "blue",
    "shell": "red",
    "web": "green",
    "python": "magenta",
    "memory": "cyan",
    "task": "yellow",
    "rag": "purple",
}


def _group_for(name: str) -> str:
    if name in {"read_file", "write_file", "edit_file", "glob", "grep", "list_dir", "read_blob"}:
        return "fs"
    if name in {"run_powershell", "run_bash"}:
        return "shell"
    if name in {"web_fetch", "web_search"}:
        return "web"
    if name == "run_python":
        return "python"
    if name in {"save_note", "recall_notes"}:
        return "memory"
    if name in {"todo_set", "todo_check"}:
        return "task"
    if name == "search_knowledge":
        return "rag"
    return "default"


class GateDecided(Message):
    def __init__(self, call_id: str, decision: str) -> None:
        super().__init__()
        self.call_id = call_id
        self.decision = decision


class UserBubble(Static):
    DEFAULT_CSS = """
    UserBubble {
        background: $accent;
        color: $background;
        padding: 0 1;
        margin: 1 4 0 12;
        text-align: right;
    }
    """


class AssistantBubble(Static):
    DEFAULT_CSS = """
    AssistantBubble {
        padding: 0 1;
        margin: 0 0 0 0;
    }
    """
    text = reactive("", layout=True)

    def append(self, chunk: str) -> None:
        self.text = self.text + chunk

    def render(self) -> str:
        return self.text


class ToolCallCardWidget(Static):
    DEFAULT_CSS = """
    ToolCallCardWidget {
        border-left: heavy $primary;
        background: $surface;
        margin: 0 0 1 0;
        padding: 0 1;
    }
    """

    def __init__(self, card: ToolCard) -> None:
        super().__init__()
        self.card = card

    def render(self) -> str:
        args_preview = ", ".join(f"{k}={v!r}" for k, v in list(self.card.args.items())[:3])
        head = f"[bold]▸ {self.card.name}[/]([dim]{args_preview}[/])"
        if self.card.result is not None:
            head += f"  [dim][{self.card.duration_ms}ms][/]"
            preview = self.card.result.splitlines()[0][:80] if self.card.result else ""
            head += f"\n  [dim]↳ {preview}[/]"
            if self.card.blob:
                head += f" [dim](blob {self.card.blob})[/]"
        else:
            head += "  [dim]…[/]"
        return head


class GateCardWidget(Static):
    DEFAULT_CSS = """
    GateCardWidget {
        border: heavy $error;
        background: $surface;
        margin: 1 2;
        padding: 1;
    }
    GateCardWidget Button { margin: 0 1 0 0; }
    """

    def __init__(self, state: GateState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.state.args.items())
        yield Static(f"[bold]Approve {self.state.name}?[/]\n{args_str}")
        yield Button("Allow", id="allow", variant="success")
        yield Button("Always", id="always", variant="primary")
        yield Button("Deny", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        decision = event.button.id or "deny"
        self.post_message(GateDecided(self.state.id, decision))


class ErrorBanner(Static):
    DEFAULT_CSS = """
    ErrorBanner {
        background: $error;
        color: $text;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """


class ConversationView(VerticalScroll):
    """Textual widget that owns the ConversationModel and renders it.

    Subscribers post GateDecided messages up to the App when a gate button is clicked.
    """

    DEFAULT_CSS = """
    ConversationView { padding: 1 2; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = ConversationModel()
        self._current_assistant_widget: Optional[AssistantBubble] = None
        self._tool_widgets: dict[str, ToolCallCardWidget] = {}

    def post_user(self, text: str) -> None:
        self.model.append_user(text)
        self._current_assistant_widget = None
        self.mount(UserBubble(text))
        self.scroll_end(animate=False)

    async def handle_event(self, ev: Event) -> None:
        self.model.handle(ev)
        if isinstance(ev, TokenEvent):
            if self._current_assistant_widget is None:
                self._current_assistant_widget = AssistantBubble()
                await self.mount(self._current_assistant_widget)
            self._current_assistant_widget.append(ev.text)
        elif isinstance(ev, ToolCallEvent):
            self._current_assistant_widget = None
            card = self.model.tool_card(ev.id)
            w = ToolCallCardWidget(card)
            self._tool_widgets[ev.id] = w
            await self.mount(w)
        elif isinstance(ev, ToolResultEvent):
            w = self._tool_widgets.get(ev.id)
            if w is not None:
                w.refresh()
        elif isinstance(ev, GateEvent):
            await self.mount(GateCardWidget(self.model.pending_gates()[-1]))
        elif isinstance(ev, ErrorEvent):
            await self.mount(ErrorBanner(f"error: {ev.message}"))
        elif isinstance(ev, DoneEvent):
            self._current_assistant_widget = None
        self.scroll_end(animate=False)
