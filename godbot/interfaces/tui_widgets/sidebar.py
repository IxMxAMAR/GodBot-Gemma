from __future__ import annotations
from typing import Iterable, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Checkbox, Label, ListItem, ListView, Select, Static


class SessionPicked(Message):
    def __init__(self, sid: str) -> None:
        super().__init__()
        self.sid = sid


class NewSessionRequested(Message):
    pass


class StopRequested(Message):
    pass


class ToolToggled(Message):
    def __init__(self, name: str, enabled: bool) -> None:
        super().__init__()
        self.name = name
        self.enabled = enabled


class RagCollectionChosen(Message):
    def __init__(self, collection: Optional[str]) -> None:
        super().__init__()
        self.collection = collection


class Sidebar(Vertical):
    DEFAULT_CSS = """
    Sidebar {
        width: 24;
        background: $surface;
        padding: 1;
    }
    Sidebar Label.title {
        color: $text-muted;
        text-style: bold;
        margin: 1 0 0 0;
    }
    Sidebar Button { width: 100%; margin: 1 0 0 0; }
    Sidebar ListView { max-height: 8; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Sessions", classes="title")
        self._session_list = ListView(id="sessions")
        yield self._session_list
        yield Button("+ New", id="new-session")
        yield Label("Tools", classes="title")
        self._tool_list = ListView(id="tools")
        yield self._tool_list
        yield Label("RAG", classes="title")
        self._rag = Select.from_values([], id="rag", prompt="(none)")
        yield self._rag
        yield Button("Stop", id="stop", variant="error")

    def populate_sessions(self, sessions: Iterable[dict]) -> None:
        self._session_list.clear()
        for s in sessions:
            self._session_list.append(ListItem(Label(s["id"]), id=f"sess-{s['id']}"))

    def populate_tools(self, tools: Iterable[dict], enabled_set: Optional[set[str]] = None) -> None:
        self._tool_list.clear()
        for t in tools:
            checked = (enabled_set is None) or (t["name"] in enabled_set)
            label = ("⚠ " if t.get("dangerous") else "") + t["name"]
            cb = Checkbox(label, value=checked, id=f"tool-{t['name']}")
            self._tool_list.append(ListItem(cb, id=f"tool-item-{t['name']}"))

    def populate_rag(self, collections: list[str], current: Optional[str]) -> None:
        self._rag.set_options([(c, c) for c in collections])
        if current and current in collections:
            self._rag.value = current

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-session":
            self.post_message(NewSessionRequested())
        elif event.button.id == "stop":
            self.post_message(StopRequested())

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id and event.item.id.startswith("sess-"):
            sid = event.item.id[len("sess-"):]
            self.post_message(SessionPicked(sid))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        cid = event.checkbox.id or ""
        if cid.startswith("tool-"):
            name = cid[len("tool-"):]
            self.post_message(ToolToggled(name, event.value))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "rag":
            value = event.value if event.value != Select.BLANK else None
            self.post_message(RagCollectionChosen(value))
