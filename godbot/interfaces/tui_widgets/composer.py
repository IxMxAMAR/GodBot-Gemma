from __future__ import annotations
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, TextArea


class SendMessage(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class MessageInput(Horizontal):
    DEFAULT_CSS = """
    MessageInput { height: 5; padding: 0 1; }
    MessageInput TextArea { height: 100%; width: 1fr; }
    MessageInput Button { width: 8; margin-left: 1; }
    """

    def compose(self) -> ComposeResult:
        self._area = TextArea(id="msg")
        yield self._area
        yield Button("Send", id="send", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            self._submit()

    async def on_key(self, event) -> None:
        if event.key == "enter" and not event.shift:
            event.prevent_default()
            event.stop()
            self._submit()

    def _submit(self) -> None:
        text = self._area.text.strip()
        if not text:
            return
        self._area.clear()
        self.post_message(SendMessage(text))
