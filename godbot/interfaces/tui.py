from __future__ import annotations
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header

from godbot.client import Session
from godbot.client.embedded import EmbeddedRunner
from godbot.config import load_config
from godbot.interfaces.tui_widgets.composer import MessageInput, SendMessage
from godbot.interfaces.tui_widgets.conversation import (
    ConversationView, GateDecided,
)
from godbot.interfaces.tui_widgets.sidebar import (
    NewSessionRequested, RagCollectionChosen, SessionPicked,
    Sidebar, StopRequested, ToolToggled,
)


class TuiApp(App):
    """Top-level Textual application for GodBot."""

    CSS_PATH = "tui.tcss"
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "stop", "Stop", priority=True),
        Binding("ctrl+n", "new_session", "New session"),
        Binding("tab", "focus_next", "Next pane"),
        Binding("shift+tab", "focus_previous", "Prev pane"),
    ]

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:7878",
        auto_launch: bool = True,
        sid: Optional[str] = None,
        sessions_root: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._auto_launch = auto_launch
        self._sid = sid
        self._sessions_root = sessions_root or (Path.cwd() / "sessions")
        self._session: Optional[Session] = None
        self._stream_task: Optional[asyncio.Task] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            yield Sidebar()
            with Vertical(id="main"):
                yield ConversationView()
                yield MessageInput()
        yield Footer()

    @property
    def conversation(self) -> ConversationView:
        return self.query_one(ConversationView)

    @property
    def sidebar(self) -> Sidebar:
        return self.query_one(Sidebar)

    async def on_mount(self) -> None:
        self.title = "GodBot"
        await self._connect_session()

    def _embedded_factory(self):
        return EmbeddedRunner.create(sessions_root=self._sessions_root, sid=self._sid)

    async def _connect_session(self) -> None:
        try:
            self._session = Session(
                sid=self._sid,
                base_url=self._base_url,
                auto_launch=self._auto_launch,
                embedded_factory=self._embedded_factory,
            )
            await self._session.connect()
        except Exception as e:
            self.conversation.post_message(_exception_to_error_event(e))
            return
        self.sub_title = f"{self._session.mode} · {self._session.sid}"
        await self._populate_sidebar()

    async def _populate_sidebar(self) -> None:
        # Daemon mode: pull from Client; embedded: pull from local registry/RAG dir.
        try:
            if self._session.mode == "daemon":
                client = self._session._client  # internal access -- small cost for v0.1
                tools = await client.list_tools()
                sessions = await client.list_sessions()
                colls = await client.list_rag_collections()
            else:
                from godbot.core.registry import DEFAULT
                tools = [
                    {"name": t.name, "description": t.description, "dangerous": t.dangerous}
                    for t in DEFAULT.all()
                ]
                sessions = []
                rag_root = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot"))) / "rag"
                colls = sorted([d.name for d in rag_root.iterdir() if d.is_dir()]) if rag_root.exists() else []
            self.sidebar.populate_sessions(sessions)
            self.sidebar.populate_tools(tools)
            self.sidebar.populate_rag(colls, current=None)
        except Exception:
            pass  # sidebar populates on a best-effort basis

    async def on_send_message(self, msg: SendMessage) -> None:
        if self._session is None:
            return
        self.conversation.post_user(msg.text)

        async def _stream():
            try:
                async for ev in self._session.run(msg.text):
                    await self.conversation.handle_event(ev)
            except Exception as e:
                await self.conversation.handle_event(_exception_to_error_event(e))

        if self._stream_task and not self._stream_task.done():
            return  # one stream at a time
        self._stream_task = asyncio.create_task(_stream())

    async def on_gate_decided(self, msg: GateDecided) -> None:
        if self._session is not None:
            await self._session.resolve_gate(msg.call_id, msg.decision)

    async def on_stop_requested(self, msg: StopRequested) -> None:
        if self._session is not None:
            await self._session.stop()

    async def on_new_session_requested(self, msg: NewSessionRequested) -> None:
        await self._switch_session(sid=None)

    async def on_session_picked(self, msg: SessionPicked) -> None:
        await self._switch_session(sid=msg.sid)

    async def on_tool_toggled(self, msg: ToolToggled) -> None:
        if self._session is None or self._session.mode != "daemon":
            return
        client = self._session._client
        await client.toggle_tool(self._session.sid, msg.name, msg.enabled)

    async def on_rag_collection_chosen(self, msg: RagCollectionChosen) -> None:
        if self._session is None or self._session.mode != "daemon":
            return
        client = self._session._client
        await client.use_rag_collection(self._session.sid, msg.collection)

    async def _switch_session(self, sid: Optional[str]) -> None:
        if self._stream_task and not self._stream_task.done():
            await self._session.stop()
        if self._session is not None:
            await self._session.close()
        self._sid = sid
        self.conversation.model.__init__()  # reset model state
        # Recreate the widget cleanly.
        view = self.conversation
        await view.remove_children()
        await self._connect_session()

    def action_stop(self) -> None:
        if self._session is not None:
            self.run_worker(self._session.stop(), exclusive=False)


def _exception_to_error_event(exc: Exception):
    from godbot.core.events import ErrorEvent
    return ErrorEvent(message=f"{type(exc).__name__}: {exc}", recoverable=False)


def main() -> int:
    parser = argparse.ArgumentParser(prog="godbot-tui")
    parser.add_argument("--base-url", default="http://127.0.0.1:7878")
    parser.add_argument("--no-daemon", action="store_true",
                        help="Skip daemon auto-launch; use embedded mode immediately.")
    parser.add_argument("--resume", help="Resume an existing session id.")
    args = parser.parse_args()

    app = TuiApp(
        base_url=args.base_url,
        auto_launch=not args.no_daemon,
        sid=args.resume,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
