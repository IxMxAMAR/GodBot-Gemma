"""Per-session event log with multi-subscriber fan-out (sub-project 21).

The original chat-stream design used a single :class:`asyncio.Queue` per
session: one consumer drained, a second got HTTP 409. That was fine when
"one user, one tab" was the contract — but a flaky network connection
or a second Studio window dropping the SSE stream meant the user lost
the in-flight turn entirely.

:class:`EventLog` replaces the queue with a sequence-numbered ring buffer
plus a wakers list. Multiple subscribers can attach concurrently; a
reconnecting subscriber passes its last-seen sequence and replays
buffered events past that cursor before tailing for new ones. The buffer
is bounded so a session that produced thousands of events doesn't hold
them forever — a subscriber that's far behind will see ``[0..head]``,
miss the elided middle, and resume from current. Sub-1k events per turn
is well within the cap.

Closing the log unblocks every waker so subscribers exit cleanly.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Optional


# Per-session event ring size. SSE events are tiny dicts; 2000 is well
# under a few MB even for chatty turns. A subscriber more than 2000
# events behind silently skips to current.
DEFAULT_MAX_EVENTS = 2000


@dataclass
class LoggedEvent:
    """One SSE event with a stable sequence id."""

    seq: int
    name: str
    data: dict


class EventLog:
    """Bounded ring buffer + multi-subscriber fanout for one session's stream.

    Append-from-runner / follow-from-subscriber. ``seq`` numbers start at
    1 and increment monotonically. ``close()`` is final — no further
    events accepted, all current waiters are released so their
    ``follow()`` generators exit.
    """

    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS) -> None:
        self._events: deque[LoggedEvent] = deque(maxlen=max_events)
        self._counter = 0
        self._wakers: list[asyncio.Event] = []
        self._closed = False
        # Used only by tests reaching across threads; the runtime is
        # single-event-loop so the GIL plus the deque's internal locking
        # is enough for the operations that matter.
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def head_seq(self) -> int:
        """The most recent published seq (0 if none)."""
        return self._counter

    def publish(self, name: str, data: dict) -> int:
        """Append an event and wake all subscribers. Returns the new seq.

        Raises :class:`RuntimeError` after :meth:`close` — late publishes
        from a runner that's already torn down would otherwise silently
        drop on the floor.
        """
        if self._closed:
            raise RuntimeError("EventLog is closed; no further publishes")
        with self._lock:
            self._counter += 1
            ev = LoggedEvent(seq=self._counter, name=name, data=data)
            self._events.append(ev)
            wakers = list(self._wakers)
        for w in wakers:
            try:
                w.set()
            except Exception:
                pass
        return ev.seq

    def close(self) -> None:
        """Mark the log closed. Wakes every subscriber so their loops exit."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            wakers = list(self._wakers)
        for w in wakers:
            try:
                w.set()
            except Exception:
                pass

    def buffered_after(self, since_seq: int) -> list[LoggedEvent]:
        """Snapshot events with ``seq > since_seq`` currently in the ring.

        Useful for a re-connecting subscriber that wants its replay
        without subscribing yet. The ring has finite capacity, so
        ``since_seq`` older than the oldest buffered event is treated as
        "you missed events; resume from current head" — i.e. we still
        return whatever is in the buffer.
        """
        with self._lock:
            return [e for e in self._events if e.seq > since_seq]

    async def follow(
        self, since_seq: int = 0, idle_timeout: Optional[float] = None,
    ) -> AsyncIterator[LoggedEvent]:
        """Async generator yielding events with ``seq > since_seq`` then tail.

        ``idle_timeout`` returns control to the caller every N seconds
        with no event yielded — used by the SSE endpoint to fire
        keepalive pings without busy-polling. ``None`` means "wait
        forever for the next event or close()".
        """
        cursor = since_seq
        while True:
            with self._lock:
                pending = [e for e in self._events if e.seq > cursor]
            if pending:
                for ev in pending:
                    yield ev
                    cursor = ev.seq
                continue
            if self._closed:
                return
            waker = asyncio.Event()
            with self._lock:
                self._wakers.append(waker)
            try:
                if idle_timeout is None:
                    await waker.wait()
                else:
                    try:
                        await asyncio.wait_for(waker.wait(), timeout=idle_timeout)
                    except asyncio.TimeoutError:
                        # Surface "idle tick" as a sentinel by yielding nothing
                        # and looping; the caller can interleave a heartbeat.
                        # We model this by yielding a synthetic ping event.
                        yield LoggedEvent(seq=cursor, name="__idle__", data={})
            finally:
                with self._lock:
                    if waker in self._wakers:
                        self._wakers.remove(waker)
