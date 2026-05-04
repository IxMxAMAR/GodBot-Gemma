"""Unit tests for godbot.core.event_log (sub-project 21)."""
from __future__ import annotations

import asyncio

import pytest

from godbot.core.event_log import EventLog


@pytest.mark.asyncio
async def test_publish_assigns_monotonic_seq():
    log = EventLog()
    s1 = log.publish("token", {"text": "a"})
    s2 = log.publish("token", {"text": "b"})
    assert s1 == 1
    assert s2 == 2
    assert log.head_seq == 2


@pytest.mark.asyncio
async def test_buffered_after_filters_by_seq():
    log = EventLog()
    log.publish("a", {"x": 1})
    log.publish("b", {"x": 2})
    log.publish("c", {"x": 3})
    out = log.buffered_after(1)
    assert [e.name for e in out] == ["b", "c"]
    assert log.buffered_after(99) == []


@pytest.mark.asyncio
async def test_follow_replays_then_tails():
    log = EventLog()
    log.publish("a", {})
    log.publish("b", {})

    seen: list[str] = []

    async def consumer():
        async for ev in log.follow(since_seq=0, idle_timeout=0.05):
            if ev.name == "__idle__":
                continue
            seen.append(ev.name)
            if ev.name == "z":
                return

    task = asyncio.create_task(consumer())
    # Give consumer a tick to drain the buffered events.
    await asyncio.sleep(0.1)
    assert seen[:2] == ["a", "b"]
    log.publish("c", {})
    log.publish("z", {})
    await asyncio.wait_for(task, timeout=2.0)
    assert seen == ["a", "b", "c", "z"]


@pytest.mark.asyncio
async def test_follow_resumes_from_cursor():
    log = EventLog()
    log.publish("a", {})
    log.publish("b", {})
    log.publish("c", {})
    seen: list[str] = []

    async def consumer():
        async for ev in log.follow(since_seq=2, idle_timeout=0.05):
            if ev.name == "__idle__":
                continue
            seen.append(ev.name)
            if ev.name == "c":
                return

    await asyncio.wait_for(consumer(), timeout=2.0)
    assert seen == ["c"]


@pytest.mark.asyncio
async def test_close_unblocks_followers():
    log = EventLog()
    out: list[str] = []

    async def consumer():
        async for ev in log.follow(since_seq=0, idle_timeout=None):
            out.append(ev.name)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.1)
    log.publish("a", {})
    await asyncio.sleep(0.05)
    log.close()
    await asyncio.wait_for(task, timeout=2.0)
    assert out == ["a"]
    # After close, publish raises.
    with pytest.raises(RuntimeError):
        log.publish("b", {})


@pytest.mark.asyncio
async def test_multiple_concurrent_subscribers_each_see_all_events():
    log = EventLog()
    a_seen: list[str] = []
    b_seen: list[str] = []

    async def consume(out: list[str]):
        async for ev in log.follow(since_seq=0, idle_timeout=0.05):
            if ev.name == "__idle__":
                continue
            out.append(ev.name)
            if ev.name == "stop":
                return

    a = asyncio.create_task(consume(a_seen))
    b = asyncio.create_task(consume(b_seen))
    await asyncio.sleep(0.05)
    log.publish("e1", {})
    log.publish("e2", {})
    log.publish("stop", {})
    await asyncio.wait_for(asyncio.gather(a, b), timeout=2.0)
    assert a_seen == ["e1", "e2", "stop"]
    assert b_seen == ["e1", "e2", "stop"]


@pytest.mark.asyncio
async def test_ring_buffer_caps_size():
    log = EventLog(max_events=3)
    for i in range(5):
        log.publish("x", {"i": i})
    # Buffer holds only the last 3.
    buf = log.buffered_after(0)
    assert len(buf) == 3
    assert [e.data["i"] for e in buf] == [2, 3, 4]


@pytest.mark.asyncio
async def test_idle_timeout_emits_synthetic_ping():
    """With no activity, follow() yields synthetic __idle__ events."""
    log = EventLog()
    saw_idle = False

    async def consumer():
        nonlocal saw_idle
        async for ev in log.follow(since_seq=0, idle_timeout=0.05):
            if ev.name == "__idle__":
                saw_idle = True
                log.close()

    await asyncio.wait_for(consumer(), timeout=2.0)
    assert saw_idle is True


@pytest.mark.asyncio
async def test_late_subscriber_replays_buffered():
    log = EventLog()
    log.publish("a", {})
    log.publish("b", {})
    seen: list[str] = []

    async def consumer():
        async for ev in log.follow(since_seq=0, idle_timeout=0.05):
            if ev.name == "__idle__":
                continue
            seen.append(ev.name)
            if ev.name == "done":
                return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.1)
    log.publish("done", {})
    await asyncio.wait_for(task, timeout=2.0)
    assert seen == ["a", "b", "done"]
