import asyncio
import pytest
from godbot.core.session import Session


@pytest.mark.asyncio
async def test_gate_resolves_with_allow(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    emitted = []

    async def emit(ev):
        emitted.append(ev)

    async def resolver():
        for _ in range(20):
            if s.has_pending_gate("c1"):
                s.resolve_gate("c1", "allow")
                return
            await asyncio.sleep(0.01)

    async def caller():
        return await s.await_gate("c1", "run_powershell", {"cmd": "ls"}, emit, timeout=2)

    decision, _ = await asyncio.gather(caller(), resolver())
    assert decision == "allow"
    assert any(e.__class__.__name__ == "GateEvent" for e in emitted)


@pytest.mark.asyncio
async def test_gate_times_out_to_deny(tmp_path):
    s = Session.create(root=tmp_path, model="m")

    async def emit(ev):
        pass

    decision = await s.await_gate("c1", "x", {}, emit, timeout=0.1)
    assert decision == "deny"
