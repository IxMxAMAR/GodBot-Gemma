"""Tests for sub-project 91 — sleep tool."""
from __future__ import annotations

import time

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import sleep as agent_sleep


def test_sleep_actually_pauses():
    started = time.monotonic()
    out = agent_sleep(seconds=0.3)
    elapsed = time.monotonic() - started
    assert "slept" in out
    assert elapsed >= 0.25  # allow small slack


def test_sleep_zero_is_instant():
    started = time.monotonic()
    agent_sleep(seconds=0)
    elapsed = time.monotonic() - started
    assert elapsed < 0.2


def test_sleep_negative_clamped_to_zero():
    started = time.monotonic()
    agent_sleep(seconds=-5)
    elapsed = time.monotonic() - started
    # Should not actually wait (clamped to 0).
    assert elapsed < 0.5


def test_sleep_caps_at_60(monkeypatch):
    """Ask for 9999 seconds; verify it clamps to 60."""
    captured = []

    def fake_sleep(s):
        captured.append(s)

    import time as _time
    monkeypatch.setattr(_time, "sleep", fake_sleep)
    agent_sleep(seconds=9999)
    assert captured == [60.0]


def test_sleep_returns_actual_duration():
    out = agent_sleep(seconds=0.5)
    assert "slept 0.50s" in out


def test_sleep_invalid_input_errors():
    out = agent_sleep(seconds="not-a-number")  # type: ignore[arg-type]
    assert out.startswith("[error]")


def test_sleep_is_registered_with_extended_timeout():
    spec = DEFAULT.spec("sleep")
    assert spec is not None
    # Tool's timeout must be larger than 60s cap so a max-duration sleep
    # doesn't get killed by the registry timeout.
    assert spec.timeout >= 70
