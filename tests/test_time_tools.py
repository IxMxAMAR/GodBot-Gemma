"""Tests for sub-project 67 — now_iso + parse_iso + time_ago."""
from __future__ import annotations

import re

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import now_iso, parse_iso, time_ago


# --- now_iso ----


def test_now_iso_local_format():
    out = now_iso()
    # Seconds-resolution ISO: YYYY-MM-DDTHH:MM:SS optional offset.
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", out)


def test_now_iso_utc_format():
    out = now_iso(tz="utc")
    # UTC always carries +00:00 offset.
    assert "+00:00" in out


# --- parse_iso ----


def test_parse_iso_full_timestamp():
    out = parse_iso(text="2026-05-05T12:34:56+00:00")
    assert "year=2026" in out
    assert "month=05" in out
    assert "day=05" in out
    assert "hour=12" in out
    assert "minute=34" in out
    assert "second=56" in out
    assert "epoch=" in out


def test_parse_iso_date_only():
    out = parse_iso(text="2026-12-25")
    assert "year=2026" in out
    assert "month=12" in out
    assert "day=25" in out


def test_parse_iso_z_suffix():
    out = parse_iso(text="2026-01-01T00:00:00Z")
    assert "year=2026" in out
    assert "month=01" in out


def test_parse_iso_weekday():
    """2026-05-05 is a Tuesday."""
    out = parse_iso(text="2026-05-05")
    assert "weekday=Tuesday" in out


def test_parse_iso_invalid_returns_error():
    out = parse_iso(text="not-a-date")
    assert out.startswith("[error]")


def test_parse_iso_empty_errors():
    out = parse_iso(text="")
    assert out.startswith("[error]")


# --- time_ago ----


def test_time_ago_recent_seconds():
    """Now-minus-10 seconds → '10 seconds ago' approximately."""
    from datetime import datetime, timezone, timedelta
    past = (datetime.now(tz=timezone.utc) - timedelta(seconds=10)).isoformat(timespec="seconds")
    out = time_ago(text=past)
    assert "ago" in out
    assert "second" in out


def test_time_ago_future_renders_in():
    from datetime import datetime, timezone, timedelta
    future = (datetime.now(tz=timezone.utc) + timedelta(hours=2)).isoformat(timespec="seconds")
    out = time_ago(text=future)
    assert out.startswith("in ")
    assert "hour" in out


def test_time_ago_days():
    from datetime import datetime, timezone, timedelta
    past = (datetime.now(tz=timezone.utc) - timedelta(days=3)).isoformat(timespec="seconds")
    out = time_ago(text=past)
    assert "day" in out


def test_time_ago_invalid_errors():
    out = time_ago(text="garbage")
    assert out.startswith("[error]")


def test_time_ago_empty_errors():
    out = time_ago(text="")
    assert out.startswith("[error]")


# --- registration ----


def test_time_tools_registered_non_dangerous():
    for name in ("now_iso", "parse_iso", "time_ago"):
        spec = DEFAULT.spec(name)
        assert spec is not None
        assert spec.dangerous is False
