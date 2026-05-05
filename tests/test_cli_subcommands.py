"""Tests for the new daemon-driven subcommands on godbot-cli.

Each subcommand mocks the corresponding HTTP endpoint with respx and
exercises the entry point via ``cli.main([...])``. The tests assert both
the exit code and the rendered output (captured via capsys).
"""
from __future__ import annotations

import httpx
import pytest
import respx

from godbot.interfaces import cli


# ---------- argv normalization (preserves backward-compat REPL flags) ----------

def test_normalize_argv_inserts_repl_when_empty():
    assert cli._normalize_argv([]) == ["repl"]


def test_normalize_argv_inserts_repl_for_legacy_flags():
    # Pre-subparser invocation: `godbot-cli --resume=last --yolo`.
    # Must keep working — argv gets `repl` prepended so it routes to REPL.
    assert cli._normalize_argv(["--resume=last", "--yolo"]) == [
        "repl", "--resume=last", "--yolo",
    ]


def test_normalize_argv_keeps_known_subcommands_intact():
    for sub in ("repl", "quickrun", "stats", "sessions"):
        assert cli._normalize_argv([sub])[0] == sub


# ---------- quickrun ----------

@respx.mock
def test_quickrun_subcommand_prints_result_and_exits_zero(capsys):
    respx.post("http://127.0.0.1:7878/api/agent/quickrun").mock(
        return_value=httpx.Response(200, json={
            "result": "the answer is 42",
            "session_id": "s-abc",
            "task_id": "t-xyz",
            "tool_calls": 2,
            "step_count": 4,
            "elapsed_ms": 1234,
            "status": "done",
        })
    )
    rc = cli.main(["quickrun", "what is the answer", "--max-wait-seconds", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "the answer is 42" in out
    assert "status=done" in out
    assert "session=s-abc" in out


@respx.mock
def test_quickrun_subcommand_returns_2_when_status_not_done(capsys):
    respx.post("http://127.0.0.1:7878/api/agent/quickrun").mock(
        return_value=httpx.Response(200, json={
            "result": "partial",
            "session_id": "s1",
            "task_id": "t1",
            "tool_calls": 0,
            "step_count": 1,
            "elapsed_ms": 100,
            "status": "running",
        })
    )
    rc = cli.main(["quickrun", "do a thing", "--max-wait-seconds", "5"])
    # status != "done" → exit 2 so shell scripts can branch on it.
    assert rc == 2


@respx.mock
def test_quickrun_subcommand_reports_http_error_and_exits_one(capsys):
    respx.post("http://127.0.0.1:7878/api/agent/quickrun").mock(
        return_value=httpx.Response(400, json={"detail": "goal required"})
    )
    rc = cli.main(["quickrun", "x", "--max-wait-seconds", "5"])
    assert rc == 1
    assert "error" in capsys.readouterr().out.lower()


# ---------- stats ----------

@respx.mock
def test_stats_subcommand_renders_summary_and_top_tools(capsys):
    respx.get("http://127.0.0.1:7878/api/stats").mock(
        return_value=httpx.Response(200, json={
            "sessions": 7,
            "turns": 42,
            "tool_calls_total": 99,
            "top_tools": [
                {"name": "read_file", "count": 33},
                {"name": "search_code", "count": 17},
            ],
            "tasks": {"total": 3, "by_status": {"done": 2, "error": 1}},
            "usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
            "estimated_cost_usd": 0.1234,
        })
    )
    rc = cli.main(["stats"])
    assert rc == 0
    out = capsys.readouterr().out
    # Summary numbers surfaced.
    assert "7" in out and "42" in out and "99" in out
    # Cost rendered with $ prefix.
    assert "$0.1234" in out
    # Top tools surfaced.
    assert "read_file" in out and "33" in out
    assert "search_code" in out


@respx.mock
def test_stats_subcommand_handles_empty_top_tools(capsys):
    respx.get("http://127.0.0.1:7878/api/stats").mock(
        return_value=httpx.Response(200, json={
            "sessions": 0,
            "turns": 0,
            "tool_calls_total": 0,
            "top_tools": [],
            "tasks": {"total": 0, "by_status": {}},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "estimated_cost_usd": 0.0,
        })
    )
    rc = cli.main(["stats"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "GodBot stats" in out
    # No "Top tools" table header should be rendered when the list is empty.
    assert "Top tools" not in out


# ---------- sessions ----------

@respx.mock
def test_sessions_subcommand_renders_table(capsys):
    respx.get("http://127.0.0.1:7878/api/sessions").mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "s-001",
                "model": "google/gemma-3-27b",
                "model_name": "gemma-3-27b",
                "provider": "lmstudio",
                "workspace_root": "C:/work",
                "started_at": "2026-04-20T10:00:00",
                "pinned": True,
            },
            {
                "id": "s-002",
                "model": "auto",
                "model_name": None,
                "provider": "openrouter",
                "workspace_root": None,
                "started_at": "2026-04-20T09:00:00",
                "pinned": False,
            },
        ])
    )
    rc = cli.main(["sessions", "--limit", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "s-001" in out
    assert "s-002" in out
    assert "lmstudio" in out
    assert "openrouter" in out
    assert "C:/work" in out
    # Pinned marker rendered for the pinned row.
    assert "★" in out


@respx.mock
def test_sessions_subcommand_handles_empty_list(capsys):
    respx.get("http://127.0.0.1:7878/api/sessions").mock(
        return_value=httpx.Response(200, json=[])
    )
    rc = cli.main(["sessions"])
    assert rc == 0
    assert "no sessions" in capsys.readouterr().out.lower()


@respx.mock
def test_sessions_subcommand_passes_limit_query_param():
    route = respx.get("http://127.0.0.1:7878/api/sessions").mock(
        return_value=httpx.Response(200, json=[])
    )
    rc = cli.main(["sessions", "--limit", "7"])
    assert rc == 0
    assert route.called
    sent_url = str(route.calls.last.request.url)
    assert "limit=7" in sent_url
