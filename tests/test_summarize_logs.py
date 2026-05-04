"""Tests for sub-project 82 — summarize_logs tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import summarize_logs


def _seed(tmp_path):
    f = tmp_path / "app.log"
    f.write_text(
        "2026-05-05 INFO startup\n"
        "2026-05-05 ERROR connection refused\n"
        "2026-05-05 WARNING slow query\n"
        "2026-05-05 INFO shutting down\n"
        "2026-05-05 ERROR oom\n",
        encoding="utf-8",
    )
    return f


def test_summarize_logs_default_levels(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = summarize_logs(path="app.log")
        assert "ERROR connection refused" in out
        assert "ERROR oom" in out
        assert "WARNING slow query" in out
        assert "INFO startup" not in out
        assert "matched (3)" in out
    finally:
        _ws_current.reset(token)


def test_summarize_logs_custom_levels(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = summarize_logs(path="app.log", levels="INFO")
        assert "INFO startup" in out
        assert "INFO shutting down" in out
        assert "ERROR" not in out.split("matched")[1]  # tolerate "ERROR" in earlier metadata
    finally:
        _ws_current.reset(token)


def test_summarize_logs_case_insensitive(tmp_path):
    f = tmp_path / "app.log"
    f.write_text("error in module x\nERROR in module y\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = summarize_logs(path="app.log", levels="error")
        assert "matched (2)" in out
    finally:
        _ws_current.reset(token)


def test_summarize_logs_max_lines_cap(tmp_path):
    f = tmp_path / "app.log"
    f.write_text("\n".join(["ERROR row %d" % i for i in range(20)]) + "\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = summarize_logs(path="app.log", levels="ERROR", max_lines=5)
        assert "stopped at max_lines=5" in out
        assert "5 capped" in out
    finally:
        _ws_current.reset(token)


def test_summarize_logs_no_matches(tmp_path):
    f = tmp_path / "app.log"
    f.write_text("INFO ok\nINFO ok\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = summarize_logs(path="app.log", levels="ERROR")
        assert "matched (0)" in out
    finally:
        _ws_current.reset(token)


def test_summarize_logs_missing_file(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = summarize_logs(path="ghost.log")
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_summarize_logs_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.log"
    outside.write_text("ERROR x\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = summarize_logs(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_summarize_logs_empty_levels_errors(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = summarize_logs(path="app.log", levels="")
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_summarize_logs_total_count(tmp_path):
    _seed(tmp_path)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = summarize_logs(path="app.log", levels="ERROR")
        assert "total lines: 5" in out
    finally:
        _ws_current.reset(token)
