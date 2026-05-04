"""Tests for sub-project 99 — parse_csv tool."""
from __future__ import annotations

import json

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import parse_csv


def test_parse_csv_basic(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(
        "name,age,city\n"
        "Alice,30,NYC\n"
        "Bob,25,SF\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = parse_csv(path="data.csv")
        rows = json.loads(out)
        assert len(rows) == 2
        assert rows[0] == {"name": "Alice", "age": "30", "city": "NYC"}
        assert rows[1] == {"name": "Bob", "age": "25", "city": "SF"}
    finally:
        _ws_current.reset(token)


def test_parse_csv_semicolon(tmp_path):
    f = tmp_path / "euro.csv"
    f.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = parse_csv(path="euro.csv")
        rows = json.loads(out)
        assert rows[0]["a"] == "1"
        assert rows[0]["b"] == "2"
    finally:
        _ws_current.reset(token)


def test_parse_csv_max_rows_truncates(tmp_path):
    f = tmp_path / "big.csv"
    rows = ["a,b"] + [f"{i},{i*2}" for i in range(50)]
    f.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = parse_csv(path="big.csv", max_rows=5)
        # Output should mention truncation.
        assert "truncated at max_rows=5" in out
        # The JSON portion should still be valid up to that point.
        json_part = out.split("\n... [")[0]
        parsed = json.loads(json_part)
        assert len(parsed) == 5
    finally:
        _ws_current.reset(token)


def test_parse_csv_only_header(tmp_path):
    f = tmp_path / "head.csv"
    f.write_text("a,b,c\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = parse_csv(path="head.csv")
        rows = json.loads(out)
        assert rows == []
    finally:
        _ws_current.reset(token)


def test_parse_csv_missing_file(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = parse_csv(path="ghost.csv")
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_parse_csv_outside_workspace(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("a,b\n1,2\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = parse_csv(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_parse_csv_empty_path():
    out = parse_csv(path="")
    assert out.startswith("[error]")


def test_parse_csv_max_rows_clamped(tmp_path):
    f = tmp_path / "small.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        # Asking for huge max_rows is fine; just clamps internally.
        out = parse_csv(path="small.csv", max_rows=99999)
        rows = json.loads(out)
        assert len(rows) == 1
    finally:
        _ws_current.reset(token)
