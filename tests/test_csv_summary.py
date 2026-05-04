"""Tests for sub-project 78 — csv_summary tool."""
from __future__ import annotations

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import csv_summary


def test_csv_summary_basic(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(
        "name,age,city\n"
        "Alice,30,NYC\n"
        "Bob,25,SF\n"
        "Carol,40,LA\n",
        encoding="utf-8",
    )
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = csv_summary(path="data.csv")
        assert "delimiter: ','" in out
        assert "columns (3): name, age, city" in out
        assert "rows: 3" in out
        assert "sample (3 of 3):" in out
        assert "Alice" in out
    finally:
        _ws_current.reset(token)


def test_csv_summary_semicolon_delimiter(tmp_path):
    f = tmp_path / "euro.csv"
    f.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = csv_summary(path="euro.csv")
        assert "delimiter: ';'" in out
        assert "columns (3): a, b, c" in out
    finally:
        _ws_current.reset(token)


def test_csv_summary_tab_delimiter(tmp_path):
    f = tmp_path / "tabs.tsv"
    f.write_text("x\ty\tz\n1\t2\t3\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = csv_summary(path="tabs.tsv")
        # Tab-detected via Sniffer.
        assert "columns (3)" in out
    finally:
        _ws_current.reset(token)


def test_csv_summary_sample_rows_cap(tmp_path):
    f = tmp_path / "big.csv"
    rows = ["a,b,c"] + [f"{i},{i*2},{i*3}" for i in range(50)]
    f.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = csv_summary(path="big.csv", sample_rows=3)
        assert "sample (3 of 50):" in out
        assert "row 4" not in out  # only 3 sample rows
    finally:
        _ws_current.reset(token)


def test_csv_summary_empty_file(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = csv_summary(path="empty.csv")
        assert "(empty)" in out
    finally:
        _ws_current.reset(token)


def test_csv_summary_only_header(tmp_path):
    f = tmp_path / "head.csv"
    f.write_text("a,b,c\n", encoding="utf-8")
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = csv_summary(path="head.csv")
        assert "rows: 0" in out
        # No sample section when no rows.
        assert "sample" not in out
    finally:
        _ws_current.reset(token)


def test_csv_summary_missing_file(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = csv_summary(path="ghost.csv")
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_csv_summary_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("a,b\n1,2\n", encoding="utf-8")
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = csv_summary(path=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)
