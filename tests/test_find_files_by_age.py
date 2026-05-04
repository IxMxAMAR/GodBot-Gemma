"""Tests for sub-project 86 — find_files_by_age tool."""
from __future__ import annotations

import os
import time

from godbot.core.workspace import Workspace, set_workspace, _current as _ws_current
from godbot.tools.workspace_meta import find_files_by_age


def _set_age(path, days_ago):
    ts = time.time() - (days_ago * 86400)
    os.utime(path, (ts, ts))


def test_find_older_than(tmp_path):
    old = tmp_path / "old.txt"
    young = tmp_path / "young.txt"
    old.write_text("x", encoding="utf-8")
    young.write_text("y", encoding="utf-8")
    _set_age(old, 60)
    _set_age(young, 1)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_files_by_age(older_than_days=30)
        assert "old.txt" in out
        assert "young.txt" not in out
        assert "total: 1 match" in out
    finally:
        _ws_current.reset(token)


def test_find_newer_than(tmp_path):
    old = tmp_path / "old.txt"
    young = tmp_path / "young.txt"
    old.write_text("x", encoding="utf-8")
    young.write_text("y", encoding="utf-8")
    _set_age(old, 60)
    _set_age(young, 1)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_files_by_age(newer_than_days=7)
        assert "young.txt" in out
        assert "old.txt" not in out
    finally:
        _ws_current.reset(token)


def test_pattern_filter(tmp_path):
    a = tmp_path / "a.py"; a.write_text("x", encoding="utf-8"); _set_age(a, 100)
    b = tmp_path / "b.md"; b.write_text("y", encoding="utf-8"); _set_age(b, 100)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_files_by_age(older_than_days=30, pattern="**/*.py")
        assert "a.py" in out
        assert "b.md" not in out
    finally:
        _ws_current.reset(token)


def test_skips_noise_dirs(tmp_path):
    real = tmp_path / "real.txt"; real.write_text("x", encoding="utf-8"); _set_age(real, 60)
    git_dir = tmp_path / ".git"; git_dir.mkdir()
    inside_git = git_dir / "config"
    inside_git.write_text("y", encoding="utf-8"); _set_age(inside_git, 60)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_files_by_age(older_than_days=30)
        assert "real.txt" in out
        assert ".git" not in out
        assert "total: 1 match" in out
    finally:
        _ws_current.reset(token)


def test_no_args_errors(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_files_by_age()
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_both_args_errors(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_files_by_age(older_than_days=10, newer_than_days=5)
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_negative_days_errors(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_files_by_age(older_than_days=-5)
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)


def test_max_results_cap(tmp_path):
    for i in range(15):
        f = tmp_path / f"f{i:02d}.txt"
        f.write_text("x", encoding="utf-8")
        _set_age(f, 60)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_files_by_age(older_than_days=30, max_results=5)
        # Capped at 5 matches.
        assert "5 match(es) (capped)" in out
    finally:
        _ws_current.reset(token)


def test_oldest_first_order(tmp_path):
    a = tmp_path / "a.txt"; a.write_text("x", encoding="utf-8"); _set_age(a, 30)
    b = tmp_path / "b.txt"; b.write_text("y", encoding="utf-8"); _set_age(b, 60)
    ws = Workspace.of(str(tmp_path))
    token = set_workspace(ws)
    try:
        out = find_files_by_age(older_than_days=10)
        # b is older, so its line should come before a's.
        b_idx = out.index("b.txt")
        a_idx = out.index("a.txt")
        assert b_idx < a_idx
    finally:
        _ws_current.reset(token)


def test_outside_workspace_blocked(tmp_path):
    ws_dir = tmp_path / "ws"; ws_dir.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    ws = Workspace.of(str(ws_dir))
    token = set_workspace(ws)
    try:
        out = find_files_by_age(older_than_days=1, root=str(outside))
        assert out.startswith("[error]")
    finally:
        _ws_current.reset(token)
