"""Tests for sub-project 27 — tool catalog hot-reload."""
from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

import godbot.tools  # noqa: F401
from godbot.core.registry import DEFAULT
from godbot.interfaces.web import build_app
from godbot.tools import discover, reload_all


def test_reload_all_returns_module_count():
    """reload_all should report a positive count after a fresh discover."""
    n = reload_all()
    assert n >= 1


def test_reload_all_is_idempotent_for_registry_keys():
    """Re-running reload should not duplicate registry entries."""
    before = sorted(DEFAULT.names())
    reload_all()
    after = sorted(DEFAULT.names())
    assert before == after


def test_reload_picks_up_module_change(tmp_path, monkeypatch):
    """Edit a tool module's docstring → reload → spec.description updates."""
    # We can't actually re-edit godbot/tools/* without polluting the repo.
    # Instead, register a custom tool, mutate its description in-place via
    # registry, and verify reload restores it (since reload re-runs the
    # decorator with the original docstring).
    from godbot.tools import knowledge

    spec = DEFAULT.spec("search_knowledge")
    assert spec is not None
    # Hand-mutate the description; reload should overwrite it back.
    DEFAULT._tools["search_knowledge"] = type(spec)(
        name=spec.name, description="MUTATED",
        fn=spec.fn, schema=spec.schema,
        dangerous=spec.dangerous, timeout=spec.timeout,
    )
    assert DEFAULT.spec("search_knowledge").description == "MUTATED"
    reload_all()
    # After reload the original docstring's first line is back.
    assert DEFAULT.spec("search_knowledge").description != "MUTATED"


def test_reload_endpoint_returns_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    r = c.post("/api/tools/reload")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["modules_reloaded"] >= 1
    assert body["tools_registered"] >= 1


def test_reload_endpoint_does_not_remove_existing_tools(tmp_path, monkeypatch):
    """The catalog count after reload should be at least as large as before."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path))
    app = build_app(sessions_root=tmp_path / "sessions")
    c = TestClient(app)
    before = len(c.get("/api/tools").json())
    c.post("/api/tools/reload")
    after = len(c.get("/api/tools").json())
    assert after >= before


def test_reload_continues_past_broken_module(tmp_path, monkeypatch):
    """A module that raises during reload should be logged + skipped, not
    fatal. We simulate by monkeypatching importlib.reload to fail for one
    module name and succeed for others."""
    import importlib
    import godbot.tools as tools_pkg

    real_reload = importlib.reload
    boom_module_name = "godbot.tools.knowledge"
    calls = {"good": 0, "bad": 0}

    def fake_reload(mod):
        if mod.__name__ == boom_module_name:
            calls["bad"] += 1
            raise RuntimeError("simulated breakage")
        calls["good"] += 1
        return real_reload(mod)

    monkeypatch.setattr(importlib, "reload", fake_reload)
    n = tools_pkg.reload_all()
    # Reload of the broken module is counted as a skip; good modules
    # increment count.
    assert calls["bad"] >= 1
    assert calls["good"] >= 1
    assert n >= 1  # at least the surviving modules
