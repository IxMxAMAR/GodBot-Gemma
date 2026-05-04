"""Tests for sub-project 75 — get_env + list_env."""
from __future__ import annotations

from godbot.core.registry import DEFAULT
from godbot.tools.workspace_meta import _redact, get_env, list_env


# --- redaction ----


def test_redact_normal_value_passthrough():
    assert _redact("PATH", "/usr/bin") == "/usr/bin"


def test_redact_secret_long_value():
    out = _redact("OPENAI_API_KEY", "sk-1234567890abcdef")
    assert "sk" in out
    assert "***" in out
    assert "ef" in out
    assert "len=" in out


def test_redact_secret_short_value():
    out = _redact("DB_PASSWORD", "abc")
    assert out == "***"


def test_redact_empty_secret():
    assert _redact("SESSION_TOKEN", "") == "(empty)"


def test_redact_case_insensitive():
    """'CREDENTIAL' and 'credential' both trigger redaction."""
    a = _redact("MY_CREDENTIAL", "abcdef")
    b = _redact("my_credential", "abcdef")
    assert "***" in a
    assert "***" in b


# --- get_env ----


def test_get_env_unset(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET_VAR_123", raising=False)
    out = get_env(name="DEFINITELY_NOT_SET_VAR_123")
    assert "(unset)" in out


def test_get_env_set_normal(monkeypatch):
    monkeypatch.setenv("MY_PUBLIC_VAR", "hello")
    out = get_env(name="MY_PUBLIC_VAR")
    assert "MY_PUBLIC_VAR: hello" in out


def test_get_env_redacted_secret(monkeypatch):
    monkeypatch.setenv("MY_SECRET_KEY", "supersecretvalue123")
    out = get_env(name="MY_SECRET_KEY")
    assert "supersecretvalue123" not in out
    assert "***" in out


def test_get_env_empty_name():
    out = get_env(name="")
    assert out.startswith("[error]")


# --- list_env ----


def test_list_env_with_prefix(monkeypatch):
    monkeypatch.setenv("GODBOT_TEST_A", "v1")
    monkeypatch.setenv("GODBOT_TEST_B", "v2")
    monkeypatch.setenv("OTHER_VAR", "v3")
    out = list_env(prefix="GODBOT_TEST_")
    assert "GODBOT_TEST_A=v1" in out
    assert "GODBOT_TEST_B=v2" in out
    assert "OTHER_VAR" not in out


def test_list_env_redacts_secrets(monkeypatch):
    monkeypatch.setenv("MYAPP_API_KEY", "sk-secretvalue")
    out = list_env(prefix="MYAPP_")
    assert "sk-secretvalue" not in out
    assert "***" in out


def test_list_env_no_matches(monkeypatch):
    out = list_env(prefix="ZZZ_NEVER_PREFIX_")
    assert "(no env vars matching" in out


def test_list_env_max_results_cap(monkeypatch):
    for i in range(50):
        monkeypatch.setenv(f"BULK_TEST_{i:03d}", "x")
    out = list_env(prefix="BULK_TEST_", max_results=5)
    rows = [ln for ln in out.splitlines() if ln.startswith("BULK_TEST_")]
    assert len(rows) == 5
    assert "truncated at max_results=5" in out


def test_env_tools_registered_non_dangerous():
    for name in ("get_env", "list_env"):
        spec = DEFAULT.spec(name)
        assert spec is not None
        assert spec.dangerous is False
