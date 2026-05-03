import os
import pytest
from unittest.mock import patch, MagicMock
from godbot.client.daemon import (
    find_daemon_pid, is_alive, launch_daemon, stop_daemon,
    pid_file_path,
)


def test_find_daemon_pid_returns_none_when_missing(tmp_godbot_home):
    assert find_daemon_pid() is None


def test_find_daemon_pid_reads_file_when_alive(tmp_godbot_home):
    pid_file_path().parent.mkdir(parents=True, exist_ok=True)
    pid_file_path().write_text(str(os.getpid()))
    # The current pytest process is alive, so this should return our pid.
    assert find_daemon_pid() == os.getpid()


def test_find_daemon_pid_returns_none_for_dead_pid(tmp_godbot_home):
    pid_file_path().parent.mkdir(parents=True, exist_ok=True)
    pid_file_path().write_text("999999")  # almost certainly dead
    assert find_daemon_pid() is None


def test_is_alive_for_self():
    assert is_alive(os.getpid()) is True


def test_is_alive_for_unlikely_pid():
    assert is_alive(999999) is False


@pytest.mark.asyncio
async def test_launch_daemon_uses_existing_alive_pid(tmp_godbot_home):
    pid_file_path().parent.mkdir(parents=True, exist_ok=True)
    pid_file_path().write_text(str(os.getpid()))
    # Should short-circuit and return the existing pid without spawning.
    pid = await launch_daemon(timeout=1.0)
    assert pid == os.getpid()


@pytest.mark.asyncio
async def test_launch_daemon_spawns_when_no_pid_file(tmp_godbot_home, monkeypatch):
    spawned = MagicMock()
    spawned.pid = 12345
    health_calls = [False, False, True]  # third call returns True

    fake_popen = MagicMock(return_value=spawned)
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    async def fake_health(self):
        return health_calls.pop(0)

    monkeypatch.setattr("godbot.client.http.Client.health", fake_health)

    pid = await launch_daemon(timeout=5.0)
    assert pid == 12345
    assert pid_file_path().read_text() == "12345"
    fake_popen.assert_called_once()


@pytest.mark.asyncio
async def test_launch_daemon_times_out(tmp_godbot_home, monkeypatch):
    spawned = MagicMock()
    spawned.pid = 12345

    fake_popen = MagicMock(return_value=spawned)
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    async def never_healthy(self):
        return False

    monkeypatch.setattr("godbot.client.http.Client.health", never_healthy)

    with pytest.raises(RuntimeError, match="timed out"):
        await launch_daemon(timeout=0.5)


def test_stop_daemon_returns_false_when_no_pid_file(tmp_godbot_home):
    import asyncio
    assert asyncio.run(stop_daemon()) is False
