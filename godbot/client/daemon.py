from __future__ import annotations
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from godbot.client.http import Client


def godbot_home() -> Path:
    return Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))


def pid_file_path() -> Path:
    return godbot_home() / "daemon.pid"


def log_file_path() -> Path:
    return godbot_home() / "daemon.log"


def lock_file_path() -> Path:
    return godbot_home() / "daemon.lock"


def is_alive(pid: int) -> bool:
    """Cross-platform check whether a process with this PID exists."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # On Windows, signal 0 isn't supported; use OpenProcess via ctypes.
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        # If we got a handle, the process exists. Close it.
        kernel32.CloseHandle(handle)
        return True
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def find_daemon_pid() -> Optional[int]:
    p = pid_file_path()
    if not p.exists():
        return None
    try:
        pid = int(p.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if is_alive(pid) else None


async def launch_daemon(
    base_url: str = "http://127.0.0.1:7878",
    timeout: float = 30.0,
) -> int:
    """Ensure the daemon is running. Returns its PID.

    Short-circuits if PID file points to a live process. Otherwise spawns
    `python -m godbot.interfaces.web` detached and polls /api/health.
    """
    existing = find_daemon_pid()
    if existing is not None:
        return existing

    home = godbot_home()
    home.mkdir(parents=True, exist_ok=True)
    log_path = log_file_path()
    log_handle = open(log_path, "ab")

    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        kwargs = {"creationflags": creationflags}
    else:
        kwargs = {"start_new_session": True}

    proc = subprocess.Popen(
        [sys.executable, "-m", "godbot.interfaces.web"],
        stdout=log_handle,
        stderr=log_handle,
        stdin=subprocess.DEVNULL,
        cwd=str(Path.cwd()),
        **kwargs,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with Client(base_url=base_url, timeout=2.0) as c:
            if await c.health():
                pid_file_path().write_text(str(proc.pid))
                return proc.pid
        await asyncio.sleep(0.5)

    raise RuntimeError(f"daemon spawn timed out after {timeout}s; check {log_path}")


async def stop_daemon() -> bool:
    """Send SIGTERM to the daemon. Returns True if a daemon was stopped."""
    pid = find_daemon_pid()
    if pid is None:
        # Clean up stale pid file if any
        try:
            pid_file_path().unlink()
        except (FileNotFoundError, OSError):
            pass
        return False
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    else:
        os.kill(pid, 15)
    try:
        pid_file_path().unlink()
    except OSError:
        pass
    return True
