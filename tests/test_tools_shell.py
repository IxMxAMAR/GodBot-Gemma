import shutil
import sys
import pytest
import godbot.tools
from godbot.core.registry import DEFAULT


def test_run_powershell_echo():
    if sys.platform != "win32":
        pytest.skip("powershell")
    out = DEFAULT.execute("run_powershell", {"cmd": "Write-Output hello"})
    assert "hello" in out


def test_run_bash_echo():
    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH")
    out = DEFAULT.execute("run_bash", {"cmd": "echo hi"})
    assert "hi" in out


def test_run_bash_failed_returns_stderr_and_code():
    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH")
    out = DEFAULT.execute("run_bash", {"cmd": "exit 7"})
    assert "exit 7" in out or "exit code 7" in out
