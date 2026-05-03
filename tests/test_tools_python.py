import sys
import godbot.tools
from godbot.core.registry import DEFAULT


def test_run_python_basic():
    out = DEFAULT.execute("run_python", {"code": "print(2 + 2)"})
    assert "4" in out


def test_run_python_captures_stderr():
    out = DEFAULT.execute("run_python", {"code": "import sys; sys.stderr.write('boom')"})
    assert "boom" in out
