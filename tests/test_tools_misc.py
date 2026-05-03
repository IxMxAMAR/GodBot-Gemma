import re
import godbot.tools
from godbot.core.registry import DEFAULT


def test_get_datetime():
    out = DEFAULT.execute("get_datetime", {})
    assert re.match(r"\d{4}-\d{2}-\d{2}", out)


def test_calculator_simple():
    out = DEFAULT.execute("calculator", {"expr": "2 + 3 * 4"})
    assert "14" in out


def test_calculator_rejects_unsafe():
    out = DEFAULT.execute("calculator", {"expr": "__import__('os').system('rm -rf /')"})
    assert "[error]" in out


def test_calculator_rejects_bare_function_call():
    # Critical-thinking regression: AST walker must block Call nodes outright.
    out = DEFAULT.execute("calculator", {"expr": "abs(-5)"})
    assert "[error]" in out


def test_calculator_rejects_name_reference():
    # Bare names (e.g. variables) must also be rejected.
    out = DEFAULT.execute("calculator", {"expr": "x + 1"})
    assert "[error]" in out


def test_take_screenshot_is_dangerous():
    # Writes a PNG to an arbitrary filesystem path — must be gated like write_file.
    spec = DEFAULT.spec("take_screenshot")
    assert spec is not None
    assert spec.dangerous is True


def test_take_screenshot_refuses_outside_workspace(tmp_path):
    from godbot.core.workspace import Workspace, set_workspace, _current
    ws = Workspace.of(str(tmp_path))
    token = _current.set(ws)
    try:
        out = DEFAULT.execute("take_screenshot", {"path": "../escape.png"})
        assert "[error] sandbox" in out
    finally:
        _current.reset(token)
