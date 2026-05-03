import sys
import pytest
from godbot.core.registry import DEFAULT
from godbot.core.workspace import Workspace, _current
import godbot.tools


@pytest.fixture
def workspace_token(tmp_path):
    ws = Workspace.of(str(tmp_path))
    token = _current.set(ws)
    try:
        yield tmp_path
    finally:
        _current.reset(token)


def test_run_python_inherits_workspace_cwd(workspace_token):
    out = DEFAULT.execute("run_python", {"code": "import os; print(os.getcwd())"})
    assert str(workspace_token).replace("\\", "/").lower() in out.replace("\\", "/").lower()


def test_run_python_workspace_root_env(workspace_token):
    out = DEFAULT.execute("run_python", {"code": "import os; print(os.environ.get('WORKSPACE_ROOT', '<unset>'))"})
    assert str(workspace_token) in out or str(workspace_token).replace("\\", "/") in out.replace("\\", "/")


def test_run_bash_inherits_workspace_cwd(workspace_token):
    import shutil
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    out = DEFAULT.execute("run_bash", {"cmd": "pwd"})
    # On Windows Git Bash translates the path; just confirm it's not a system path.
    assert "Windows" not in out and "/c/Windows" not in out


def test_run_powershell_inherits_workspace_cwd(workspace_token):
    if sys.platform != "win32":
        pytest.skip("powershell")
    out = DEFAULT.execute("run_powershell", {"cmd": "$pwd.Path"})
    assert str(workspace_token).replace("/", "\\") in out
