import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_godbot_home(tmp_path, monkeypatch):
    """Redirect ~/.godbot to a tmpdir so tests don't pollute real home."""
    monkeypatch.setenv("GODBOT_HOME", str(tmp_path / ".godbot"))
    return tmp_path / ".godbot"
