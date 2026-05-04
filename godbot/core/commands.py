"""Custom slash commands (sub-project 18).

Users drop ``~/.godbot/commands/<name>.toml`` files to register reusable
"skills" — a system-prompt suffix plus an optional tool subset. When the
user message starts with ``/<name> `` (strict-prefix, case-sensitive,
trailing space required), the agent loop loads the command, appends the
suffix to the system prompt for that turn, and constrains the tool
catalog if ``tool_overrides`` is set.

This generalises the ``/plan`` pattern from sub-project 10.4 without
hard-coding more commands into Python.

Example file format:

    name = "refactor"
    description = "Refactor with focus on readability + DRY"
    system_prompt_suffix = '''
    The user is asking for a refactor. Constraints:
    - Preserve all existing public APIs.
    - Don't introduce new dependencies.
    - Run tests at the end (use run_tests).
    '''
    tool_overrides = [
        "read_file", "edit_file", "write_file", "run_tests",
        "list_directory", "git_status", "git_diff",
    ]
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("godbot.commands")


@dataclass
class CommandSpec:
    """One user-defined slash command."""

    name: str
    description: str = ""
    system_prompt_suffix: str = ""
    tool_overrides: list[str] = field(default_factory=list)
    source_path: Optional[str] = None


_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,30}$", re.IGNORECASE)


def _commands_dir() -> Path:
    home = Path(os.environ.get("GODBOT_HOME", str(Path.home() / ".godbot")))
    return home / "commands"


def _load_one(path: Path) -> Optional[CommandSpec]:
    """Parse one TOML command file, return ``None`` on any malformedness.

    Uses tomllib (3.11+). If a file fails to parse or has the wrong
    shape, log and skip — never raise from here, since startup paths
    iterate over every file in the directory.
    """
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        log.warning("tomllib missing; custom commands disabled")
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("failed to parse command file %s", path)
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name") or path.stem
    if not isinstance(name, str) or not _NAME_PATTERN.match(name):
        log.warning("command file %s has invalid name %r; skipping", path, name)
        return None
    description = data.get("description") or ""
    suffix = data.get("system_prompt_suffix") or ""
    if not isinstance(description, str):
        description = ""
    if not isinstance(suffix, str):
        suffix = ""
    raw_overrides = data.get("tool_overrides") or []
    if not isinstance(raw_overrides, list):
        raw_overrides = []
    overrides = [t for t in raw_overrides if isinstance(t, str)]
    return CommandSpec(
        name=name,
        description=description.strip(),
        system_prompt_suffix=suffix.strip(),
        tool_overrides=overrides,
        source_path=str(path),
    )


def load_commands(commands_dir: Optional[Path] = None) -> dict[str, CommandSpec]:
    """Load every ``*.toml`` in the commands dir into a name→spec map.

    Empty/missing directory returns an empty map (cheap fast-path so
    importers don't pay for filesystem reads when no commands exist).
    """
    d = commands_dir or _commands_dir()
    if not d.exists():
        return {}
    out: dict[str, CommandSpec] = {}
    for p in sorted(d.glob("*.toml")):
        spec = _load_one(p)
        if spec is not None:
            out[spec.name] = spec
    return out


def match_command(user_message: str, commands: Optional[dict[str, CommandSpec]] = None) -> Optional[CommandSpec]:
    """If ``user_message`` is ``/<name> ...``, return the matching spec.

    Strict-prefix: requires a leading ``/``, an alphanumeric name, then
    whitespace (or end-of-string for ``/<name>`` with no args). Returns
    ``None`` when the message doesn't start with a slash, when the
    extracted name isn't a known command, or when ``commands`` is empty.

    The built-in ``/plan`` is intentionally NOT matched here — it has
    its own dedicated handling in agent.py. Custom commands fill the
    gap for any other ``/<name>``.
    """
    if not user_message or not user_message.startswith("/"):
        return None
    # Strip the leading "/", split on first whitespace.
    rest = user_message[1:]
    # Match name and require whitespace or EOL after it.
    m = re.match(r"^([a-z][a-z0-9_-]{0,30})(\s|$)", rest, re.IGNORECASE)
    if not m:
        return None
    name = m.group(1)
    if name == "plan":
        # Reserved for the built-in plan-mode handler; never claim it.
        return None
    cmds = commands if commands is not None else load_commands()
    return cmds.get(name)
