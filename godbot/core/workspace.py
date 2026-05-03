from __future__ import annotations
import contextvars
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


class WorkspaceEscape(Exception):
    """Raised when a path resolves to a target outside the workspace."""

    def __init__(self, requested: str, resolved: Path, root: Path) -> None:
        super().__init__(
            f"path {requested!r} resolves to {resolved} which escapes workspace {root}"
        )
        self.requested = requested
        self.resolved = resolved
        self.root = root


@dataclass(frozen=True)
class Workspace:
    """A user-confined directory the agent may not read/write outside.

    `root` is always an absolute, resolved path. FS tools call `confine(p)` to
    map their inputs into the workspace; raises `WorkspaceEscape` on escape.
    Shell tools use `root` as cwd; they are NOT confined (soft sandbox).
    """
    root: Path
    auto_approve_in_sandbox: bool = False

    @classmethod
    def of(
        cls,
        root: Union[str, Path, None],
        auto_approve: bool = False,
    ) -> Optional["Workspace"]:
        if root is None or root == "":
            return None
        p = Path(root)
        if not p.exists():
            raise FileNotFoundError(p)
        if not p.is_dir():
            raise NotADirectoryError(p)
        return cls(root=p.resolve(), auto_approve_in_sandbox=auto_approve)

    def confine(self, path: Union[str, Path]) -> Path:
        """Return the absolute resolved path inside this workspace.

        Raises WorkspaceEscape if the resolved target lies outside.
        """
        requested = str(path)
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        if not _is_relative_to(resolved, self.root):
            raise WorkspaceEscape(requested, resolved, self.root)
        return resolved

    def is_inside(self, path: Union[str, Path]) -> bool:
        try:
            return _is_relative_to(Path(path).resolve(strict=False), self.root)
        except OSError:
            return False


def _is_relative_to(child: Path, parent: Path) -> bool:
    """Path.is_relative_to backport-safe; resolves once, compares strings."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


_current: contextvars.ContextVar[Optional[Workspace]] = contextvars.ContextVar(
    "godbot_workspace", default=None
)


def current_workspace() -> Optional[Workspace]:
    return _current.get()


def set_workspace(ws: Optional[Workspace]) -> contextvars.Token:
    return _current.set(ws)
