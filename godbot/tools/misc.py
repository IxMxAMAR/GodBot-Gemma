from __future__ import annotations
import ast
import datetime as _dt
import operator as op
from typing import Any

from godbot.core.registry import tool


@tool()
def get_datetime() -> str:
    """Current local date and time as ISO 8601."""
    return _dt.datetime.now().isoformat(timespec="seconds")


_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _eval(node: Any) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"disallowed: {ast.dump(node)}")


@tool()
def calculator(expr: str) -> str:
    """Evaluate an arithmetic expression: +, -, *, /, //, %, **. Rejects anything else."""
    try:
        tree = ast.parse(expr, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"[error] {e}"


@tool()
def get_clipboard() -> str:
    """Read the system clipboard."""
    try:
        import pyperclip
        return pyperclip.paste()
    except Exception as e:
        return f"[error] {e}"


@tool()
def set_clipboard(text: str) -> str:
    """Write text to the system clipboard."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return f"ok: copied {len(text)} chars"
    except Exception as e:
        return f"[error] {e}"


@tool(dangerous=True)
def take_screenshot(path: str) -> str:
    """Capture the primary monitor and save to PNG path."""
    from godbot.core.workspace import current_workspace, WorkspaceEscape
    ws = current_workspace()
    if ws is not None:
        try:
            path = str(ws.confine(path))
        except WorkspaceEscape as e:
            return f"[error] sandbox: {e}"
    try:
        import mss
        with mss.mss() as sct:
            sct.shot(mon=1, output=path)
        return f"ok: saved {path}"
    except Exception as e:
        return f"[error] {e}"
