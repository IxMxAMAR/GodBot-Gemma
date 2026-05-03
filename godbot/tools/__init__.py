from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path


def discover() -> int:
    """Import every module in this package so @tool decorators register.

    Returns count of modules loaded. Idempotent within a single Python process
    because importlib caches imports. If any tool module raises at import time,
    the exception propagates — fail-fast keeps a broken tool from silently
    disappearing from the registry.
    """
    pkg_dir = Path(__file__).parent
    count = 0
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")
        count += 1
    return count


# Auto-discover on import.
discover()
