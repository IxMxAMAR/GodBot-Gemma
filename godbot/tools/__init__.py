from __future__ import annotations
import importlib
import pkgutil
import sys
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


def reload_all() -> int:
    """Re-import every ``godbot.tools.*`` module so @tool decorators re-fire (sub-project 27).

    Used by the daemon's POST /api/tools/reload endpoint and by users
    iterating on a custom tool module during a session. The registry's
    name-keyed dict means the second registration of a tool overwrites
    the first, so reload picks up code changes without a daemon restart.

    Returns the count of modules reloaded. Modules that were never
    imported are added via ``discover``; modules that vanished from
    disk are NOT removed from the registry — that's a v2 concern (would
    need a "tools-known-from-last-discover" set).

    Failure in any single module aborts the reload of that module (logs
    the exception, continues with the rest), so a broken tool can't
    take the daemon down.
    """
    import logging
    logger = logging.getLogger("godbot.tools")
    pkg_dir = Path(__file__).parent
    count = 0
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name.startswith("_"):
            continue
        full = f"{__name__}.{info.name}"
        try:
            mod = sys.modules.get(full)
            if mod is None:
                importlib.import_module(full)
            else:
                importlib.reload(mod)
            count += 1
        except Exception:
            logger.exception("reload of %s failed; skipping", full)
    return count


# Auto-discover on import.
discover()
