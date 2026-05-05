"""End-to-end smoke test against a running GodBot daemon (sub-project 108).

Exercises the most-used endpoints + a couple of common provider flows.
Intended to be run by hand (or a CI hook) after a release. Fails fast on
the first non-trivial issue and prints a numbered checklist of what
worked / what didn't.

Usage:

    python scripts/smoke_test.py [--base-url http://127.0.0.1:7878]
                                 [--token <bearer>]
                                 [--provider lmstudio|openai|anthropic]
                                 [--model <id>]
                                 [--skip-llm]      # skip the live LLM turn

Exits 0 if every checked phase passes, 1 if any fails. The probe is
read-mostly: it creates ONE temporary session, runs ONE turn through it
when --skip-llm is not set, and deletes the session at the end.

Requires only `httpx` from the daemon's dep set.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any

try:
    import httpx
except ImportError:
    print("[FATAL] httpx not installed; install godbot first", file=sys.stderr)
    sys.exit(2)


def _color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m"


GREEN = lambda s: _color(s, "32")
RED = lambda s: _color(s, "31")
YELLOW = lambda s: _color(s, "33")


class SmokeRunner:
    def __init__(self, base_url: str, token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        headers: dict[str, str] = {"User-Agent": "godbot-smoke/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=60.0)
        self.results: list[tuple[str, bool, str]] = []
        self._session_id: str | None = None

    def step(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, ok, detail))
        glyph = GREEN("OK ") if ok else RED("FAIL")
        line = f"  {glyph}  {name}"
        if detail:
            line += f"  ({detail})"
        print(line)

    # ---- phases ----

    def phase_health(self) -> bool:
        try:
            r = self.client.get("/api/health")
            ok = r.status_code == 200 and r.json().get("status") == "ok"
            self.step("health endpoint", ok, f"HTTP {r.status_code}")
            return ok
        except Exception as e:
            self.step("health endpoint", False, str(e))
            return False

    def phase_version(self) -> bool:
        try:
            r = self.client.get("/api/version")
            v = r.json().get("version", "?")
            self.step("version endpoint", r.status_code == 200, f"version={v}")
            return r.status_code == 200
        except Exception as e:
            self.step("version endpoint", False, str(e))
            return False

    def phase_tools_listed(self) -> bool:
        try:
            r = self.client.get("/api/tools")
            count = len(r.json()) if r.status_code == 200 else 0
            ok = r.status_code == 200 and count > 0
            self.step("tools registered", ok, f"{count} tools")
            return ok
        except Exception as e:
            self.step("tools registered", False, str(e))
            return False

    def phase_providers_listed(self) -> bool:
        try:
            r = self.client.get("/api/providers")
            body = r.json() if r.status_code == 200 else {}
            providers = [p["name"] for p in body.get("providers", [])]
            ok = r.status_code == 200 and len(providers) > 0
            self.step("providers configured", ok, f"{', '.join(providers[:8])}")
            return ok
        except Exception as e:
            self.step("providers configured", False, str(e))
            return False

    def phase_session_lifecycle(self) -> bool:
        try:
            r = self.client.post("/api/sessions/new", json={})
            sid = r.json().get("session_id")
            self._session_id = sid
            self.step("session create", bool(sid), f"sid={sid}")
            r = self.client.get(f"/api/sessions/{sid}")
            ok2 = r.status_code == 200
            self.step("session fetch", ok2, f"HTTP {r.status_code}")
            return bool(sid) and ok2
        except Exception as e:
            self.step("session lifecycle", False, str(e))
            return False

    def phase_one_turn(self, provider: str, model: str) -> bool:
        if not self._session_id:
            self.step("LLM turn", False, "no session id from earlier phase")
            return False
        try:
            r = self.client.post("/api/agent/quickrun", json={
                "goal": "Reply with exactly the word: PONG",
                "provider": provider,
                "model": model,
                "max_wait_seconds": 60,
                "max_steps": 3,
                "safe_only": True,
            })
            if r.status_code != 200:
                self.step("LLM turn", False, f"HTTP {r.status_code}: {r.text[:120]}")
                return False
            body = r.json()
            status = body.get("status")
            result_excerpt = (body.get("result") or "").strip()[:60]
            ok = status == "done" and "PONG" in result_excerpt.upper()
            self.step(
                f"LLM turn ({provider}/{model})",
                ok,
                f"status={status} result={result_excerpt!r}",
            )
            return ok
        except Exception as e:
            self.step("LLM turn", False, str(e))
            return False

    def phase_session_cleanup(self) -> bool:
        if not self._session_id:
            return True
        try:
            r = self.client.delete(f"/api/sessions/{self._session_id}")
            ok = r.status_code == 200
            self.step("session delete", ok, f"HTTP {r.status_code}")
            return ok
        except Exception as e:
            self.step("session delete", False, str(e))
            return False

    def phase_snapshot(self) -> bool:
        try:
            r = self.client.get("/api/admin/snapshot")
            body = r.json() if r.status_code == 200 else {}
            ok = (
                r.status_code == 200
                and "version" in body
                and "health" in body
                and "stats" in body
            )
            self.step("admin snapshot", ok, f"HTTP {r.status_code}")
            return ok
        except Exception as e:
            self.step("admin snapshot", False, str(e))
            return False

    def run(self, provider: str, model: str, skip_llm: bool) -> int:
        print(f"\n{YELLOW('GodBot smoke test')} → {self.base_url}\n")
        phases = [
            self.phase_health,
            self.phase_version,
            self.phase_tools_listed,
            self.phase_providers_listed,
            self.phase_session_lifecycle,
            self.phase_snapshot,
        ]
        for p in phases:
            p()
        if not skip_llm:
            self.phase_one_turn(provider, model)
        self.phase_session_cleanup()

        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        print(f"\n{passed}/{total} checks passed")
        return 0 if passed == total else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="godbot-smoke", description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:7878")
    p.add_argument("--token", default="")
    p.add_argument("--provider", default="lmstudio")
    p.add_argument("--model", default="auto")
    p.add_argument("--skip-llm", action="store_true",
                   help="Skip the live LLM round-trip (useful when no provider is up)")
    args = p.parse_args(argv)
    runner = SmokeRunner(args.base_url, token=args.token)
    return runner.run(args.provider, args.model, args.skip_llm)


if __name__ == "__main__":
    sys.exit(main())
