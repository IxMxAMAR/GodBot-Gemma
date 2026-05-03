# GodBot

Local agent harness for Gemma 3n E4B (or any Gemma) served by LM Studio. ReAct + JSON-schema tool calls. Web UI primary, CLI secondary, chromadb-backed RAG.

**Trust model:** all tools execute against your real machine. Confirmation gates protect dangerous tools (`write_file`, `edit_file`, `run_powershell`, `run_bash`, `run_python`). The web port binds to `127.0.0.1` only.

## Setup

1. Install LM Studio. Load a Gemma 3n E4B (or compatible Gemma) model. Set context length >= 28k. Start the server on `localhost:1234`.
2. (Optional) load an embedding model in LM Studio for RAG (e.g. `nomic-embed-text`). If absent, GodBot falls back to `BAAI/bge-small-en-v1.5` via `sentence-transformers`.
3. Install:
   ```bash
   pip install -e ".[dev]"
   ```

## Run

- `godbot-web` — web UI on http://127.0.0.1:7878
- `godbot-cli` — CLI REPL
- `godbot-index <path>` — build/refresh a RAG collection from a directory

## Config

`~/.godbot/config.toml` is created on first run. Tweak model preference, max_steps, port, theme, RAG settings there.

## Notes

- Sessions land in `<cwd>/sessions/`. Don't put it inside OneDrive — fsync gets slow.
- Tool toggles per session: web sidebar checkboxes, or `[tools] enabled = ["..."]` in config.
- The default `web_search` uses DuckDuckGo HTML and breaks regularly. Drop in `tools/web_search_brave.py` (you provide the API key) when it does.

## Live smoke test

After upgrading LM Studio:
```bash
GODBOT_LIVE=1 pytest tests/test_lmstudio_smoke.py -v
```

## Client library

GodBot ships a Python client at `godbot.client` for building new surfaces (TUI, Discord, VSCode, custom integrations).

Two layers:

- **`Client`** — slim async HTTP/SSE wrapper around the daemon. Use directly when you know the daemon is up.
- **`Session`** — mode-transparent helper. Probes the daemon on connect; falls back to an embedded in-process agent loop if no daemon. Optional auto-launch.

```python
from godbot.client import Session

async with Session(auto_launch=True) as s:
    async for ev in s.run("hello"):
        print(ev)
```

Embedded mode (no daemon, no auto-launch):

```python
from pathlib import Path
from godbot.client import Session
from godbot.client.embedded import EmbeddedRunner

def factory():
    return EmbeddedRunner.create(sessions_root=Path("sessions"))

async with Session(embedded_factory=factory) as s:
    async for ev in s.run("hello"):
        print(ev)
```

## TUI

A full-screen terminal app:

```bash
godbot-tui                       # daemon-preferred (auto-launch if not running)
godbot-tui --no-daemon           # embedded only
godbot-tui --resume <session-id> # resume a session
godbot-tui --base-url http://127.0.0.1:7878
```

Three-pane layout: sessions/tools/RAG sidebar on the left, conversation in the middle, message input on the bottom. `Ctrl+C` stops the running turn (NOT quit); `Ctrl+Q` quits. `Ctrl+N` starts a new session.

Dangerous tool calls (write_file, run_powershell, etc.) display a gate card with Allow / Always / Deny buttons inline in the conversation.

## Known issues / follow-ups

Discovered during implementation; track for the next pass:

- **SSE stream is single-consumer** (Phase 9). Reloading the browser mid-tool-call closes the original SSE connection but the new tab cannot re-attach to the in-flight run — the server returns HTTP 409 on the second stream request. Wait for the current run to finish (or POST `/api/stop`) before reloading. Proper fix: buffer events per session and replay them to a reconnecting client.
- **`save_note` blocks ~10s when LM Studio is absent** (Phase 10). The notes embedder probes LM Studio synchronously on first use. When the server is down, the connect timeout dominates the call. Workaround: set `[rag] embedder = "st"` in config to force the local sentence-transformers path. Proper fix: make the probe async with a short timeout and cache the negative result for the session.
- **`tool_overrides` semantics footgun** (Phase 9). Empty list means "no tools enabled", `None` means "all tools enabled". Easy to send the wrong one from the web UI when the user toggles every tool off — the registry then exposes nothing and the model has no recourse. Consider a sentinel or explicit `disabled_tools` field instead.
- **Module-level state in `web/app.py` is unbounded** (Phase 9). `_streams`, `_cancels`, and `_sessions_cache` accumulate per session id and never evict. Long-running web servers will leak memory proportional to total session count. Needs an LRU or session-close hook.
- **`tree-sitter-language-pack` pinned `<1.0`** (Phase 0.1). The 1.0 release changed the loader API in a way that breaks our chunker. Unpin and migrate when we have a quiet afternoon.
- **Windows `bash.exe` resolves to the WSL stub, not Git Bash** (Phase 7.3). `run_bash` on a fresh Windows install will hit the "Windows Subsystem for Linux has no installed distributions" error instead of running the script. Workaround: prepend Git Bash to PATH or set `GODBOT_BASH=C:\Program Files\Git\bin\bash.exe`. Proper fix: probe known Git Bash install locations in the tool.
- **DuckDuckGo HTML scraper is fragile** (Phase 7.4). DDG rotates anti-bot measures every few months and `web_search` will silently start returning empty results. The `tools/web_search_brave.py` drop-in is the recommended replacement once a Brave API key is in hand.
