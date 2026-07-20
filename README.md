# GodBot

Local agent harness for Gemma 3n E4B (or any Gemma) served by LM Studio. ReAct + JSON-schema tool calls. Web UI primary, CLI secondary, chromadb-backed RAG.

**Trust model:** all tools execute against your real machine. Confirmation gates protect dangerous tools (`write_file`, `edit_file`, `run_powershell`, `run_bash`, `run_python`). The web port binds to `127.0.0.1` only.

## Setup

1. Install LM Studio. Load a Gemma 3n E4B (or compatible Gemma) model. Set context length >= 28k. Start the server on `localhost:1234`.
2. (Optional) load an embedding model in LM Studio for RAG (e.g. `nomic-embed-text`). If absent, GodBot falls back to `BAAI/bge-small-en-v1.5` via `sentence-transformers`.
3. Create a dedicated venv (Python 3.11 or 3.12) and install:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate      # POSIX
   pip install -e ".[dev]"          # add ",discord" to enable the Discord bot extra
   ```

   GodBot pulls heavy deps (chromadb, sentence-transformers, torch via transitive). Don't install into your system Python or share a venv with another project — pin one venv per checkout.

## Run

- `godbot-web` — web UI on http://127.0.0.1:7878
- `godbot-cli` — CLI REPL
- `godbot-index <path>` — build/refresh a RAG collection from a directory

## Config

`~/.godbot/config.toml` is created on first run. Tweak model preference, max_steps, port, theme, RAG settings there.

## Providers (any model that can do tool calls)

GodBot's daemon talks to LLMs through a provider abstraction. Out of the box every session defaults to the legacy LM Studio path with ReAct + JSON-schema tool calls — that is byte-for-byte identical to v0.1. To use a different backend, pick one in `[providers.<name>]` and select it per session.

Supported providers:

| Name | Class | Notes |
|---|---|---|
| `lmstudio` (default) | `GenericOpenAICompatProvider` | OpenAI-compatible at `localhost:1234/v1`. ReAct + JSON-schema. |
| `ollama` | `GenericOpenAICompatProvider` | OpenAI-compat shim at `localhost:11434/v1`. |
| `openai` | `GenericOpenAICompatProvider` | Native `tools=[...]` for GPT-4o/etc. Set `OPENAI_API_KEY`. |
| `anthropic` | `AnthropicProvider` | Native `tool_use` blocks. Set `ANTHROPIC_API_KEY`. |
| `gemini` | `GeminiProvider` | Native `functionDeclarations` via raw httpx (no Google SDK dep). Set `GOOGLE_API_KEY`. |
| `groq`, `together`, `openrouter`, `cerebras`, `mistral`, `fireworks`, `vllm` | `GenericOpenAICompatProvider` | Drop-in OpenAI-compat — set the matching `*_API_KEY` env var. |
| `openai_compat` | `GenericOpenAICompatProvider` | For custom self-hosted endpoints — set `base_url` and (optionally) `api_key`. |

Per-session selection:

```bash
curl -X POST http://localhost:7878/api/sessions/new \
  -H 'Content-Type: application/json' \
  -d '{"provider":"anthropic","model_name":"claude-3-5-sonnet-latest"}'
```

Two tool-call protocols are auto-selected per model and overridable per-session:

- `react_json` — ReAct envelope + JSON-schema constrained output (legacy Gemma path; works on any model but slower)
- `native` — Claude `tool_use` / OpenAI `tool_calls` / Gemini `functionCall`

The provider chooses based on the model id (small/legacy models default to ReAct; cloud frontier models default to native). Pin manually with `protocol = "react_json"` or `protocol = "native"` either in `[providers.<name>]` or the session-create body.

API endpoints added:

- `GET /api/providers` — list configured providers + connection state
- `GET /api/providers/<name>/models` — list available models for a provider
- `POST /api/sessions/new` — accepts `provider`, `model_name`, `protocol` in body
- `GET /api/sessions/<sid>` — response includes `provider`, `model_name`, `protocol`

The optional `[providers]` extra in `pyproject.toml` declares `anthropic>=0.40` (used only as a future SDK convenience; the wire path is raw httpx). Gemini stays SDK-free.

## Notes

- Sessions land in `<cwd>/sessions/`. Don't put it inside OneDrive — fsync gets slow.
- Tool toggles per session: web sidebar checkboxes, or `[tools] enabled = ["..."]` in config.
- The default `web_search` uses DuckDuckGo HTML and breaks regularly. Drop in `tools/web_search_brave.py` (you provide the API key) when it does.

## Workspace sandbox

Confine the agent's filesystem reach to one directory. Combined with `--auto-approve`, the agent runs autonomously inside the workspace without confirming every file write.

```bash
godbot-tui --workspace ./scratch --auto-approve
godbot-cli --workspace ./scratch --auto-approve
```

In the web UI: `POST /api/sessions/new` with body `{"workspace": "/abs/path", "auto_approve_in_sandbox": true}`.

In VSCode: workspace defaults to your current VSCode workspace folder; toggle `godbot.autoApproveInSandbox` in settings to enable autonomous mode.

In Discord: per-channel workspace via the (forthcoming) `/godbot workspace <path>` command.

### What's confined and what isn't

| Tool | Sandboxed? |
|---|---|
| `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `list_dir` | **HARD** — paths confined to workspace, escapes refused |
| `read_blob` | by construction (only reads from active session's blobs) |
| `run_python`, `run_powershell`, `run_bash` | **SOFT** — `cwd` set to workspace + `WORKSPACE_ROOT` env var, but the model can still `cd C:\Windows` mid-script |
| `take_screenshot` | NOT sandboxed — writes to any path the model passes |
| `web_fetch`, `web_search` | NOT sandboxed (network is out of scope) |

### `--auto-approve` semantics

When the workspace is active AND `--auto-approve` (or the equivalent setting) is on, dangerous tools in the **safe set** (`write_file`, `edit_file`) skip the gate prompt. Shell tools (`run_powershell`, `run_bash`, `run_python`) and `take_screenshot` STILL prompt for approval — they aren't in the safe set because the soft sandbox can't actually contain them.

To skip ALL gates regardless of sandbox: use `--yolo`. That's the unrestricted mode.

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

## Discord bot

Self-hosted single-owner Discord bot. Surfaces GodBot in any channel or DM.

### Setup

1. Create a bot at https://discord.com/developers/applications
2. Enable `Message Content Intent` under "Bot"
3. Generate an invite link with the `bot` scope and `Send Messages` + `Embed Links` permissions; invite to a server you own
4. Add to `~/.godbot/config.toml`:
   ```toml
   [discord]
   token = "<bot token from the developer portal>"
   owner_id = <your discord user id, right-click yourself with developer mode on>
   ```
5. Install the optional dep:
   ```bash
   pip install -e ".[dev,discord]"
   ```
6. Run:
   ```bash
   godbot-discord                  # daemon-preferred
   godbot-discord --no-daemon      # embedded only (single channel at a time; warning logged)
   ```

### Usage

- **In a channel:** mention the bot — `@GodBot read main.py`
- **In a DM:** just send a message
- Each channel is its own session; sessions persist across bot restarts via `~/.godbot/discord-sessions.json`
- Tool gates appear inline as button views — Allow / Always / Deny (owner only; 5-min auto-deny)

### Limits

- Solo-owner only — other users are silently ignored
- Embedded mode (`--no-daemon`) serializes ALL channels through one global lock due to a known shared-env-var limitation (see "Known issues" above)
- Discord rate-limits message edits; the bot throttles streaming to ~750ms / 400-char cadence

## VSCode Extension

Lives at `extensions/vscode/`. Build:

```bash
cd extensions/vscode
npm install
npm run build
```

Install in VSCode:
- Easiest path: open this folder in a separate VSCode window, press F5 to run an Extension Development Host
- Packaged install: `npx vsce package --allow-missing-repository` produces `godbot-vscode-0.1.0.vsix` → `code --install-extension godbot-vscode-0.1.0.vsix`

Settings (set via Settings → Extensions → GodBot):
- `godbot.baseUrl` — daemon URL
- `godbot.autoLaunch` — start `godbot-web` if down
- `godbot.pythonPath` — interpreter to use when auto-launching

The extension talks to the same `godbot-web` daemon as the TUI / Discord bot, so sessions are visible across all three surfaces.

### Known issues (v0.1)

- No session picker widget — `New Session` button works, but switching to an existing daemon session requires a future widget
- No `vscode-test` integration framework — manual smoke + `tsc --noEmit` + Node SSE-parser test (3 tests via `tsx`) is the v0.1 floor
- Node tests run via `tsx`: `node --import tsx --test src/__tests__/client.test.ts` (Node's built-in runner doesn't natively understand TypeScript, so `tsx` is loaded as an importer)
- VSIX packaging needs `--allow-missing-repository` since the `package.json` has no `repository` field
- The webview button glyphs (⊕ ⏹ ⚙) are inline Unicode in the provider's HTML; if your VSCode build renders them as boxes, switch to text labels in `chatProvider.ts`

## MCP servers

GodBot is an [MCP](https://modelcontextprotocol.io) **client**: it can connect to any third-party MCP server (filesystem, github, postgres, brave-search, puppeteer, ...) over stdio and surface its tools to the agent under namespaced names `mcp_<server>_<tool>`. No Python wrappers required.

Configure servers in `~/.godbot/config.toml`:

```toml
[mcp.servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/some/workspace"]

[mcp.servers.github]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]

[mcp.servers.github.env]
GITHUB_PERSONAL_ACCESS_TOKEN = "ghp_..."
```

On daemon (or CLI) startup each configured server is spawned as a subprocess and its tools are registered into the agent's tool registry. Failures are logged and skipped — a broken server entry never blocks startup. Use the built-in `mcp_status` tool to inspect connection state and tool inventory at runtime.

Notes:
- Stdio transport only in v1; HTTP/SSE servers are not yet supported.
- All MCP tools are marked `dangerous=True` so the user gates each call.
- Restart the daemon to pick up changes to `[mcp.servers.*]`.
- On Windows, `npx` resolves to `npx.cmd` automatically when Node is on PATH.
