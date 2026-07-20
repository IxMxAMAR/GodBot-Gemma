# Smoke-test runbook

A 5-minute sanity pass to run by hand (or wire into CI) before tagging a release.

The script is `scripts/smoke_test.py`. It exercises the most-used endpoints and one live LLM round-trip. Read-mostly: creates ONE temporary session, runs ONE turn, deletes the session.

## Prereqs

1. The daemon is running: `godbot-web` (or `python -m godbot.interfaces.web --port 7878`).
2. At least one provider is reachable. Pick one:
   - **LM Studio** at `localhost:1234` with a chat model loaded (default: looks for "gemma" in id, override with `--model <id>`).
   - **OpenAI** with `OPENAI_API_KEY` set in env.
   - **Anthropic** with `ANTHROPIC_API_KEY` set.
3. `httpx` available (it's in the daemon's deps).

## Run

```bash
# default (LM Studio)
python scripts/smoke_test.py

# OpenAI
python scripts/smoke_test.py --provider openai --model gpt-4o-mini

# Anthropic
python scripts/smoke_test.py --provider anthropic --model claude-haiku-4-5-20251001

# Skip the LLM call (just exercise the daemon HTTP surface)
python scripts/smoke_test.py --skip-llm

# Auth-enabled daemon
python scripts/smoke_test.py --token $GODBOT_API_TOKEN
```

Exit code 0 if every check passes, 1 otherwise.

## What it checks

| Phase | Endpoint(s) | Pass criterion |
|---|---|---|
| health | `GET /api/health` | HTTP 200, `status: "ok"` |
| version | `GET /api/version` | HTTP 200, version string present |
| tools registered | `GET /api/tools` | At least one tool |
| providers configured | `GET /api/providers` | At least one provider |
| session create + fetch | `POST /api/sessions/new` + `GET /api/sessions/{sid}` | session_id returned, fetch 200 |
| admin snapshot | `GET /api/admin/snapshot` | HTTP 200, `version` + `health` + `stats` keys |
| LLM turn | `POST /api/agent/quickrun` (skipped with `--skip-llm`) | status="done", reply contains "PONG" |
| session delete | `DELETE /api/sessions/{sid}` | HTTP 200 |

## Manual checks (not automated)

These require a UI or out-of-process action:

- **Studio launch**: run `npm run tauri dev` from your GodBot-Studio checkout — confirm window opens, daemon dot turns green.
- **Discord bot**: bring up the bot, run `/stats` in a server channel — confirm it returns an embed.
- **VSCode extension**: install `extensions/vscode/godbot-vscode-*.vsix`, run `Godbot: Open Workspace` from the Command Palette — confirm the quick-pick lists known workspaces.
- **MCP host**: add `godbot-mcp-server` to Claude Desktop's `claude_desktop_config.json` (see the "MCP servers" section of the README), restart Claude Desktop, ask it to call a GodBot tool.

## When something fails

The script prints `OK` / `FAIL` per phase with a short reason. Common failures:

- `health endpoint` FAIL → daemon isn't running, or wrong `--base-url` (default 7878).
- `LLM turn` FAIL with `status=running` → quickrun timed out; provider is slow or model isn't loaded. Bump `max_wait_seconds` in the script if needed, or use `--skip-llm`.
- `LLM turn` FAIL with `result` not containing "PONG" → model is too small to follow simple instructions, or you picked a non-chat model.
- All API checks FAIL with `HTTP 401` → daemon has auth on; pass `--token`.
