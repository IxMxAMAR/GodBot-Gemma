# GodBot for VSCode

Chat with GodBot from a sidebar panel; ask about your current selection; apply tool calls (read/write/edit) to your workspace.

## Install

1. Build: `npm install && npm run build`
2. Package: `npm run package` — produces `godbot-vscode-0.1.0.vsix`
3. Install in VSCode: `code --install-extension godbot-vscode-0.1.0.vsix`

## Setup

You also need GodBot itself installed (Python):

```bash
pip install -e <godbot-repo>
```

Configure in VSCode settings:
- `godbot.baseUrl` — daemon URL (default `http://127.0.0.1:7878`)
- `godbot.autoLaunch` — start `godbot-web` if down (default true)
- `godbot.pythonPath` — interpreter to use when auto-launching

## Usage

- Click the GodBot icon in the activity bar
- `Ctrl+Alt+G` — send the current editor selection as a message
- `Ctrl+Alt+N` — start a new session

## Tests

Node tests use `tsx` to load TypeScript directly:

```bash
node --import tsx --test src/__tests__/client.test.ts
```
