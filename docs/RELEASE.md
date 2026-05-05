# GodBot daemon — release runbook

Manual checklist for cutting a daemon release. Studio has its own runbook.

## 1. Pre-flight

```bash
# from repo root
cd <repo-root>

# 1.1 confirm working tree clean
git status

# 1.2 confirm tests pass
.venv/Scripts/python -m pytest tests/ --ignore=tests/test_lmstudio_smoke.py -q
# expected: ~1270 passed, 2 skipped

# 1.3 smoke test against a running daemon (separate terminal)
.venv/Scripts/godbot-web --port 7878 &
.venv/Scripts/python scripts/smoke_test.py --skip-llm
# expected: 7/7 checks passed
```

## 2. Bump version

Edit **two** places:

- `pyproject.toml` → `[project] version = "X.Y.Z"`
- (optional) `godbot/__init__.py` if a `__version__` literal exists

Commit:

```bash
git add pyproject.toml
git commit -m "chore: bump version to X.Y.Z"
```

## 3. Tag + push

```bash
git tag -a vX.Y.Z -m "GodBot vX.Y.Z"
git push origin master
git push origin vX.Y.Z
```

## 4. GitHub release

Cut the release on GitHub against the tag. Past three commits' messages are usually the changelog. Keep release notes short — link to the diff for detail.

The Studio's auto-updater + the daemon's `/api/version/check_update` both query `api.github.com/repos/IxMxAMAR/GodBot-Gemma/releases/latest`, so the moment the release is published the next probe sees the new version.

## 5. Optional: build distributable wheels

```bash
.venv/Scripts/python -m pip install build
.venv/Scripts/python -m build  # produces dist/godbot-X.Y.Z-py3-none-any.whl
```

Attach the `.whl` to the GitHub release if you want users to be able to `pip install` from the release page directly.

## 6. Console scripts cheat-sheet

After `pip install -e .`, the venv exposes:

| Script | Module |
|---|---|
| `godbot-web` | `godbot.interfaces.web:main` |
| `godbot-cli` | `godbot.interfaces.cli:main` |
| `godbot-tui` | `godbot.interfaces.tui:main` |
| `godbot-index` | `godbot.index:main` |
| `godbot-discord` | `godbot.interfaces.discord_bot.bot:main` |
| `godbot-mcp-server` | `godbot.mcp.server:main` |

If a release adds a new entry point, add it to `pyproject.toml`'s `[project.scripts]` and re-install with `pip install -e .` so the venv shim regenerates.

## 7. Code signing (Windows)

The daemon ships as Python source. No signing needed for the wheel.

The Studio Tauri build produces `.msi` and `.exe` installers that DO benefit from a signing cert. See `GodBotStudio/docs/RELEASE.md` for the Studio-side signing flow.
