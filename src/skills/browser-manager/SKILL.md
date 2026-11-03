---
name: browser-manager
description: >-
  Drives Playwright Chromium via the BrowserManager CLI (navigate, scrape text/HTML,
  JavaScript, screenshots, LinkedIn feed-posts). Use when automating a real browser
  headlessly or with a persistent profile, in CLI/CI/production, or when
  cursor-ide-browser is unavailable. For LinkedIn hiring workflow steps 1–4, use
  the linkedin-job-search skill instead.
disable-model-invocation: true
---

# Browser Manager CLI

**All project skills and CLI commands:** [.cursor/skills/SKILL.md](../SKILL.md)

## Setup (one-time)

```bash
poetry run playwright install chromium
```

## Session modes

| Mode | How | When |
|------|-----|------|
| `ephemeral` | Single subcommand (`navigate`, `title`, …) | One-off actions; **tab state does not persist** to the next CLI call |
| `multi_step` | `run --steps '[...]'` | Navigate → JS → scrape in one browser lifetime |

Responses include `"session_mode"`. Empty `title` on a standalone `title` command usually means you need `run --steps`.

## Invoke

Prefer from repo root:

```bash
poetry run python scripts/browser_manager.py [GLOBAL_FLAGS] COMMAND ...
poetry run python -m src.utility.browser_manager [GLOBAL_FLAGS] COMMAND ...
```

## BrowserManager constructor flags

Every flag maps to `BrowserManager.__init__`:

| Flag | Default | Parameter |
|------|---------|-----------|
| `--headless` | on (if neither headless flag set) | `headless=True` |
| `--no-headless` | — | `headless=False` |
| `--viewport WxH` | `1280x720` | `viewport` |
| `--user-agent STRING` | built-in Chrome UA | `user_agent` |
| `--user-data-dir PATH` | none | `user_data_dir` (persistent profile) |
| `--profile-directory NAME` | none | `profile_directory` (e.g. `Default`) |

**Persistent profile:** do not open the same `--user-data-dir` in regular Chrome at the same time (SingletonLock).

## Commands

| Command | Purpose |
|---------|---------|
| `navigate URL` | Open URL |
| `url` | Current URL |
| `title` | Document title |
| `text [--max-chars N]` | `document.body.innerText` |
| `html [--selector SEL] [--max-chars N]` | Page or element HTML |
| `js CODE` or `js --file path.js` | Run JS in page |
| `screenshot [--out path.png]` | PNG (base64 preview in JSON if no `--out`) |
| `scroll PIXELS` | Vertical scroll |
| `wait [--selector SEL] [--timeout-ms N]` | Wait for selector or sleep |
| `feed-posts [--max-posts N] [--scroll-rounds N]` | LinkedIn feed JSON (compact); scrolls down until `max_posts`, plateau, or budget |
| `hiring-posts [--max-posts N] [--scroll-px N] [--wait-ms N] [--max-scrolls N]` | Hiring-announcement posts; structured JSON with company/job/author/post URLs |
| `check-auth` | Check LinkedIn auth state for profile (errors if not signed in) |
| `sign-in [--timeout-s N]` | Open LinkedIn login and wait for manual sign-in |
| `run --steps JSON` | Multiple ops in one browser session |

### `run --steps` example

```bash
poetry run python scripts/browser_manager.py run --steps '[
  {"op":"navigate","url":"https://example.com"},
  {"op":"js","code":"return document.title"},
  {"op":"text","max_chars":2000}
]'
```

Supported `op` values: `navigate`, `url`, `title`, `text`, `html`, `js`, `screenshot`, `scroll`, `wait`, `scroll-until`, `feed-posts`, `hiring-posts`, `check-auth`, `sign-in`.

## JSON stdout

Success (single line on stdout):

```json
{"ok": true, "command": "navigate", "browser": {"headless": true, "viewport": [1280, 720], ...}, "data": {...}}
```

Failure (stderr, non-zero exit):

```json
{"ok": false, "command": "text", "error": "..."}
```

Do not use `cursor-ide-browser` for headless, CI, or production — use this CLI.

## Related

- LinkedIn hiring steps 1–4: **linkedin-job-search** skill
- LinkedIn manual login only: **linkedin-sign-in** skill
- Full LangChain LinkedIn agent: `poetry run python -m src.utility.browser`
