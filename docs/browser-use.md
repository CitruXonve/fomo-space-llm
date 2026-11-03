# Browser Use & Browserless: Substitution Analysis

Evaluation of **Browserless.io** and the **browser-use** agent skill/CLI as
replacements or complements for this repo’s Playwright stack:

| Local component | Role |
|-----------------|------|
| `src/utility/browser_manager.py` | Async Playwright wrapper (ephemeral / persistent profile) |
| `src/utility/browser_manager_cli.py` | Thin entry; shared CLI lives in `browser_cli_common.py` |
| `src/skills/browser-manager/SKILL.md` | Agent skill → `scripts/browser_manager.py` JSON CLI |
| Consumers | `LinkedInWebAgent`, `LinkedInFeedPipeline`, auth helpers, tests |

Related design docs: [linkedin-feed-access.md](./linkedin-feed-access.md),
[browser-remote-view.md](./browser-remote-view.md).

**Verdict up front:** neither Browserless nor browser-use is a drop-in
replacement for the pair. Both can replace **local Chromium launch / generic
CLI browsing**; neither replaces LinkedIn-specific ops, the JSON step runner
contract, or the imperative Python API used by agents without a redesign.

---

## Current stack (what would be replaced)

```
Agent skills / Claude
        │
        ▼
scripts/browser_manager.py  →  browser_manager_cli / browser_cli_common
        │                         (JSON: navigate, js, feed-posts, run --steps, …)
        ▼
BrowserManager  (Playwright Chromium)
  • ephemeral launch  OR  launch_persistent_context(user_data_dir=.browser_profile)
  • navigate, screenshot, evaluate (+ LLM JS sanitize), HTML/text, scroll, wait
  • LinkedIn helpers elsewhere: check-auth, sign-in, feed-posts, hiring-posts
        │
        ▼
LinkedInWebAgent / LinkedInFeedPipeline / crawl4ai_linkedin_helper
```

Hard dependencies on this design:

1. **Local persistent profile** (`.browser_profile`) + SingletonLock /
   `linkedin_profile_lock`
2. **Manual headed sign-in** (`make sign-in-linkedin`, `sign-in` CLI)
3. **Stable JSON CLI** for skills (`ok` / `command` / `data`)
4. **Deterministic `run --steps`** (not an LLM action loop)
5. **Project-specific logic** — LLM JS fence/TS strip, `#workspace` scroll,
   `stream_html_snapshots`, feed/hiring parsers

---

## 1. Browserless.io

Docs: [AI integrations start](https://docs.browserless.io/ai-integrations/start).
Product: managed headless browsers (cloud or self-hosted) with Playwright /
Puppeteer over WebSocket, REST APIs, BrowserQL (GraphQL), MCP.

### Open-source project

| Item | Detail |
|------|--------|
| Repo | [browserless/browserless](https://github.com/browserless/browserless) |
| Deploy | Docker image (GHCR); Playwright/Puppeteer WS + REST |
| License | **SSPL-1.0** — free for OSS/dev/eval; closed-source commercial/CI often needs a paid license |
| Cloud extras | Stealth, CAPTCHA, proxies, BrowserQL, session recording (plan-dependent) |

### Comparable open-source alternatives

| Project | Notes |
|---------|--------|
| [Steel Browser](https://github.com/steel-dev/steel-browser) | Browser API for AI agents; Docker; Playwright-compatible CDP; typed SDKs |
| [BlitzBrowser](https://github.com/blitzbrowser/blitzbrowser) | Headful CDP in Docker; Apache-2.0 |
| [HeadlessX](https://github.com/SaifyXPRO/HeadlessX) | Self-hosted stealth scraping platform |

### Can Browserless substitute `BrowserManager` / CLI?

**Partially — as a remote browser backend, not as a drop-in.**

Browserless sits **below** Playwright (managed browsers). Your CLI and skills
sit **above** `BrowserManager`. Typical integration is only changing launch:

```python
# Conceptual — not implemented in this repo
browser = await playwright.chromium.connect_over_cdp(
    "wss://production-sfo.browserless.io?token=..."
)
```

Most thin wrappers (`navigate`, `screenshot`, `evaluate`, `click`, `wait`)
still work on the remote `Page`.

| Capability | Fit |
|------------|-----|
| Replace local Chromium in CI/prod | **Yes** — `connect_over_cdp` or self-hosted image |
| Keep `BrowserManager` methods as-is | **Mostly** — same Playwright surface |
| Replace entire CLI + skills | **No** — different product; keep CLI, swap backend |
| Local `.browser_profile` reuse | **Poor** — Session API / Authenticated Profiles ≠ mount local user-data-dir |
| Manual headed LinkedIn sign-in | **Workflow change** — LiveURL / hybrid automation, not `--no-headless` |
| LLM JS sanitize, feed-posts, etc. | **Unchanged** — still yours |
| Playwright version coupling | **Risk** — client must match worker Playwright |
| License / cost | SSPL self-host or cloud API keys |

### Recommended Browserless adoption path

Keep `BrowserManager` as the abstraction; add a backend mode:

```text
BrowserManager.start()
  ├── local      → launch / launch_persistent_context  (today)
  └── browserless → connect_over_cdp(+ Session API for auth state)
```

Keep `browser_manager_cli.py` and skills unchanged so the JSON contract stays
stable. Move LinkedIn auth off local `user_data_dir` onto Browserless Session
API, LiveURL, or cookie import/export if you leave local profiles behind.

---

## 2. browser-use (agent skill + CLI + library)

Skill listing:
[skillsmp.com/skills/browser-use-…](https://skillsmp.com/skills/browser-use-browser-use-skills-browser-use-skill-md)

Upstream: [browser-use/browser-use](https://github.com/browser-use/browser-use)
(~98k stars).

### Install the skill

```bash
npx skills add https://github.com/browser-use/browser-use --skill browser-use
```

### Skills in the same repo

| Skill | Purpose |
|-------|---------|
| **browser-use** | Drive the **`browser-use` CLI** (daemon; index-based clicks) |
| **open-source** | Python `from browser_use import Agent, Browser` (LLM agent loop) |
| **cloud** | Browser Use Cloud REST/SDK |
| **remote-browser** | Sandbox + tunnels for local dev servers |
| **x402** | Pay-per-use cloud credits via USDC |

This analysis focuses on the **browser-use skill/CLI** (closest to our
`browser-manager` skill). The open-source **library** is a separate
substitution path for `LinkedInWebAgent`-style orchestration.

### How the CLI works

Background daemon keeps Chromium open (~50ms per command):

1. `browser-use open <url>`
2. `browser-use state` → clickable elements with **numeric indices**
3. `browser-use click 5` / `input 3 "text"` / …
4. `browser-use close`

Auth / profile modes: headless Chromium, `--headed`, `connect` (user Chrome +
remote debugging), `--profile "Default"`, `cloud connect`.

### Can browser-use substitute the CLI?

**Closest overlap with `browser_manager_cli` / the browser-manager skill —
still not drop-in.**

| Aspect | This repo | browser-use skill |
|--------|-----------|-------------------|
| Invoke | `poetry run python scripts/browser_manager.py …` | `browser-use …` |
| Output | Stable JSON (`ok`, `command`, `data`) | Text; optional `--json` |
| Multi-step | `run --steps '[{op…}]'` | `&&` chaining or separate calls |
| Targeting | CSS selectors + raw JS | **Indices from `state`** |
| LinkedIn | `feed-posts`, `hiring-posts`, `check-auth`, `sign-in` | None |
| Wiring | Poetry, `src/skills/`, tests, Makefile | External CLI; `Bash(browser-use:*)` |
| Profile | `--user-data-dir .browser_profile` | `--profile`, `connect`, cloud |

Generic navigate / text / HTML / JS / screenshot / scroll / wait can be
re-taught to skills via browser-use. LinkedIn commands and the step JSON
runner must be reimplemented or abandoned.

### Can browser-use substitute `browser_manager.py`?

**Poor fit as a library substitute.**

`BrowserManager` is an **imperative** Playwright API imported by Python:

- `LinkedInWebAgent` — LangChain tools wrapping `BrowserManager`
- Feed pipeline / auth / Crawl4AI helpers
- Deterministic, testable steps

browser-use **library** is an **`Agent(task=…, llm=…).run()`** loop: the model
chooses actions from DOM/vision. That replaces orchestration, not just the
browser backend.

| `BrowserManager` | browser-use library / CLI |
|------------------|---------------------------|
| Explicit `navigate()`, `execute_javascript()` | LLM picks next action |
| Deterministic | Probabilistic |
| `async with BrowserManager()` in-process | Agent run or subprocess CLI |
| Tied to Poetry tests | Separate install/runtime |

Shelling out to `browser-use` from Python is a rewrite, not a swap.

### Gains vs losses

**Would gain:** index-based `state` (LLM-friendly), daemon persistence without
`run --steps`, Chrome profile / cloud connect, MCP (`browser-use --mcp`),
large ecosystem.

**Would lose:** LinkedIn pipeline commands and parsers, in-process agent
imports, CI/tests against the JSON contract, deterministic step replay, Poetry
monorepo consistency.

---

## Side-by-side summary

```
                    Browserless              browser-use skill/CLI
                    ───────────              ─────────────────────
Primary role        Remote Chromium host     Agent-facing browser CLI
Replaces            Local launch in BM       browser-manager skill + CLI
Keeps BM methods?   Yes (connect_over_cdp)   No (different CLI; rewrite)
Keeps Python API?   Yes with thin change     No (use Agent library instead)
LinkedIn profile    Redesign sessions        Different profile/connect model
Best as             Backend mode for BM      Complementary exploratory skill
Drop-in for both
files?              No                       No
```

| Goal | Browserless | browser-use skill |
|------|-------------|-------------------|
| Drop-in replace BM + CLI | No | No |
| Replace local Chromium only | Yes | Indirect (CLI owns browser) |
| Replace generic CLI/skills | Keep CLI, swap BM backend | Rewrite skills to `browser-use` |
| Replace `LinkedInWebAgent` | Keep agent; remote browser | Only via `Agent` redesign |
| Keep `.browser_profile` as-is | Poor fit | Partial (`--profile` / `connect`) |
| Add exploratory agent browsing | Optional MCP/REST | **Good fit** |

---

## Practical recommendations

1. **Do not delete** `browser_manager.py` / CLI while LinkedIn feed/auth and
   skills depend on the JSON contract and persistent profile.
2. **If you need managed browsers in CI/prod:** add a Browserless (or Steel)
   backend inside `BrowserManager.start()`; leave CLI/skills alone.
3. **If you want better agentic CLI UX:** install the browser-use skill
   **alongside** `browser-manager` for ad-hoc browsing; keep project skills on
   the Poetry CLI for LinkedIn/production.
4. **If you want an LLM-driven browser agent:** evaluate the browser-use
   **open-source library** as an alternative to `LinkedInWebAgent`’s tool
   loop — that is an agent rewrite, not a backend swap.
5. **Remote viewing / WS control** of the existing server Playwright session
   is a separate track — see [browser-remote-view.md](./browser-remote-view.md).

### Hybrid architecture (if adopting either)

```text
Skills / scripts
  └── browser_manager CLI (JSON contract)     ← keep for LinkedIn + CI
        └── BrowserManager
              ├── local Chromium + .browser_profile
              └── optional: Browserless / Steel CDP

Optional parallel:
  └── browser-use skill CLI                   ← exploratory / cloud / MCP
```

---

## References

- Browserless AI start: https://docs.browserless.io/ai-integrations/start
- Browserless OSS: https://github.com/browserless/browserless
- browser-use skill: https://skillsmp.com/skills/browser-use-browser-use-skills-browser-use-skill-md
- browser-use repo: https://github.com/browser-use/browser-use
- Local: `src/utility/browser_manager.py`, `src/utility/browser_cli_common.py`,
  `src/skills/browser-manager/SKILL.md`
