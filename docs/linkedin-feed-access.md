# LinkedIn Feed Access

Deterministic feed collection for production/background use — no LLM, no token
usage. Sibling of `LinkedInWebAgent` (LangChain + tools). Prefer this path for
cron, workers, and A/B experiments that only need raw post rows.

## Why this exists

| | `LinkedInWebAgent` | `LinkedInFeedPipeline` |
|---|---|---|
| Control flow | LangChain `ainvoke`, tool calls | Fixed phases |
| Intelligence | LLM-driven | None |
| Loop | AI turns | Scroll until target / plateau / budget |
| Typical use | Interactive job-search agent | Background feed scrape + JSON export |

Naming: the class was formerly `LinkedInFeedAgent`. It was renamed to
**`LinkedInFeedPipeline`** so it is not confused with an LLM agent loop. The
scroll/parse loop lives in `LiveFeedCollector`, not in the pipeline orchestrator.

## Current architecture

```
LinkedInFeedPipeline(LinkedInFeedPipelineConfig)
  1. authenticate     → LinkedInBrowserSession.ensure_authenticated
  2. collect          → crawl_linkedin_feed(options)  and/or  LiveFeedCollector.run()
  3. dedupe + validate → dedupe_feed_posts, arm_quality_metrics
  4. export           → write_feed_export

LiveFeedCollector.run()
  prepare_live_feed_page
    → try in-browser JS IIFE (scroll + extract)
    → fallback HTML parse + Python scroll loop
    → resolve author-only stubs (Send → Copy link)
```

Default path (`use_crawl4ai=False`): one locked live Chromium session.

Opt-in Crawl4AI (`use_crawl4ai=True`): virtual-scroll HTML snapshot. When
`resolve_missing_urls=True` and the crawl returns empty, the pipeline falls back
to the live collector (`crawl4ai+live_fallback`). When crawl returns rows with
missing `post_url`, a separate live session fills URLs via Send → Copy link.

## Module map

| Module | Role |
|--------|------|
| `linkedin_feed_pipeline.py` | Top-level orchestrator (`LinkedInFeedPipeline`) |
| `linkedin_feed_config.py` | Dataclasses: profile, scroll, selectors, pipeline config, results |
| `linkedin_feed_collector.py` | `LiveFeedCollector` — scroll/parse loop |
| `linkedin_feed_prepare.py` | Navigate, sort Recent (or `?feedType=recent`), hydrate cards |
| `linkedin_feed_scroll_js.py` | In-browser JS IIFEs (feed posts + hiring) |
| `linkedin_feed_url_resolve.py` | Clipboard spy + Send → Copy link |
| `linkedin_browser_session.py` | Locked `BrowserManager` + profile lock for one session |
| `linkedin_feed_parser.py` | HTML → post dicts; URL validation helpers |
| `crawl4ai_linkedin_helper.py` | Crawl4AI virtual scroll + `crawl_linkedin_feed(options)` |
| `linkedin_feed_collect.py` | Thin backward-compat facade over collector / prepare / resolve |
| `linkedin_feed_agent.py` | Shim: `LinkedInFeedAgent = LinkedInFeedPipeline` |
| `linkedin_feed_cli.py` | CLI entry (`python -m src.utility.linkedin_feed_cli`) |

Related: `browser_manager.py` (`safe_evaluate`, `wait_for_page_ready`),
`linkedin_auth.py`, `makefile` targets `sign-in-linkedin` / `collect-linkedin-feed`.

## Configuration

Callers pass one object instead of a long kwargs list:

```python
from src.utility.linkedin_feed_pipeline import (
    LinkedInFeedPipeline,
    LinkedInFeedPipelineConfig,
)

config = LinkedInFeedPipelineConfig.from_kwargs(
    user_data_dir=".browser_profile",
    profile_directory="Default",
    headless=True,
    max_posts=25,
    scroll_rounds=50,
    check_auth=True,
    resolve_missing_urls=True,
    use_crawl4ai=False,
)
result = await LinkedInFeedPipeline(config).run()
```

Flat kwargs still work (`LinkedInFeedPipeline(user_data_dir=..., ...)`), and
`LinkedInFeedPipeline.from_kwargs(...)` is available.

### Nested types (`linkedin_feed_config.py`)

- **`LinkedInBrowserProfile`** — `user_data_dir`, `profile_directory`, `headless`, `viewport`
- **`FeedScrollPolicy`** — `max_posts`, `scroll_rounds`, scroll px / pause / plateau / cap; optional `agent_max_posts`
- **`FeedDomSelectors`** — `container_selector`, `post_card_selector`, `html_selector`
- **`FeedCollectionOptions`** — profile + scroll + selectors + `sort_js_path`, `resolve_missing_urls`, `require_activity_id`
- **`LinkedInFeedPipelineConfig`** — `collection` + `export_dir`, `login_timeout_s`, `check_auth`, `use_crawl4ai`
- **`FeedCollectionResult`** — posts, sort meta, `collection_strategy` (`js` | `html_scroll` | `js+resolve`), scroll / resolve counts

## Runtime flow (live path)

1. **Auth** (optional): open profile → navigate feed → `is_linkedin_authenticated`. Failures ask for `make sign-in-linkedin`.
2. **Prepare**: if sort JS file exists, load feed → run sort script; on failure navigate to `?feedType=recent`. If sort JS is **missing**, navigate **directly** to Recent (avoids a useless feed → Recent double hop).
3. **Collect**: prefer JS IIFE on lazy-mount cards; keep author-only stubs. On JS failure / empty feed container, HTML parse + scroll until target, plateau, or budget.
4. **Resolve**: `wait_for_page_ready` then Send → Copy link via `BrowserManager.safe_evaluate` (retries when Playwright reports execution context destroyed after SPA navigation).
5. **Dedupe / quality / export**: cap to `max_posts`, compute valid `post_url` stats, write timestamped JSON under the export directory.

## Operational constraints

- Uses `linkedin_profile_lock` under `user_data_dir`. Do not run `LinkedInFeedPipeline`, `LinkedInWebAgent`, and ad-hoc `browser_manager` against the same profile concurrently.
- Headless runs need a pre-warmed signed-in profile (`make sign-in-linkedin` / `scripts/linkedin_sign_in.py`).
- Not for inline FastAPI request handlers — treat as a background job.
- Default post cards: `div[data-lazy-mount-id]`.
- `resolve_missing_urls=False` skips Send → Copy link (may yield author-only rows without `post_url`).
- Default sort JS path points at `src/skills/linkedin-job-search/examples/linkedin_sort_recent.js`. If that file is absent, prepare uses the Recent URL only.

## CLI

```bash
make sign-in-linkedin
make collect-linkedin-feed
# or
poetry run python -m src.utility.linkedin_feed_cli \
  --user-data-dir .browser_profile --profile-directory Default \
  --headless --max-posts 25 --scroll-rounds 50
```

Useful flags: `--no-headless`, `--use-crawl4ai`, `--no-resolve-missing-urls`,
`--skip-auth-check`, `--no-export`, `--require-activity-id`.

Export files land under `settings.EXPORT_DIRECTORY` (typically `.export/`) as
`linkedin_feed_<timestamp>.json` with `params`, `posts`, `valid_posts`, `quality`.

## Reliability notes (known failure mode)

**Symptom:** `Page.evaluate: Execution context was destroyed, most likely because of a navigation` during **collecting feed**, often after LinkedIn React hydration noise (`Minified React error #418`).

**Cause:** The in-page feed-scroll IIFE can finish successfully; LinkedIn then reloads/navigates. A subsequent `page.evaluate` (clipboard spy / URL resolve) hits a destroyed context. Missing sort JS previously also forced feed → Recent navigation, increasing churn.

**Mitigations in tree:**

- Navigate straight to Recent when sort JS is missing.
- `BrowserManager.wait_for_page_ready` before URL-resolution evaluates.
- `BrowserManager.safe_evaluate` retries on context-destroyed errors.

## Refactor history and proposals

### Done

1. **Rename** `LinkedInFeedAgent` → `LinkedInFeedPipeline` (compat alias retained).
2. **Config objects** — flatten kwargs into `FeedCollectionOptions` / `LinkedInFeedPipelineConfig`.
3. **Extract collector** — scroll loop owned by `LiveFeedCollector` with explicit strategies.
4. **Split helpers** — prepare / scroll JS / URL resolve / browser session modules; `linkedin_feed_collect.py` remains a thin facade.
5. **Crawl4AI API** — `crawl_linkedin_feed(FeedCollectionOptions)` instead of mirrored kwargs.
6. **Context-destroyed hardening** — as above.

### Still reasonable follow-ups

1. **Unify backends behind a protocol** — e.g. `FeedCollector` with `LiveFeedCollector`, `Crawl4AIFeedCollector`, `Crawl4AIWithLiveFallbackCollector` so `run()` picks a strategy instead of an if/else tree.
2. **Single browser session for the whole live run** — auth + collect + resolve already use `LinkedInBrowserSession`; Crawl4AI fallback still opens separate sessions by design (different Chromium stacks / lock). Further collapsing live-only phases is optional.
3. **Restore or relocate sort JS** — if Recent-via-menu is preferred over `?feedType=recent`, check in `linkedin_sort_recent.js` (or point `sort_js_path` at a maintained copy).
4. **Retire the agent shim** — after callers migrate off `linkedin_feed_agent.LinkedInFeedAgent`, delete the alias module.
5. **Hiring collector** — `collect_hiring_posts_from_page` still lives beside feed collection; could share `FeedScrollPolicy` more explicitly or move under a dedicated hiring module.

## Tests

- `tests/test_linkedin_feed_agent.py` — pipeline modes (live, crawl4ai, fallback, auth, URL resolve).
- `tests/test_linkedin_feed_collect.py` — collector / scroll budget / JS vs HTML paths (mocked browser).

## Import cheat sheet

```python
# Preferred
from src.utility.linkedin_feed_pipeline import LinkedInFeedPipeline, LinkedInFeedPipelineConfig

# Legacy alias
from src.utility.linkedin_feed_agent import LinkedInFeedAgent  # == LinkedInFeedPipeline

# Lower-level (CLI steps, experiments)
from src.utility.linkedin_feed_collect import (
    collect_raw_feed_posts_from_page,
    prepare_live_feed_page,
    resolve_post_urls_via_send_button,
)
```
