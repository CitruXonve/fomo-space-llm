---
name: linkedin-feed-collect
description: >-
  Scrapes the LinkedIn feed for hiring-announcement posts and returns structured
  JSON with company URLs, job listing URLs, author profile URLs, and post URLs.
  Use when the user wants to collect hiring signals from their LinkedIn feed.
  Requires an authenticated LinkedIn session (run linkedin-sign-in first if needed).
disable-model-invocation: true
---

# LinkedIn Feed Collect

**All project skills and CLI commands:** [SKILL.md](../SKILL.md)

Collects hiring-announcement posts from the LinkedIn feed using an in-browser
scroll-and-extract loop. Detection uses job listing link presence plus keyword
matching (hiring, open role, now hiring, …).

## Parameters

| Param                 | Default | CLI flag          | `run --steps` field        |
| --------------------- | ------- | ----------------- | -------------------------- |
| `max_posts`           | `10`    | `--max-posts N`   | `"max_posts": N`           |
| `scroll_px`           | `800`   | `--scroll-px N`   | `"scroll_px": N`           |
| `wait_ms`             | `2000`  | `--wait-ms N`     | `"wait_ms": N`             |
| `max_scrolls`         | `25`    | `--max-scrolls N` | `"max_scrolls": N`         |
| `max_scroll_per_post` | `8`     | _(step only)_     | `"max_scroll_per_post": N` |

## Prerequisites

- [ ] `poetry run playwright install chromium`
- [ ] Authenticated LinkedIn session — run **linkedin-sign-in** if `check-auth` returns `"authenticated": false`

## Shell alias

```bash
BM="poetry run python src/utility/browser_manager_cli.py \
  --user-data-dir .browser_profile --profile-directory Default --no-headless"
```

## Step 0 — Authentication gate

```bash
$BM check-auth
```

If unauthenticated, run **linkedin-sign-in**, then recheck before continuing.

## Step 1 — Collect hiring posts

### Standalone command (opens and closes browser)

```bash
$BM hiring-posts --max-posts 10
```

### Inside `run --steps` (shares session with navigate / wait)

```bash
$BM run --steps '[
  {"op": "navigate", "url": "https://www.linkedin.com/feed/"},
  {"op": "wait", "timeout_ms": 4000},
  {"op": "hiring-posts", "max_posts": 10}
]'
```

The `hiring-posts` op runs its own internal scroll loop — no separate `scroll-until`
step is needed. Use `run --steps` when you need to navigate first or chain with
other ops in the same browser session.

## Output schema

`data.posts` is an array; one entry per detected hiring post:

```json
[
  {
    "company_urls": ["https://www.linkedin.com/company/..."],
    "job_listing_urls": ["https://www.linkedin.com/jobs/view/..."],
    "author_profile_url": "https://www.linkedin.com/in/...",
    "post_url": "https://www.linkedin.com/feed/update/..."
  }
]
```

- `company_urls` — all `/company/` links found in the post; `[]` if none
- `job_listing_urls` — all `/jobs/view/` links found; `[]` if none
- `author_profile_url` — first `/in/` link (post author); `null` if not found
- `post_url` — resolved via Send → Copy link to post; `null` only if the post card has no Send button (ads, sponsored); expect ~1/12 null in a typical feed run

## How `post_url` is resolved

LinkedIn's React DOM does not embed `/feed/update/` or `/posts/` anchors directly
in feed cards, so DOM parsing always returns `null`. `post_url` is resolved after
the scroll loop by `resolve_post_urls_via_send_button` for each post:

1. JS (`_FIND_SEND_BTN_JS`) climbs the DOM from the author's `/in/` anchor to find
   the **Send `<a>`** in the post's action bar. The Send action is an `<a>` tag
   (not `<button>`), so the selector is `querySelectorAll("button, a")`.
2. The Send link's `href` is set to `javascript:void(0)` before `.click()` to
   prevent it from navigating away (its natural href is `https://www.linkedin.com/feed/`).
3. Playwright's **native `page.locator("text=Copy link to post").first.click()`**
   is used — NOT `element.click()` inside `page.evaluate()`. This is critical:
   `document.execCommand("copy")` requires a user-activation task; `await
setTimeout(...)` inside a single `page.evaluate()` breaks that chain, silently
   failing the copy. Each Playwright call gets its own activation context.
4. The URL is captured via a `document.addEventListener('copy', ...)` spy
   (`window.__clipboardLastUrl`). LinkedIn uses `execCommand("copy")`, not
   `navigator.clipboard.writeText`, so only the `copy` event path works.
5. `page.keyboard.press("Escape")` closes the share dialog before the next post.

The "Copy link to post" menu item has no `role` attribute — it is a plain `<button>`
inside the share dialog. Do **not** include `div` in the `copyBtn` selector: the
outer `<div>` wrapper also matches the text and may contain a navigating `<a>`,
which destroys the execution context.

## Scroll-forward resolution (`resolve_post_urls_via_send_button`)

After the IIFE finishes, the feed viewport is at the **bottom** of the scroll
journey. Without scrolling, `_FIND_SEND_BTN_JS` may find the author's URL in a
_different_ visible card (a repost, @mention, or sidebar reference), climb to that
card's Send button, and copy the **wrong** URL — repeating the same URL for every
unresolved post.

The resolver therefore:

1. **Scrolls back to the top** (`scrollTop = 0`, 1 s settle) so the earliest
   collected posts re-enter the DOM.
2. Iterates posts **in order** (top-to-bottom, matching the IIFE's collection
   order) and scrolls forward ~600 px (one card height) after each failed lookup,
   up to `max_scroll_per_post` attempts per post (default 8). This keeps the target card in the viewport when its
   Send button is clicked.
3. Maintains a **`resolved_urls` set** — a candidate URL already in the set means
   the wrong card was found. This is treated as a miss (scroll forward continues)
   rather than stopping, so the resolver keeps advancing to find the correct card.

## Wrong-card guards in `_FIND_SEND_BTN_JS`

Three guards prevent clicking the Send button of the wrong post:

1. **Feed-scoped anchor search** — anchors are queried inside
   `div[data-testid=mainFeed]` only, excluding sidebar "People you may know" and
   recommendation links that appear before feed cards in DOM order.

2. **First-`/in/`-link verification** — after climbing up from the matched anchor
   to a container that has a Send button, the script checks that the **first
   `/in/` link** inside that container starts with `authorHref`. If it doesn't,
   the author is only a body @mention or reposter attribution in someone else's
   card — `_FIND_SEND_BTN_JS` returns `false` rather than clicking the wrong
   Send button.

   Without this check, the top post (if it remains visible after scroll-to-top)
   can be repeatedly mis-selected for every subsequent post whose author is
   @mentioned in it, causing the same URL to be proposed over and over.

3. **`data-lazy-mount-id` circuit breaker** — `div[data-lazy-mount-id]` elements
   are LinkedIn's top-level feed card containers, direct children of
   `div[data-testid=mainFeed]`. Before clicking Send, `_FIND_SEND_BTN_JS` looks
   up the card via `match.closest("div[data-lazy-mount-id]")` and checks
   `window.__resolvedCardIds`. If the card ID is present, the function returns
   `false` without opening the share dialog.

   On a successful copy, Python calls `_MARK_CARD_RESOLVED_JS` to add the card
   ID to `window.__resolvedCardIds` (marked only after URL is confirmed — a failed
   copy leaves the card unmarked so scroll-forward retries still work).

   `_FIND_SEND_BTN_JS` stores the card ID in `window.__lastClickedCardId` before
   calling `sendBtn.click()` and always returns `true` on success (never the ID
   string directly — an empty `data-lazy-mount-id` would be falsy in Python).
   Python reads `window.__lastClickedCardId` after confirming the URL, then calls
   `_MARK_CARD_RESOLVED_JS`. `window.__resolvedCardIds` is initialised alongside
   `window.__clipboardSpyInstalled` in `_CLIPBOARD_SPY_JS`.

   Unlike a custom `data-*` attribute, the Set lives on `window` and survives
   React re-renders.

## Observed scroll behavior during resolution

Confirmed by live debug tracing: **LinkedIn resets the feed scroll position to
~744 px after every successful Send → Copy dialog interaction** (closing via
Escape after "Copy link to post" is clicked). This means:

- After each success, the next post's resolution attempt starts from ~744 px
  (near the top), not from wherever the previous card was found.
- Posts whose cards are accessible near the top of the feed are resolved in
  quick succession without needing scroll increments.
- Posts whose author links are absent from the DOM at all scroll positions
  within the 8-attempt × 600 px budget remain null.

All confirmed null results produce `candidate=None` (author link not found in
DOM), not wrong-URL duplicates — the circuit breaker and first-`/in/`-link guard
are working correctly. The remaining null rate is bounded by LinkedIn DOM
virtualization, not code bugs. Cards like `holzmillermichael` are consistently
unreachable across sessions — they are likely sponsored posts (no Send button) or
deep feed cards that are fully virtualized at all reachable scroll positions.

## Step 2 — Save export

```bash
mkdir -p .export
printf '%s' "$JSON" > ".export/linkedin_hiring_$(date +%Y-%m-%dT%H:%M:%S%z).json"
```

## Guardrails

- Never invent URLs — only use values returned from the `hiring-posts` step.
- Standalone `hiring-posts` navigates to the **current page** — navigate first or
  use `run --steps` if you need to land on the feed.
- The internal loop scrolls `window` (not a scrollable container); run in
  `--no-headless` mode so the viewport is real and lazy-loaded cards render.
