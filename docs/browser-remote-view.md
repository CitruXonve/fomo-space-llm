# Browser Remote View

Architecture notes for remote viewing of the server-side Playwright browser
(`BrowserManager`) and for streaming page content to a client over WebSocket.
Companion to [linkedin-feed-access.md](./linkedin-feed-access.md).

> Formerly `docs/browser-use.md`. Renamed so `docs/browser-use.md` can cover
> the [browser-use](https://github.com/browser-use/browser-use) product and
> related substitution analysis.

**Status:** Design only. No WebSocket browser API, CDP screencast, or client
viewer exists in the FastAPI app yet. SSE in this project is used only for LLM
chat tokens (`src/api/chat.py`).

---

## Current state

### Server-local Playwright only

`BrowserManager` launches and owns Chromium via Playwright on the **same host**
as the Python process (ephemeral or persistent `.browser_profile`). Page
inspection is **pull-based** inside that process:

| Method | What it returns |
|--------|-----------------|
| `get_page_html()` / `_get_html_snapshot()` | Full document or selector `outerHTML` |
| `extract_visible_text()` | `document.body.innerText` |
| `get_screenshot()` | Viewport PNG as base64 |
| `execute_javascript()` | JSON-serializable evaluate result |
| `stream_html_snapshots()` | Periodic HTML string yields (unused by API/CLI) |

There is no network listener, no subscription API, and no CDP screencast
wrapper on `BrowserManager` today.

### Batch feed collection vs interactive viewing

`LinkedInFeedPipeline` (formerly `LinkedInFeedAgent`) is a **batch worker**:
open browser → collect → export → close. Progress is string labels only
(`on_progress`); the result is structured post JSON, not a live page mirror.

Constraints that matter for any remote viewer:

- Use `linkedin_profile_lock` under `user_data_dir`. Do not run the pipeline,
  `LinkedInWebAgent`, and `browser_manager` concurrently against the same
  profile.
- Integrate feed collection as a background worker, not inline in FastAPI
  request handlers.
- Headless runs need a pre-warmed signed-in profile (`make sign-in-linkedin`).

### FastAPI surface

`src/api/` exposes chat and context routes only. There are **no** routes for
browser state, screenshots, HTML streams, or WebSockets for browser control.

### CLI is one-shot

`browser_cli_common.py` supports ops like `html`, `text`, `screenshot` as
discrete JSON steps — useful for scripts, not a live viewer.

### Can a client browser see the page today?

**No.** A client cannot request or watch current page content in real time.
`stream_html_snapshots` exists on `BrowserManager` but is unused and unexposed.

| Question | Answer |
|----------|--------|
| Client see server Playwright page **right now**? | No |
| Latent near-real-time HTML support? | Partially — unused `stream_html_snapshots` |
| True real-time DOM mirroring? | No — only snapshot polling unless you add push |
| Pipeline expose the page during a run? | No — progress labels + final post JSON only |

---

## Two remote models

```
Server Playwright → client viewer     User's Chrome as source of truth
───────────────────────────────       ────────────────────────────────
Long-lived BrowserManager             Extension or CDP attach to user Chrome
WS/SSE of frames or HTML              Content script / MutationObserver
Cookies in .browser_profile           Cookies in the user's profile
Fits existing BrowserManager          Different architecture; rewrite auth/DOM
```

This document focuses on **server Playwright → client viewer**. Client-side
automation (extension / `connect_over_cdp` to the user’s Chrome) is a separate
product shape.

---

## Target architecture: WebSocket remote-browser bridge

Turn server-side Playwright into a **remotely viewable and operable** browser:

- **Server → client:** page snapshots (and metadata) over WebSocket
- **Client → server:** navigation, clicks, scroll, keyboard
- **Reuse:** `BrowserManager` as the automation layer; add session + protocol
  on top

Keep `LinkedInFeedPipeline` as a batch worker. Do not run feed collection and
interactive viewing on the same profile concurrently (`linkedin_profile_lock`).

### Layering

```
Client browser UI
  canvas / HTML pane + pointer/keyboard capture
        │ WebSocket (JSON + optional binary frames)
        ▼
FastAPI  /api/browser/ws
  BrowserSessionManager  →  BrowserRemoteSession
        │
        ▼
BrowserManager  →  Chromium (+ optional persistent profile)
```

| Layer | Responsibility |
|-------|----------------|
| `BrowserManager` | Playwright primitives (exists) |
| `BrowserRemoteSession` | One long-lived browser + stream loop + input serialization |
| `BrowserSessionManager` | Create/attach/destroy; profile locking; TTL |
| `src/api/browser_ws.py` | Auth, protocol, fan-out (new) |
| Client UI | Render frames; map coordinates; send commands |

### Session model

One remote session ≈ one Playwright context:

```text
BrowserRemoteSession
  session_id
  browser: BrowserManager
  profile_lock          # wrap linkedin_profile_lock when using .browser_profile
  subscribers: set[WebSocket]
  stream_mode: screenshot | html | hybrid | screencast
  stream_interval_s
  input_lock: asyncio.Lock   # serialize Playwright mutations
  stream_task
```

**Lifecycle**

1. Client opens WebSocket (auth → `session_id` assign/attach).
2. Server creates or attaches `BrowserRemoteSession`.
3. Start `BrowserManager` (persistent profile for LinkedIn).
4. Stream task runs until last subscriber disconnects + idle TTL.
5. Shutdown: cancel stream, `bm.stop()`, release profile lock.

**Concurrency**

- One Chromium per `user_data_dir` (profile lock).
- One writer to Playwright at a time (`input_lock`).
- Multiple viewers OK (fan-out); one controller (or first-come lease).

**Storage**

- Dev: `app.state.browser_sessions`
- Prod: prefer a dedicated browser worker process (Playwright is not
  multi-tenant friendly in API worker pools)

### Streaming modes (frame producers)

| Mode | Mechanism | Quality | Notes |
|------|-----------|---------|-------|
| Screenshot poll | `get_screenshot()` on interval | Visual, ~2–5 fps | Simplest; already on `BrowserManager` |
| HTML snapshots | `stream_html_snapshots` / `get_page_html` | Structure/text | Good for selectors; not visual fidelity |
| Hybrid | Screenshots + slow HTML of `#workspace` | Best default | Metadata (`url`, `title`) every tick |
| CDP screencast | `Page.startScreencast` via CDP session | Visual, push | See [CDP screencast](#cdp-screencast) |

None of these are true MutationObserver-style DOM push. HTML mode is still
periodic snapshots.

### Input relay (client → server)

| Client message | Server action | Exists on `BrowserManager`? |
|----------------|---------------|-----------------------------|
| `navigate` `{url}` | `bm.navigate(url)` | Yes |
| `click` `{x, y}` | `page.mouse.click(x, y)` | **No** — only `click(selector)` |
| `wheel` `{deltaY}` | mouse wheel or `bm.scroll()` | Partial |
| `keydown` / `press` | `bm.press_key(key)` | Partial |
| `type` `{text}` | `page.keyboard.type(text)` | **No** |
| `click_selector` | `bm.click(selector)` | Yes |
| `eval` `{code}` | `bm.execute_javascript(code)` | Yes — **gate heavily** |

Scale client display coordinates to server `viewport` (default 1280×720). Send
viewport in the first server `ready` message.

Serialize: client message → queue → `input_lock` → Playwright → optional
immediate frame.

### Suggested WebSocket protocol (v1)

**Client → server**

```json
{ "v": 1, "type": "subscribe", "session_id": "...", "mode": "hybrid", "token": "..." }
{ "v": 1, "type": "navigate", "url": "https://www.linkedin.com/feed/" }
{ "v": 1, "type": "pointer", "action": "click", "x": 640, "y": 360, "button": "left" }
{ "v": 1, "type": "wheel", "deltaX": 0, "deltaY": 120 }
{ "v": 1, "type": "keyboard", "action": "press", "key": "Enter" }
{ "v": 1, "type": "keyboard", "action": "type", "text": "hello" }
{ "v": 1, "type": "ping" }
```

**Server → client**

```json
{ "v": 1, "type": "ready", "session_id": "...", "viewport": {"width": 1280, "height": 720} }
{ "v": 1, "type": "frame", "ts": "...", "url": "...", "title": "...",
  "image": {"mime": "image/jpeg", "data_base64": "..." } }
{ "v": 1, "type": "html", "ts": "...", "selector": "#workspace", "html": "...", "truncated": true }
{ "v": 1, "type": "navigated", "url": "..." }
{ "v": 1, "type": "error", "code": "PROFILE_LOCKED", "message": "..." }
{ "v": 1, "type": "pong" }
```

Prefer **keep-latest** under backpressure (do not unbounded-queue frames).
Binary WebSocket frames for JPEG are optional; base64 JSON is easier to debug.

### Suggested module layout

```text
src/
  service/
    browser_session_manager.py
    browser_remote_session.py
  api/
    browser_ws.py
  utility/
    browser_manager.py   # add click_at, type_text, wheel; optional JPEG / screencast
```

Keep protocol parsing out of `BrowserManager`.

### Security (required)

Remote control equals access to a logged-in LinkedIn session.

1. Authenticate before/at `websocket.accept()` (JWT/session).
2. Authorize `session_id` ownership; no anonymous guessing.
3. Origin allowlist (CORS already lists `localhost:3000` in `main.py`).
4. Disable or tightly restrict `eval` in v1.
5. Optional URL allowlist (`linkedin.com` for LinkedIn use).
6. Rate-limit messages per connection.
7. Audit log navigate / click / session start-stop.
8. Production: `wss://` only; idle TTL closes unused browsers.

### Relation to feed pipeline

| Use case | Approach |
|----------|----------|
| Manual sign-in helper | WS viewer on server browser; replace/local UX of `scripts/linkedin_sign_in.py` |
| Watch feed collection | Needs shared long-lived session injected into collect path (refactor) |
| Parallel job + viewer | Blocked by profile lock — queue or exclude |

Treat interactive WS as a **separate service** from `LinkedInFeedPipeline`.

### Phased rollout

1. **View only** — WS + screenshot stream + url/title.
2. **Input** — navigate, click-at, scroll, keyboard; coordinate scaling; lock.
3. **Sign-in UX** — wire to `.browser_profile`.
4. **Optional** — CDP screencast producer, HTML channel, controller lease.

---

## `--remote-debugging-port` vs screenshots

### What the flag does

`--remote-debugging-port=9222` (or Playwright `connect_over_cdp`) exposes
Chrome’s **CDP HTTP/WebSocket endpoint** so tools can attach for debugging and
control. It is **not** a ready-made “stream this page to my web app” API.

This repo already treats exposed `:9222` as fragile when sharing a profile
(Crawl4AI managed-browser relaunch cycles).

### Does exposing the port help the WebSocket viewer?

Usually **no**.

```text
Desired:  Playwright ──frames──► your WS ──► client canvas

Port path: Chromium :9222 ◄── CDP ── (you'd still need a proxy) ──► client
```

- CDP is full browser control; exposing it is a security risk.
- The UI client does not speak CDP; you would proxy anyway.
- In-process Playwright already can open a CDP session **without** binding a
  public debugging port.
- Neither the port nor screenshots stream live DOM by themselves — both pixel
  (or separate HTML) paths still need your protocol.

**When the port *is* useful:** attach to a user’s local Chrome for
dev/sign-in; open DevTools against a server browser for debugging; rare
cross-process attach (avoid with `linkedin_profile_lock`).

**Recommendation:** keep Playwright as the sole browser owner; use in-process
CDP screencast or screenshots; **do not** expose `:9222` to clients.

---

## CDP screencast

### What it is

CDP command **`Page.startScreencast`** asks Chromium’s compositor to encode the
**visible viewport** and **push** `Page.screencastFrame` events (base64 JPEG/PNG
+ metadata). Each frame must be acknowledged with **`Page.screencastFrameAck`**
or Chrome stops sending (built-in backpressure).

Stop with `Page.stopScreencast`.

### Server-to-client streaming?

**Yes — as a middle layer.** Common remote-browser pattern:

```text
Chromium compositor
  → Page.screencastFrame (CDP, in-process)
  → Python (decode / drop stale / rate-limit)
  → WebSocket binary or base64 JSON
  → Client <canvas>
```

The **client never speaks CDP**. CDP stays inside the server process via
Playwright `context.new_cdp_session(page)`. Your WebSocket is the transport —
same client experience as screenshot polling, often with better timing.

Playwright has no high-level `page.start_screencast()`; use raw CDP:

```python
cdp = await page.context.new_cdp_session(page)
await cdp.send("Page.startScreencast", {
    "format": "jpeg",
    "quality": 75,
    "maxWidth": 1280,
    "maxHeight": 720,
    "everyNthFrame": 1,
})
# on Page.screencastFrame: broadcast → Page.screencastFrameAck
```

### Screencast vs `page.screenshot()` poll

| | CDP screencast | Polling `page.screenshot()` |
|--|----------------|-----------------------------|
| Trigger | Push when viewport updates | Timer pull |
| Idle page | Few/no frames | Captures every N ms |
| Busy page | Better keep-up; `everyNthFrame` | Fixed interval |
| Backpressure | Native frame ack | DIY keep-latest |
| API | Raw CDP lifecycle | Already on `BrowserManager` |
| Client payload | Still raster images | Still raster images |
| Needs `--remote-debugging-port`? | **No** | No |

Both are **pixel streams**. Neither replaces HTML snapshots for parsing.

### Pros

- Better “live view” semantics (push on change).
- Often lower CPU than blind polling when idle.
- Native backpressure via ack.
- Tunable: format, quality, max dimensions, every Nth frame.
- Metadata (device size, scroll offsets) helps click mapping.
- In-process — no exposed debugging port.
- Works headless (sign-in still needs headed once for profile warm-up).

### Cons

- Still heavy bandwidth (JPEG frames; LinkedIn is image-rich).
- Not DOM / not searchable; feed parse stays server-side.
- More lifecycle code (nav, crash, multi-subscriber fan-out, stop/restart).
- Semi-internal CDP API surface; worth integration tests across upgrades.
- JPEG softens small text; PNG is sharper but larger.
- Practical remote fps often 2–10, not 60.
- Shares compositor with Playwright actions — use `input_lock`.
- Same profile-lock / single-Chromium constraints.
- Slow ack stalls Chromium; prefer ack promptly + keep-latest to clients.

### Recommended frame policy

```text
One BrowserRemoteSession
  ├── one CDP screencast subscription
  ├── N WebSocket subscribers (fan-out)
  ├── input_lock for commands
  └── keep-latest: CDP → latest_frame; send to WS ≤ every 200 ms
```

Ack CDP quickly so the compositor does not stall; rate-limit what clients get.

### When to choose which producer

| Choose screencast when… | Choose screenshots when… |
|-------------------------|--------------------------|
| Live viewer during long sessions | Occasional snapshots / Phase 1 prototype |
| Page changes a lot (scroll, SPA) | Mostly static between actions |
| Care about latency/CPU at steady state | Minimal code first |
| Building the WS viewer properly | Debug / agent tool thumbnails |

Same WebSocket `type: "frame"` shape can switch producers later without a
client redesign.

### What screencast does not give you

- True real-time DOM (use HTML snapshots or a browser extension).
- Smaller-than-image payloads for rich SPAs.
- Client-side LinkedIn cookies (session stays in server profile).
- Safe unauthenticated exposure (still full visual access to a login).

---

## Practical defaults for this repo

```text
Phase 1:  page.screenshot JPEG @ 2–5 fps over authenticated WebSocket (view-only)
Phase 2:  Input (click-at, type, wheel) + coordinate scaling + profile lock
Phase 3:  Sign-in UX against .browser_profile
Phase 4:  Swap producer to CDP Page.startScreencast; optional HTML channel
Avoid:    Exposing --remote-debugging-port to clients or the public network
```

### Minimal `BrowserManager` extensions (when implementing)

- `click_at(x, y, button="left")`
- `type_text(text)`
- `wheel(delta_x, delta_y)`
- Optional: `screenshot_jpeg(quality=70)`
- Optional: screencast start/stop + frame callback helpers

### Operational notes

| Topic | Guidance |
|-------|----------|
| Headless | Streaming works headless; headed mainly for interactive sign-in |
| Memory | ~300–800 MB per Chromium; cap concurrent sessions |
| Deploy | Prefer browser worker separate from API workers |
| Health | Expose active session count |
| Reconnect | Resubscribe with same `session_id`; TTL covers brief disconnects |

---

## Summary

| Topic | Verdict |
|-------|---------|
| Today | Server-only Playwright; no client live view |
| WebSocket bridge | Good fit; add session manager + protocol + input path |
| `stream_html_snapshots` | Useful HTML producer; unused today |
| `remote-debugging-port` | Attach/debug only; not the viewer transport |
| CDP screencast | Better live **frame producer** than screenshot poll; still pixels over your WS |
| Feed pipeline | Keep batch; exclusive profile lock vs interactive session |

Related: [linkedin-feed-access.md](./linkedin-feed-access.md),
`src/utility/browser_manager.py`, `src/utility/linkedin_feed_pipeline.py`,
`scripts/linkedin_sign_in.py`.
