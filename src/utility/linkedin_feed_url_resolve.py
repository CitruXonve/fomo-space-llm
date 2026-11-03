"""Send → Copy link URL resolution for author-only feed stubs."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.utility.browser_manager import BrowserManager

logger = logging.getLogger(__name__)

CLIPBOARD_SPY_JS = """
if (!window.__clipboardSpyInstalled) {
    window.__clipboardLastUrl = null;
    window.__resolvedCardIds = new Set();
    document.addEventListener('copy', (e) => {
        const txt = e.clipboardData && e.clipboardData.getData('text/plain');
        if (txt) window.__clipboardLastUrl = txt;
    }, true);
    if (navigator.clipboard && navigator.clipboard.writeText) {
        const _orig = navigator.clipboard.writeText.bind(navigator.clipboard);
        navigator.clipboard.writeText = async (t) => { window.__clipboardLastUrl = t; return _orig(t).catch(() => {}); };
    }
    window.__clipboardSpyInstalled = true;
}
"""

FIND_SEND_BTN_JS = """
(authorHref) => {
    const feed = document.querySelector("div[data-testid=mainFeed]") || document;
    const anchors = [...feed.querySelectorAll("a[href]")];
    const match = anchors.find(a => a.href && (a.href === authorHref || a.href.startsWith(authorHref)));
    if (!match) return false;
    const card = match.closest("div[data-lazy-mount-id]");
    if (card) {
        const cardId = card.getAttribute("data-lazy-mount-id");
        if (cardId && window.__resolvedCardIds && window.__resolvedCardIds.has(cardId)) return false;
    }
    let node = match.parentElement;
    for (let i = 0; i < 25; i++) {
        if (!node || node === document.body) break;
        const elems = [...node.querySelectorAll("button, a")];
        const sendBtn = elems.find(b => {
            const label = (b.getAttribute("aria-label") || "").toLowerCase();
            const text = (b.innerText || b.textContent || "").trim().toLowerCase();
            return /\\bsend\\b/.test(label) || text === "send";
        });
        if (sendBtn) {
            const firstInLink = node.querySelector("a[href*='/in/']");
            if (!firstInLink || !firstInLink.href.startsWith(authorHref)) return false;
            window.__lastClickedCardId = (card && card.getAttribute("data-lazy-mount-id")) || null;
            sendBtn.click();
            return true;
        }
        node = node.parentElement;
    }
    return false;
}
"""

MARK_CARD_RESOLVED_JS = (
    "(cardId) => { if (window.__resolvedCardIds && cardId) window.__resolvedCardIds.add(cardId); }"
)

RESOLVE_SCROLL_STEP = 600
RESOLVE_SCROLL_ATTEMPTS = 8


async def install_clipboard_spy(browser: BrowserManager) -> None:
    await browser.safe_evaluate(CLIPBOARD_SPY_JS)


async def resolve_one_post_url(
    browser: BrowserManager,
    author_url: str,
    *,
    menu_timeout_ms: int = 3000,
    pause_s: float = 0.5,
) -> str | None:
    """Click Send → Copy link to post for one post; return the URL or None."""
    await browser.safe_evaluate("window.__clipboardLastUrl = null")
    clicked = await browser.safe_evaluate(FIND_SEND_BTN_JS, author_url)
    if not clicked:
        return None
    page = browser._ensure_started()
    try:
        await page.wait_for_selector("text=Copy link to post", timeout=menu_timeout_ms)
        await page.locator("text=Copy link to post").first.click(timeout=2000)
    except Exception:
        await page.keyboard.press("Escape")
        return None
    await asyncio.sleep(pause_s)
    url = await browser.safe_evaluate("window.__clipboardLastUrl")
    await page.keyboard.press("Escape")
    await asyncio.sleep(0.2)
    if url and isinstance(url, str) and "linkedin.com" in url:
        card_id = await browser.safe_evaluate("() => window.__lastClickedCardId || null")
        if card_id:
            await browser.safe_evaluate(MARK_CARD_RESOLVED_JS, card_id)
        return url
    return None


async def resolve_author_only_rows(
    browser: BrowserManager,
    rows: list[dict[str, str]],
) -> int:
    """Fill missing ``post_url`` via Send → Copy link; return count resolved."""
    resolved = 0
    for entry in rows:
        if entry.get("post_url") or not entry.get("author_profile_url"):
            continue
        url = await resolve_one_post_url(browser, entry["author_profile_url"])
        if url:
            entry["post_url"] = url
            resolved += 1
    return resolved


async def resolve_post_urls_via_send_button(
    browser: BrowserManager,
    posts: list[dict[str, Any]],
    *,
    menu_timeout_ms: int = 3000,
    pause_s: float = 0.5,
    scroll_step: int = RESOLVE_SCROLL_STEP,
    max_attempt_per_post: int = RESOLVE_SCROLL_ATTEMPTS,
) -> list[dict[str, Any]]:
    """Fill in null ``post_url`` entries by clicking Send → Copy link to post."""
    await browser.wait_for_page_ready()
    await install_clipboard_spy(browser)

    await browser.safe_evaluate(
        "(document.querySelector('#workspace') || document.documentElement).scrollTop = 0"
    )
    await asyncio.sleep(1.0)

    resolved_urls: set[str] = set()
    for post in posts:
        if post.get("post_url"):
            resolved_urls.add(post["post_url"])
            continue
        author_url = post.get("author_profile_url")
        if not author_url:
            continue

        url: str | None = None
        for _ in range(max_attempt_per_post):
            candidate = await resolve_one_post_url(
                browser, author_url, menu_timeout_ms=menu_timeout_ms, pause_s=pause_s
            )
            if candidate and candidate not in resolved_urls:
                url = candidate
                break
            await browser.scroll(scroll_step)
            await asyncio.sleep(0.5)

        if url:
            post["post_url"] = url
            resolved_urls.add(url)

    return posts
