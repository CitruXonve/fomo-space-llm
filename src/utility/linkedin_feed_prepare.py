"""Prepare a live LinkedIn feed page (navigate, sort Recent, hydrate cards)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utility.linkedin_feed_parser import DEFAULT_POST_CARD_SELECTOR

if TYPE_CHECKING:
    from src.utility.browser_manager import BrowserManager

logger = logging.getLogger(__name__)

LINKEDIN_FEED_URL = "https://www.linkedin.com/feed/"
LINKEDIN_FEED_RECENT_URL = "https://www.linkedin.com/feed/?feedType=recent"
DEFAULT_SORT_JS_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "linkedin-job-search"
    / "examples"
    / "linkedin_sort_recent.js"
)
SORT_SETTLE_MS = 2000


async def prepare_live_feed_page(
    browser: BrowserManager,
    *,
    sort_js_path: str | Path | None = None,
    post_card_selector: str | None = DEFAULT_POST_CARD_SELECTOR,
) -> dict[str, Any]:
    """Navigate, wait for hydration, sort Recent, settle, then wait for feed cards.

    Shared by :class:`LinkedInFeedPipeline` and the A/B control arm so production
    and experiments use the same ordering + lazy-mount hydration sequence.
    Falls back to ``?feedType=recent`` when the sort-menu JS cannot click Recent.
    """
    sort_applied = False
    sort_method = "none"
    sort_path = Path(sort_js_path or DEFAULT_SORT_JS_PATH)

    if not sort_path.is_file():
        logger.info("Sort JS missing; navigating directly to Recent feed URL")
        await browser.navigate(LINKEDIN_FEED_RECENT_URL)
        sort_method = "feedType=recent"
        sort_applied = True
    else:
        await browser.navigate(LINKEDIN_FEED_URL)
        try:
            await browser.wait_for(
                selector="div[data-testid=mainFeed], div[data-lazy-mount-id]",
                timeout_ms=60_000,
            )
        except Exception as exc:
            logger.warning("Live feed container wait failed: %s", exc)

        result = await browser.execute_javascript(sort_path.read_text(encoding="utf-8"))
        if isinstance(result, dict) and result.get("sorted"):
            sort_applied = True
            sort_method = str(result.get("method") or "menu")

        if not sort_applied:
            logger.info("Sort menu failed; navigating to Recent feed URL fallback")
            await browser.navigate(LINKEDIN_FEED_RECENT_URL)
            sort_method = "feedType=recent"
            sort_applied = True

    try:
        await browser.wait_for(
            selector="div[data-testid=mainFeed], div[data-lazy-mount-id]",
            timeout_ms=60_000,
        )
    except Exception as exc:
        logger.warning("Live feed container wait failed: %s", exc)

    await browser.wait_for(timeout_ms=SORT_SETTLE_MS)

    if post_card_selector:
        try:
            await browser.wait_for(selector=post_card_selector, timeout_ms=15_000)
        except Exception:
            pass

    return {"sort_applied": sort_applied, "sort_method": sort_method}
