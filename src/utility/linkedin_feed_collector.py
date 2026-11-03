"""Live-browser feed collection: prepare → JS extract → HTML scroll loop → resolve.

:class:`LiveFeedCollector` owns the scroll/parse loop that was previously hidden
inside ``collect_raw_feed_posts_from_page``. One session phase::

    prepare_live_feed_page  →  try JS IIFE  →  fallback HTML while-loop  →  resolve stubs
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.utility.linkedin_feed_config import (
    CollectionStrategy,
    FeedCollectionOptions,
    FeedCollectionResult,
    scroll_budget,
)
from src.utility.linkedin_feed_parser import parse_feed_posts, post_dedupe_id_from_url
from src.utility.linkedin_feed_prepare import prepare_live_feed_page
from src.utility.linkedin_feed_scroll_js import format_feed_posts_js, format_hiring_js
from src.utility.linkedin_feed_url_resolve import (
    install_clipboard_spy,
    resolve_author_only_rows,
    resolve_one_post_url,
)

if TYPE_CHECKING:
    from src.utility.browser_manager import BrowserManager

logger = logging.getLogger(__name__)


def row_dedupe_key(row: dict[str, str]) -> str:
    post_url = row.get("post_url") or ""
    urn_id = post_dedupe_id_from_url(post_url)
    if urn_id:
        return f"urn:{urn_id}"
    post = post_url.split("?", 1)[0]
    if post:
        return post[:512]
    author = row.get("author_profile_url") or ""
    return f"author:{author}" if author else ""


class LiveFeedCollector:
    """Scroll the live feed until target posts, plateau, or scroll budget."""

    def __init__(self, options: FeedCollectionOptions) -> None:
        self._options = options

    async def run(self, browser: BrowserManager) -> FeedCollectionResult:
        """Prepare page, collect posts, optionally resolve author-only stubs."""
        sort_meta = await prepare_live_feed_page(
            browser,
            sort_js_path=self._options.sort_js_path,
            post_card_selector=self._options.selectors.post_card_selector,
        )
        posts, strategy, scroll_count, urls_resolved = await self._collect_posts(browser)
        return FeedCollectionResult(
            posts=posts,
            sort_meta=sort_meta,
            collection_strategy=strategy,
            scroll_count=scroll_count,
            urls_resolved_count=urls_resolved,
        )

    async def _collect_posts(
        self,
        browser: BrowserManager,
    ) -> tuple[list[dict[str, str]], CollectionStrategy, int, int]:
        scroll = self._options.scroll
        selectors = self._options.selectors
        target = scroll.effective_target()
        urls_resolved = 0

        js_rows = await self._collect_via_js(browser, target=target)
        if js_rows:
            strategy: CollectionStrategy = "js"
            if self._options.resolve_missing_urls:
                await browser.wait_for_page_ready()
                await install_clipboard_spy(browser)
                urls_resolved = await resolve_author_only_rows(browser, js_rows)
                if urls_resolved:
                    strategy = "js+resolve"
            return js_rows[:target], strategy, 0, urls_resolved

        posts, scroll_count, urls_resolved = await self._collect_via_html_scroll(
            browser, target=target
        )
        return posts, "html_scroll", scroll_count, urls_resolved

    async def _collect_via_js(
        self,
        browser: BrowserManager,
        *,
        target: int,
    ) -> list[dict[str, str]] | None:
        """In-browser feed extraction (scroll + URN/href). Returns None when feed missing."""
        scroll = self._options.scroll
        max_scrolls = scroll_budget(scroll_rounds=scroll.scroll_rounds, target=target)
        if max_scrolls <= 0:
            max_scrolls = 1
        js = format_feed_posts_js(
            min_posts=max(1, target),
            max_scrolls=max_scrolls,
            scroll_px=max(1, scroll.scroll_px),
            wait_ms=max(100, int(scroll.scroll_pause_s * 1000)),
            plateau_limit=max(1, scroll.plateau_rounds),
        )
        result = await browser.execute_javascript(js)
        if isinstance(result, dict) and result.get("error"):
            logger.warning("collect_raw_feed_posts: JS error: %s", result["error"])
            return None
        if not isinstance(result, list):
            logger.warning(
                "collect_raw_feed_posts: unexpected JS result type: %s", type(result)
            )
            return None
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in result:
            if not isinstance(item, dict):
                continue
            row = {
                "post_url": str(item.get("post_url") or ""),
                "author_profile_url": str(item.get("author_profile_url") or ""),
                "text_snippet": str(item.get("text_snippet") or ""),
                "relative_time": str(item.get("relative_time") or ""),
            }
            if not row["post_url"] and not row["author_profile_url"]:
                continue
            key = row_dedupe_key(row)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(row)
        return rows[:target]

    async def _collect_via_html_scroll(
        self,
        browser: BrowserManager,
        *,
        target: int,
    ) -> tuple[list[dict[str, str]], int, int]:
        """Parse HTML snapshots in a scroll loop until target, plateau, or budget."""
        scroll = self._options.scroll
        selectors = self._options.selectors
        max_scrolls = scroll_budget(scroll_rounds=scroll.scroll_rounds, target=target)

        page = browser._ensure_started()
        if self._options.resolve_missing_urls:
            await browser.wait_for_page_ready()
            await install_clipboard_spy(browser)

        merged: list[dict[str, str]] = []
        seen: set[str] = set()
        plateau = 0
        scroll_count = 0
        urls_resolved = 0

        while True:
            html = await browser.get_page_html(selectors.html_selector)
            if not html:
                logger.warning("collect_raw_feed_posts: empty HTML")
                break

            prev_len = len(merged)
            batch = parse_feed_posts(
                html, post_card_selector=selectors.post_card_selector
            )
            for row in batch:
                k = row_dedupe_key(row)
                if not k or k in seen:
                    continue
                seen.add(k)
                entry = dict(row)
                if (
                    self._options.resolve_missing_urls
                    and not entry.get("post_url")
                    and entry.get("author_profile_url")
                ):
                    resolved = await resolve_one_post_url(
                        browser, entry["author_profile_url"]
                    )
                    if resolved:
                        entry["post_url"] = resolved
                        urls_resolved += 1
                        seen.discard(k)
                        new_k = row_dedupe_key(entry)
                        if new_k and new_k != k:
                            if new_k in seen:
                                continue
                            seen.add(new_k)
                        else:
                            seen.add(k)
                merged.append(entry)
                if len(merged) >= target:
                    break

            if len(merged) >= target:
                break

            if len(merged) == prev_len:
                plateau += 1
            else:
                plateau = 0

            if plateau >= scroll.plateau_rounds:
                logger.info(
                    "collect_raw_feed_posts: feed plateau after %d scrolls (%d/%d posts)",
                    scroll_count,
                    len(merged),
                    target,
                )
                break

            if scroll_count >= max_scrolls:
                logger.info(
                    "collect_raw_feed_posts: scroll budget exhausted (%d/%d posts)",
                    len(merged),
                    target,
                )
                break

            await browser.scroll(scroll.scroll_px)
            scroll_count += 1
            await asyncio.sleep(scroll.scroll_pause_s)

        return merged[:target], scroll_count, urls_resolved


async def collect_hiring_posts_from_page(
    browser: BrowserManager,
    *,
    max_posts: int = 10,
    scroll_px: int = 800,
    wait_ms: int = 2000,
    max_scrolls: int = 25,
    plateau_limit: int = 7,
) -> list[dict]:
    """Scroll the LinkedIn feed and return hiring-announcement posts."""
    js = format_hiring_js(
        min_posts=max(1, int(max_posts)),
        max_scrolls=max(1, int(max_scrolls)),
        scroll_px=max(1, int(scroll_px)),
        wait_ms=max(100, int(wait_ms)),
        plateau_limit=max(1, int(plateau_limit)),
    )
    result = await browser.execute_javascript(js)
    if isinstance(result, dict) and result.get("error"):
        logger.warning("collect_hiring_posts: JS error: %s", result["error"])
        return []
    if not isinstance(result, list):
        logger.warning(
            "collect_hiring_posts: unexpected JS result type: %s", type(result)
        )
        return []
    return result
