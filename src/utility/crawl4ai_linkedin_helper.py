"""Crawl4AI virtual-scroll LinkedIn feed collection (standalone, shared user_data_dir)."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, VirtualScrollConfig

from src.utility.linkedin_feed_config import FeedCollectionOptions, scroll_budget
from src.utility.linkedin_feed_parser import (
    DEFAULT_POST_CARD_SELECTOR,
    activity_id_from_url,
    parse_feed_posts,
)

logger = logging.getLogger(__name__)

LINKEDIN_FEED_URL = "https://www.linkedin.com/feed/"
DEFAULT_SORT_JS_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "linkedin-job-search"
    / "examples"
    / "linkedin_sort_recent.js"
)
PROFILE_LOCK_NAME = ".linkedin_feed_ab.lock"
_WAIT_AFTER_SORT_S = 2.0
_SCROLL_PAUSE_S = 0.75
_LAZY_MOUNT_SCROLL_PAUSE_S = 1.0
_LAZY_MOUNT_HTML_DELAY_S = 1.5


@dataclass
class CrawlFeedResult:
    """Outcome of a Crawl4AI virtual-scroll feed crawl."""

    posts: list[dict[str, str]]
    elapsed_seconds: float
    html_length: int
    scroll_count: int
    crawl4ai_success: bool
    error_message: str | None = None


def _dedupe_key(row: dict[str, str]) -> str:
    uid = activity_id_from_url(row.get("post_url", "") or "")
    if uid:
        return f"activity:{uid}"
    post = (row.get("post_url") or "").split("?", 1)[0]
    if post:
        return post[:512]
    author = row.get("author_profile_url") or ""
    return f"author:{author}" if author else ""


def dedupe_feed_posts(
    rows: list[dict[str, str]],
    *,
    max_posts: int,
) -> list[dict[str, str]]:
    """Dedupe parsed rows and cap to ``max_posts`` (same keys as feed-posts collector)."""
    target = max(1, int(max_posts))
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        k = _dedupe_key(row)
        if not k or k in seen:
            continue
        seen.add(k)
        merged.append(dict(row))
        if len(merged) >= target:
            break
    return merged


def build_browser_config(
    *,
    user_data_dir: str | Path,
    profile_directory: str | None = "Default",
    headless: bool = True,
    viewport: tuple[int, int] = (1280, 720),
) -> BrowserConfig:
    """Build Crawl4AI browser config sharing a Chromium user-data directory."""
    resolved = Path(user_data_dir).resolve()
    extra_args: list[str] | None = None
    if profile_directory:
        extra_args = [f"--profile-directory={profile_directory}"]
    return BrowserConfig(
        browser_type="chromium",
        headless=headless,
        # Playwright launch_persistent_context (same model as BrowserManager).
        # Avoid use_managed_browser subprocess + CDP :9222 relaunch cycle.
        use_persistent_context=True,
        use_managed_browser=False,
        user_data_dir=str(resolved),
        extra_args=extra_args,
        viewport_width=int(viewport[0]),
        viewport_height=int(viewport[1]),
    )


def build_virtual_scroll_config(
    *,
    scroll_rounds: int,
    max_posts: int,
    container_selector: str = "#workspace",
    post_card_selector: str | None = DEFAULT_POST_CARD_SELECTOR,
) -> VirtualScrollConfig:
    """Map CLI scroll budget to Crawl4AI virtual scroll settings."""
    scroll_count = scroll_budget(scroll_rounds=scroll_rounds, target=max_posts)
    if scroll_count <= 0:
        scroll_count = 1
    wait_after_scroll = (
        _LAZY_MOUNT_SCROLL_PAUSE_S if post_card_selector else _SCROLL_PAUSE_S
    )
    return VirtualScrollConfig(
        container_selector=container_selector,
        scroll_count=scroll_count,
        scroll_by="container_height",
        wait_after_scroll=wait_after_scroll,
    )


def _feed_ready_and_sort_wait_js(
    post_card_selector: str | None = DEFAULT_POST_CARD_SELECTOR,
) -> str:
    """Poll until LinkedIn feed hydrates, apply sort-once, then settle before virtual scroll."""
    ms = int(_WAIT_AFTER_SORT_S * 1000)
    lazy_probe = ""
    if post_card_selector:
        lazy_probe = f"""
  const lazyCards = document.querySelectorAll({post_card_selector!r});
  if (lazyCards.length > 0) window.__li_ab_lazy_seen = true;
"""
    return f"""() => {{
  const feed = document.querySelector("div[data-testid=mainFeed]")
    || document.querySelector("#workspace");
  const nav = document.querySelector("nav.global-nav, [data-test-global-nav]");
  if (!feed && !nav) return false;
{lazy_probe}
  if (!window.__li_ab_sorted) {{
    window.__li_ab_sorted = true;
    const sortBtn = document.querySelector(
      'button[aria-label*="Sort"], button[aria-label*="sort"], [data-test-sort-toggle]'
    );
    if (sortBtn) {{
      sortBtn.click();
      const recent = Array.from(document.querySelectorAll('button, [role="menuitem"], li'))
        .find((el) => /recent/i.test(el.textContent || ""));
      if (recent) recent.click();
    }}
    window.__li_ab_sort_at = Date.now();
    return false;
  }}
  if (!window.__li_ab_sort_at) window.__li_ab_sort_at = Date.now();
  return Date.now() - window.__li_ab_sort_at >= {ms};
}}"""


@asynccontextmanager
async def linkedin_profile_lock(user_data_dir: str | Path) -> AsyncIterator[None]:
    """Exclusive lock under ``user_data_dir`` so two Chromium stacks do not clash."""
    lock_path = Path(user_data_dir).resolve() / PROFILE_LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


async def crawl_linkedin_feed(options: FeedCollectionOptions) -> CrawlFeedResult:
    """Crawl the LinkedIn feed with Crawl4AI virtual scroll and parse posts."""
    profile = options.profile
    scroll = options.scroll
    selectors = options.selectors
    post_card_selector = selectors.post_card_selector

    start = time.perf_counter()
    scroll_count = scroll_budget(
        scroll_rounds=scroll.scroll_rounds, target=scroll.max_posts
    )
    if scroll_count <= 0:
        scroll_count = 1

    browser_config = build_browser_config(
        user_data_dir=profile.user_data_dir,
        profile_directory=profile.profile_directory,
        headless=profile.headless,
        viewport=profile.viewport,
    )
    virtual_config = build_virtual_scroll_config(
        scroll_rounds=scroll.scroll_rounds,
        max_posts=scroll.max_posts,
        container_selector=selectors.container_selector,
        post_card_selector=post_card_selector,
    )
    html_delay = _LAZY_MOUNT_HTML_DELAY_S if post_card_selector else 1.0
    run_config = CrawlerRunConfig(
        virtual_scroll_config=virtual_config,
        cache_mode=CacheMode.BYPASS,
        wait_for=f"js:{_feed_ready_and_sort_wait_js(post_card_selector)}",
        wait_for_timeout=60_000,
        wait_until="load",
        page_timeout=120_000,
        delay_before_return_html=html_delay,
    )

    posts: list[dict[str, str]] = []
    html_length = 0
    error_message: str | None = None
    success = False

    async with linkedin_profile_lock(profile.user_data_dir):
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=LINKEDIN_FEED_URL, config=run_config)
            if not result.success:
                error_message = getattr(
                    result, "error_message", None) or "crawl_failed"
                logger.warning("Crawl4AI feed crawl failed: %s", error_message)
            else:
                html = result.html or ""
                html_length = len(html)
                parsed = parse_feed_posts(
                    html,
                    post_card_selector=post_card_selector,
                )
                posts = dedupe_feed_posts(parsed, max_posts=scroll.max_posts)
                success = True
        except Exception as exc:
            error_message = str(exc)
            logger.exception("Crawl4AI feed crawl error")

    elapsed = time.perf_counter() - start
    return CrawlFeedResult(
        posts=posts,
        elapsed_seconds=elapsed,
        html_length=html_length,
        scroll_count=scroll_count,
        crawl4ai_success=success,
        error_message=error_message,
    )


__all__ = [
    "CrawlFeedResult",
    "PROFILE_LOCK_NAME",
    "LINKEDIN_FEED_URL",
    "build_browser_config",
    "build_virtual_scroll_config",
    "crawl_linkedin_feed",
    "dedupe_feed_posts",
    "linkedin_profile_lock",
]
