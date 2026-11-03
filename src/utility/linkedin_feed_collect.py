"""Collect LinkedIn feed posts from a ``BrowserManager`` page.

Backward-compatible facade. Implementation lives in:

- :mod:`linkedin_feed_collector` — :class:`LiveFeedCollector` (scroll/parse loop)
- :mod:`linkedin_feed_prepare` — page hydration and Recent sort
- :mod:`linkedin_feed_url_resolve` — Send → Copy link resolution
"""

from __future__ import annotations

import json
from typing import Any

from src.utility.linkedin_feed_collector import (
    LiveFeedCollector,
    collect_hiring_posts_from_page,
    row_dedupe_key,
)
from src.utility.linkedin_feed_config import (
    FeedCollectionOptions,
    FeedDomSelectors,
    FeedScrollPolicy,
    LinkedInBrowserProfile,
    scroll_budget,
)
from src.utility.linkedin_feed_prepare import (
    DEFAULT_SORT_JS_PATH,
    LINKEDIN_FEED_RECENT_URL,
    LINKEDIN_FEED_URL,
    SORT_SETTLE_MS,
    prepare_live_feed_page,
)
from src.utility.linkedin_feed_url_resolve import resolve_post_urls_via_send_button

# Backward-compat alias used by tests and crawl4ai helper imports.
_scroll_budget = scroll_budget


async def collect_raw_feed_posts_from_page(
    browser,
    *,
    max_posts: int = 5,
    scroll_rounds: int = 3,
    html_selector: str | None = None,
    post_card_selector: str | None = None,
    agent_max_posts: int | None = None,
    resolve_missing_urls: bool = True,
) -> list[dict[str, str]]:
    """Scroll the feed and return compact post dicts (delegates to LiveFeedCollector).

    Does not navigate or sort — callers (e.g. ``linkedin_browser_cli``) must
    already be on the feed page. Pipeline code uses ``LiveFeedCollector.run()``
    which includes ``prepare_live_feed_page``.
    """
    from src.utility.linkedin_feed_parser import DEFAULT_POST_CARD_SELECTOR

    if post_card_selector is None:
        post_card_selector = DEFAULT_POST_CARD_SELECTOR

    options = FeedCollectionOptions(
        profile=LinkedInBrowserProfile(user_data_dir=".browser_profile"),
        scroll=FeedScrollPolicy(
            max_posts=max_posts,
            scroll_rounds=scroll_rounds,
            agent_max_posts=agent_max_posts,
        ),
        selectors=FeedDomSelectors(
            post_card_selector=post_card_selector,
            html_selector=html_selector,
        ),
        resolve_missing_urls=resolve_missing_urls,
    )
    result = await LiveFeedCollector(options)._collect_posts(browser)
    posts, _strategy, _scroll_count, _urls_resolved = result
    return posts


async def collect_raw_feed_posts_json(
    browser,
    **kwargs: Any,
) -> str:
    """Return JSON string of feed posts (``ensure_ascii=False``)."""
    rows = await collect_raw_feed_posts_from_page(browser, **kwargs)
    return json.dumps(rows, ensure_ascii=False)


__all__ = [
    "DEFAULT_SORT_JS_PATH",
    "LINKEDIN_FEED_RECENT_URL",
    "LINKEDIN_FEED_URL",
    "SORT_SETTLE_MS",
    "_scroll_budget",
    "collect_hiring_posts_from_page",
    "collect_raw_feed_posts_from_page",
    "collect_raw_feed_posts_json",
    "prepare_live_feed_page",
    "resolve_post_urls_via_send_button",
    "row_dedupe_key",
    "scroll_budget",
]
