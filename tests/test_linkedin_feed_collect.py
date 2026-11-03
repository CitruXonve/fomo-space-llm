"""Unit tests for ``linkedin_feed_collect`` (mocked browser, HTML fixtures)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.utility.linkedin_feed_collect import (
    _scroll_budget,
    collect_raw_feed_posts_from_page,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestCollectRawFeedPostsFromPage(unittest.IsolatedAsyncioTestCase):
    def _browser_with_html_fallback(self, html: str) -> MagicMock:
        browser = MagicMock()
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.url = "https://www.linkedin.com/feed/"
        browser._ensure_started = MagicMock(return_value=page)
        browser.get_page_html = AsyncMock(return_value=html)
        browser.scroll = AsyncMock()
        browser.wait_for_page_ready = AsyncMock()
        browser.safe_evaluate = AsyncMock(return_value=None)
        browser.execute_javascript = AsyncMock(return_value={"error": "no feed"})
        return browser

    async def test_parses_fixture_without_scroll(self) -> None:
        html = (_FIXTURES / "linkedin_feed_dom_cards.html").read_text(encoding="utf-8")
        browser = self._browser_with_html_fallback(html)

        rows = await collect_raw_feed_posts_from_page(
            browser,
            max_posts=5,
            scroll_rounds=0,
        )

        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("linkedin.com", rows[0]["post_url"])
        browser.scroll.assert_not_called()

    async def test_dedupes_across_scroll_rounds(self) -> None:
        html = (_FIXTURES / "linkedin_feed_dom_cards.html").read_text(encoding="utf-8")
        browser = self._browser_with_html_fallback(html)

        rows = await collect_raw_feed_posts_from_page(
            browser,
            max_posts=10,
            scroll_rounds=2,
        )

        urls = [r["post_url"] for r in rows]
        self.assertEqual(len(urls), len(set(urls)))
        self.assertGreaterEqual(browser.scroll.await_count, 1)
        self.assertLessEqual(browser.scroll.await_count, 5)

    async def test_scrolls_until_max_posts(self) -> None:
        html = (_FIXTURES / "linkedin_feed_dom_cards.html").read_text(encoding="utf-8")
        browser = self._browser_with_html_fallback(html)

        rows = await collect_raw_feed_posts_from_page(
            browser,
            max_posts=1,
            scroll_rounds=10,
        )

        self.assertEqual(len(rows), 1)
        browser.scroll.assert_not_called()

    async def test_stops_on_plateau_when_target_not_met(self) -> None:
        html = (_FIXTURES / "linkedin_feed_dom_cards.html").read_text(encoding="utf-8")
        browser = self._browser_with_html_fallback(html)

        rows = await collect_raw_feed_posts_from_page(
            browser,
            max_posts=50,
            scroll_rounds=10,
        )

        self.assertLess(len(rows), 50)
        self.assertEqual(browser.scroll.await_count, 5)

    async def test_scrolls_until_target_with_growing_html(self) -> None:
        browser = MagicMock()
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        browser._ensure_started = MagicMock(return_value=page)
        browser.get_page_html = AsyncMock(return_value="<html></html>")
        browser.scroll = AsyncMock()
        browser.wait_for_page_ready = AsyncMock()
        browser.safe_evaluate = AsyncMock(return_value=None)
        browser.execute_javascript = AsyncMock(return_value={"error": "no feed"})

        def _row(n: int) -> dict[str, str]:
            return {
                "post_url": f"https://www.linkedin.com/feed/update/urn:li:activity:{n}",
                "author_profile_url": "",
                "text_snippet": f"post {n}",
                "relative_time": "",
            }

        with patch(
            "src.utility.linkedin_feed_collector.parse_feed_posts",
            side_effect=[[_row(1)], [_row(2)], [_row(3)]],
        ):
            rows = await collect_raw_feed_posts_from_page(
                browser,
                max_posts=3,
                scroll_rounds=10,
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual(browser.scroll.await_count, 2)

    async def test_js_path_returns_posts_with_urn(self) -> None:
        browser = MagicMock()
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.url = "https://www.linkedin.com/feed/"
        browser._ensure_started = MagicMock(return_value=page)
        browser.wait_for_page_ready = AsyncMock()
        browser.safe_evaluate = AsyncMock(return_value=None)
        browser.execute_javascript = AsyncMock(
            return_value=[
                {
                    "post_url": "https://www.linkedin.com/feed/update/urn:li:share:111/",
                    "author_profile_url": "https://www.linkedin.com/in/alice/",
                    "text_snippet": "Hello",
                    "relative_time": "1d",
                },
                {
                    "post_url": "",
                    "author_profile_url": "https://www.linkedin.com/in/bob/",
                    "text_snippet": "skip me",
                    "relative_time": "",
                },
            ]
        )
        rows = await collect_raw_feed_posts_from_page(
            browser,
            max_posts=5,
            scroll_rounds=3,
        )
        self.assertEqual(len(rows), 2)
        with_url = [r for r in rows if r.get("post_url")]
        self.assertEqual(len(with_url), 1)
        self.assertIn("urn:li:share:111", with_url[0]["post_url"])
        browser.get_page_html.assert_not_called()


class TestScrollBudget(unittest.TestCase):
    def test_zero_means_no_scroll(self) -> None:
        self.assertEqual(_scroll_budget(scroll_rounds=0, target=5), 0)

    def test_at_least_target_when_scroll_enabled(self) -> None:
        self.assertEqual(_scroll_budget(scroll_rounds=3, target=5), 5)
        self.assertEqual(_scroll_budget(scroll_rounds=10, target=5), 10)


if __name__ == "__main__":
    unittest.main()
