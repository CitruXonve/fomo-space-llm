"""Unit tests for ``linkedin_feed_parser``."""

import unittest
from pathlib import Path

from src.utility.linkedin_feed_parser import (
    activity_id_from_url,
    feed_update_url_from_urn,
    normalize_linkedin_url,
    parse_feed_posts,
    post_dedupe_id_from_url,
    post_urn_from_url,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestLinkedinFeedParser(unittest.TestCase):
    def test_normalize_linkedin_url(self) -> None:
        self.assertEqual(
            normalize_linkedin_url("//www.linkedin.com/in/foo/"),
            "https://www.linkedin.com/in/foo/",
        )
        self.assertEqual(
            normalize_linkedin_url("/feed/update/urn:li:activity:1"),
            "https://www.linkedin.com/feed/update/urn:li:activity:1",
        )

    def test_activity_id_from_url(self) -> None:
        self.assertEqual(
            activity_id_from_url(
                "https://www.linkedin.com/feed/update/urn:li:activity:7454467825173172224"
            ),
            "7454467825173172224",
        )
        self.assertEqual(
            activity_id_from_url(
                "https://www.linkedin.com/feed/update/urn:li:share:7468370974543888384/"
            ),
            "7468370974543888384",
        )
        self.assertEqual(
            activity_id_from_url(
                "https://www.linkedin.com/feed/update/urn:li:ugcPost:7468076248452210688/"
            ),
            "7468076248452210688",
        )

    def test_post_urn_and_dedupe(self) -> None:
        url = "https://www.linkedin.com/feed/update/urn:li:share:1234567890/"
        self.assertEqual(post_urn_from_url(url), "urn:li:share:1234567890")
        self.assertEqual(post_dedupe_id_from_url(url), "share:1234567890")
        self.assertEqual(
            feed_update_url_from_urn("urn:li:ugcPost:99"),
            "https://www.linkedin.com/feed/update/urn:li:ugcPost:99/",
        )

    def test_parse_bare_urn_in_html(self) -> None:
        html = (
            '<html><body>{"entity":"urn:li:share:5555555555555555555"}'
            '<a href="/feed/update/urn:li:activity:1">x</a></body></html>'
        )
        rows = parse_feed_posts(html)
        urls = {r["post_url"] for r in rows}
        self.assertTrue(
            any("urn:li:share:5555555555555555555" in u for u in urls),
            urls,
        )

    def test_parse_voyager_snippet(self) -> None:
        html = (_FIXTURES / "linkedin_feed_voyager_snippet.html").read_text(
            encoding="utf-8"
        )
        rows = parse_feed_posts(html)
        urls = {r["post_url"] for r in rows}
        self.assertTrue(
            any("7454467825173172224" in u for u in urls),
            f"expected activity id in rows, got {urls}",
        )
        snips = [r["text_snippet"] for r in rows if r["text_snippet"]]
        self.assertTrue(
            any("hiring" in s.lower() for s in snips),
            snips,
        )

    def test_parse_dom_cards(self) -> None:
        html = (_FIXTURES / "linkedin_feed_dom_cards.html").read_text(encoding="utf-8")
        rows = parse_feed_posts(html)
        self.assertGreaterEqual(len(rows), 1)
        first = next(r for r in rows if "9998887776665554443" in r["post_url"])
        self.assertIn("linkedin.com/in/", first["author_profile_url"])
        self.assertIn("backend", first["text_snippet"].lower())

    def test_parse_lazy_mount_cards_from_data_urn(self) -> None:
        html = (_FIXTURES / "linkedin_feed_lazy_mount_cards.html").read_text(
            encoding="utf-8"
        )
        rows = parse_feed_posts(html)
        urls = {r["post_url"] for r in rows}
        self.assertIn(
            "https://www.linkedin.com/feed/update/urn:li:activity:1111222233334444555/",
            urls,
        )
        self.assertIn(
            "https://www.linkedin.com/feed/update/urn:li:share:6666777788889999000/",
            urls,
        )
        activity_row = next(
            r for r in rows if "1111222233334444555" in r["post_url"]
        )
        self.assertIn("janedoe", activity_row["author_profile_url"])
        self.assertIn("hiring", activity_row["text_snippet"].lower())

    def test_parse_lazy_mount_author_only_stub(self) -> None:
        html = """
        <div data-lazy-mount-id="stub1">
          <a href="https://www.linkedin.com/in/bobengineer/">Bob Engineer</a>
          <div>We are hiring platform engineers.</div>
          <time datetime="2026-06-08T12:00:00Z">2h</time>
        </div>
        """
        rows = parse_feed_posts(html)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["post_url"], "")
        self.assertIn("bobengineer", rows[0]["author_profile_url"])
        self.assertIn("hiring", rows[0]["text_snippet"].lower())

    def test_parse_lazy_mount_without_post_href(self) -> None:
        html = """
        <div data-lazy-mount-id="x1">
          <a href="https://www.linkedin.com/in/alice/">Alice</a>
          <span data-urn="urn:li:ugcPost:1234567890123456789"></span>
          <div>Now hiring SREs.</div>
        </div>
        """
        rows = parse_feed_posts(html)
        self.assertEqual(len(rows), 1)
        self.assertIn("ugcPost:1234567890123456789", rows[0]["post_url"])
        self.assertIn("alice", rows[0]["author_profile_url"])
        self.assertIn("hiring", rows[0]["text_snippet"].lower())


if __name__ == "__main__":
    unittest.main()
