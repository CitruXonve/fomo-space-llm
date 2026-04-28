"""Unit tests for ``linkedin_feed_parser``."""

import unittest
from pathlib import Path

from src.utility.linkedin_feed_parser import (
    activity_id_from_url,
    normalize_linkedin_url,
    parse_feed_posts,
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


if __name__ == "__main__":
    unittest.main()
