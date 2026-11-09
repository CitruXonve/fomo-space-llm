"""Unit tests for the Firecrawl parse arm (no network, no browser)."""

import ast
import inspect
import unittest
from types import SimpleNamespace

from src.utility.firecrawl_linkedin_helper import (
    PARSE_FORMATS,
    map_parse_to_posts,
    parse_options,
    parse_redacted_feed,
)
from src.utility.linkedin_feed_redact import redact_feed_html

_FEED_HTML = """
<html>
  <script>li_at=secretvalue; JSESSIONID=abc123</script>
  <style>.hidden { color: red; }</style>
  <!-- csrf=supersecret -->
  <div id="workspace" nonce="tok" integrity="sha256-abc">
    <a href="https://www.linkedin.com/in/janedoe/">Jane Doe</a>
    <a href="https://www.linkedin.com/feed/update/urn:li:activity:111/?trk=feed">post</a>
    <a href="https://www.linkedin.com/posts/jane_activity-222">slug</a>
    <a href="/in/bobsmith/">Bob Smith</a>
    <a href="https://www.linkedin.com/messaging/">msg</a>
    <p>Email me at jane@example.com about ajax:1234567890</p>
    <iframe src="https://evil.example/track"></iframe>
  </div>
</html>
"""


class TestRedactFeedHtml(unittest.TestCase):
    def test_strips_secrets_and_keeps_feed_hrefs(self) -> None:
        redacted = redact_feed_html(_FEED_HTML)

        self.assertNotIn("<script", redacted)
        self.assertNotIn("<style", redacted)
        self.assertNotIn("<iframe", redacted)
        self.assertNotIn("csrf=supersecret", redacted)
        self.assertNotIn("nonce", redacted)
        self.assertNotIn("integrity", redacted)
        self.assertNotIn("li_at", redacted)
        self.assertNotIn("JSESSIONID", redacted)
        self.assertNotIn("secretvalue", redacted)
        self.assertNotIn("jane@example.com", redacted)
        self.assertNotIn("ajax:", redacted)
        self.assertNotIn("/messaging/", redacted)

        self.assertIn("Jane Doe", redacted)
        self.assertIn("Bob Smith", redacted)
        self.assertIn("https://www.linkedin.com/in/janedoe/", redacted)
        self.assertIn("/in/bobsmith/", redacted)
        self.assertIn("/feed/update/urn:li:activity:111/", redacted)
        self.assertIn("/posts/jane_activity-222", redacted)
        self.assertNotIn("trk=feed", redacted)

    def test_empty_html(self) -> None:
        self.assertEqual(redact_feed_html("   "), "")


class TestMapParseToPosts(unittest.TestCase):
    def test_pairs_author_snippet_and_relative_time(self) -> None:
        author = "https://www.linkedin.com/in/janedoe/"
        post = "https://www.linkedin.com/feed/update/urn:li:activity:111/"
        markdown = (
            f"[Jane Doe]({author})\n"
            "3h\n"
            "We are hiring backend engineers.\n"
            f"[post]({post})\n"
        )
        rows = map_parse_to_posts(markdown, [author, post], max_posts=10)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["post_url"], post)
        self.assertEqual(rows[0]["author_profile_url"], author)
        self.assertIn("hiring backend engineers", rows[0]["text_snippet"])
        self.assertEqual(rows[0]["relative_time"], "3h")

    def test_uses_links_when_markdown_has_no_urls(self) -> None:
        author = "https://www.linkedin.com/in/janedoe/"
        post = "https://www.linkedin.com/posts/jane_activity-222"
        rows = map_parse_to_posts(
            "Jane Doe\n3h\nOpen role on our platform team.",
            [{"url": author}, {"href": post}],
            max_posts=5,
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("/posts/jane_activity-222", rows[0]["post_url"])
        self.assertEqual(rows[0]["author_profile_url"], author)
        self.assertEqual(rows[0]["text_snippet"], "")

    def test_caps_and_dedupes_posts(self) -> None:
        first = "https://www.linkedin.com/feed/update/urn:li:activity:1/"
        second = "https://www.linkedin.com/feed/update/urn:li:activity:2/"
        rows = map_parse_to_posts("", [first, first, second], max_posts=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["post_url"], first)


class _FakeFirecrawl:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, dict]] = []

    def parse(self, payload: bytes, **kwargs: object) -> SimpleNamespace:
        self.calls.append((payload, kwargs))
        return SimpleNamespace(
            markdown=(
                "[Jane Doe](https://www.linkedin.com/in/janedoe/)\n"
                "3h\n"
                "We are hiring backend engineers.\n"
                "https://www.linkedin.com/feed/update/urn:li:activity:111/\n"
            ),
            links=[
                "https://www.linkedin.com/in/janedoe/",
                "https://www.linkedin.com/feed/update/urn:li:activity:111/",
            ],
        )


class TestParseRedactedFeed(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_redacted_html_with_markdown_and_links(self) -> None:
        client = _FakeFirecrawl()
        posts, nbytes = await parse_redacted_feed(
            _FEED_HTML,
            client=client,
            max_posts=10,
        )

        self.assertEqual(len(client.calls), 1)
        payload, kwargs = client.calls[0]
        self.assertNotIn(b"li_at", payload)
        self.assertNotIn(b"jane@example.com", payload)
        self.assertIn(b"Jane Doe", payload)
        self.assertEqual(kwargs["filename"], "feed.html")
        self.assertEqual(kwargs["content_type"], "text/html")
        self.assertEqual(list(kwargs["options"].formats), list(PARSE_FORMATS))
        self.assertIs(kwargs["options"].only_main_content, False)
        self.assertNotIn("zeroDataRetention", kwargs)
        self.assertNotIn("profile", kwargs)
        self.assertGreater(nbytes, 0)
        self.assertEqual(len(posts), 1)
        self.assertIn("activity:111", posts[0]["post_url"])
        self.assertIn("janedoe", posts[0]["author_profile_url"])

    def test_parse_options_are_not_llm_formats(self) -> None:
        options = parse_options()
        self.assertEqual(list(options.formats), ["markdown", "links"])
        self.assertIs(options.only_main_content, False)

    def test_modules_do_not_import_crawl4ai_pipeline_or_interact(self) -> None:
        import src.utility.firecrawl_linkedin_helper as helper
        import src.utility.linkedin_feed_firecrawl_cli as cli
        import src.utility.linkedin_feed_redact as redact

        forbidden = ("crawl4ai", "linkedin_feed_pipeline", "interact")
        for module in (helper, cli, redact):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    for blocked in forbidden:
                        self.assertNotIn(blocked, name)


if __name__ == "__main__":
    unittest.main()
