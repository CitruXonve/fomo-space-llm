"""Unit tests for ``linkedin_browser_cli`` (no Playwright launch)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.utility.linkedin_browser_cli import (
    build_linkedin_parser,
    execute_combined_step,
    execute_linkedin_step,
)


class TestLinkedInParser(unittest.TestCase):
    def test_feed_posts_flags(self) -> None:
        parser = build_linkedin_parser()
        ns = parser.parse_args(
            [
                "--no-headless",
                "--user-data-dir",
                ".browser_profile",
                "feed-posts",
                "--max-posts",
                "10",
                "--scroll-rounds",
                "2",
            ]
        )
        self.assertEqual(ns.command, "feed-posts")
        self.assertEqual(ns.max_posts, 10)
        self.assertEqual(ns.scroll_rounds, 2)
        self.assertTrue(ns.no_headless)

    def test_check_auth_command(self) -> None:
        parser = build_linkedin_parser()
        ns = parser.parse_args(["check-auth"])
        self.assertEqual(ns.command, "check-auth")

    def test_sign_in_timeout_arg(self) -> None:
        parser = build_linkedin_parser()
        ns = parser.parse_args(["sign-in", "--timeout-s", "45"])
        self.assertEqual(ns.command, "sign-in")
        self.assertEqual(ns.timeout_s, 45)


class TestLinkedInExecuteStep(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_linkedin_op(self) -> None:
        bm = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            await execute_linkedin_step(bm, {"op": "nope"})
        self.assertIn("unknown op", str(ctx.exception))

    async def test_combined_rejects_unknown(self) -> None:
        bm = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            await execute_combined_step(bm, {"op": "nope"})
        self.assertIn("unknown op", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
