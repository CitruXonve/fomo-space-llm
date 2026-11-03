"""Tests for LinkedIn authentication helpers."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.utility.linkedin_auth import is_linkedin_authenticated


class TestLinkedinAuth(unittest.IsolatedAsyncioTestCase):
    async def test_false_on_login_url(self) -> None:
        bm = MagicMock()
        bm.get_current_url = AsyncMock(return_value="https://www.linkedin.com/login")
        bm.execute_javascript = AsyncMock()
        self.assertFalse(await is_linkedin_authenticated(bm))
        bm.execute_javascript.assert_not_awaited()

    async def test_true_on_logged_in_dom_hints(self) -> None:
        bm = MagicMock()
        bm.get_current_url = AsyncMock(return_value="https://www.linkedin.com/feed/")
        bm.execute_javascript = AsyncMock(
            return_value={
                "hasGlobalNav": True,
                "hasSearchInput": False,
                "hasNetworkLink": False,
                "hasSignInForm": False,
                "hasJoinLink": False,
            }
        )
        self.assertTrue(await is_linkedin_authenticated(bm))


if __name__ == "__main__":
    unittest.main()
