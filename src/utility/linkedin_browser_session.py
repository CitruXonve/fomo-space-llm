"""Single Chromium session for LinkedIn feed pipeline phases.

Collapses repeated ``BrowserManager`` construction across auth check, live
collection, and Crawl4AI URL-resolution fallback into one locked session
when the pipeline runs live-browser phases back-to-back.
"""

from __future__ import annotations

from types import TracebackType
from typing import Self

from src.utility.browser_manager import BrowserManager
from src.utility.crawl4ai_linkedin_helper import LINKEDIN_FEED_URL, linkedin_profile_lock
from src.utility.linkedin_auth import is_linkedin_authenticated
from src.utility.linkedin_feed_config import LinkedInBrowserProfile


class LinkedInBrowserSession:
    """Locked ``BrowserManager`` context for one or more pipeline phases."""

    def __init__(self, profile: LinkedInBrowserProfile) -> None:
        self._profile = profile
        self._browser = BrowserManager(
            headless=profile.headless,
            viewport=profile.viewport,
            user_data_dir=profile.user_data_dir,
            profile_directory=profile.profile_directory,
        )
        self._lock_ctx = linkedin_profile_lock(profile.user_data_dir)

    @property
    def browser(self) -> BrowserManager:
        return self._browser

    async def __aenter__(self) -> Self:
        await self._lock_ctx.__aenter__()
        await self._browser.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._browser.__aexit__(exc_type, exc, tb)
        await self._lock_ctx.__aexit__(exc_type, exc, tb)

    async def ensure_authenticated(self) -> None:
        """Navigate to feed and raise if the profile is not signed in."""
        await self._browser.navigate(LINKEDIN_FEED_URL)
        if await is_linkedin_authenticated(self._browser):
            return
        raise RuntimeError(
            "LinkedIn session not authenticated. "
            "Run `make sign-in-linkedin` or `scripts/linkedin_sign_in.py` first."
        )
