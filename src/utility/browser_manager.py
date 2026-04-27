"""Async Playwright wrapper for managing a Chromium browser session.

Supports two launch modes:

1. Ephemeral (default): a fresh Chromium instance with no saved state.
2. Persistent: reuses cookies, login state, localStorage, and extensions from a
   Chrome user-data directory by passing ``user_data_dir`` (e.g. ``.browser_profile``).
   In this mode the persistent context owns the browser lifecycle, so ``self.browser``
   stays ``None`` and ``stop()`` only closes the context.

Caveat: the persistent profile must NOT be opened by another Chrome process at the
same time -- Chromium uses a singleton lock per user-data directory and Playwright
will fail to launch if the lock is held.

Setup (one-time):

    poetry run playwright install chromium

Ephemeral usage:

    async with BrowserManager() as bm:
        await bm.navigate("https://example.com")
        title = await bm.get_page_title()

Persistent usage (reuses a saved Chrome profile):

    async with BrowserManager(user_data_dir=".browser_profile") as bm:
        await bm.navigate("https://www.linkedin.com/feed/")
"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

logger = logging.getLogger(__name__)


_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_DEFAULT_BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-web-security",
    "--disable-features=VizDisplayCompositor",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
]


class BrowserManager:
    """Manages a Playwright Chromium instance with optional persistent profile."""

    def __init__(
        self,
        headless: bool = True,
        viewport: tuple[int, int] = (1280, 720),
        user_agent: str | None = None,
        user_data_dir: str | Path | None = None,
        profile_directory: str | None = None,
    ) -> None:
        self.headless = headless
        self.viewport = viewport
        self.user_agent = user_agent or _DEFAULT_USER_AGENT
        self.user_data_dir: Path | None = (
            Path(user_data_dir) if user_data_dir is not None else None
        )
        self.profile_directory = profile_directory

        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    @property
    def is_persistent(self) -> bool:
        return self.user_data_dir is not None

    async def __aenter__(self) -> "BrowserManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def start(self) -> None:
        """Initialize Playwright and launch Chromium (ephemeral or persistent)."""
        logger.info("Starting Playwright")
        self.playwright = await async_playwright().start()

        viewport_dict = {"width": self.viewport[0], "height": self.viewport[1]}
        launch_args = list(_DEFAULT_BROWSER_ARGS)
        if self.profile_directory:
            launch_args.append(f"--profile-directory={self.profile_directory}")

        if self.is_persistent:
            assert self.user_data_dir is not None
            self._warn_if_singleton_lock_present()
            logger.info(
                "Launching Chromium with persistent context at %s "
                "(profile_directory=%s, headless=%s)",
                self.user_data_dir,
                self.profile_directory,
                self.headless,
            )
            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.user_data_dir),
                headless=self.headless,
                args=launch_args,
                viewport=viewport_dict,
                user_agent=self.user_agent,
                ignore_https_errors=True,
                bypass_csp=True,
            )
            self.browser = None
            existing_pages = self.context.pages
            self.page = existing_pages[0] if existing_pages else await self.context.new_page()
        else:
            logger.info(
                "Launching Chromium (ephemeral, headless=%s)", self.headless
            )
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=launch_args,
            )
            self.context = await self.browser.new_context(
                viewport=viewport_dict,
                user_agent=self.user_agent,
                ignore_https_errors=True,
                bypass_csp=True,
            )
            self.page = await self.context.new_page()

        self._attach_page_log_handlers(self.page)
        logger.info("BrowserManager started")

    async def stop(self) -> None:
        """Close all Playwright resources, swallowing errors via the logger."""
        logger.info("Stopping BrowserManager")
        try:
            if self.context is not None:
                try:
                    await self.context.close()
                except Exception as exc:
                    logger.warning("Error closing context: %s", exc)
                self.context = None

            if self.browser is not None:
                try:
                    await self.browser.close()
                except Exception as exc:
                    logger.warning("Error closing browser: %s", exc)
                self.browser = None

            if self.playwright is not None:
                try:
                    await self.playwright.stop()
                except Exception as exc:
                    logger.warning("Error stopping playwright: %s", exc)
                self.playwright = None
        finally:
            self.page = None
            logger.info("BrowserManager stopped")

    async def navigate(self, url: str, timeout_ms: int = 30_000) -> None:
        """Navigate to ``url``, waiting for ``domcontentloaded``."""
        page = self._ensure_started()
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        logger.info("Navigating to %s", url)
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

    async def get_screenshot(self) -> str:
        """Capture a PNG screenshot. Returns base64 string ('' on failure)."""
        page = self._ensure_started()
        try:
            data = await page.screenshot(type="png", full_page=False)
            return base64.b64encode(data).decode("utf-8")
        except Exception as exc:
            logger.error("Screenshot failed: %s", exc)
            return ""

    async def get_current_url(self) -> str:
        page = self._ensure_started()
        try:
            return page.url
        except Exception as exc:
            logger.error("Failed to read current URL: %s", exc)
            return ""

    async def get_page_title(self) -> str:
        page = self._ensure_started()
        try:
            return await page.title()
        except Exception as exc:
            logger.error("Failed to read page title: %s", exc)
            return ""

    async def execute_javascript(self, code: str) -> Any:
        """Run ``code`` via ``page.evaluate`` and return the JSON-serializable result."""
        page = self._ensure_started()
        snippet = code if len(code) <= 100 else f"{code[:100]}..."
        logger.debug("Executing JavaScript: %s", snippet)
        return await page.evaluate(code)

    async def scroll(self, pixels: int) -> None:
        """Scroll the page by ``pixels`` (positive = down, negative = up)."""
        page = self._ensure_started()
        await page.evaluate(f"window.scrollBy(0, {int(pixels)})")

    async def wait_for(
        self,
        selector: str | None = None,
        timeout_ms: int = 10_000,
    ) -> None:
        """Wait for ``selector`` to appear, or sleep ``timeout_ms`` if no selector."""
        page = self._ensure_started()
        if selector:
            await page.wait_for_selector(selector, timeout=timeout_ms)
        else:
            await page.wait_for_timeout(timeout_ms)

    async def extract_visible_text(self) -> str:
        """Return ``document.body.innerText`` for the current page."""
        page = self._ensure_started()
        try:
            text = await page.evaluate("document.body.innerText")
        except Exception as exc:
            logger.error("Failed to extract visible text: %s", exc)
            return ""
        if isinstance(text, str):
            preview = text[:200].replace("\n", " ")
            logger.debug(
                "Extracted %d chars of visible text (preview: %s...)",
                len(text),
                preview,
            )
            return text
        return str(text) if text is not None else ""

    async def stream_screenshots(
        self,
        interval_seconds: float = 2.0,
    ) -> AsyncIterator[str]:
        """Yield base64 PNG screenshots every ``interval_seconds`` until cancelled."""
        self._ensure_started()
        try:
            while self.page is not None:
                shot = await self.get_screenshot()
                if not shot:
                    logger.warning("Empty screenshot; stopping stream")
                    return
                yield shot
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Screenshot stream cancelled")
            raise

    def _ensure_started(self) -> Page:
        if self.page is None:
            raise RuntimeError("Browser not initialized")
        return self.page

    def _attach_page_log_handlers(self, page: Page) -> None:
        page.on("console", lambda msg: logger.debug("Console: %s", msg.text))
        page.on("pageerror", lambda err: logger.warning("Page error: %s", err))

    def _warn_if_singleton_lock_present(self) -> None:
        if self.user_data_dir is None:
            return
        lock_path = self.user_data_dir / "SingletonLock"
        if lock_path.exists() or os.path.islink(lock_path):
            logger.warning(
                "SingletonLock present at %s -- another Chrome process may be "
                "using this profile; Playwright launch will fail if so.",
                lock_path,
            )
