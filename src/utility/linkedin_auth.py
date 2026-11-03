"""LinkedIn authentication helpers shared by CLI and agents."""

from __future__ import annotations

import asyncio
from typing import Any

from src.utility.browser_manager import BrowserManager


async def is_linkedin_authenticated(browser: BrowserManager) -> bool:
    """Heuristic auth check used by LinkedIn browser automations."""
    current_url = await browser.get_current_url()
    blocked_markers = ("/login", "/uas/login", "/checkpoint", "/authwall")
    if any(marker in current_url for marker in blocked_markers):
        return False

    try:
        probe = await browser.execute_javascript(
            """(() => {
                const hasGlobalNav = Boolean(
                    document.querySelector('nav.global-nav, [data-test-global-nav], .global-nav')
                );
                const hasSearchInput = Boolean(
                    document.querySelector('input[placeholder*="Search"], input[aria-label*="Search"]')
                );
                const hasNetworkLink = Boolean(
                    document.querySelector('a[href*="/mynetwork/"], a[href*="/jobs/"], a[href*="/messaging/"]')
                );
                const hasSignInForm = Boolean(
                    document.querySelector('form[action*="login"], input[name="session_key"], input[name="session_password"]')
                );
                const hasJoinLink = Boolean(
                    document.querySelector('a[href*="signup"], a[data-tracking-control-name*="join"]')
                );
                return {
                    hasGlobalNav,
                    hasSearchInput,
                    hasNetworkLink,
                    hasSignInForm,
                    hasJoinLink,
                };
            })()"""
        )
    except Exception:
        return False

    if not isinstance(probe, dict):
        probe = {}

    p: dict[str, Any] = probe
    has_logged_in_hint = bool(p.get("hasGlobalNav")) or bool(p.get("hasSearchInput")) or bool(
        p.get("hasNetworkLink")
    )
    has_signed_out_hint = bool(p.get("hasSignInForm")) or bool(p.get("hasJoinLink"))
    on_feed = "/feed/" in current_url
    return has_logged_in_hint or (on_feed and not has_signed_out_hint)


async def wait_for_linkedin_sign_in(
    browser: BrowserManager,
    timeout_s: int = 120,
    poll_interval_s: float = 2.0,
) -> bool:
    """Navigate to login page and wait for manual user sign-in."""
    await browser.navigate("https://www.linkedin.com/login")
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if await is_linkedin_authenticated(browser):
            return True
        await asyncio.sleep(poll_interval_s)
    return False
