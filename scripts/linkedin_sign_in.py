#!/usr/bin/env python3
"""
Open LinkedIn login in a persistent Chromium profile and wait for manual sign-in.

Prerequisites::

    poetry install
    poetry run playwright install chromium

Examples::

    poetry run python scripts/linkedin_sign_in.py \\
      --user-data-dir .browser_profile --profile-directory Default

    poetry run python scripts/linkedin_sign_in.py --check-only

See ``.cursor/skills/linkedin-sign-in/SKILL.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from src.utility.browser_manager import BrowserManager
from src.utility.linkedin_auth import is_linkedin_authenticated, wait_for_linkedin_sign_in

LINKEDIN_FEED_URL = "https://www.linkedin.com/feed/"


def _parse_viewport(raw: str) -> tuple[int, int]:
    if "x" not in raw.lower():
        raise ValueError(f"viewport must be WIDTHxHEIGHT, got: {raw!r}")
    w_s, h_s = raw.lower().split("x", 1)
    w, h = int(w_s), int(h_s)
    if w < 1 or h < 1:
        raise ValueError(f"viewport dimensions must be positive: {raw!r}")
    return (w, h)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LinkedIn manual sign-in with persistent browser profile.",
    )
    headless = parser.add_mutually_exclusive_group()
    headless.add_argument("--headless", action="store_true", default=None)
    headless.add_argument("--no-headless", action="store_true", default=None)
    parser.add_argument("--viewport", default="1280x720", metavar="WxH")
    parser.add_argument(
        "--user-data-dir",
        default=".browser_profile",
        dest="user_data_dir",
    )
    parser.add_argument(
        "--profile-directory",
        default="Default",
        dest="profile_directory",
    )
    parser.add_argument("--timeout-s", type=int, default=120, dest="timeout_s")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run auth check; do not open the login page.",
    )
    parser.add_argument(
        "--skip-precheck",
        action="store_true",
        help="Skip feed pre-check and open login immediately.",
    )
    return parser


async def _check_auth(bm: BrowserManager) -> dict[str, object]:
    await bm.navigate(LINKEDIN_FEED_URL)
    authed = await is_linkedin_authenticated(bm)
    return {
        "authenticated": authed,
        "current_url": await bm.get_current_url(),
    }


async def _run_sign_in(ns: argparse.Namespace) -> dict[str, object]:
    if ns.headless is True:
        headless = True
    elif ns.no_headless is True:
        headless = False
    else:
        headless = False

    viewport = _parse_viewport(ns.viewport)
    bm = BrowserManager(
        headless=headless,
        viewport=viewport,
        user_data_dir=ns.user_data_dir,
        profile_directory=ns.profile_directory,
    )

    async with bm:
        if ns.check_only:
            data = await _check_auth(bm)
            return {
                "ok": bool(data["authenticated"]),
                "mode": "check_only",
                **data,
            }

        already_authenticated = False
        if not ns.skip_precheck:
            pre = await _check_auth(bm)
            if pre["authenticated"]:
                already_authenticated = True
                return {
                    "ok": True,
                    "mode": "sign_in",
                    "already_authenticated": True,
                    "authenticated": True,
                    "current_url": pre["current_url"],
                }

        authed = await wait_for_linkedin_sign_in(bm, timeout_s=ns.timeout_s)
        current_url = await bm.get_current_url()
        if not authed:
            return {
                "ok": False,
                "mode": "sign_in",
                "error": "linkedin_sign_in_timeout",
                "authenticated": False,
                "already_authenticated": already_authenticated,
                "timeout_s": ns.timeout_s,
                "current_url": current_url,
            }

        verify = await _check_auth(bm)
        return {
            "ok": bool(verify["authenticated"]),
            "mode": "sign_in",
            "already_authenticated": already_authenticated,
            "authenticated": verify["authenticated"],
            "current_url": verify["current_url"],
            "timeout_s": ns.timeout_s,
        }


async def _async_main(ns: argparse.Namespace) -> int:
    try:
        result = await _run_sign_in(ns)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1

    stream = sys.stdout if result.get("ok") else sys.stderr
    print(json.dumps(result, ensure_ascii=False, default=str), file=stream)
    return 0 if result.get("ok") else 1


def main() -> int:
    parser = _build_parser()
    ns = parser.parse_args()
    try:
        _parse_viewport(ns.viewport)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    return asyncio.run(_async_main(ns))


if __name__ == "__main__":
    raise SystemExit(main())
