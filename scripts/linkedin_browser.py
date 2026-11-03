#!/usr/bin/env python3
"""
LinkedIn BrowserManager CLI — generic browser commands plus feed/auth helpers.

Examples::

    poetry run python scripts/linkedin_browser.py --no-headless \\
        --user-data-dir .browser_profile --profile-directory Default \\
        check-auth

    poetry run python scripts/linkedin_browser.py --no-headless \\
        --user-data-dir .browser_profile feed-posts --max-posts 10
"""

from __future__ import annotations

from src.utility.linkedin_browser_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
