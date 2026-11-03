#!/usr/bin/env python3
"""
Collect LinkedIn feed posts via LinkedInFeedPipeline.

Prerequisites::

    poetry install
    poetry run playwright install chromium
    make sign-in-linkedin   # one-time authenticated profile

Example::

    poetry run python scripts/linkedin_feed_agent.py \\
      --user-data-dir .browser_profile --profile-directory Default \\
      --headless --max-posts 25 --scroll-rounds 50
"""

from __future__ import annotations

from src.utility.linkedin_feed import main

if __name__ == "__main__":
    raise SystemExit(main())
