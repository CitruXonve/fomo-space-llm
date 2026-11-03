#!/usr/bin/env python3
"""
Generic Playwright BrowserManager CLI (no LinkedIn-specific commands).

For LinkedIn feed/auth commands use ``scripts/linkedin_browser.py``.

One-time setup::

    poetry run playwright install chromium

Examples::

    poetry run python scripts/browser_manager.py navigate https://example.com
    poetry run python scripts/browser_manager.py run --steps \\
        '[{"op":"navigate","url":"https://example.com"},{"op":"title"}]'
"""

from __future__ import annotations

from src.utility.browser_manager_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
