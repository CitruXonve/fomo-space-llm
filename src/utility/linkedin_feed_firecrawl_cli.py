"""CLI for the Firecrawl parse arm of LinkedIn feed collection.

Usage::

    poetry run python -m src.utility.linkedin_feed_firecrawl_cli \\
      --user-data-dir .browser_profile --profile-directory Default \\
      --headless --max-posts 25 --scroll-rounds 50

The browser profile stays local. One redacted HTML fragment is uploaded to
Firecrawl ``/v2/parse``. This command does not call the live pipeline or Crawl4AI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from src.utility.firecrawl_linkedin_helper import (
    COLLECTION_MODE,
    collect_firecrawl_feed,
    default_firecrawl_export_path,
    export_params,
    write_feed_export,
)
from src.utility.linkedin_feed_config import FeedCollectionOptions
from src.utility.linkedin_feed_parser import DEFAULT_POST_CARD_SELECTOR


def _parse_viewport(raw: str) -> tuple[int, int]:
    if "x" not in raw.lower():
        raise ValueError(f"viewport must be WIDTHxHEIGHT, got: {raw!r}")
    width_s, height_s = raw.lower().split("x", 1)
    width, height = int(width_s), int(height_s)
    if width < 1 or height < 1:
        raise ValueError(f"viewport dimensions must be positive: {raw!r}")
    return (width, height)


def _format_duration(total_seconds: float) -> str:
    minutes, seconds = divmod(total_seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect LinkedIn feed posts with a local browser profile and "
            "Firecrawl /v2/parse (markdown + links)."
        ),
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
    parser.add_argument("--max-posts", type=int, default=25)
    parser.add_argument("--scroll-rounds", type=int, default=10)
    parser.add_argument(
        "--container-selector",
        default="#workspace",
        dest="container_selector",
    )
    parser.add_argument(
        "--post-card-selector",
        default=DEFAULT_POST_CARD_SELECTOR,
        dest="post_card_selector",
    )
    parser.add_argument("--export-dir", default=None, dest="export_dir")
    parser.add_argument("--require-activity-id", action="store_true")
    parser.add_argument("--no-export", action="store_true", dest="no_export")
    parser.add_argument(
        "--skip-auth-check",
        action="store_true",
        dest="skip_auth_check",
    )
    return parser


async def _async_main(ns: argparse.Namespace) -> int:
    if ns.headless is True:
        headless = True
    elif ns.no_headless is True:
        headless = False
    else:
        headless = True

    options = FeedCollectionOptions.from_pipeline_kwargs(
        user_data_dir=ns.user_data_dir,
        profile_directory=ns.profile_directory,
        headless=headless,
        max_posts=ns.max_posts,
        scroll_rounds=ns.scroll_rounds,
        container_selector=ns.container_selector,
        post_card_selector=ns.post_card_selector,
        viewport=_parse_viewport(ns.viewport),
        require_activity_id=ns.require_activity_id,
        resolve_missing_urls=False,
    )
    started = time.perf_counter()
    result = await collect_firecrawl_feed(
        options,
        check_auth=not ns.skip_auth_check,
    )
    if result.error_message:
        print(
            json.dumps({"ok": False, "error": result.error_message, "collection_mode": COLLECTION_MODE}),
            file=sys.stderr,
        )
        return 1

    export_path: str | None = None
    if not ns.no_export:
        written = write_feed_export(
            default_firecrawl_export_path(ns.export_dir),
            posts=result.posts,
            valid_posts=result.valid_posts,
            quality=result.quality,
            params=export_params(options, result),
        )
        export_path = str(written)

    elapsed = time.perf_counter() - started
    ok = result.quality["valid_post_url_count"] > 0
    print(
        f"LinkedIn Firecrawl parse complete in {_format_duration(elapsed)}",
        file=sys.stderr,
    )
    summary = {
        "ok": ok,
        "export_path": export_path,
        "quality": result.quality,
        "elapsed_seconds": result.elapsed_seconds,
        "collection_mode": result.collection_mode,
        "sort_applied": result.sort_meta.get("sort_applied", False),
        "sort_method": result.sort_meta.get("sort_method", "none"),
        "scroll_count": result.scroll_count,
        "redacted_html_bytes": result.redacted_html_bytes,
        "error_message": result.error_message,
    }
    print(json.dumps(summary, ensure_ascii=False, default=str))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    try:
        _parse_viewport(ns.viewport)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    return asyncio.run(_async_main(ns))


if __name__ == "__main__":
    raise SystemExit(main())
