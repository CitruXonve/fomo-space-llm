"""CLI entrypoint that runs ``LinkedInFeedPipeline`` exactly once.

Usage::

    poetry run python -m src.utility.linkedin_feed
    poetry run python -m src.utility.linkedin_feed \\
      --user-data-dir .browser_profile --profile-directory Default \\
      --headless --max-posts 25 --scroll-rounds 50

Each invocation constructs a :class:`LinkedInFeedPipeline`, renders a live spinner
whose suffix updates as collection progresses, prints one JSON summary to stdout,
and on success prints elapsed time to stderr (human-readable).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from src.utility.linkedin_feed_parser import DEFAULT_POST_CARD_SELECTOR
from src.utility.linkedin_feed_pipeline import LinkedInFeedPipeline, LinkedInFeedPipelineConfig
from src.utility.linkedin_web_agent import _spin_until
from src.utility.spinner import Spinner


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
        description="Collect LinkedIn feed posts via LinkedInFeedPipeline (live browser by default).",
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
    parser.add_argument(
        "--no-post-card-selector",
        action="store_true",
        dest="no_post_card_selector",
        help="Disable lazy-mount card parsing (div[data-lazy-mount-id]).",
    )
    parser.add_argument("--export-dir", default=None, dest="export_dir")
    parser.add_argument("--require-activity-id", action="store_true")
    parser.add_argument("--no-export", action="store_true", dest="no_export")
    parser.add_argument(
        "--skip-auth-check",
        action="store_true",
        dest="skip_auth_check",
    )
    resolve_urls = parser.add_mutually_exclusive_group()
    resolve_urls.add_argument(
        "--resolve-missing-urls",
        action="store_true",
        default=None,
        dest="resolve_missing_urls",
        help="Resolve missing post_url via Send → Copy link (default).",
    )
    resolve_urls.add_argument(
        "--no-resolve-missing-urls",
        action="store_true",
        default=None,
        dest="no_resolve_missing_urls",
        help="Skip Send → Copy link resolution (degraded; may return author-only rows).",
    )
    parser.add_argument(
        "--use-crawl4ai",
        action="store_true",
        dest="use_crawl4ai",
        help="Use Crawl4AI virtual-scroll instead of live BrowserManager (A/B / experiment).",
    )
    return parser


async def _async_main(ns: argparse.Namespace) -> int:
    if ns.headless is True:
        headless = True
    elif ns.no_headless is True:
        headless = False
    else:
        headless = True

    viewport = _parse_viewport(ns.viewport)
    if ns.no_resolve_missing_urls:
        resolve_missing_urls = False
    elif ns.resolve_missing_urls:
        resolve_missing_urls = True
    else:
        resolve_missing_urls = True

    config = LinkedInFeedPipelineConfig.from_kwargs(
        user_data_dir=ns.user_data_dir,
        profile_directory=ns.profile_directory,
        headless=headless,
        max_posts=ns.max_posts,
        scroll_rounds=ns.scroll_rounds,
        container_selector=ns.container_selector,
        post_card_selector=(
            None if ns.no_post_card_selector else ns.post_card_selector
        ),
        export_dir=ns.export_dir,
        require_activity_id=ns.require_activity_id,
        check_auth=not ns.skip_auth_check,
        viewport=viewport,
        resolve_missing_urls=resolve_missing_urls,
        use_crawl4ai=ns.use_crawl4ai,
    )
    pipeline = LinkedInFeedPipeline(config)

    spinner = Spinner()
    current_stage = ["starting"]
    stop = asyncio.Event()
    spin_task = asyncio.create_task(_spin_until(stop, current_stage, spinner))

    try:
        result = await pipeline.run(
            on_progress=lambda msg: current_stage.__setitem__(0, msg),
            write_export=not ns.no_export,
        )
    except Exception as exc:
        stop.set()
        await spin_task
        spinner.finish("LinkedIn feed collection failed")
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1

    stop.set()
    await spin_task
    spinner.finish(
        f"LinkedIn feed collection complete in {result['elapsed_pretty']}"
    )

    summary = {
        "ok": result["ok"],
        "export_path": result.get("export_path"),
        "quality": result.get("quality"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "crawl4ai_success": result.get("crawl4ai_success"),
        "error_message": result.get("error_message"),
        "urls_resolved_count": result.get("urls_resolved_count"),
        "resolve_missing_urls": result.get("resolve_missing_urls"),
        "use_crawl4ai": result.get("use_crawl4ai"),
        "collection_mode": result.get("collection_mode"),
        "sort_applied": result.get("sort_applied"),
        "sort_method": result.get("sort_method"),
        "browser_fallback_used": result.get("browser_fallback_used"),
    }
    print(json.dumps(summary, ensure_ascii=False, default=str))
    return 0 if result["ok"] else 1


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
