"""LinkedIn feed arm that parses a local HTML snapshot with Firecrawl ``/v2/parse``.

The Chromium profile stays on disk. One redacted feed fragment is uploaded per
run. Parse formats are markdown and links only — ``json`` and ``summary`` call
an LLM and are refused. This module does not import Crawl4AI or
``linkedin_feed_pipeline``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from src.utility.browser_manager import BrowserManager
from src.utility.linkedin_auth import is_linkedin_authenticated
from src.utility.linkedin_feed_config import FeedCollectionOptions, scroll_budget
from src.utility.linkedin_feed_parser import (
    MAIN_FEED_SELECTOR,
    activity_id_from_url,
    normalize_linkedin_url,
    post_dedupe_id_from_url,
    post_url_validation_reason,
)
from src.utility.linkedin_feed_prepare import LINKEDIN_FEED_URL, prepare_live_feed_page
from src.utility.linkedin_feed_redact import redact_feed_html

logger = logging.getLogger(__name__)

COLLECTION_MODE = "firecrawl_parse"
PARSE_FORMATS = ("markdown", "links")
_MAX_PARSE_BYTES = 50 * 1024 * 1024
_MAX_INVALID_SAMPLES = 5
_SNIPPET_MAX = 400
_PROFILE_LOCK_NAME = ".linkedin_feed_ab.lock"
_REFUSED_PARSE_FORMATS = frozenset({"json", "summary", "question", "highlights"})
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
_RELATIVE_TIME_RE = re.compile(
    r"\b\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|sec|min|hr|mo|yr|[smhdw])\b",
    re.IGNORECASE,
)
_MARKDOWN_NOISE_RE = re.compile(r"[\[\]()#*_>`]+")


@dataclass
class FirecrawlFeedResult:
    """Outcome of one local-browser + Firecrawl parse run."""

    posts: list[dict[str, str]]
    valid_posts: list[dict[str, Any]]
    quality: dict[str, Any]
    sort_meta: dict[str, Any]
    scroll_count: int
    redacted_html_bytes: int
    elapsed_seconds: float
    collection_mode: str = COLLECTION_MODE
    error_message: str | None = None


def build_firecrawl_client() -> Any:
    """Build a Firecrawl client from the process environment.

    ``FIRECRAWL_API_KEY`` is required. ``FIRECRAWL_API_URL`` is optional and
    points at a self-hosted API. The key is not read from application settings.
    """
    from firecrawl import Firecrawl

    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "FIRECRAWL_API_KEY is required for the Firecrawl parse arm."
        )
    api_url = os.environ.get("FIRECRAWL_API_URL", "").strip()
    if api_url:
        return Firecrawl(api_key=api_key, api_url=api_url)
    return Firecrawl(api_key=api_key)


def parse_options() -> Any:
    """Parse options for one feed fragment. LLM formats are refused."""
    from firecrawl.v2.types import ParseOptions

    formats = list(PARSE_FORMATS)
    refused = _REFUSED_PARSE_FORMATS.intersection(formats)
    if refused:
        raise RuntimeError(f"Firecrawl parse arm refuses LLM formats: {sorted(refused)}")
    # The fragment is already the feed container. Leave it intact.
    return ParseOptions(formats=formats, only_main_content=False)


def map_parse_to_posts(
    markdown: str | None,
    links: list[Any] | None,
    *,
    max_posts: int,
) -> list[dict[str, str]]:
    """Map Firecrawl markdown and links into feed post rows.

    Author profile URLs are the nearest preceding ``/in/`` link. Snippets and
    relative times come from markdown near the post URL when that URL, its
    path, or its activity id appears in the markdown.
    """
    target = max(1, int(max_posts))
    text = markdown or ""
    ordered = _ordered_links(text, links or [])
    posts: list[dict[str, str]] = []
    seen: set[str] = set()
    pending_author = ""

    for kind, url in ordered:
        if kind == "author":
            pending_author = url
            continue
        key = _post_key(url)
        if not key or key in seen:
            continue
        seen.add(key)
        window = _snippet_window(text, url)
        posts.append(
            {
                "post_url": url,
                "author_profile_url": pending_author,
                "text_snippet": _clean_snippet(window),
                "relative_time": _relative_time(window),
            }
        )
        pending_author = ""
        if len(posts) >= target:
            break
    return posts


async def parse_redacted_feed(
    html: str,
    *,
    client: Any | None = None,
    max_posts: int,
) -> tuple[list[dict[str, str]], int]:
    """Redact ``html``, upload it once, and return ``(posts, redacted_bytes)``."""
    redacted = redact_feed_html(html)
    payload = redacted.encode("utf-8")
    if len(payload) > _MAX_PARSE_BYTES:
        raise ValueError(
            f"Redacted feed HTML is {len(payload)} bytes; "
            "Firecrawl parse accepts at most 50 MB."
        )
    if not payload.strip():
        return [], 0

    firecrawl = client if client is not None else build_firecrawl_client()
    options = parse_options()
    document = await asyncio.to_thread(
        firecrawl.parse,
        payload,
        filename="feed.html",
        content_type="text/html",
        options=options,
    )
    markdown, links = _document_parts(document)
    return map_parse_to_posts(markdown, links, max_posts=max_posts), len(payload)


async def collect_firecrawl_feed(
    options: FeedCollectionOptions,
    *,
    client: Any | None = None,
    check_auth: bool = True,
    on_progress: Any | None = None,
) -> FirecrawlFeedResult:
    """Own one local Chromium session, scroll the feed, and parse the snapshot."""
    emit = on_progress if on_progress is not None else (lambda _msg: None)
    start = time.perf_counter()
    profile = options.profile
    sort_meta: dict[str, Any] = {"sort_applied": False, "sort_method": "none"}
    scroll_count = 0
    error_message: str | None = None
    posts: list[dict[str, str]] = []
    redacted_bytes = 0

    try:
        async with _profile_lock(profile.user_data_dir):
            async with BrowserManager(
                headless=profile.headless,
                viewport=profile.viewport,
                user_data_dir=profile.user_data_dir,
                profile_directory=profile.profile_directory,
            ) as browser:
                if check_auth:
                    emit("checking auth")
                    await browser.navigate(LINKEDIN_FEED_URL)
                    if not await is_linkedin_authenticated(browser):
                        raise RuntimeError(
                            "LinkedIn session not authenticated. "
                            "Run `make sign-in-linkedin` or `scripts/linkedin_sign_in.py` first."
                        )
                emit("preparing feed")
                sort_meta = await prepare_live_feed_page(
                    browser,
                    sort_js_path=options.sort_js_path,
                    post_card_selector=options.selectors.post_card_selector,
                )
                emit("scrolling feed")
                scroll_count = await _scroll_feed(browser, options)
                emit("capturing feed html")
                html = await _capture_feed_html(browser, options)
            emit("parsing feed")
            posts, redacted_bytes = await parse_redacted_feed(
                html,
                client=client,
                max_posts=options.scroll.max_posts,
            )
    except Exception as exc:
        error_message = str(exc)
        logger.exception("Firecrawl feed parse failed")

    quality_raw = arm_quality_metrics(
        posts,
        require_activity_id=options.require_activity_id,
    )
    elapsed = time.perf_counter() - start
    return FirecrawlFeedResult(
        posts=posts,
        valid_posts=quality_raw["valid_rows"],
        quality=_public_quality(quality_raw),
        sort_meta=sort_meta,
        scroll_count=scroll_count,
        redacted_html_bytes=redacted_bytes,
        elapsed_seconds=elapsed,
        error_message=error_message,
    )


def arm_quality_metrics(
    posts: list[dict[str, Any]],
    *,
    require_activity_id: bool = False,
) -> dict[str, Any]:
    """Compute valid ``post_url`` stats. Same counts as the live feed export."""
    total = len(posts)
    valid_rows: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, str]] = []
    empty_count = 0

    for row in posts:
        url = (row.get("post_url") or "").strip()
        if not url:
            empty_count += 1
        reason = post_url_validation_reason(url, require_activity_id=require_activity_id)
        if reason is None:
            valid_rows.append(row)
        elif len(invalid_samples) < _MAX_INVALID_SAMPLES:
            invalid_samples.append(
                {
                    "post_url": url,
                    "author_profile_url": row.get("author_profile_url") or "",
                    "reason": reason,
                }
            )

    valid_count = len(valid_rows)
    ratio = (valid_count / total) if total else 0.0
    return {
        "total_count": total,
        "valid_post_url_count": valid_count,
        "valid_post_url_ratio": round(ratio, 4),
        "empty_post_url_count": empty_count,
        "invalid_post_url_samples": invalid_samples,
        "valid_rows": valid_rows,
    }


def write_feed_export(
    path: str | Path,
    *,
    posts: list[dict[str, Any]],
    valid_posts: list[dict[str, Any]],
    quality: dict[str, Any],
    params: dict[str, Any],
) -> Path:
    """Write feed collection JSON. Shape matches the live arm export."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "posts": posts,
        "valid_posts": valid_posts,
        "quality": quality,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return out


def default_firecrawl_export_path(export_dir: str | Path | None = None) -> Path:
    """Timestamped Firecrawl-arm export path under the export directory."""
    if export_dir is None:
        from src.config.settings import settings

        export_dir = settings.EXPORT_DIRECTORY
    base = Path(export_dir)
    stamp = datetime.now(ZoneInfo("US/Pacific")).strftime("%Y-%m-%dT%H:%M:%S%z")
    return base / f"linkedin_feed_firecrawl_{stamp}.json"


def export_params(
    options: FeedCollectionOptions,
    result: FirecrawlFeedResult,
) -> dict[str, Any]:
    """JSON ``params`` for one Firecrawl parse export."""
    return {
        **options.export_params(),
        "collection_mode": COLLECTION_MODE,
        "use_firecrawl_parse": True,
        "sort_applied": result.sort_meta.get("sort_applied", False),
        "sort_method": result.sort_meta.get("sort_method", "none"),
        "scroll_count": result.scroll_count,
        "redacted_html_bytes": result.redacted_html_bytes,
    }


@asynccontextmanager
async def _profile_lock(user_data_dir: str | Path) -> AsyncIterator[None]:
    """Exclusive lock on the same file the live and Crawl4AI arms use.

    Copied here so this arm does not import the Crawl4AI helper. The lock
    name is ``.linkedin_feed_ab.lock`` under ``user_data_dir``.
    """
    lock_path = Path(user_data_dir).resolve() / _PROFILE_LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


async def _scroll_feed(browser: BrowserManager, options: FeedCollectionOptions) -> int:
    scroll = options.scroll
    max_scrolls = scroll_budget(
        scroll_rounds=scroll.scroll_rounds,
        target=scroll.effective_target(),
    )
    if max_scrolls <= 0:
        return 0
    for _ in range(max_scrolls):
        await browser.scroll(scroll.scroll_px)
        await asyncio.sleep(scroll.scroll_pause_s)
    return max_scrolls


async def _capture_feed_html(browser: BrowserManager, options: FeedCollectionOptions) -> str:
    selectors = options.selectors
    html = await browser.get_page_html(selectors.container_selector)
    if html and html.strip():
        return html
    html = await browser.get_page_html(MAIN_FEED_SELECTOR)
    if html and html.strip():
        return html
    raise RuntimeError("Feed container HTML was empty.")


def _document_parts(document: Any) -> tuple[str, list[Any]]:
    if isinstance(document, dict):
        markdown = document.get("markdown") or ""
        links = document.get("links") or []
    else:
        markdown = getattr(document, "markdown", None) or ""
        links = getattr(document, "links", None) or []
    return str(markdown), list(links)


def _ordered_links(markdown: str, links: list[Any]) -> list[tuple[str, str]]:
    positioned: list[tuple[int, str, str]] = []
    seen_urls: set[str] = set()

    for match in _URL_RE.finditer(markdown):
        classified = _classify_link(match.group(0))
        if classified is None:
            continue
        kind, url = classified
        positioned.append((match.start(), kind, url))
        seen_urls.add(url)

    ordered = [(kind, url) for _index, kind, url in sorted(positioned, key=lambda item: item[0])]
    for item in links:
        classified = _classify_link(_link_url(item))
        if classified is None:
            continue
        kind, url = classified
        if url in seen_urls:
            continue
        seen_urls.add(url)
        ordered.append((kind, url))
    return ordered


def _link_url(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("url") or item.get("href") or "")
    return str(getattr(item, "url", None) or getattr(item, "href", None) or "")


def _classify_link(raw: str) -> tuple[str, str] | None:
    normalized = normalize_linkedin_url(raw.strip().rstrip(").,;"))
    if not normalized:
        return None
    path = urlparse(normalized).path.lower()
    if "/feed/update/" in path or "/posts/" in path:
        return "post", normalized
    if "/in/" in path:
        return "author", normalized
    return None


def _post_key(url: str) -> str:
    urn = post_dedupe_id_from_url(url)
    if urn:
        return f"urn:{urn}"
    bare = url.split("?", 1)[0]
    return bare[:512]


def _snippet_window(markdown: str, url: str) -> str:
    index = _find_index(markdown, url)
    if index is None:
        return ""
    start = max(0, index - 250)
    end = min(len(markdown), index + 400)
    return markdown[start:end]


def _find_index(markdown: str, url: str) -> int | None:
    if url and url in markdown:
        return markdown.find(url)
    activity_id = activity_id_from_url(url)
    if activity_id and activity_id in markdown:
        return markdown.find(activity_id)
    path = urlparse(url).path
    if path and path in markdown:
        return markdown.find(path)
    return None


def _clean_snippet(window: str) -> str:
    text = _URL_RE.sub(" ", window)
    text = _MARKDOWN_NOISE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= _SNIPPET_MAX:
        return text
    return text[:_SNIPPET_MAX].rstrip() + "…"


def _relative_time(window: str) -> str:
    match = _RELATIVE_TIME_RE.search(window)
    if not match:
        return ""
    return re.sub(r"\s+", "", match.group(0))


def _public_quality(quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_count": quality["total_count"],
        "valid_post_url_count": quality["valid_post_url_count"],
        "valid_post_url_ratio": quality["valid_post_url_ratio"],
        "empty_post_url_count": quality["empty_post_url_count"],
        "invalid_post_url_samples": quality["invalid_post_url_samples"],
    }
