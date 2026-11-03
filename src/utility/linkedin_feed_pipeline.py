"""Deterministic LinkedIn feed collection pipeline for production/background use.

Production-oriented sibling of :class:`LinkedInWebAgent` — no LLM, no token
usage. Orchestrates feed scraping for background jobs, cron, or downstream
processing.

End-state architecture::

    LinkedInFeedPipeline(LinkedInFeedPipelineConfig)
      1. authenticate          (LinkedInBrowserSession.ensure_authenticated)
      2. collect               (Crawl4AI and/or LiveFeedCollector)
      3. dedupe + validate     (dedupe_feed_posts, arm_quality_metrics)
      4. export                (write_feed_export)

    LiveFeedCollector.run()  — scroll/parse loop (see linkedin_feed_collector.py)
      prepare_live_feed_page → JS extract → HTML scroll loop → resolve stubs

Default path (``use_crawl4ai=False``): one ``LinkedInBrowserSession`` —
hydrate feed → sort Recent → scroll → extract lazy-mount cards → inline
Send → Copy link for author-only stubs.

Opt-in Crawl4AI (``use_crawl4ai=True``): virtual-scroll HTML snapshot for A/B
or experimentation. Crawl4AI often returns an unhydrated shell; enable
``resolve_missing_urls`` to fall back to the live session when the crawl is empty.

Constraints:

- Uses ``linkedin_profile_lock`` under ``user_data_dir``. Do not run
  ``LinkedInFeedPipeline``, ``LinkedInWebAgent``, or ``browser_manager.py``
  concurrently against the same profile.
- Headless runs require a pre-warmed signed-in profile (``make sign-in-linkedin``).
- Integrate as a background worker, not inline in FastAPI request handlers.
- Feed cards use ``post_card_selector`` (default ``div[data-lazy-mount-id]``).
- ``resolve_missing_urls=False`` skips Send → Copy link (degraded; may yield
  author-only rows without ``post_url``).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.config.settings import settings
from src.utility.crawl4ai_linkedin_helper import (
    CrawlFeedResult,
    crawl_linkedin_feed,
    dedupe_feed_posts,
)
from src.utility.linkedin_browser_session import LinkedInBrowserSession
from src.utility.linkedin_feed_config import (
    FeedCollectionResult,
    LinkedInFeedPipelineConfig,
)
from src.utility.linkedin_feed_collector import LiveFeedCollector
from src.utility.linkedin_feed_parser import post_url_validation_reason
from src.utility.linkedin_feed_prepare import prepare_live_feed_page
from src.utility.linkedin_feed_url_resolve import resolve_post_urls_via_send_button
from src.utility.spinner import make_progress_emitter

logger = logging.getLogger(__name__)

_MAX_INVALID_SAMPLES = 5


def _format_duration(total_seconds: float) -> str:
    """Format a duration as 'Xm Y.YYs' (always shows minutes, even when 0)."""
    minutes, seconds = divmod(total_seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def default_export_path(export_dir: str | Path | None = None) -> Path:
    """Timestamped export path under the export directory."""
    base = Path(export_dir or settings.EXPORT_DIRECTORY)
    stamp = datetime.now(ZoneInfo("US/Pacific")).strftime("%Y-%m-%dT%H:%M:%S%z")
    return base / f"linkedin_feed_{stamp}.json"


def _public_quality(q: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_count": q["total_count"],
        "valid_post_url_count": q["valid_post_url_count"],
        "valid_post_url_ratio": q["valid_post_url_ratio"],
        "empty_post_url_count": q["empty_post_url_count"],
        "invalid_post_url_samples": q["invalid_post_url_samples"],
    }


def write_feed_export(
    path: str | Path,
    *,
    posts: list[dict[str, Any]],
    valid_posts: list[dict[str, Any]],
    quality: dict[str, Any],
    params: dict[str, Any],
) -> Path:
    """Write feed collection JSON to ``path``."""
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


def arm_quality_metrics(
    posts: list[dict[str, Any]],
    *,
    require_activity_id: bool = False,
) -> dict[str, Any]:
    """Compute valid ``post_url`` stats for one arm."""
    total = len(posts)
    valid_rows: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, str]] = []
    empty_count = 0

    for row in posts:
        url = (row.get("post_url") or "").strip()
        if not url:
            empty_count += 1
        reason = post_url_validation_reason(
            url, require_activity_id=require_activity_id
        )
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


class LinkedInFeedPipeline:
    """LinkedIn feed scraper pipeline — live browser by default, Crawl4AI opt-in."""

    def __init__(
        self,
        config: LinkedInFeedPipelineConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("pass config or keyword arguments, not both")
        self.config = config or LinkedInFeedPipelineConfig.from_kwargs(**kwargs)
        self._export_dir = self.config.export_dir or settings.EXPORT_DIRECTORY

    @property
    def options(self) -> Any:
        """Collection options (shortcut for ``config.collection``)."""
        return self.config.collection

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> LinkedInFeedPipeline:
        """Construct from flat keyword arguments."""
        return cls(LinkedInFeedPipelineConfig.from_kwargs(**kwargs))

    async def run(
        self,
        *,
        on_progress: Callable[[str], None] | None = None,
        write_export: bool = True,
        export_filename: str | None = None,
    ) -> dict[str, Any]:
        """Run the pipeline once: auth → collect → validate → export."""
        emit = make_progress_emitter(on_progress)
        start = time.perf_counter()

        if self.config.check_auth:
            emit("checking auth")
            async with LinkedInBrowserSession(self.config.collection.profile) as session:
                await session.ensure_authenticated()
            logger.info("LinkedIn session already authenticated")

        collected = await self._collect(emit)
        posts = dedupe_feed_posts(
            collected["posts"],
            max_posts=self.config.collection.scroll.max_posts,
        )

        emit("validating posts")
        quality_raw = arm_quality_metrics(
            posts,
            require_activity_id=self.config.collection.require_activity_id,
        )
        valid_posts = quality_raw["valid_rows"]
        quality = _public_quality(quality_raw)

        export_path: str | None = None
        if write_export:
            emit("writing export")
            out_path = (
                Path(self._export_dir) / export_filename
                if export_filename
                else default_export_path(self._export_dir)
            )
            params = {
                **self.config.collection.export_params(),
                "use_crawl4ai": self.config.use_crawl4ai,
                "collection_mode": collected["collection_mode"],
                "sort_applied": collected["sort_meta"].get("sort_applied", False),
                "sort_method": collected["sort_meta"].get("sort_method", "none"),
                "urls_resolved_count": collected["urls_resolved_count"],
                "browser_fallback_used": collected["browser_fallback_used"],
            }
            written = write_feed_export(
                out_path,
                posts=posts,
                valid_posts=valid_posts,
                quality=quality,
                params=params,
            )
            export_path = str(written)

        elapsed_seconds = time.perf_counter() - start
        elapsed_pretty = _format_duration(elapsed_seconds)
        ok = quality["valid_post_url_count"] > 0
        crawl_result: CrawlFeedResult | None = collected.get("crawl_result")

        logger.info(
            "LinkedInFeedPipeline run complete in %s — mode=%s posts=%d valid=%d",
            elapsed_pretty,
            collected["collection_mode"],
            len(posts),
            quality["valid_post_url_count"],
        )

        return {
            "ok": ok,
            "posts": posts,
            "valid_posts": valid_posts,
            "quality": quality,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_pretty": elapsed_pretty,
            "export_path": export_path,
            "crawl4ai_success": crawl_result.crawl4ai_success if crawl_result else None,
            "error_message": crawl_result.error_message if crawl_result else None,
            "urls_resolved_count": collected["urls_resolved_count"],
            "resolve_missing_urls": self.config.collection.resolve_missing_urls,
            "use_crawl4ai": self.config.use_crawl4ai,
            "collection_mode": collected["collection_mode"],
            "sort_applied": collected["sort_meta"].get("sort_applied", False),
            "sort_method": collected["sort_meta"].get("sort_method", "none"),
            "browser_fallback_used": collected["browser_fallback_used"],
        }

    async def _collect(self, emit: Callable[[str], None]) -> dict[str, Any]:
        """Phase 2: Crawl4AI and/or live browser collection."""
        if self.config.use_crawl4ai:
            return await self._collect_crawl4ai(emit)
        emit("collecting feed")
        result = await self._collect_live()
        urls_resolved_count = sum(
            1 for p in result.posts if (p.get("post_url") or "").strip()
        )
        return {
            "posts": result.posts,
            "sort_meta": result.sort_meta,
            "collection_mode": "live",
            "urls_resolved_count": urls_resolved_count,
            "browser_fallback_used": False,
            "crawl_result": None,
        }

    async def _collect_crawl4ai(self, emit: Callable[[str], None]) -> dict[str, Any]:
        emit("collecting feed (crawl4ai)")
        crawl_result = await crawl_linkedin_feed(self.config.collection)
        posts: list[dict[str, Any]] = list(crawl_result.posts)
        collection_mode = "crawl4ai"
        sort_meta: dict[str, Any] = {"sort_applied": False, "sort_method": "none"}
        urls_resolved_count = 0
        browser_fallback_used = False

        if self.config.collection.resolve_missing_urls:
            if not posts:
                emit("collecting feed live")
                live = await self._collect_live()
                posts = live.posts
                sort_meta = live.sort_meta
                collection_mode = "crawl4ai+live_fallback"
                browser_fallback_used = True
                urls_resolved_count = sum(
                    1 for p in posts if (p.get("post_url") or "").strip()
                )
            else:
                urls_resolved_count = await self._resolve_missing_post_urls(
                    posts, emit
                )

        return {
            "posts": posts,
            "sort_meta": sort_meta,
            "collection_mode": collection_mode,
            "urls_resolved_count": urls_resolved_count,
            "browser_fallback_used": browser_fallback_used,
            "crawl_result": crawl_result,
        }

    async def _collect_live(self) -> FeedCollectionResult:
        """One locked browser session: prepare → LiveFeedCollector.run."""
        async with LinkedInBrowserSession(self.config.collection.profile) as session:
            return await LiveFeedCollector(self.config.collection).run(session.browser)

    async def _resolve_missing_post_urls(
        self,
        posts: list[dict[str, Any]],
        emit: Callable[[str], None],
    ) -> int:
        """Fill missing ``post_url`` via Send → Copy link (Crawl4AI partial rows only)."""
        candidates = [
            p
            for p in posts
            if not (p.get("post_url") or "").strip()
            and (p.get("author_profile_url") or "").strip()
        ]
        if not candidates:
            return 0

        before = sum(1 for p in posts if (p.get("post_url") or "").strip())
        emit("resolving post urls")
        async with LinkedInBrowserSession(self.config.collection.profile) as session:
            await prepare_live_feed_page(
                session.browser,
                sort_js_path=self.config.collection.sort_js_path,
                post_card_selector=self.config.collection.selectors.post_card_selector,
            )
            await resolve_post_urls_via_send_button(session.browser, posts)

        after = sum(1 for p in posts if (p.get("post_url") or "").strip())
        resolved = after - before
        if resolved:
            logger.info(
                "Resolved %d post URL(s) via Send → Copy link", resolved
            )
        return resolved


__all__ = [
    "LinkedInFeedPipeline",
    "LinkedInFeedPipelineConfig",
    "default_export_path",
    "write_feed_export",
]
