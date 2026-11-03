"""Configuration objects for LinkedIn feed collection.

End-state layering::

    LinkedInFeedPipeline(LinkedInFeedPipelineConfig)
      → FeedCollectionOptions (browser + scroll + DOM + collection flags)
      → LiveFeedCollector / crawl_linkedin_feed (collection backends)
      → FeedCollectionResult (posts + sort meta + strategy)

Callers build one :class:`LinkedInFeedPipelineConfig` (via
:meth:`LinkedInFeedPipelineConfig.from_kwargs` or explicit fields) instead of
threading a dozen keyword arguments into :class:`LinkedInFeedPipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from src.utility.linkedin_feed_parser import DEFAULT_POST_CARD_SELECTOR

_MAX_SCROLL_CAP = 30

CollectionStrategy = Literal["js", "html_scroll", "js+resolve"]


def scroll_budget(*, scroll_rounds: int, target: int) -> int:
    """Max scroll iterations allowed before giving up (0 = parse current viewport only)."""
    sr = int(scroll_rounds)
    if sr <= 0:
        return 0
    return min(max(sr, target), _MAX_SCROLL_CAP)


@dataclass(frozen=True)
class LinkedInBrowserProfile:
    """Chromium persistent-profile settings shared by live and Crawl4AI paths."""

    user_data_dir: Path
    profile_directory: str | None = "Default"
    headless: bool = True
    viewport: tuple[int, int] = (1280, 720)


@dataclass(frozen=True)
class FeedScrollPolicy:
    """Scroll-until-target parameters for feed collection loops."""

    max_posts: int = 25
    scroll_rounds: int = 10
    scroll_px: int = 800
    scroll_pause_s: float = 0.75
    plateau_rounds: int = 5
    max_scroll_cap: int = _MAX_SCROLL_CAP
    agent_max_posts: int | None = None

    def effective_target(self) -> int:
        """Cap ``max_posts`` when ``agent_max_posts`` is set (LangChain tool limit)."""
        cap = self.agent_max_posts if self.agent_max_posts is not None else self.max_posts
        return max(1, min(int(self.max_posts), int(cap)))


@dataclass(frozen=True)
class FeedDomSelectors:
    """DOM selectors for feed container, cards, and optional HTML snapshot scope."""

    container_selector: str = "#workspace"
    post_card_selector: str | None = DEFAULT_POST_CARD_SELECTOR
    html_selector: str | None = None


@dataclass(frozen=True)
class FeedCollectionOptions:
    """All knobs for one feed collection run (live browser or Crawl4AI)."""

    profile: LinkedInBrowserProfile
    scroll: FeedScrollPolicy = field(default_factory=FeedScrollPolicy)
    selectors: FeedDomSelectors = field(default_factory=FeedDomSelectors)
    sort_js_path: Path | None = None
    resolve_missing_urls: bool = True
    require_activity_id: bool = False

    @classmethod
    def from_pipeline_kwargs(
        cls,
        *,
        user_data_dir: str | Path,
        profile_directory: str | None = "Default",
        headless: bool = True,
        max_posts: int = 25,
        scroll_rounds: int = 10,
        container_selector: str = "#workspace",
        post_card_selector: str | None = DEFAULT_POST_CARD_SELECTOR,
        viewport: tuple[int, int] = (1280, 720),
        require_activity_id: bool = False,
        sort_js_path: str | Path | None = None,
        resolve_missing_urls: bool = True,
    ) -> FeedCollectionOptions:
        """Build options from :class:`LinkedInFeedPipeline` constructor kwargs."""
        sort_path = Path(sort_js_path) if sort_js_path is not None else None
        return cls(
            profile=LinkedInBrowserProfile(
                user_data_dir=Path(user_data_dir),
                profile_directory=profile_directory,
                headless=headless,
                viewport=viewport,
            ),
            scroll=FeedScrollPolicy(max_posts=max_posts, scroll_rounds=scroll_rounds),
            selectors=FeedDomSelectors(
                container_selector=container_selector,
                post_card_selector=post_card_selector,
            ),
            sort_js_path=sort_path,
            resolve_missing_urls=resolve_missing_urls,
            require_activity_id=require_activity_id,
        )

    def export_params(self) -> dict[str, Any]:
        """Serialize options for JSON export metadata."""
        return {
            "user_data_dir": str(self.profile.user_data_dir),
            "profile_directory": self.profile.profile_directory,
            "headless": self.profile.headless,
            "max_posts": self.scroll.max_posts,
            "scroll_rounds": self.scroll.scroll_rounds,
            "container_selector": self.selectors.container_selector,
            "post_card_selector": self.selectors.post_card_selector,
            "require_activity_id": self.require_activity_id,
            "resolve_missing_urls": self.resolve_missing_urls,
        }


@dataclass(frozen=True)
class LinkedInFeedPipelineConfig:
    """All knobs for :class:`LinkedInFeedPipeline` (collection + pipeline flags)."""

    collection: FeedCollectionOptions
    export_dir: str | Path | None = None
    login_timeout_s: int = 120
    check_auth: bool = True
    use_crawl4ai: bool = False

    @classmethod
    def from_kwargs(
        cls,
        *,
        user_data_dir: str | Path = ".browser_profile",
        profile_directory: str | None = "Default",
        headless: bool = True,
        max_posts: int = 25,
        scroll_rounds: int = 10,
        container_selector: str = "#workspace",
        post_card_selector: str | None = DEFAULT_POST_CARD_SELECTOR,
        viewport: tuple[int, int] = (1280, 720),
        export_dir: str | Path | None = None,
        require_activity_id: bool = False,
        login_timeout_s: int = 120,
        check_auth: bool = True,
        sort_js_path: str | Path | None = None,
        resolve_missing_urls: bool = True,
        use_crawl4ai: bool = False,
    ) -> Self:
        """Build pipeline config from flat keyword arguments (CLI / tests)."""
        return cls(
            collection=FeedCollectionOptions.from_pipeline_kwargs(
                user_data_dir=user_data_dir,
                profile_directory=profile_directory,
                headless=headless,
                max_posts=max_posts,
                scroll_rounds=scroll_rounds,
                container_selector=container_selector,
                post_card_selector=post_card_selector,
                viewport=viewport,
                require_activity_id=require_activity_id,
                sort_js_path=sort_js_path,
                resolve_missing_urls=resolve_missing_urls,
            ),
            export_dir=export_dir,
            login_timeout_s=login_timeout_s,
            check_auth=check_auth,
            use_crawl4ai=use_crawl4ai,
        )


@dataclass
class FeedCollectionResult:
    """Outcome of one live-browser collection session (prepare → collect → resolve)."""

    posts: list[dict[str, str]]
    sort_meta: dict[str, Any]
    collection_strategy: CollectionStrategy
    scroll_count: int = 0
    urls_resolved_count: int = 0


__all__ = [
    "CollectionStrategy",
    "FeedCollectionOptions",
    "FeedCollectionResult",
    "FeedDomSelectors",
    "FeedScrollPolicy",
    "LinkedInBrowserProfile",
    "LinkedInFeedPipelineConfig",
    "scroll_budget",
]
