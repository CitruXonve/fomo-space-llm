"""Backward-compatible alias for :class:`LinkedInFeedPipeline`.

Prefer importing from :mod:`linkedin_feed_pipeline` directly. This module
re-exports the renamed pipeline class under the legacy ``LinkedInFeedAgent`` name.
"""

from __future__ import annotations

from src.utility.linkedin_feed_pipeline import (
    LinkedInFeedPipeline,
    LinkedInFeedPipelineConfig,
    default_export_path,
    write_feed_export,
)

LinkedInFeedAgent = LinkedInFeedPipeline

__all__ = [
    "LinkedInFeedAgent",
    "LinkedInFeedPipeline",
    "LinkedInFeedPipelineConfig",
    "default_export_path",
    "write_feed_export",
]
