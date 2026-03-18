import logging
from src.type.enums import ContextScope, ContextPersistence

logger = logging.getLogger(__name__)


class ContextStore:
    """
    In-memory store that accumulates KB contexts across conversation turns.

    Scoping determines which sessions share a context bucket:
      LOCAL    — one bucket per session_id
      CATEGORY — one bucket per category_id (multiple sessions may share it)
      GLOBAL   — a single shared bucket for all sessions

    Persistence controls what happens to the bucket on each update:
      EPHEMERAL  — replace the bucket with the current call's contexts only
      PERSISTENT — merge new contexts into the bucket (deduplicated)
    """

    def __init__(self, max_contexts_per_key: int = 40) -> None:
        self._store: dict[str, list[dict]] = {}
        self._max = max_contexts_per_key

    def get_scope_key(
        self,
        session_id: str | None,
        context_scope: ContextScope,
        category_id: str | None,
    ) -> str:
        """
        Derive the storage key from the scope configuration.

        LOCAL    → session_id (falls back to "anonymous" if absent)
        CATEGORY → category_id if provided, else session_id
        GLOBAL   → the literal string "global"
        """
        if context_scope == ContextScope.GLOBAL:
            return "global"
        if context_scope == ContextScope.CATEGORY:
            return category_id or session_id or "anonymous"
        # LOCAL (default)
        return session_id or "anonymous"

    def get(self, scope_key: str) -> list[dict]:
        """Return the accumulated contexts for *scope_key* (empty list if none)."""
        return list(self._store.get(scope_key, []))

    def update(
        self,
        scope_key: str,
        new_contexts: list[dict],
        persistence: ContextPersistence,
    ) -> list[dict]:
        """
        Update the context bucket and return the effective context list.

        EPHEMERAL  — Discard any previously accumulated contexts; store and
                     return only *new_contexts*.
        PERSISTENT — Merge *new_contexts* into the existing bucket.
                     Deduplication key: (source_file, chunk_index).
                     When a duplicate exists, keep the entry with the higher
                     similarity_score.  Trim to *max_contexts_per_key* by
                     descending similarity_score.
        """
        if persistence == ContextPersistence.EPHEMERAL:
            self._store[scope_key] = list(new_contexts)
            logger.debug(
                "ContextStore [%s] ephemeral: replaced with %d context(s)",
                scope_key,
                len(new_contexts),
            )
            return list(new_contexts)

        # PERSISTENT: merge
        existing = {
            (c["source_file"], c["chunk_index"]): c
            for c in self._store.get(scope_key, [])
        }

        for ctx in new_contexts:
            key = (ctx["source_file"], ctx["chunk_index"])
            # Update the existing context by similarity score comparison if higher
            if key not in existing or ctx["similarity_score"] > existing[key]["similarity_score"]:
                existing[key] = ctx

        # Reorder by similarity score in descending order and keep only the top N
        merged = sorted(existing.values(),
                        key=lambda c: c["similarity_score"], reverse=True)
        merged = merged[: self._max]

        self._store[scope_key] = merged
        logger.debug(
            "ContextStore [%s] persistent: %d unique context(s) (added %d new)",
            scope_key,
            len(merged),
            len(new_contexts),
        )
        return list(merged)

    def clear(self, scope_key: str) -> None:
        """Explicitly remove the context bucket for *scope_key*."""
        self._store.pop(scope_key, None)
        logger.debug("ContextStore [%s] cleared", scope_key)
