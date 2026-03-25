from __future__ import annotations

"""
Knowledge Base File Service
Manages the lifecycle (metadata, CRUD, scoping) of files in the KB directory.

Components
----------
KnowledgeItemRecord    – on-disk metadata for a single KB file
KnowledgeManifest      – per-scope metadata manifest stored in Redis
FileInfo               – API response model (manifest data + scope context)
KnowledgeMetadataStore – Redis-backed reads/writes of KnowledgeManifest
KnowledgeRegistry  – manages per-scope knowledge items, routes multi-tier search
KnowledgeFileService – file CRUD, delegates metadata to KnowledgeMetadataStore

Redis key schema
----------------
  kb:manifest:global
  kb:manifest:category:{category_id}
  kb:manifest:session:{session_id}

Each key holds a JSON string of KnowledgeManifest.model_dump_json().
Manifest ``items`` keys are **content_hash** (SHA-256 hex), not display filenames.
"""

import hashlib
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import redis
from pydantic import BaseModel

from src.config.settings import settings
from src.service.file_parser import ParserFactory
from src.service.knowledge_base import (
    KnowledgeBaseService,
    KnowledgeBaseServiceMultiFormat,
)
from src.type.enums import ContentFormat, ContextPersistence, ContextScope

logger = logging.getLogger(__name__)

_parser_factory = ParserFactory()

# Under KB_DIRECTORY (e.g. ``.knowledge_sources/_kb_context/``): scoped upload roots.
KB_CONTEXT_SCOPED_SUBDIR = "_kb_context"
# Under each scope directory (e.g. ``…/_kb_context/_content_blobs/<sha>.pdf``): CAS blob prefix in storage_relpath.
KB_CONTENT_BLOB_SUBDIR = "_content_blobs"

# Pre-rename on-disk layout (read/unlink compatibility only).
_LEGACY_CONTEXT_SCOPED_SUBDIR = "_objects"
_LEGACY_CONTENT_BLOB_SUBDIR = "_objects"


# Data models
_EXTENSION_FORMAT: dict[str, ContentFormat] = {
    ".md": ContentFormat.MARKDOWN,
    ".pdf": ContentFormat.PDF,
    ".txt": ContentFormat.TXT,
}
_ALLOWED_SUFFIXES: frozenset[str] = frozenset(_EXTENSION_FORMAT)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_SHA256_HEX_LEN = 64


def _is_sha256_hex_key(s: str) -> bool:
    if len(s) != _SHA256_HEX_LEN:
        return False
    return all(c in "0123456789abcdef" for c in s.lower())


def _normalize_manifest_item_keys(manifest: KnowledgeManifest) -> bool:
    """
    Migrate Redis manifest ``items`` from legacy filename keys to content_hash keys.
    Returns True if the manifest was mutated (caller may want to save).
    """
    if not manifest.items:
        return False
    if all(_is_sha256_hex_key(k) for k in manifest.items):
        return False

    new_items: dict[str, KnowledgeItemRecord] = {}
    for _old_key, rec in manifest.items.items():
        nk = rec.content_hash.lower() if _is_sha256_hex_key(rec.content_hash) else None
        if nk is None:
            logger.warning(
                "Skipping manifest row with non-hash content_hash during migration; "
                "key was %r filename=%r",
                _old_key,
                rec.filename,
            )
            continue
        if nk in new_items:
            cur = new_items[nk]
            if rec.updated_at > cur.updated_at:
                new_items[nk] = rec
        else:
            new_items[nk] = rec

    manifest.items = new_items
    manifest.version = "2"
    return True


class KnowledgeItemRecord(BaseModel):
    """Persistent metadata for a single knowledge file."""
    filename: str
    persistence: ContextPersistence = ContextPersistence.PERSISTENT
    format: ContentFormat
    size_bytes: int
    content_hash: str           # SHA-256 of file contents at write time
    created_at: str             # ISO 8601 UTC
    updated_at: str             # ISO 8601 UTC
    title: str | None = None
    description: str | None = None
    tags: list[str] = []
    # Path relative to scope dir, e.g. "_content_blobs/{sha256}.md". None = legacy (filename is the on-disk name).
    storage_relpath: str | None = None


class KnowledgeManifest(BaseModel):
    """
    Per-scope metadata manifest.

    One manifest lives in Redis per scope key. All items in a scope share the
    same scope/category_id/session_id context at the manifest level.
    """
    version: str = "1"
    scope: ContextScope
    category_id: str | None = None
    session_id: str | None = None
    # keyed by content_hash (SHA-256 hex)
    items: dict[str, KnowledgeItemRecord] = {}


def _effective_storage_relpath(record: KnowledgeItemRecord) -> str:
    """Canonical storage key for refcounting (legacy rows use filename)."""
    if record.storage_relpath:
        return record.storage_relpath
    return record.filename


def _storage_relpath_for_hash(content_hash: str, suffix: str) -> str:
    return f"{KB_CONTENT_BLOB_SUBDIR}/{content_hash}{suffix}"


def _legacy_scope_directory_parallel(primary: Path, kb_root: Path) -> Path | None:
    """
    If *primary* uses {KB_CONTEXT_SCOPED_SUBDIR}, return the same relative path
    under _LEGACY_CONTEXT_SCOPED_SUBDIR for deployments that have not renamed
    the scoped tree yet.
    """
    try:
        tail = primary.relative_to(kb_root)
    except ValueError:
        return None
    if not tail.parts or tail.parts[0] != KB_CONTEXT_SCOPED_SUBDIR:
        return None
    rest = tail.parts[1:]
    if not rest:
        return kb_root / _LEGACY_CONTEXT_SCOPED_SUBDIR
    return kb_root / _LEGACY_CONTEXT_SCOPED_SUBDIR / Path(*rest)


def _iter_scope_directories(kb_root: Path, primary_scope_dir: Path) -> tuple[Path, ...]:
    out: list[Path] = [primary_scope_dir]
    leg = _legacy_scope_directory_parallel(primary_scope_dir, kb_root)
    if leg is not None and leg.resolve() != primary_scope_dir.resolve():
        out.append(leg)
    return tuple(out)


def _is_global_scope_directory(scope_dir: Path, kb_root: Path) -> bool:
    """True when *scope_dir* is the global context root (``{KB_CONTEXT_SCOPED_SUBDIR}`` under *kb_root*)."""
    try:
        return scope_dir.resolve() == (kb_root / KB_CONTEXT_SCOPED_SUBDIR).resolve()
    except OSError:
        return False


def _legacy_global_flat_path(
    kb_root: Path,
    scope_dir: Path,
    record: KnowledgeItemRecord,
    rel: str,
) -> Path | None:
    """
    Bootstrap-global registered files directly under ``KB_DIRECTORY`` with no
    ``storage_relpath``; they are not under ``{KB_CONTEXT_SCOPED_SUBDIR}/``.
    """
    if record.storage_relpath is not None:
        return None
    if not _is_global_scope_directory(scope_dir, kb_root):
        return None
    if rel != record.filename or len(Path(rel).parts) != 1:
        return None
    if rel in (".", ".."):
        return None
    p = kb_root / rel
    return p if p.is_file() else None


def _count_storage_refs(manifest: KnowledgeManifest, relpath: str) -> int:
    return sum(
        1 for r in manifest.items.values()
        if _effective_storage_relpath(r) == relpath
    )


_CAS_SUBDIR_NAMES = frozenset(
    {KB_CONTENT_BLOB_SUBDIR, _LEGACY_CONTENT_BLOB_SUBDIR})


def _bootstrap_relative_skips_cas(rel: Path) -> bool:
    return bool(_CAS_SUBDIR_NAMES.intersection(rel.parts))


def _iter_bootstrap_file_paths(directory: Path, *, recursive: bool) -> list[Path]:
    if not directory.exists():
        return []
    if not recursive:
        return sorted(p for p in directory.iterdir() if p.is_file())
    out: list[Path] = []
    for p in sorted(directory.rglob("*"), key=lambda x: str(x).lower()):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(directory)
        except ValueError:
            continue
        if _bootstrap_relative_skips_cas(rel):
            continue
        out.append(p)
    return out


def _bootstrap_logical_filename(directory: Path, file_path: Path) -> str:
    if file_path.parent == directory:
        return file_path.name
    return str(file_path.relative_to(directory)).replace("\\", "/")


def _candidate_blob_paths(scope_dir: Path, relpath: str) -> tuple[Path, ...]:
    """
    Ordered paths to try for a manifest storage_relpath under *scope_dir*.

    Includes: primary ``scope_dir / relpath``, legacy duplicate segment layout
    (old bug: ``scope_dir / _objects / _objects / …``), and very old blobs stored
    flat in the scope dir while the manifest still used a blob-prefix relpath.
    """
    ordered: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            ordered.append(p)

    primary = scope_dir / relpath
    add(primary)

    for blob_seg in (_LEGACY_CONTENT_BLOB_SUBDIR, KB_CONTENT_BLOB_SUBDIR):
        prefix = f"{blob_seg}/"
        if relpath.startswith(prefix):
            dup = scope_dir / blob_seg / relpath
            try:
                if dup.resolve() != primary.resolve():
                    add(dup)
            except OSError:
                add(dup)
            rest = relpath[len(prefix):]
            if rest:
                add(scope_dir / rest)

    return tuple(ordered)


def _physical_path(
    scope_dir: Path,
    record: KnowledgeItemRecord,
    *,
    kb_root: Path,
) -> Path:
    rel = record.storage_relpath or record.filename
    for sd in _iter_scope_directories(kb_root, scope_dir):
        for candidate in _candidate_blob_paths(sd, rel):
            if candidate.is_file():
                return candidate
    legacy_flat = _legacy_global_flat_path(kb_root, scope_dir, record, rel)
    if legacy_flat is not None:
        return legacy_flat
    return scope_dir / rel


class FileInfo(BaseModel):
    """API response model — manifest record expanded with scope context."""
    filename: str
    scope: ContextScope
    persistence: ContextPersistence
    format: ContentFormat
    size_bytes: int        # live from filesystem
    content_hash: str
    created_at: str
    updated_at: str
    category_id: str | None = None
    session_id: str | None = None
    title: str | None = None
    description: str | None = None
    tags: list[str] = []


# KnowledgeMetadataStore (Redis-backed)
class KnowledgeMetadataStore:
    """
    Stores KnowledgeManifest records in Redis as JSON strings.

    Key format:  kb:manifest:{scope_key}
    Value format: KnowledgeManifest.model_dump_json()

    The scope_key is produced by KnowledgeBaseRegistry.scope_key() and matches:
      "global"             → kb:manifest:global
      "category:{id}"      → kb:manifest:category:{id}
      "session:{id}"       → kb:manifest:session:{id}
    """

    KEY_PREFIX = "kb:manifest:"

    def __init__(self, redis_client: redis.Redis) -> None:
        self._r = redis_client

    def _key(self, scope_key: str) -> str:
        return f"{self.KEY_PREFIX}{scope_key}"

    def load(
        self,
        scope_key: str,
        scope: ContextScope,
        category_id: str | None = None,
        session_id: str | None = None,
    ) -> KnowledgeManifest:
        """Load manifest from Redis; return an empty default if the key is absent."""
        raw = self._r.get(self._key(scope_key))
        if raw:
            try:
                manifest = KnowledgeManifest.model_validate_json(raw)
                if _normalize_manifest_item_keys(manifest):
                    self.save(scope_key, manifest)
                return manifest
            except Exception as e:
                logger.warning(
                    "Corrupt manifest for '%s' (%s); resetting.", scope_key, e)
        return KnowledgeManifest(scope=scope, category_id=category_id, session_id=session_id)

    def save(self, scope_key: str, manifest: KnowledgeManifest) -> None:
        """Persist manifest to Redis."""
        self._r.set(self._key(scope_key), manifest.model_dump_json())
        logger.debug("Manifest saved → Redis key '%s'", self._key(scope_key))

    def delete(self, scope_key: str) -> None:
        """Remove a manifest key (e.g. when all files in a scope are deleted)."""
        self._r.delete(self._key(scope_key))
        logger.debug("Manifest deleted → Redis key '%s'", self._key(scope_key))

    def bootstrap(
        self,
        scope_key: str,
        directory: Path,
        scope: ContextScope,
        category_id: str | None = None,
        session_id: str | None = None,
        *,
        recursive: bool = False,
    ) -> KnowledgeManifest:
        """
        Load (or create) the manifest and auto-register any untracked files
        found on disk with default metadata (persistent, format inferred from
        extension). Saves back to Redis if any new entries were added.

        *recursive* **False**: only regular files **directly under** *directory*
        (``Path.iterdir``, not subfolders).

        *recursive* **True**: all files under *directory* via ``rglob``, except
        any path whose relative path contains a CAS segment (skipped:
        ``_content_blobs/`` or legacy ``_objects/`` as a path component). Nested
        files are stored with a slash-separated logical ``filename`` (e.g.
        ``notes/doc.md``).
        """
        if not directory.exists():
            return self.load(scope_key, scope, category_id, session_id)

        manifest = self.load(scope_key, scope, category_id, session_id)
        changed = False

        for file_path in _iter_bootstrap_file_paths(directory, recursive=recursive):
            try:
                rel = file_path.relative_to(directory)
            except ValueError:
                continue
            if not recursive and _bootstrap_relative_skips_cas(rel):
                continue
            suffix = file_path.suffix.lower()
            if suffix not in _ALLOWED_SUFFIXES:
                continue
            stat = file_path.stat()
            content_hash = _sha256(file_path.read_bytes())
            if content_hash in manifest.items:
                continue

            logical_name = _bootstrap_logical_filename(directory, file_path)
            ts = datetime.fromtimestamp(
                stat.st_ctime, tz=timezone.utc).isoformat(timespec="seconds")
            manifest.items[content_hash] = KnowledgeItemRecord(
                filename=logical_name,
                persistence=ContextPersistence.PERSISTENT,
                format=_EXTENSION_FORMAT[suffix],
                size_bytes=stat.st_size,
                content_hash=content_hash,
                created_at=ts,
                updated_at=ts,
            )
            changed = True
            logger.debug(
                "Bootstrap: registered '%s' (hash=%s) in Redis manifest '%s'",
                logical_name,
                content_hash[:12],
                scope_key,
            )

        if changed:
            self.save(scope_key, manifest)

        return manifest


# KnowledgeRegistry
class KnowledgeRegistry:
    """
    Manages one global knowledge item (pre-loaded at startup) and lazily-created
    scoped knowledge items (per category or per session).

    Scope → key → on-disk roots (dirname ``_kb_context`` = ``KB_CONTEXT_SCOPED_SUBDIR``):

    GLOBAL   → "global"         → ``{KB_DIRECTORY}/_kb_context/``
    CATEGORY → "category:{id}"  → ``{KB_DIRECTORY}/_kb_context/category/{id}/``  (``default`` if missing)
    LOCAL    → "session:{id}"   → ``{KB_DIRECTORY}/_kb_context/session/{id}/``  (``anonymous`` if missing)
    """

    def __init__(self, global_kb: KnowledgeBaseService) -> None:
        self._global = global_kb
        self._scoped: dict[str, KnowledgeBaseService] = {}
        self._lock = threading.Lock()
        self._file_service: KnowledgeFileService | None = None

    def attach_file_service(self, file_service: KnowledgeFileService) -> None:
        """Wire back-reference so KB reloads can use manifest-driven file lists."""
        self._file_service = file_service

    @staticmethod
    def parse_scope_key(key: str) -> tuple[ContextScope, str | None, str | None]:
        if key == "global":
            return ContextScope.GLOBAL, None, None
        if key.startswith("category:"):
            return ContextScope.CATEGORY, None, key.split(":", 1)[1]
        if key.startswith("session:"):
            return ContextScope.LOCAL, key.split(":", 1)[1], None
        return ContextScope.GLOBAL, None, None

    # Static helpers
    @staticmethod
    def scope_key(
        scope: ContextScope,
        session_id: str | None,
        category_id: str | None,
    ) -> str:
        if scope == ContextScope.GLOBAL:
            return "global"
        if scope == ContextScope.CATEGORY:
            return f"category:{category_id or session_id or 'default'}"
        return f"session:{session_id or 'anonymous'}"

    @staticmethod
    def scope_directory(
        kb_root: str,
        scope: ContextScope,
        session_id: str | None,
        category_id: str | None,
    ) -> Path:
        root = Path(kb_root) / KB_CONTEXT_SCOPED_SUBDIR
        if scope == ContextScope.GLOBAL:
            return root
        if scope == ContextScope.CATEGORY:
            return root / "category" / (category_id or "default")
        return root / "session" / (session_id or "anonymous")

    # Lazy creation / invalidation
    def get_or_create(self, key: str, directory: str) -> KnowledgeBaseService | None:
        """
        Return cached KB instance for *key*, creating one if needed.
        Returns None when the directory is empty or missing.
        """
        with self._lock:
            if key not in self._scoped:
                safe_prefix = key.replace(":", "_")
                file_sources = None
                if self._file_service is not None:
                    scope, session_id, category_id = KnowledgeRegistry.parse_scope_key(
                        key)
                    file_sources = self._file_service.iter_kb_sources(
                        scope, session_id, category_id)
                try:
                    instance = KnowledgeBaseServiceMultiFormat(
                        kb_directory=directory,
                        cache_prefix=safe_prefix,
                        file_sources=file_sources,
                    )
                    self._scoped[key] = instance
                except (FileNotFoundError, ValueError) as e:
                    logger.debug("No scoped KB for '%s': %s", key, e)
                    return None
            return self._scoped[key]

    def invalidate(self, key: str) -> None:
        """Evict scoped instance so the next search reloads from disk."""
        with self._lock:
            self._scoped.pop(key, None)
        logger.debug("Registry: invalidated scoped KB '%s'", key)

    def invalidate_global(self) -> None:
        """Replace the global KB instance (called after a GLOBAL file upload)."""
        with self._lock:
            file_sources = None
            if self._file_service is not None:
                file_sources = self._file_service.iter_kb_sources(
                    ContextScope.GLOBAL, None, None)
            self._global = KnowledgeBaseServiceMultiFormat(
                kb_directory=str(settings.KB_DIRECTORY),
                cache_prefix="",
                file_sources=file_sources,
            )
        logger.info("Registry: global KB reloaded")

    # Multi-tier search

    def search(
        self,
        query: str,
        session_id: str | None,
        context_scope: ContextScope,
        category_id: str | None,
        top_k: int = settings.DEFAULT_TOP_K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Search across applicable KB tiers and return deduplicated results.

        Always searches GLOBAL. CATEGORY and LOCAL tiers are added based on
        *context_scope* and whether the relevant IDs are provided.
        """
        results_map: dict[tuple, dict] = {}

        def _merge(new_results: list[dict]) -> None:
            for r in new_results:
                k = (r["source_file"], r["chunk_index"])
                if k not in results_map or r["similarity_score"] > results_map[k]["similarity_score"]:
                    results_map[k] = r

        # Tier 1: GLOBAL (always)
        _merge(self._global.search(query, top_k=top_k,
               similarity_threshold=similarity_threshold))

        # Tier 2: CATEGORY
        if category_id and context_scope in (ContextScope.CATEGORY, ContextScope.LOCAL):
            cat_key = self.scope_key(
                ContextScope.CATEGORY, session_id, category_id)
            cat_dir = self.scope_directory(
                settings.KB_DIRECTORY, ContextScope.CATEGORY, session_id, category_id)
            cat_kb = self.get_or_create(cat_key, str(cat_dir))
            if cat_kb:
                _merge(cat_kb.search(query, top_k=top_k,
                       similarity_threshold=similarity_threshold))

        # Tier 3: LOCAL / SESSION
        if context_scope == ContextScope.LOCAL and session_id:
            sess_key = self.scope_key(
                ContextScope.LOCAL, session_id, category_id)
            sess_dir = self.scope_directory(
                settings.KB_DIRECTORY, ContextScope.LOCAL, session_id, category_id)
            sess_kb = self.get_or_create(sess_key, str(sess_dir))
            if sess_kb:
                _merge(sess_kb.search(query, top_k=top_k,
                       similarity_threshold=similarity_threshold))

        merged = sorted(results_map.values(),
                        key=lambda r: r["similarity_score"], reverse=True)
        return merged[:top_k]


# KnowledgeFileService
class KnowledgeFileService:
    """
    File CRUD for the knowledge item.

    Manages files in scoped subdirectories and keeps Redis manifests in sync.
    Delegates knowledge item instance lifecycle to KnowledgeRegistry.
    """

    ALLOWED_SUFFIXES: frozenset[str] = _ALLOWED_SUFFIXES

    def __init__(
        self,
        kb_directory: str,
        registry: KnowledgeRegistry,
        redis_client: redis.Redis,
    ) -> None:
        self._kb_root = Path(kb_directory)
        self._registry = registry
        self._metadata = KnowledgeMetadataStore(redis_client)

    def rebind_registry(self, registry: KnowledgeRegistry) -> None:
        """Replace registry (e.g. swap stub for real instance after startup wiring)."""
        self._registry = registry

    # if the blob has no references, delete it physically
    def _unlink_storage_if_orphaned(
        self,
        scope_dir: Path,
        manifest: KnowledgeManifest,
        relpath: str,
        *,
        orphan_record: KnowledgeItemRecord | None = None,
    ) -> None:
        if _count_storage_refs(manifest, relpath) > 0:
            return
        for sd in _iter_scope_directories(self._kb_root, scope_dir):
            for blob in _candidate_blob_paths(sd, relpath):
                try:
                    if blob.is_file():
                        blob.unlink()
                        return
                except OSError as e:
                    logger.warning(
                        "Could not remove orphan blob %s: %s", blob, e)
        if (
            orphan_record is not None
            and orphan_record.storage_relpath is None
            and _is_global_scope_directory(scope_dir, self._kb_root)
        ):
            name = orphan_record.filename
            if len(Path(name).parts) == 1 and name not in (".", ".."):
                p = self._kb_root / name
                try:
                    if p.is_file():
                        p.unlink()
                except OSError as e:
                    logger.warning(
                        "Could not remove legacy global flat file %s: %s", p, e
                    )

    def iter_kb_sources(
        self,
        scope: ContextScope,
        session_id: str | None = None,
        category_id: str | None = None,
    ) -> list[tuple[str, Path]]:
        """
        (logical_filename, absolute_path) for manifest entries with a readable
        file on disk and a supported extension. Used to build KB embeddings.
        """
        scope_dir = self._scope_directory(scope, session_id, category_id)
        sk = KnowledgeRegistry.scope_key(scope, session_id, category_id)
        manifest = self._metadata.load(sk, scope, category_id, session_id)
        out: list[tuple[str, Path]] = []
        for content_hash_key, record in sorted(manifest.items.items()):
            path = _physical_path(scope_dir, record, kb_root=self._kb_root)
            if not path.is_file():
                logger.warning(
                    "KB manifest entry %s missing on disk: %s",
                    content_hash_key[:12],
                    path,
                )
                continue
            suffix = path.suffix.lower()
            if suffix not in self.ALLOWED_SUFFIXES:
                logger.warning(
                    "KB manifest entry %s has unsupported suffix: %s",
                    content_hash_key[:12],
                    path,
                )
                continue
            expected_fmt = _EXTENSION_FORMAT.get(suffix)
            if expected_fmt is not None and record.format != expected_fmt:
                logger.warning(
                    "KB manifest entry %s format mismatch (record=%s, file=%s)",
                    content_hash_key[:12],
                    record.format,
                    expected_fmt,
                )
                continue
            out.append((record.filename, path.resolve()))
        return out

    # Helpers
    def _scope_directory(
        self,
        scope: ContextScope,
        session_id: str | None,
        category_id: str | None,
    ) -> Path:
        # Scope root: ``{kb_root}/_kb_context/`` or ``…/_kb_context/category/{id}/`` or ``…/_kb_context/session/{id}/``.
        # New uploads set ``storage_relpath`` to ``_content_blobs/<hash>.<ext>`` under that root.
        return KnowledgeRegistry.scope_directory(
            str(self._kb_root), scope, session_id, category_id
        )

    def _to_file_info(
        self,
        record: KnowledgeItemRecord,
        file_path: Path,
        scope: ContextScope,
        category_id: str | None,
        session_id: str | None,
    ) -> FileInfo:
        try:
            live_size = file_path.stat().st_size
        except OSError:
            live_size = record.size_bytes

        return FileInfo(
            filename=record.filename,
            scope=scope,
            persistence=record.persistence,
            format=record.format,
            size_bytes=live_size,
            content_hash=record.content_hash,
            created_at=record.created_at,
            updated_at=record.updated_at,
            category_id=category_id,
            session_id=session_id,
            title=record.title,
            description=record.description,
            tags=record.tags,
        )

    def _sync_manifest_from_disk(
        self,
        scope: ContextScope,
        session_id: str | None,
        category_id: str | None,
    ) -> None:
        """
        Register untracked files from disk into the Redis manifest for *scope*
        (same Redis key as ``KnowledgeMetadataStore.load`` / ``kb:manifest:{scope_key}``).

        **GLOBAL** (``scope_key`` ``global``) — two passes, both updating the same manifest:

        1. ``bootstrap(..., directory={KB_DIRECTORY}, recursive=False)`` — only files
           sitting **directly** under ``{KB_DIRECTORY}/`` (e.g. ``…/foo.pdf``), not
           inside subdirectories such as ``_kb_context/``.
        2. If ``{KB_DIRECTORY}/_kb_context/`` exists:
           ``bootstrap(..., directory={KB_DIRECTORY}/_kb_context/, recursive=True)`` —
           every file under that tree **except** paths under ``_content_blobs/`` or
           ``_objects/`` (CAS blob dirs are manifest-driven, not auto-imported).

        **CATEGORY** — ``bootstrap(..., directory={KB_DIRECTORY}/_kb_context/category/{id}/, recursive=True)``
        with the same CAS-dir skips.

        **LOCAL** — ``bootstrap(..., directory={KB_DIRECTORY}/_kb_context/session/{id}/, recursive=True)``
        with the same CAS-dir skips.

        Here ``{KB_DIRECTORY}`` is ``KnowledgeFileService._kb_root`` (settings ``KB_DIRECTORY``).
        """
        sk = KnowledgeRegistry.scope_key(scope, session_id, category_id)
        if scope == ContextScope.GLOBAL:
            self._metadata.bootstrap(sk, self._kb_root, ContextScope.GLOBAL)
            ctx = self._scope_directory(
                ContextScope.GLOBAL, None, None)
            if ctx.exists():
                self._metadata.bootstrap(
                    sk, ctx, ContextScope.GLOBAL, recursive=True)
            logger.warning(
                f"Context directory - {scope} scanned -> {ctx}")
            return
        d = self._scope_directory(scope, session_id, category_id)
        self._metadata.bootstrap(
            sk, d, scope, category_id, session_id, recursive=True)
        logger.warning(
            f"Scope directory - {scope} scanned -> {d}")

    # Startup bootstrap
    def bootstrap_global(self) -> None:
        """
        Sync global manifest from disk: see ``_sync_manifest_from_disk`` for
        exact roots (flat ``{KB_DIRECTORY}/*.{md,pdf,txt}`` then recursive
        ``{KB_DIRECTORY}/_kb_context/**`` excluding CAS subtrees). Called once
        at application startup.
        """
        self._sync_manifest_from_disk(ContextScope.GLOBAL, None, None)
        scope_key = KnowledgeRegistry.scope_key(
            ContextScope.GLOBAL, None, None)
        logger.info(
            "KB metadata bootstrap complete for global scope (Redis key: kb:manifest:%s)", scope_key)

    # CRUD
    def list_files(
        self,
        scope: ContextScope,
        session_id: str | None = None,
        category_id: str | None = None,
        include_higher_scopes: bool = False,
    ) -> list[FileInfo]:
        """
        List files for the requested scope.

        When *include_higher_scopes* is True, also returns files from higher
        scopes (e.g. LOCAL shows GLOBAL + CATEGORY + SESSION).
        """
        results: list[FileInfo] = []

        def _add_from(s: ContextScope, sid: str | None, cid: str | None) -> None:
            self._sync_manifest_from_disk(s, sid, cid)
            directory = self._scope_directory(s, sid, cid)
            sk = KnowledgeRegistry.scope_key(s, sid, cid)
            manifest = self._metadata.load(sk, s, cid, sid)
            for record in manifest.items.values():
                file_path = _physical_path(
                    directory, record, kb_root=self._kb_root)
                results.append(self._to_file_info(
                    record, file_path, s, cid, sid))

        if include_higher_scopes or scope == ContextScope.GLOBAL:
            _add_from(ContextScope.GLOBAL, None, None)

        if (include_higher_scopes and category_id) or scope == ContextScope.CATEGORY:
            _add_from(ContextScope.CATEGORY, session_id, category_id)

        if scope == ContextScope.LOCAL:
            _add_from(ContextScope.LOCAL, session_id, category_id)

        return results

    def save_file(
        self,
        filename: str,
        content: bytes,
        scope: ContextScope,
        persistence: ContextPersistence,
        session_id: str | None = None,
        category_id: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> FileInfo:
        """
        Write *content* to the appropriate scoped directory, update the Redis
        manifest, and invalidate the KB registry entry for that scope.

        Raises
        ------
        ValueError  — unsupported file extension
        """
        suffix = Path(filename).suffix.lower()
        if suffix not in self.ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported file type: '{suffix}'")

        target_dir = self._scope_directory(scope, session_id, category_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        content_hash = _sha256(content)
        storage_relpath = _storage_relpath_for_hash(content_hash, suffix)
        blob_path = target_dir / storage_relpath
        blob_path.parent.mkdir(parents=True, exist_ok=True)

        now = _now_iso()
        scope_key = KnowledgeRegistry.scope_key(
            scope, session_id, category_id)
        manifest = self._metadata.load(
            scope_key, scope, category_id, session_id)

        # Replace-by-display-name: drop other rows with same filename, different content
        for hk, rec in list(manifest.items.items()):
            if rec.filename == filename and hk != content_hash:
                del manifest.items[hk]
                self._unlink_storage_if_orphaned(
                    target_dir,
                    manifest,
                    _effective_storage_relpath(rec),
                    orphan_record=rec,
                )

        old_record = manifest.items.get(content_hash)
        old_relpath = (
            _effective_storage_relpath(old_record) if old_record else None
        )

        if blob_path.exists():
            if blob_path.stat().st_size != len(content):
                blob_path.write_bytes(content)
            elif _sha256(blob_path.read_bytes()) != content_hash:
                blob_path.write_bytes(content)
        else:
            blob_path.write_bytes(content)

        if content_hash in manifest.items:
            existing = manifest.items[content_hash]
            record = existing.model_copy(update={
                "filename": filename,
                "persistence": persistence,
                "size_bytes": len(content),
                "content_hash": content_hash,
                "storage_relpath": storage_relpath,
                "updated_at": now,
                "title": title if title is not None else existing.title,
                "description": description if description is not None else existing.description,
                "tags": tags if tags is not None else existing.tags,
            })
        else:
            record = KnowledgeItemRecord(
                filename=filename,
                persistence=persistence,
                format=_EXTENSION_FORMAT[suffix],
                size_bytes=len(content),
                content_hash=content_hash,
                created_at=now,
                updated_at=now,
                title=title,
                description=description,
                tags=tags or [],
                storage_relpath=storage_relpath,
            )

        manifest.items[content_hash] = record
        self._metadata.save(scope_key, manifest)

        if old_relpath and old_relpath != storage_relpath:
            self._unlink_storage_if_orphaned(
                target_dir,
                manifest,
                old_relpath,
                orphan_record=old_record,
            )

        file_path = blob_path

        # Invalidate registry so next search reloads from disk
        if scope == ContextScope.GLOBAL:
            self._registry.invalidate_global()
        else:
            self._registry.invalidate(scope_key)

        return self._to_file_info(record, file_path, scope, category_id, session_id)

    def delete_file(
        self,
        content_hash: str,
        scope: ContextScope,
        session_id: str | None = None,
        category_id: str | None = None,
    ) -> bool:
        """
        Delete a file and remove it from the Redis manifest.

        *content_hash* is the manifest key (SHA-256 hex of file contents).

        Returns True if the file was found and deleted, False otherwise.
        """
        target_dir = self._scope_directory(scope, session_id, category_id)
        scope_key = KnowledgeRegistry.scope_key(
            scope, session_id, category_id)
        manifest = self._metadata.load(
            scope_key, scope, category_id, session_id)
        key = content_hash.lower()
        record = manifest.items.get(key)
        if record is None:
            return False

        relpath = _effective_storage_relpath(record)
        del manifest.items[key]
        self._metadata.save(scope_key, manifest)
        self._unlink_storage_if_orphaned(
            target_dir, manifest, relpath, orphan_record=record)

        if scope == ContextScope.GLOBAL:
            self._registry.invalidate_global()
        else:
            self._registry.invalidate(scope_key)

        return True

    def get_file_info(
        self,
        content_hash: str,
        scope: ContextScope,
        session_id: str | None = None,
        category_id: str | None = None,
    ) -> FileInfo | None:
        """Return FileInfo for *content_hash* (manifest key) in the given scope."""
        self._sync_manifest_from_disk(scope, session_id, category_id)
        target_dir = self._scope_directory(scope, session_id, category_id)
        scope_key = KnowledgeRegistry.scope_key(
            scope, session_id, category_id)
        manifest = self._metadata.load(
            scope_key, scope, category_id, session_id)
        key = content_hash.lower()
        record = manifest.items.get(key)
        if record is None:
            return None

        file_path = _physical_path(
            target_dir, record, kb_root=self._kb_root)
        if not file_path.is_file():
            return None

        return self._to_file_info(record, file_path, scope, category_id, session_id)

    def get_stats(
        self,
        scope: ContextScope | None = None,
        session_id: str | None = None,
        category_id: str | None = None,
    ) -> dict:
        """
        Return aggregate or per-scope statistics.

        When *scope* is None, aggregates all three scope directories.
        """
        total_files = 0
        total_bytes = 0
        ephemeral_count = 0
        by_scope: dict[str, dict] = {}

        scopes = (
            [ContextScope.GLOBAL, ContextScope.CATEGORY, ContextScope.LOCAL]
            if scope is None
            else [scope]
        )

        for s in scopes:
            self._sync_manifest_from_disk(s, session_id, category_id)
            sk = KnowledgeRegistry.scope_key(s, session_id, category_id)
            manifest = self._metadata.load(sk, s, category_id, session_id)
            file_count = len(manifest.items)
            scope_bytes = sum(r.size_bytes for r in manifest.items.values())
            scope_ephemeral = sum(
                1 for r in manifest.items.values()
                if r.persistence == ContextPersistence.EPHEMERAL
            )

            total_files += file_count
            total_bytes += scope_bytes
            ephemeral_count += scope_ephemeral
            by_scope[s.value] = {
                "file_count": file_count,
                "total_bytes": scope_bytes,
                "ephemeral_count": scope_ephemeral,
            }

        return {
            "total_files": total_files,
            "total_bytes": total_bytes,
            "ephemeral_file_count": ephemeral_count,
            "by_scope": by_scope,
        }

    def cleanup_ephemeral(self, session_id: str) -> int:
        """
        Delete all ephemeral files in the session directory and update the
        Redis manifest. Returns the number of files deleted.
        """
        scope_key = KnowledgeRegistry.scope_key(
            ContextScope.LOCAL, session_id, None)
        sess_dir = self._scope_directory(ContextScope.LOCAL, session_id, None)
        manifest = self._metadata.load(
            scope_key, ContextScope.LOCAL, None, session_id)

        to_delete = [
            hk for hk, rec in manifest.items.items()
            if rec.persistence == ContextPersistence.EPHEMERAL
        ]

        count = 0
        for hk in to_delete:
            rec = manifest.items[hk]
            relpath = _effective_storage_relpath(rec)
            del manifest.items[hk]
            self._unlink_storage_if_orphaned(
                sess_dir, manifest, relpath, orphan_record=rec)
            count += 1

        if count > 0:
            self._metadata.save(scope_key, manifest)
            self._registry.invalidate(scope_key)
            logger.info(
                "Cleaned up %d ephemeral file(s) for session '%s'", count, session_id)

        return count
