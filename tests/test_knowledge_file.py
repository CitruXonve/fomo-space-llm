"""
Unit tests for src/service/knowledge_file_service.py

Mocking strategy
----------------
- redis.Redis  → unittest.mock.MagicMock (no live Redis required)
- KnowledgeRegistry  → MagicMock for file-service tests (avoids loading
  the sentence-transformers model); static methods are tested directly.
- Filesystem  → tempfile.TemporaryDirectory (real I/O, consistent with other
  tests in this project).
"""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.service.knowledge_file_service import (
    KnowledgeFileService,
    KnowledgeRegistry,
    KnowledgeItemRecord,
    KnowledgeManifest,
    KnowledgeMetadataStore,
    _now_iso,
    _sha256,
)
from src.type.enums import ContentFormat, ContextPersistence, ContextScope


# Helper factories
def _make_record(
    filename: str = "doc.md",
    persistence: ContextPersistence = ContextPersistence.PERSISTENT,
    fmt: ContentFormat = ContentFormat.MARKDOWN,
    size_bytes: int = 100,
    content_hash: str = "abc123",
    created_at: str = "2026-01-01T00:00:00+00:00",
    updated_at: str = "2026-01-01T00:00:00+00:00",
) -> KnowledgeItemRecord:
    return KnowledgeItemRecord(
        filename=filename,
        persistence=persistence,
        format=fmt,
        size_bytes=size_bytes,
        content_hash=content_hash,
        created_at=created_at,
        updated_at=updated_at,
    )


def _make_manifest(
    scope: ContextScope = ContextScope.GLOBAL,
    items: dict | None = None,
) -> KnowledgeManifest:
    return KnowledgeManifest(scope=scope, items=items or {})


class TestHelpers(unittest.TestCase):

    def test_now_iso_format(self):
        ts = _now_iso()
        self.assertIsInstance(ts, str)
        # Must encode UTC offset (+00:00 or Z)
        self.assertTrue(ts.endswith("+00:00") or ts.endswith("Z"), ts)

    def test_sha256_known_input(self):
        expected = hashlib.sha256(b"hello").hexdigest()
        self.assertEqual(_sha256(b"hello"), expected)

    def test_sha256_empty_bytes(self):
        expected = hashlib.sha256(b"").hexdigest()
        self.assertEqual(_sha256(b""), expected)


class TestKnowledgeMetadataStore(unittest.TestCase):

    def setUp(self):
        self.mock_redis = MagicMock()
        self.mock_redis.get.return_value = None
        self.store = KnowledgeMetadataStore(self.mock_redis)

    def test_key_global(self):
        self.assertEqual(self.store._key("global"), "kb:manifest:global")

    def test_key_category(self):
        self.assertEqual(self.store._key("category:cat1"),
                         "kb:manifest:category:cat1")

    def test_key_session(self):
        self.assertEqual(self.store._key("session:s1"),
                         "kb:manifest:session:s1")

    def test_load_existing_manifest(self):
        record = _make_record()
        manifest = _make_manifest(scope=ContextScope.GLOBAL, items={
                                  record.filename: record})
        self.mock_redis.get.return_value = manifest.model_dump_json()

        result = self.store.load("global", ContextScope.GLOBAL)

        self.assertEqual(result.scope, ContextScope.GLOBAL)
        self.assertIn("doc.md", result.items)
        self.mock_redis.get.assert_called_once_with("kb:manifest:global")

    def test_load_missing_key_returns_empty_manifest(self):
        self.mock_redis.get.return_value = None

        result = self.store.load(
            "global", ContextScope.GLOBAL, category_id="c1", session_id="s1")

        self.assertEqual(result.scope, ContextScope.GLOBAL)
        self.assertEqual(result.category_id, "c1")
        self.assertEqual(result.session_id, "s1")
        self.assertEqual(result.items, {})

    def test_load_corrupt_json_returns_empty_manifest(self):
        self.mock_redis.get.return_value = "this is not valid json {"

        # Must not raise; silently resets to empty
        result = self.store.load("global", ContextScope.GLOBAL)

        self.assertEqual(result.items, {})

    def test_save_calls_redis_set_with_correct_key(self):
        manifest = _make_manifest(scope=ContextScope.GLOBAL)

        self.store.save("global", manifest)

        self.mock_redis.set.assert_called_once()
        key, value = self.mock_redis.set.call_args[0]
        self.assertEqual(key, "kb:manifest:global")
        # Value must be valid JSON that round-trips back to the manifest
        roundtripped = KnowledgeManifest.model_validate_json(value)
        self.assertEqual(roundtripped.scope, ContextScope.GLOBAL)

    def test_delete_calls_redis_delete_with_correct_key(self):
        self.store.delete("global")

        self.mock_redis.delete.assert_called_once_with("kb:manifest:global")

    def test_bootstrap_nonexistent_directory(self):
        result = self.store.bootstrap(
            "global", Path("/nonexistent/path/xyz"), ContextScope.GLOBAL
        )

        self.assertEqual(result.items, {})
        self.mock_redis.set.assert_not_called()

    def test_bootstrap_registers_new_md_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "guide.md").write_bytes(b"# Hello")

            result = self.store.bootstrap("global", tmp, ContextScope.GLOBAL)

        self.assertIn("guide.md", result.items)
        rec = result.items["guide.md"]
        self.assertEqual(rec.format, ContentFormat.MARKDOWN)
        self.assertEqual(rec.persistence, ContextPersistence.PERSISTENT)
        self.mock_redis.set.assert_called_once()

    def test_bootstrap_skips_already_tracked_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "guide.md").write_bytes(b"# Hello")

            # Pre-load manifest that already tracks the file
            existing = _make_manifest(
                scope=ContextScope.GLOBAL,
                items={"guide.md": _make_record(filename="guide.md")},
            )
            self.mock_redis.get.return_value = existing.model_dump_json()

            result = self.store.bootstrap("global", tmp, ContextScope.GLOBAL)

        # File is in manifest but was already there — no new save
        self.assertIn("guide.md", result.items)
        self.mock_redis.set.assert_not_called()

    def test_bootstrap_ignores_unsupported_extensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "script.exe").write_bytes(b"\x00\x01")
            (tmp / "image.png").write_bytes(b"\x89PNG")

            result = self.store.bootstrap("global", tmp, ContextScope.GLOBAL)

        self.assertEqual(result.items, {})
        self.mock_redis.set.assert_not_called()

    def test_bootstrap_registers_multiple_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "doc.md").write_bytes(b"# Markdown")
            (tmp / "notes.txt").write_bytes(b"plain text")

            result = self.store.bootstrap("global", tmp, ContextScope.GLOBAL)

        self.assertIn("doc.md", result.items)
        self.assertIn("notes.txt", result.items)
        self.assertEqual(result.items["doc.md"].format, ContentFormat.MARKDOWN)
        self.assertEqual(result.items["notes.txt"].format, ContentFormat.TXT)
        self.mock_redis.set.assert_called_once()


class TestKnowledgeRegistryScopeHelpers(unittest.TestCase):
    """Static helper methods — pure functions, no external dependencies."""

    # scope_key
    def test_scope_key_global_ignores_ids(self):
        self.assertEqual(
            KnowledgeRegistry.scope_key(ContextScope.GLOBAL, "s1", "c1"),
            "global",
        )

    def test_scope_key_category_with_category_id(self):
        self.assertEqual(
            KnowledgeRegistry.scope_key(ContextScope.CATEGORY, None, "cat1"),
            "category:cat1",
        )

    def test_scope_key_category_falls_back_to_session_id(self):
        # When category_id is None, falls back to session_id
        self.assertEqual(
            KnowledgeRegistry.scope_key(ContextScope.CATEGORY, "sess1", None),
            "category:sess1",
        )

    def test_scope_key_category_defaults_to_default(self):
        self.assertEqual(
            KnowledgeRegistry.scope_key(ContextScope.CATEGORY, None, None),
            "category:default",
        )

    def test_scope_key_local_with_session_id(self):
        self.assertEqual(
            KnowledgeRegistry.scope_key(ContextScope.LOCAL, "sess1", None),
            "session:sess1",
        )

    def test_scope_key_local_anonymous_when_no_ids(self):
        self.assertEqual(
            KnowledgeRegistry.scope_key(ContextScope.LOCAL, None, None),
            "session:anonymous",
        )

    # scope_directory

    def test_scope_directory_global(self):
        result = KnowledgeRegistry.scope_directory(
            "/kb", ContextScope.GLOBAL, None, None)
        self.assertEqual(result, Path("/kb"))

    def test_scope_directory_category(self):
        result = KnowledgeRegistry.scope_directory(
            "/kb", ContextScope.CATEGORY, None, "cat1")
        self.assertEqual(result, Path("/kb/category/cat1"))

    def test_scope_directory_category_default(self):
        result = KnowledgeRegistry.scope_directory(
            "/kb", ContextScope.CATEGORY, None, None)
        self.assertEqual(result, Path("/kb/category/default"))

    def test_scope_directory_local(self):
        result = KnowledgeRegistry.scope_directory(
            "/kb", ContextScope.LOCAL, "sess1", None)
        self.assertEqual(result, Path("/kb/session/sess1"))

    def test_scope_directory_local_anonymous(self):
        result = KnowledgeRegistry.scope_directory(
            "/kb", ContextScope.LOCAL, None, None)
        self.assertEqual(result, Path("/kb/session/anonymous"))


class TestKnowledgeFileService(unittest.TestCase):
    """
    File CRUD tests.

    Uses a real temp directory for filesystem operations and mocks for Redis
    and KnowledgeRegistry (to avoid loading the embeddings model).
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.kb_root = cls.tmp.name

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        # Fresh mocks per test
        self.mock_redis = MagicMock()
        self.mock_redis.get.return_value = None  # empty manifests by default

        self.mock_registry = MagicMock()
        # Delegate static methods to real implementations
        self.mock_registry.scope_key = KnowledgeRegistry.scope_key
        self.mock_registry.scope_directory = KnowledgeRegistry.scope_directory

        self.svc = KnowledgeFileService(
            kb_directory=self.kb_root,
            registry=self.mock_registry,
            redis_client=self.mock_redis,
        )

    def test_save_file_unsupported_extension_raises(self):
        with self.assertRaises(ValueError):
            self.svc.save_file(
                filename="payload.exe",
                content=b"bad",
                scope=ContextScope.GLOBAL,
                persistence=ContextPersistence.PERSISTENT,
            )

    def test_save_file_creates_file_on_disk(self):
        content = b"# Hello world"
        info = self.svc.save_file(
            filename="new.md",
            content=content,
            scope=ContextScope.GLOBAL,
            persistence=ContextPersistence.PERSISTENT,
        )

        expected_path = Path(self.kb_root) / "new.md"
        self.assertTrue(expected_path.exists())
        self.assertEqual(expected_path.read_bytes(), content)
        self.assertEqual(info.filename, "new.md")
        self.assertEqual(info.format, ContentFormat.MARKDOWN)
        self.assertEqual(info.size_bytes, len(content))
        self.assertEqual(info.content_hash, _sha256(content))

    def test_save_file_new_record_has_matching_created_and_updated_at(self):
        info = self.svc.save_file(
            filename="fresh.txt",
            content=b"text",
            scope=ContextScope.GLOBAL,
            persistence=ContextPersistence.PERSISTENT,
        )
        self.assertEqual(info.created_at, info.updated_at)

    def test_save_file_updates_existing_record_preserves_created_at(self):
        old_record = _make_record(
            filename="update_me.md",
            content_hash="oldhash",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        old_manifest = _make_manifest(
            scope=ContextScope.GLOBAL,
            items={"update_me.md": old_record},
        )
        self.mock_redis.get.return_value = old_manifest.model_dump_json()

        new_content = b"updated content"
        info = self.svc.save_file(
            filename="update_me.md",
            content=new_content,
            scope=ContextScope.GLOBAL,
            persistence=ContextPersistence.PERSISTENT,
        )

        # created_at must be preserved; updated_at and hash must change
        self.assertEqual(info.created_at, "2025-01-01T00:00:00+00:00")
        self.assertNotEqual(info.updated_at, "2025-01-01T00:00:00+00:00")
        self.assertEqual(info.content_hash, _sha256(new_content))

    def test_save_file_global_calls_invalidate_global(self):
        self.svc.save_file(
            filename="kb.md",
            content=b"x",
            scope=ContextScope.GLOBAL,
            persistence=ContextPersistence.PERSISTENT,
        )
        self.mock_registry.invalidate_global.assert_called_once()
        self.mock_registry.invalidate.assert_not_called()

    def test_save_file_local_calls_invalidate_with_scope_key(self):
        self.svc.save_file(
            filename="session.md",
            content=b"x",
            scope=ContextScope.LOCAL,
            persistence=ContextPersistence.EPHEMERAL,
            session_id="sess1",
        )
        self.mock_registry.invalidate.assert_called_once_with("session:sess1")
        self.mock_registry.invalidate_global.assert_not_called()

    def test_save_file_persists_manifest_to_redis(self):
        self.svc.save_file(
            filename="persist.md",
            content=b"data",
            scope=ContextScope.GLOBAL,
            persistence=ContextPersistence.PERSISTENT,
        )
        self.mock_redis.set.assert_called()
        key, json_str = self.mock_redis.set.call_args[0]
        self.assertEqual(key, "kb:manifest:global")
        saved = KnowledgeManifest.model_validate_json(json_str)
        self.assertIn("persist.md", saved.items)

    def test_delete_file_existing_returns_true(self):
        # Create file on disk first
        target = Path(self.kb_root) / "to_delete.md"
        target.write_bytes(b"bye")

        record = _make_record(filename="to_delete.md")
        manifest = _make_manifest(
            scope=ContextScope.GLOBAL,
            items={"to_delete.md": record},
        )
        self.mock_redis.get.return_value = manifest.model_dump_json()

        result = self.svc.delete_file("to_delete.md", ContextScope.GLOBAL)

        self.assertTrue(result)
        self.assertFalse(target.exists())
        self.mock_redis.set.assert_called_once()  # manifest updated

    def test_delete_file_missing_returns_false(self):
        result = self.svc.delete_file("ghost.md", ContextScope.GLOBAL)

        self.assertFalse(result)
        self.mock_redis.set.assert_not_called()

    def test_delete_file_global_calls_invalidate_global(self):
        target = Path(self.kb_root) / "del_global.md"
        target.write_bytes(b"x")
        self.mock_redis.get.return_value = _make_manifest(
            scope=ContextScope.GLOBAL,
            items={"del_global.md": _make_record(filename="del_global.md")},
        ).model_dump_json()

        self.svc.delete_file("del_global.md", ContextScope.GLOBAL)

        self.mock_registry.invalidate_global.assert_called_once()

    def test_get_file_info_returns_correct_file_info(self):
        target = Path(self.kb_root) / "info.md"
        target.write_bytes(b"content")

        record = _make_record(filename="info.md")
        manifest = _make_manifest(
            scope=ContextScope.GLOBAL,
            items={"info.md": record},
        )
        self.mock_redis.get.return_value = manifest.model_dump_json()

        info = self.svc.get_file_info("info.md", ContextScope.GLOBAL)

        self.assertIsNotNone(info)
        self.assertEqual(info.filename, "info.md")
        self.assertEqual(info.scope, ContextScope.GLOBAL)
        self.assertEqual(info.format, ContentFormat.MARKDOWN)

    def test_get_file_info_missing_file_returns_none(self):
        result = self.svc.get_file_info("no_such_file.md", ContextScope.GLOBAL)
        self.assertIsNone(result)

    def test_get_file_info_not_in_manifest_returns_none(self):
        # File exists on disk but is not in the Redis manifest
        orphan = Path(self.kb_root) / "orphan.md"
        orphan.write_bytes(b"orphan")
        # manifest is empty (mock_redis.get returns None → empty manifest)

        result = self.svc.get_file_info("orphan.md", ContextScope.GLOBAL)

        self.assertIsNone(result)
        orphan.unlink()  # cleanup

    def test_list_files_global_scope(self):
        record = _make_record(filename="listed.md")
        manifest = _make_manifest(
            scope=ContextScope.GLOBAL,
            items={"listed.md": record},
        )
        self.mock_redis.get.return_value = manifest.model_dump_json()

        results = self.svc.list_files(ContextScope.GLOBAL)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].filename, "listed.md")
        self.assertEqual(results[0].scope, ContextScope.GLOBAL)

    def test_list_files_include_higher_scopes_adds_global(self):
        global_record = _make_record(filename="global_doc.md")
        global_manifest = _make_manifest(
            scope=ContextScope.GLOBAL,
            items={"global_doc.md": global_record},
        )

        def get_side_effect(key):
            if key == "kb:manifest:global":
                return global_manifest.model_dump_json()
            return None

        self.mock_redis.get.side_effect = get_side_effect

        results = self.svc.list_files(
            ContextScope.LOCAL,
            session_id="sess1",
            include_higher_scopes=True,
        )

        filenames = [r.filename for r in results]
        self.assertIn("global_doc.md", filenames)

    def test_list_files_local_scope_without_higher_scopes(self):
        # Only LOCAL scope requested — global files must not appear
        global_record = _make_record(filename="global_only.md")
        global_manifest = _make_manifest(
            scope=ContextScope.GLOBAL,
            items={"global_only.md": global_record},
        )

        local_record = _make_record(filename="local_only.md")
        local_manifest = _make_manifest(
            scope=ContextScope.LOCAL,
            items={"local_only.md": local_record},
        )

        def get_side_effect(key):
            if key == "kb:manifest:global":
                return global_manifest.model_dump_json()
            if key == "kb:manifest:session:sess1":
                return local_manifest.model_dump_json()
            return None

        self.mock_redis.get.side_effect = get_side_effect

        results = self.svc.list_files(ContextScope.LOCAL, session_id="sess1")

        filenames = [r.filename for r in results]
        self.assertIn("local_only.md", filenames)
        self.assertNotIn("global_only.md", filenames)

    def test_get_stats_single_scope(self):
        record = _make_record(filename="stats.md", size_bytes=512)
        manifest = _make_manifest(
            scope=ContextScope.GLOBAL,
            items={"stats.md": record},
        )
        self.mock_redis.get.return_value = manifest.model_dump_json()

        stats = self.svc.get_stats(scope=ContextScope.GLOBAL)

        self.assertEqual(stats["total_files"], 1)
        self.assertEqual(stats["total_bytes"], 512)
        self.assertIn("global", stats["by_scope"])
        self.assertEqual(stats["by_scope"]["global"]["file_count"], 1)

    def test_get_stats_aggregate_covers_all_scopes(self):
        # No scope filter → all three scopes queried
        stats = self.svc.get_stats(scope=None)

        self.assertIn("by_scope", stats)
        by_scope_keys = set(stats["by_scope"].keys())
        self.assertIn("global", by_scope_keys)
        self.assertIn("category", by_scope_keys)
        self.assertIn("local", by_scope_keys)

    def test_get_stats_ephemeral_count(self):
        ephemeral_rec = _make_record(
            filename="eph.md", persistence=ContextPersistence.EPHEMERAL
        )
        manifest = _make_manifest(
            scope=ContextScope.GLOBAL,
            items={"eph.md": ephemeral_rec},
        )
        self.mock_redis.get.return_value = manifest.model_dump_json()

        stats = self.svc.get_stats(scope=ContextScope.GLOBAL)

        self.assertEqual(stats["ephemeral_file_count"], 1)

    def test_cleanup_ephemeral_deletes_only_ephemeral_files(self):
        sess_dir = Path(self.kb_root) / "session" / "cleanup_sess"
        sess_dir.mkdir(parents=True, exist_ok=True)

        eph_file = sess_dir / "temp.md"
        keep_file = sess_dir / "keep.md"
        eph_file.write_bytes(b"ephemeral")
        keep_file.write_bytes(b"persistent")

        eph_rec = _make_record(
            filename="temp.md", persistence=ContextPersistence.EPHEMERAL
        )
        keep_rec = _make_record(
            filename="keep.md", persistence=ContextPersistence.PERSISTENT
        )
        manifest = KnowledgeManifest(
            scope=ContextScope.LOCAL,
            session_id="cleanup_sess",
            items={"temp.md": eph_rec, "keep.md": keep_rec},
        )
        self.mock_redis.get.return_value = manifest.model_dump_json()

        count = self.svc.cleanup_ephemeral("cleanup_sess")

        self.assertEqual(count, 1)
        self.assertFalse(eph_file.exists())
        self.assertTrue(keep_file.exists())
        self.mock_redis.set.assert_called_once()
        self.mock_registry.invalidate.assert_called_once_with(
            "session:cleanup_sess")

        # cleanup
        keep_file.unlink()

    def test_cleanup_ephemeral_empty_session_returns_zero(self):
        count = self.svc.cleanup_ephemeral("empty_sess")

        self.assertEqual(count, 0)
        self.mock_redis.set.assert_not_called()

    def test_bootstrap_global_registers_pre_existing_file(self):
        # Place a file in the kb_root before bootstrap
        pre_existing = Path(self.kb_root) / "pre_existing.md"
        pre_existing.write_bytes(b"# Pre-existing")

        self.svc.bootstrap_global()

        # Redis SET must have been called (file was discovered and saved)
        self.mock_redis.set.assert_called()
        key = self.mock_redis.set.call_args[0][0]
        self.assertEqual(key, "kb:manifest:global")

        # cleanup to not affect other tests
        pre_existing.unlink()


if __name__ == "__main__":
    unittest.main()
