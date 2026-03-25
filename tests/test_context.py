"""
Integration tests for /api/context/* endpoints (src/api/context.py).

Approach
--------
- FastAPI dependency override: `app.dependency_overrides[get_knowledge_file_service]`
  injects a MagicMock instead of the real service, so no Redis or embeddings
  model is needed.
- `TestClient(app)` without a `with` block — avoids triggering startup_event,
  which would attempt Redis.ping() and load sentence-transformers.
- Each test class resets the mock in setUp() so tests are independent.
"""

import hashlib
import io
import unittest
from unittest.mock import MagicMock

# Valid 64-char SHA-256 hex for path params on GET/DELETE /api/context/{content_hash}
_CTX_HASH = hashlib.sha256(b"context-test").hexdigest()
_CTX_HASH_2 = hashlib.sha256(b"context-test-2").hexdigest()
_MISSING_HASH = "0" * 64

from fastapi.testclient import TestClient

from src.api.context import get_knowledge_file_service
from src.main import app
from src.service.knowledge_file_service import FileInfo, KnowledgeFileService
from src.type.enums import ContentFormat, ContextPersistence, ContextScope


# Test helpers
def _make_file_info(
    filename: str = "doc.md",
    scope: ContextScope = ContextScope.GLOBAL,
    persistence: ContextPersistence = ContextPersistence.PERSISTENT,
    fmt: ContentFormat = ContentFormat.MARKDOWN,
    size_bytes: int = 128,
    content_hash: str | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
    updated_at: str = "2026-01-01T00:00:00+00:00",
    **kwargs,
) -> FileInfo:
    ch = content_hash if content_hash is not None else _CTX_HASH
    return FileInfo(
        filename=filename,
        scope=scope,
        persistence=persistence,
        format=fmt,
        size_bytes=size_bytes,
        content_hash=ch,
        created_at=created_at,
        updated_at=updated_at,
        **kwargs,
    )


def _make_mock_service() -> MagicMock:
    """Return a MagicMock with the ALLOWED_SUFFIXES class attribute set."""
    svc = MagicMock(spec=KnowledgeFileService)
    svc.ALLOWED_SUFFIXES = KnowledgeFileService.ALLOWED_SUFFIXES
    return svc


# GET /api/context
class TestListKBFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_svc = _make_mock_service()
        app.dependency_overrides[get_knowledge_file_service] = lambda: cls.mock_svc
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.pop(get_knowledge_file_service, None)

    def setUp(self):
        self.mock_svc.reset_mock()
        self.mock_svc.ALLOWED_SUFFIXES = KnowledgeFileService.ALLOWED_SUFFIXES
        self.mock_svc.list_files.return_value = []

    # response shape

    def test_list_returns_200_with_files_key(self):
        resp = self.client.get("/api/context")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertIn("files", body)

    def test_list_empty_service_returns_empty_list(self):
        resp = self.client.get("/api/context")
        self.assertEqual(resp.json()["files"], [])

    def test_list_returns_serialized_file_info(self):
        self.mock_svc.list_files.return_value = [
            _make_file_info(filename="a.md")]
        resp = self.client.get("/api/context")
        files = resp.json()["files"]
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["filename"], "a.md")
        self.assertEqual(files[0]["scope"], "global")

    # scope routing

    def test_list_global_scope_passes_include_higher_scopes_false(self):
        self.client.get("/api/context?scope=global")
        _, kwargs = self.mock_svc.list_files.call_args
        self.assertFalse(kwargs["include_higher_scopes"])

    def test_list_local_scope_passes_include_higher_scopes_true(self):
        self.client.get("/api/context?scope=local&session_id=sess1")
        _, kwargs = self.mock_svc.list_files.call_args
        self.assertTrue(kwargs["include_higher_scopes"])

    def test_list_category_scope_passes_include_higher_scopes_true(self):
        self.client.get("/api/context?scope=category&category_id=cat1")
        _, kwargs = self.mock_svc.list_files.call_args
        self.assertTrue(kwargs["include_higher_scopes"])

    def test_list_passes_session_id_and_category_id(self):
        self.client.get(
            "/api/context?scope=local&session_id=s1&category_id=c1")
        _, kwargs = self.mock_svc.list_files.call_args
        self.assertEqual(kwargs["session_id"], "s1")
        self.assertEqual(kwargs["category_id"], "c1")


# POST /api/context/upload
class TestUploadKBFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_svc = _make_mock_service()
        app.dependency_overrides[get_knowledge_file_service] = lambda: cls.mock_svc
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.pop(get_knowledge_file_service, None)

    def setUp(self):
        self.mock_svc.reset_mock()
        self.mock_svc.ALLOWED_SUFFIXES = KnowledgeFileService.ALLOWED_SUFFIXES
        # clear any side_effect from prior test
        self.mock_svc.save_file.side_effect = None
        self.mock_svc.save_file.return_value = _make_file_info()

    def _upload(self, filename="doc.md", content=b"# Hello", **form_fields):
        return self.client.post(
            "/api/context/upload",
            files={"file": (filename, io.BytesIO(content), "text/plain")},
            data={"scope": "global", "persistence": "persistent", **form_fields},
        )

    # happy path

    def test_upload_valid_md_returns_201(self):
        resp = self._upload("guide.md", b"# Guide")
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertIn("file", body)

    def test_upload_valid_txt_returns_201(self):
        self.mock_svc.save_file.return_value = _make_file_info(
            filename="notes.txt", fmt=ContentFormat.TXT
        )
        resp = self._upload("notes.txt", b"some text")
        self.assertEqual(resp.status_code, 201)

    def test_upload_calls_save_file_with_correct_filename(self):
        self._upload("my_doc.md")
        _, kwargs = self.mock_svc.save_file.call_args
        self.assertEqual(kwargs["filename"], "my_doc.md")

    def test_upload_passes_title_description(self):
        self._upload("doc.md", title="My Title", description="My Desc")
        _, kwargs = self.mock_svc.save_file.call_args
        self.assertEqual(kwargs["title"], "My Title")
        self.assertEqual(kwargs["description"], "My Desc")

    def test_upload_parses_valid_tags(self):
        self._upload("doc.md", tags='["faq","hr"]')
        _, kwargs = self.mock_svc.save_file.call_args
        self.assertEqual(kwargs["tags"], ["faq", "hr"])

    def test_upload_empty_tags_passes_empty_list(self):
        self._upload("doc.md")
        _, kwargs = self.mock_svc.save_file.call_args
        self.assertEqual(kwargs["tags"], [])

    # validation: filename

    def test_upload_unsupported_extension_returns_400(self):
        resp = self._upload("payload.exe")
        self.assertEqual(resp.status_code, 400)

    def test_upload_path_traversal_filename_returns_400(self):
        resp = self._upload("../../etc/passwd")
        self.assertEqual(resp.status_code, 400)

    def test_upload_filename_with_special_chars_returns_400(self):
        resp = self._upload("bad;name.md")
        self.assertEqual(resp.status_code, 400)

    # validation: scope context

    def test_upload_local_without_session_id_returns_400(self):
        resp = self.client.post(
            "/api/context/upload",
            files={"file": ("doc.md", b"x", "text/plain")},
            data={"scope": "local", "persistence": "persistent"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("session_id", resp.json()["detail"])

    def test_upload_category_without_category_id_returns_400(self):
        resp = self.client.post(
            "/api/context/upload",
            files={"file": ("doc.md", b"x", "text/plain")},
            data={"scope": "category", "persistence": "persistent"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("category_id", resp.json()["detail"])

    # validation: size

    def test_upload_oversized_file_returns_413(self):
        from src.config.settings import settings
        oversized = b"x" * (settings.KB_MAX_UPLOAD_BYTES + 1)
        resp = self._upload("big.md", oversized)
        self.assertEqual(resp.status_code, 413)

    # validation: tags JSON

    def test_upload_invalid_tags_json_returns_400(self):
        resp = self._upload("doc.md", tags="not-json")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("tags", resp.json()["detail"])

    def test_upload_tags_not_string_array_returns_400(self):
        resp = self._upload("doc.md", tags="[1, 2, 3]")  # ints, not strings
        self.assertEqual(resp.status_code, 400)

    # service error propagation

    def test_upload_service_value_error_returns_400(self):
        self.mock_svc.save_file.side_effect = ValueError("bad extension")
        resp = self._upload("doc.md")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("bad extension", resp.json()["detail"])


# GET /api/context/stats

class TestGetKBStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_svc = _make_mock_service()
        app.dependency_overrides[get_knowledge_file_service] = lambda: cls.mock_svc
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.pop(get_knowledge_file_service, None)

    def setUp(self):
        self.mock_svc.reset_mock()
        self.mock_svc.ALLOWED_SUFFIXES = KnowledgeFileService.ALLOWED_SUFFIXES
        self.mock_svc.get_stats.return_value = {
            "total_files": 0,
            "total_bytes": 0,
            "ephemeral_file_count": 0,
            "by_scope": {"global": {"file_count": 0, "total_bytes": 0, "ephemeral_count": 0}},
        }

    def test_stats_returns_200_with_stats_key(self):
        resp = self.client.get("/api/context/stats")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertIn("stats", body)

    def test_stats_no_scope_param_calls_service_with_none(self):
        self.client.get("/api/context/stats")
        _, kwargs = self.mock_svc.get_stats.call_args
        self.assertIsNone(kwargs["scope"])

    def test_stats_with_scope_param_passes_it_to_service(self):
        self.client.get("/api/context/stats?scope=global")
        _, kwargs = self.mock_svc.get_stats.call_args
        self.assertEqual(kwargs["scope"], ContextScope.GLOBAL)

    def test_stats_returns_service_data(self):
        self.mock_svc.get_stats.return_value = {
            "total_files": 3,
            "total_bytes": 1024,
            "ephemeral_file_count": 1,
            "by_scope": {},
        }
        resp = self.client.get("/api/context/stats")
        stats = resp.json()["stats"]
        self.assertEqual(stats["total_files"], 3)
        self.assertEqual(stats["total_bytes"], 1024)


# GET /api/context/{content_hash}
class TestGetKBFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_svc = _make_mock_service()
        app.dependency_overrides[get_knowledge_file_service] = lambda: cls.mock_svc
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.pop(get_knowledge_file_service, None)

    def setUp(self):
        self.mock_svc.reset_mock()
        self.mock_svc.ALLOWED_SUFFIXES = KnowledgeFileService.ALLOWED_SUFFIXES
        self.mock_svc.get_file_info.return_value = None

    def test_get_existing_file_returns_200(self):
        h = hashlib.sha256(b"guide").hexdigest()
        self.mock_svc.get_file_info.return_value = _make_file_info(
            filename="guide.md", content_hash=h)
        resp = self.client.get(f"/api/context/{h}")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["file"]["filename"], "guide.md")

    def test_get_missing_file_returns_404(self):
        self.mock_svc.get_file_info.return_value = None
        resp = self.client.get(f"/api/context/{_MISSING_HASH}")
        self.assertEqual(resp.status_code, 404)

    def test_get_invalid_hash_returns_400(self):
        resp = self.client.get("/api/context/not-a-sha256")
        self.assertEqual(resp.status_code, 400)

    def test_get_passes_scope_and_ids_to_service(self):
        self.mock_svc.get_file_info.return_value = _make_file_info()
        self.client.get(
            f"/api/context/{_CTX_HASH}?scope=local&session_id=s1&category_id=c1")
        _, kwargs = self.mock_svc.get_file_info.call_args
        self.assertEqual(kwargs["content_hash"], _CTX_HASH)
        self.assertEqual(kwargs["scope"], ContextScope.LOCAL)
        self.assertEqual(kwargs["session_id"], "s1")
        self.assertEqual(kwargs["category_id"], "c1")

    def test_get_default_scope_is_global(self):
        self.mock_svc.get_file_info.return_value = _make_file_info()
        self.client.get(f"/api/context/{_CTX_HASH}")
        _, kwargs = self.mock_svc.get_file_info.call_args
        self.assertEqual(kwargs["scope"], ContextScope.GLOBAL)


# DELETE /api/context/{content_hash}
class TestDeleteKBFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_svc = _make_mock_service()
        app.dependency_overrides[get_knowledge_file_service] = lambda: cls.mock_svc
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.pop(get_knowledge_file_service, None)

    def setUp(self):
        self.mock_svc.reset_mock()
        self.mock_svc.ALLOWED_SUFFIXES = KnowledgeFileService.ALLOWED_SUFFIXES
        self.mock_svc.delete_file.return_value = True

    def test_delete_existing_file_returns_204(self):
        resp = self.client.delete(f"/api/context/{_CTX_HASH}")
        self.assertEqual(resp.status_code, 204)

    def test_delete_missing_file_returns_404(self):
        self.mock_svc.delete_file.return_value = False
        resp = self.client.delete(f"/api/context/{_MISSING_HASH}")
        self.assertEqual(resp.status_code, 404)

    def test_delete_passes_content_hash_scope_ids_to_service(self):
        self.client.delete(
            f"/api/context/{_CTX_HASH_2}?scope=local&session_id=s1")
        _, kwargs = self.mock_svc.delete_file.call_args
        self.assertEqual(kwargs["content_hash"], _CTX_HASH_2)
        self.assertEqual(kwargs["scope"], ContextScope.LOCAL)
        self.assertEqual(kwargs["session_id"], "s1")

    def test_delete_local_without_session_id_returns_400(self):
        resp = self.client.delete(f"/api/context/{_CTX_HASH}?scope=local")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("session_id", resp.json()["detail"])

    def test_delete_category_without_category_id_returns_400(self):
        resp = self.client.delete(f"/api/context/{_CTX_HASH}?scope=category")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("category_id", resp.json()["detail"])

    def test_delete_default_scope_is_global(self):
        self.client.delete(f"/api/context/{_CTX_HASH}")
        _, kwargs = self.mock_svc.delete_file.call_args
        self.assertEqual(kwargs["scope"], ContextScope.GLOBAL)


if __name__ == "__main__":
    unittest.main()
