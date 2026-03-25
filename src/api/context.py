import json
import re
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status

from src.config.settings import settings
from src.service.knowledge_file_service import KnowledgeFileService, FileInfo
from src.type.enums import ContextPersistence, ContextScope

context_router = APIRouter(prefix="/api/context", tags=["context"])

_SAFE_FILENAME_RE = re.compile(r'^[\w\-. ]+$')
_SHA256_HEX_LEN = 64


# Dependency
def get_knowledge_file_service(request: Request) -> KnowledgeFileService:
    return request.app.state.knowledge_file_service


# Helpers
def _parse_content_hash(content_hash: str) -> str:
    h = content_hash.lower().strip()
    if (
        len(h) != _SHA256_HEX_LEN
        or any(c not in "0123456789abcdef" for c in h)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content_hash: expected 64-character lowercase SHA-256 hex",
        )
    return h


def _validate_scope_context(scope: ContextScope, session_id: str | None, category_id: str | None) -> None:
    if scope == ContextScope.LOCAL and not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id is required for scope=local",
        )
    if scope == ContextScope.CATEGORY and not category_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="category_id is required for scope=category",
        )


# Endpoints
# NOTE: /stats and /upload are declared BEFORE /{content_hash} to prevent FastAPI
# from treating "stats" or "upload" as path parameters.
@context_router.get("", status_code=status.HTTP_200_OK)
async def list_knowledge_files(
    scope: ContextScope = Query(default=ContextScope.GLOBAL),
    session_id: Optional[str] = Query(default=None),
    category_id: Optional[str] = Query(default=None),
    knowledge_file_service: KnowledgeFileService = Depends(
        get_knowledge_file_service),
):
    """
    GET /api/context

    List knowledge base files accessible from the requested scope.
    - scope=global  → global files only
    - scope=category → global + category files
    - scope=local   → global + category (if category_id given) + session files
    """
    include_higher = scope != ContextScope.GLOBAL
    files = knowledge_file_service.list_files(
        scope=scope,
        session_id=session_id,
        category_id=category_id,
        include_higher_scopes=include_higher,
    )
    return {"status": "success", "files": [f.model_dump() for f in files]}


@context_router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_knowledge_file(
    file: UploadFile = File(...),
    scope: ContextScope = Form(default=ContextScope.GLOBAL),
    persistence: ContextPersistence = Form(
        default=ContextPersistence.PERSISTENT),
    session_id: Optional[str] = Form(default=None),
    category_id: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
    # JSON array string, e.g. '["faq","hr"]'
    tags: Optional[str] = Form(default=None),
    knowledge_file_service: KnowledgeFileService = Depends(
        get_knowledge_file_service),
):
    """
    POST /api/context/upload  (multipart/form-data)

    Upload a file (.md, .pdf, .txt) to the appropriate scope directory.
    The file is immediately indexed and available for KB search in subsequent
    chat requests with a matching scope.
    """
    filename = file.filename or ""

    # Validate filename
    if not _SAFE_FILENAME_RE.match(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename. Use alphanumeric characters, hyphens, underscores, spaces, and dots only.",
        )

    if not any(filename.lower().endswith(s) for s in knowledge_file_service.ALLOWED_SUFFIXES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(knowledge_file_service.ALLOWED_SUFFIXES)}",
        )

    # Validate scope context
    _validate_scope_context(scope, session_id, category_id)

    # Read and size-check
    content = await file.read()
    if len(content) > settings.KB_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File too large ({len(content)} bytes). Maximum allowed size is {settings.KB_MAX_UPLOAD_BYTES} bytes.",
        )

    # Parse optional tags (JSON array)
    parsed_tags: list[str] = []
    if tags:
        try:
            parsed_tags = json.loads(tags)
            if not isinstance(parsed_tags, list) or not all(isinstance(t, str) for t in parsed_tags):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="tags must be a JSON array of strings, e.g. '[\"faq\",\"hr\"]'",
            )

    # Save file
    try:
        file_info = knowledge_file_service.save_file(
            filename=filename,
            content=content,
            scope=scope,
            persistence=persistence,
            session_id=session_id,
            category_id=category_id,
            title=title,
            description=description,
            tags=parsed_tags,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return {"status": "success", "file": file_info.model_dump()}


@context_router.get("/stats", status_code=status.HTTP_200_OK)
async def get_kb_stats(
    scope: Optional[ContextScope] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    category_id: Optional[str] = Query(default=None),
    knowledge_file_service: KnowledgeFileService = Depends(
        get_knowledge_file_service),
):
    """
    GET /api/context/stats

    Return aggregate statistics. When scope is omitted, aggregates all scopes.
    """
    stats = knowledge_file_service.get_stats(
        scope=scope, session_id=session_id, category_id=category_id)
    return {"status": "success", "stats": stats}


@context_router.get("/{content_hash}", status_code=status.HTTP_200_OK)
async def get_knowledge_file(
    content_hash: str,
    scope: ContextScope = Query(default=ContextScope.GLOBAL),
    session_id: Optional[str] = Query(default=None),
    category_id: Optional[str] = Query(default=None),
    knowledge_file_service: KnowledgeFileService = Depends(
        get_knowledge_file_service),
):
    """
    GET /api/context/{content_hash}

    Return metadata for a knowledge item. *content_hash* is the SHA-256 hex
    manifest id (same as ``file.content_hash`` from list/upload responses).
    """
    h = _parse_content_hash(content_hash)
    file_info = knowledge_file_service.get_file_info(
        content_hash=h,
        scope=scope,
        session_id=session_id,
        category_id=category_id,
    )
    if file_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return {"status": "success", "file": file_info.model_dump()}


@context_router.delete("/{content_hash}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_file(
    content_hash: str,
    scope: ContextScope = Query(default=ContextScope.GLOBAL),
    session_id: Optional[str] = Query(default=None),
    category_id: Optional[str] = Query(default=None),
    knowledge_file_service: KnowledgeFileService = Depends(
        get_knowledge_file_service),
):
    """
    DELETE /api/context/{content_hash}

    Delete a knowledge item by SHA-256 content hash (manifest key).
    """
    _validate_scope_context(scope, session_id, category_id)
    h = _parse_content_hash(content_hash)

    deleted = knowledge_file_service.delete_file(
        content_hash=h,
        scope=scope,
        session_id=session_id,
        category_id=category_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)  # no response body
