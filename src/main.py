import logging
from contextlib import asynccontextmanager

import redis
from redis import exceptions as redis_exceptions

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.chat import router as chat_router, stream_router
from src.api.context import context_router
from src.service.session_service import SessionService
from src.service.llm_service import ClaudeLLMService
from src.service.knowledge_base import KnowledgeBaseServiceMultiFormat
from src.service.knowledge_file_service import KnowledgeFileService, KnowledgeRegistry
from src.config.settings import settings
from src.type.enums import ContextScope
from src.utility.in_memory_redis import InMemoryRedis

# Apply compatibility patches before any Anthropic streaming (handles dict container/context_management)
from src.compat import apply_langchain_anthropic_patch

apply_langchain_anthropic_patch()


logger = logging.getLogger(__name__)


class _NoopKbRegistry:
    """Placeholder until KnowledgeFileService is wired to the real KnowledgeRegistry."""

    def invalidate_global(self) -> None:
        pass

    def invalidate(self, key: str) -> None:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app.state on startup (replaces deprecated on_event startup)."""
    logger.info("Starting up...")
    app.state.session_service = SessionService()
    try:
        redis_client = redis.from_url(
            settings.REDIS_URL, decode_responses=True)
        redis_client.ping()
    except (
        redis_exceptions.ConnectionError,
        redis_exceptions.TimeoutError,
        OSError,
    ) as e:
        logger.warning(
            "Redis unavailable at %s (%s); using in-memory KB manifest store "
            "(not shared across processes).",
            settings.REDIS_URL,
            e,
        )
        redis_client = InMemoryRedis()

    noop_registry = _NoopKbRegistry()
    knowledge_file_service = KnowledgeFileService(
        settings.KB_DIRECTORY,
        noop_registry,
        redis_client,
    )
    knowledge_file_service.bootstrap_global()
    global_sources = knowledge_file_service.iter_kb_sources(
        ContextScope.GLOBAL, None, None
    )
    logger.warning(
        f"Global sources after bootstrap: \n{"\n".join([f"- {source[0]} from {source[1]}" for source in global_sources])}")
    kb_service = KnowledgeBaseServiceMultiFormat(
        kb_directory=settings.KB_DIRECTORY,
        cache_prefix="",
        file_sources=global_sources,
    )
    registry = KnowledgeRegistry(kb_service)
    registry.attach_file_service(knowledge_file_service)
    knowledge_file_service.rebind_registry(registry)
    app.state.knowledge_file_service = knowledge_file_service
    app.state.llm_service = ClaudeLLMService(kb_service)
    logger.info("Resources initialized successfully")
    yield


app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost:3000", "127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(stream_router)
app.include_router(context_router)


@app.get("/api/health")
async def root():
    """
    Root endpoint to verify the API is running.
    """
    return {"message": settings.APP_TITLE + " API is running"}
