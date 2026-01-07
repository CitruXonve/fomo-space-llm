import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # App settings
    APP_TITLE: str = "CitruXonve FAQ Backend"
    APP_DESCRIPTION: str = "A FastAPI backend of knowledge base, LLM and RAG for retrieval"
    APP_VERSION: str = "1.0.0"

    # Knowledge base settings
    project_root: Path = Path(__file__).parent.parent.parent
    KB_DIRECTORY: str = os.path.join(
        project_root, '.knowledge_sources')
    KB_POST_TTL: int = 3600 * 24 * 7  # 7 days

    # Export settings
    EXPORT_DIRECTORY: str = os.path.join(project_root, '.export')

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # 384-dimensional, fast, good for FAQ
    EMBEDDING_MODEL_CACHE_DIR: str = os.path.join(project_root, '.models')
    EMBEDDING_CACHE_DIR: str = os.path.join(project_root, '.embedding_cache')
    EMBEDDING_MODEL_CHUNK_SIZE: int = 500
    EMBEDDING_MODEL_CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL_BATCH_SIZE: int = 32
    EMBEDDING_MODEL_SHOW_PROGRESS_BAR: bool = True
    EMBEDDING_MODEL_CONVERT_TO_NUMPY: bool = True
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.3
    DEFAULT_TOP_K: int = 3

    """Claude API settings"""
    CLAUDE_MODEL: str
    CLAUDE_TEMPERATURE: float = 0.7  # Balanced - not too creative, not too rigid
    CLAUDE_MAX_TOKENS: int

    model_config = SettingsConfigDict(
        env_file=os.path.join(Path(__file__).parent.parent.parent, '.env'),
        env_file_encoding='utf-8',
        extra='ignore'
    )


# Create a global settings instance
# settings = (
#     Settings(_env_file=".env.test")
#     if os.getenv("RUN_HANDLER_ENV") == "test"
#     else Settings()
# )
settings = Settings()
