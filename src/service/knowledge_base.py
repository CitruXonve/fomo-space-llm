"""
Knowledge base: load sources, chunk text, embed locally (sentence-transformers),
and retrieve by cosine similarity. See ``KnowledgeBaseServiceMultiFormat`` for the
lifecycle and which public entrypoints trigger a sync with disk and cache.
"""

import re
import json
import hashlib
import numpy as np
import logging
from abc import ABC, abstractmethod

from typing import Optional, Sequence
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import settings
from src.service.file_parser import ParserFactory

logger = logging.getLogger(__name__)


class Chunk:
    """One searchable slice of a source file: text, provenance, optional embedding vector."""

    def __init__(
        self,
        content: str,
        source_file: str,
        heading: str = "",
        chunk_index: int = 0
    ):
        self.content = content.strip()
        self.source_file = source_file
        self.heading = heading
        self.chunk_index = chunk_index
        self.embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "source_file": self.source_file,
            "heading": self.heading,
            "chunk_index": self.chunk_index
        }


class KnowledgeBaseService(ABC):
    """Contract for KB load, embed, stats, and search. See ``KnowledgeBaseServiceMultiFormat``."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _load_knowledge_base(self) -> None:
        """Load KB from text file"""
        pass

    @abstractmethod
    def _create_embeddings(self) -> None:
        """Create embeddings for the knowledge base"""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get stats for the knowledge base"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = settings.DEFAULT_TOP_K, similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD) -> list[dict]:
        """Semantic search using embeddings"""
        # Returns relevant KB chunks
        pass


class KnowledgeBaseServiceMultiFormat(KnowledgeBaseService):
    """
    Multi-format KB: ``.md``, ``.pdf``, and ``.txt`` sources parsed through
    ``ParserFactory``. Supports both directory mode (``iterdir``) and manifest
    mode (explicit ``file_sources`` list). Chunking respects section boundaries
    and size limits; embeddings are batched through the configured SentenceTransformer.

    Lifecycle (what happens over time)
    ----------------------------------
    1. ``__init__`` loads the embedding model, then calls ``_load_from_cache`` for the
       current KB hash. On success, ``chunks`` and ``embeddings`` are restored from disk
       cache and the instance is ready to search.
    2. On cache miss or first use, callers rely on ``refresh_embeddings`` (directly or
       via ``get_all_sources``, ``get_stats``, or ``search``) to align memory with sources.

    State machine (conceptual)
    --------------------------
    * **uninitialized** — right after ``__init__`` if cache miss: ``chunks`` may be empty,
      ``embeddings`` None until a successful refresh.
    * **ready** — ``chunks`` and ``embeddings`` are populated and row-aligned; search and
      stats are meaningful.
    * **refreshing** — during ``refresh_embeddings`` when cache is invalid: chunks are
      cleared, files are re-read, embeddings recomputed, cache rewritten.

    ``refresh_embeddings`` first tries ``_load_from_cache`` for the computed hash; if that
    succeeds it returns immediately without re-parsing source files. Otherwise it replaces
    ``chunks``, reloads from disk, embeds, and saves cache.

    ``_load_knowledge_base`` only appends to ``chunks``; ``refresh_embeddings`` clears
    ``chunks`` before calling it so reloads are never duplicated.
    """
    embeddings: Optional[np.ndarray]

    def __init__(
        self,
        kb_directory: str | None = None,
        cache_prefix: str = "",
        file_sources: Sequence[tuple[str, Path]] | None = None,
    ):
        self.kb_directory = Path(kb_directory) if kb_directory else Path(
            settings.KB_DIRECTORY)
        # When set, load only these (logical_name, path) pairs instead of globbing.
        self._file_sources: Sequence[tuple[str, Path]] | None = file_sources
        self.chunk_size = settings.EMBEDDING_MODEL_CHUNK_SIZE
        self.chunk_overlap = settings.EMBEDDING_MODEL_CHUNK_OVERLAP

        # Embedding cache paths — namespaced by cache_prefix to support multiple instances
        self.cache_dir = Path(settings.EMBEDDING_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{cache_prefix}_" if cache_prefix else ""
        self.embeddings_cache_file = self.cache_dir / f"{prefix}embeddings.npy"
        self.chunks_cache_file = self.cache_dir / f"{prefix}chunks.json"
        self.hash_cache_file = self.cache_dir / f"{prefix}kb_hash.txt"

        # Load embedding model (cached locally in EMBEDDING_MODEL_CACHE_DIR)
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model: SentenceTransformer = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            cache_folder=settings.EMBEDDING_MODEL_CACHE_DIR
        )
        logger.info("Embedding model loaded successfully")

        # Initialize storage
        self.chunks: list[Chunk] = []
        self.embeddings = None

        # Compute current KB hash
        current_hash = self._compute_kb_hash()

        # Try to load from cache if KB unchanged;
        # Otherwise, leave it as-is for lazy-loading or on-demand loading
        if self._load_from_cache(current_hash):
            logger.info(f"Loaded {len(self.chunks)} chunks from cache")

    def refresh_embeddings(self, current_hash: Optional[str] = None) -> str:
        """
        Bring ``chunks`` and ``embeddings`` in sync with the KB on disk and embedding settings.

        **When to use:** After adding, editing, or removing source files; when you need
        guaranteed-fresh vectors without constructing a new service; or implicitly—this
        is invoked by ``search``, ``get_stats``, and ``get_all_sources``.

        **How it works:** Computes (or accepts) a content hash. If on-disk cache matches,
        loads cache and returns. Otherwise clears ``chunks``, re-reads files, runs
        ``_create_embeddings``, writes cache files, and returns the hash.

        **Returns:** The KB hash string used for cache validation.
        """
        hash = current_hash or self._compute_kb_hash()
        if self._load_from_cache(hash):
            return hash
        # Full reload must replace in-memory chunks; _load_knowledge_base only appends.
        self.chunks = []
        self._load_knowledge_base()
        self._create_embeddings()
        self._save_to_cache(hash)
        logger.info(
            f"Knowledge base loaded and embeddings refreshed successfully: {len(self.chunks)} chunks")
        return hash

    def get_all_sources(self) -> list[str]:
        """
        Distinct logical / file names that appear in ``chunk.source_file``.

        **When to use:** UI lists, metrics, or debugging which files contributed chunks.

        **Behavior:** Calls ``refresh_embeddings`` first, so this can be expensive on a
        cache miss (full re-embed). Result order is not guaranteed (built from a ``set``).
        """
        self.refresh_embeddings()
        return list(set(chunk.source_file for chunk in self.chunks))

    def get_chunk_by_index(self, index: int) -> Optional[Chunk]:
        """
        Random access to ``self.chunks[index]`` if in range.

        **When to use:** Inspecting a single chunk after you know its index (e.g. from
        search ordering or tests).

        **Caveat:** Does **not** call ``refresh_embeddings``; data may be stale relative to
        disk until another method refreshes the service.
        """
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None

    def get_stats(self) -> dict:
        """
        Snapshot of size and configuration: chunk count, source count, embedding width,
        source list, and model object.

        **When to use:** Health checks, admin dashboards, or tests asserting the KB loaded.

        **Behavior:** Calls ``refresh_embeddings`` then ``get_all_sources`` (which refreshes
        again internally). Prefer a single ``refresh_embeddings`` if you are batching work.
        """
        self.refresh_embeddings()
        return {
            "total_chunks": len(self.chunks),
            "total_sources": len(self.get_all_sources()),
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "sources": self.get_all_sources(),
            "model_details": self.model,
        }

    def _compute_kb_hash(self) -> str:
        """Compute a hash of the KB state, including all supported file types."""
        supported_suffixes = {".md", ".pdf", ".txt"}

        if self._file_sources is not None:
            hash_data = ["mode:manifest_mf"]
            for logical, path in sorted(self._file_sources, key=lambda x: x[0]):
                if path.is_file():
                    stat = path.stat()
                    hash_data.append(
                        f"{logical}:{path.resolve()}:{stat.st_size}:{stat.st_mtime}"
                    )
                else:
                    hash_data.append(f"{logical}:missing")
            hash_data.append(f"chunk_size:{self.chunk_size}")
            hash_data.append(f"chunk_overlap:{self.chunk_overlap}")
            hash_data.append(f"model:{settings.EMBEDDING_MODEL}")
            return hashlib.sha256("|".join(hash_data).encode()).hexdigest()

        if not self.kb_directory.exists():
            return ""

        hash_data = ["mode:iterdir_mf"]
        for f in sorted(self.kb_directory.iterdir()):
            if f.is_file() and f.suffix.lower() in supported_suffixes:
                stat = f.stat()
                hash_data.append(f"{f.name}:{stat.st_size}:{stat.st_mtime}")

        hash_data.append(f"chunk_size:{self.chunk_size}")
        hash_data.append(f"chunk_overlap:{self.chunk_overlap}")
        hash_data.append(f"model:{settings.EMBEDDING_MODEL}")

        return hashlib.sha256("|".join(hash_data).encode()).hexdigest()

    def _load_from_cache(self, current_hash: str) -> bool:
        """
        Load embeddings and chunks from local cache if valid.
        Returns True if cache was loaded successfully, False otherwise.
        """
        try:
            # Check if all cache files exist
            if not all([
                self.embeddings_cache_file.exists(),
                self.chunks_cache_file.exists(),
                self.hash_cache_file.exists()
            ]):
                logger.info("Cache files not found")
                return False

            # Check if hash matches
            cached_hash = self.hash_cache_file.read_text().strip()
            if cached_hash != current_hash:
                logger.info("KB has changed - cache invalidated")
                return False

            # Load embeddings
            self.embeddings = np.load(self.embeddings_cache_file)

            # Load chunks
            with open(self.chunks_cache_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            self.chunks = []
            for i, chunk_dict in enumerate(chunks_data):
                chunk = Chunk(
                    content=chunk_dict["content"],
                    source_file=chunk_dict["source_file"],
                    heading=chunk_dict["heading"],
                    chunk_index=chunk_dict["chunk_index"]
                )
                chunk.embedding = self.embeddings[i]
                self.chunks.append(chunk)

            logger.info(
                f"Cache loaded: {len(self.chunks)} chunks, embeddings shape {self.embeddings.shape}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return False

    def _save_to_cache(self, current_hash: str) -> None:
        """Save embeddings and chunks to cache."""
        try:
            # Save embeddings
            np.save(self.embeddings_cache_file, self.embeddings)

            # Save chunks (without embeddings - they're in the .npy file)
            chunks_data = [chunk.to_dict() for chunk in self.chunks]
            with open(self.chunks_cache_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False)

            # Save hash
            self.hash_cache_file.write_text(current_hash)

            logger.info(f"Cache saved to {self.cache_dir}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_knowledge_base(self) -> None:
        factory = ParserFactory()
        supported_suffixes = {".md", ".pdf", ".txt"}

        if self._file_sources is not None:
            if not self._file_sources:
                logger.warning("No manifest-driven KB sources (multi-format)")
                return
            for logical, file_path in sorted(self._file_sources, key=lambda x: x[0]):
                if not file_path.is_file():
                    logger.warning(
                        "Skipping missing KB file for logical name '%s': %s",
                        logical,
                        file_path,
                    )
                    continue
                if file_path.suffix.lower() not in supported_suffixes:
                    logger.warning(
                        "Skipping unsupported suffix for logical name '%s': %s",
                        logical,
                        file_path,
                    )
                    continue
                try:
                    sections = factory.parse_file(file_path)
                except Exception as e:
                    logger.warning(
                        "Failed to parse %s (logical '%s'): %s",
                        file_path,
                        logical,
                        e,
                    )
                    continue

                for section_index, (heading, section_content) in enumerate(sections):
                    section_chunks = self._chunk_section(section_content)
                    for chunk_index, chunk_text in enumerate(section_chunks):
                        chunk_with_context = self._add_context(
                            heading, chunk_text)
                        chunk = Chunk(
                            content=chunk_with_context,
                            source_file=logical,
                            heading=heading,
                            chunk_index=section_index *
                            len(section_chunks) + chunk_index,
                        )
                        self.chunks.append(chunk)
            return

        if not self.kb_directory.exists():
            raise FileNotFoundError(
                f"Knowledge base directory not found: {self.kb_directory}")

        all_files = [
            f for f in sorted(self.kb_directory.iterdir())
            if f.is_file() and f.suffix.lower() in supported_suffixes
        ]

        if not all_files:
            raise ValueError(
                f"No supported files found in {self.kb_directory}")

        logger.info(f"Found {len(all_files)} file(s) in {self.kb_directory}")

        for file_path in all_files:
            try:
                sections = factory.parse_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to parse {file_path.name}: {e}")
                continue

            for section_index, (heading, section_content) in enumerate(sections):
                section_chunks = self._chunk_section(section_content)
                for chunk_index, chunk_text in enumerate(section_chunks):
                    chunk_with_context = self._add_context(heading, chunk_text)
                    chunk = Chunk(
                        content=chunk_with_context,
                        source_file=file_path.name,
                        heading=heading,
                        chunk_index=section_index *
                        len(section_chunks) + chunk_index,
                    )
                    self.chunks.append(chunk)

    def _chunk_section(self, content: str) -> list[str]:
        """
        Split a section into smaller chunks if needed.

        Strategy:
        - If section < chunk_size: return as-is
        - If section > chunk_size: split by paragraphs with overlap
        """
        if len(content) <= self.chunk_size:
            return [content]

        # Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', content)

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = len(para)

            # If single paragraph exceeds chunk_size, split by sentences
            if para_length > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = []
                temp_length = 0

                for sentence in sentences:
                    if temp_length + len(sentence) > self.chunk_size and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        # Keep overlap
                        overlap_sentences = temp_chunk[-2:] if len(
                            temp_chunk) >= 2 else temp_chunk
                        temp_chunk = overlap_sentences
                        temp_length = sum(len(s) for s in temp_chunk)

                    temp_chunk.append(sentence)
                    temp_length += len(sentence)

                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))

                continue

            # Add paragraph to current chunk
            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))

                # Add overlap: keep last paragraph
                if current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(para)
            current_length += para_length

        # Add remaining content
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _add_context(self, heading: str, content: str) -> str:
        """
        Add header context to chunk for better retrieval.

        Example:
        Input: heading="Password Reset", content="Click forgot password..."
        Output: "Password Reset: Click forgot password..."
        """
        if heading and heading.lower() != "introduction":
            return f"{heading}: {content}"
        return content

    def _create_embeddings(self) -> None:
        """Generate embeddings for all chunks"""
        if not self.chunks:
            logger.warning("No chunks to embed")
            return

        logger.info(f"Generating embeddings for {len(self.chunks)} chunks...")

        # Extract content from chunks
        texts = [chunk.content for chunk in self.chunks]

        # Generate embeddings in batch (faster)
        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=settings.EMBEDDING_MODEL_BATCH_SIZE,
            show_progress_bar=settings.EMBEDDING_MODEL_SHOW_PROGRESS_BAR,
            convert_to_numpy=settings.EMBEDDING_MODEL_CONVERT_TO_NUMPY
        )

        # Store embeddings
        self.embeddings = embeddings

        # Also store in individual chunks for reference
        for chunk, embedding in zip(self.chunks, embeddings):
            chunk.embedding = embedding

        logger.info(f"Embeddings created: shape {self.embeddings.shape}")

    def search(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD
    ) -> list[dict]:
        """
        Encode ``query`` and return up to ``top_k`` chunks whose cosine similarity meets
        ``similarity_threshold``, each as ``chunk.to_dict()`` plus ``similarity_score``.

        **When to use:** RAG retrieval or any semantic match over indexed text.

        **Behavior:** Calls ``refresh_embeddings`` first so results match current files and
        cache. Returns ``[]`` if the KB has no chunks or embeddings after refresh.

        Args:
            query: Natural-language question or keywords.
            top_k: Maximum number of candidates considered before threshold filtering.
            similarity_threshold: Minimum similarity in [0, 1]; lower values yield more hits.
        """
        self.refresh_embeddings()
        if not self.chunks or self.embeddings is None:
            logger.error("Knowledge base not initialized")
            return []

        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter by threshold and prepare results
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])

            if similarity < similarity_threshold:
                continue

            chunk = self.chunks[idx]
            result = chunk.to_dict()
            result['similarity_score'] = similarity
            results.append(result)

        logger.info(
            f"Search query: '{query}' - Found {len(results)} relevant chunks")

        return results

# on-demand
if __name__ == "__main__":
    KnowledgeBaseServiceMultiFormat().refresh_embeddings()
