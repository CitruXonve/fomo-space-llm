"""
Knowledge Base Service
Purpose: Load, index, and search through FAQ content using semantic search.
Key Features:
- Simple text file loader (<500KB as per requirement)
- Chunk text into semantic sections
- Use sentence-transformers for embeddings (local, no API cost)
- Cosine similarity search for retrieval
- Embedding cache for faster startup when KB unchanged
"""

import re
import json
import hashlib
import numpy as np
import logging
from abc import ABC, abstractmethod

from typing import Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import settings

logger = logging.getLogger(__name__)


class MarkdownChunk:
    """Represents a chunk of knowledge base content with metadata"""

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
    def search(self, query: str) -> list[dict]:
        """Semantic search using embeddings"""
        # Returns relevant KB chunks
        pass


class KnowledgeBaseServiceMarkdown(KnowledgeBaseService):
    """
    Handles markdown knowledge base loading, chunking, and semantic search.

    Strategy:
    1. Load all .md files from knowledge_base/ directory
    2. Parse markdown structure (headers, sections)
    3. Chunk by sections while preserving context
    4. Generate embeddings using sentence-transformers
    5. Perform semantic search using cosine similarity
    """

    def __init__(self):
        self.kb_directory = Path(settings.KB_DIRECTORY)
        self.chunk_size = settings.EMBEDDING_MODEL_CHUNK_SIZE
        self.chunk_overlap = settings.EMBEDDING_MODEL_CHUNK_OVERLAP

        # Embedding cache paths
        self.cache_dir = Path(settings.EMBEDDING_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_cache_file = self.cache_dir / "embeddings.npy"
        self.chunks_cache_file = self.cache_dir / "chunks.json"
        self.hash_cache_file = self.cache_dir / "kb_hash.txt"

        # Load embedding model (cached locally in EMBEDDING_MODEL_CACHE_DIR)
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            cache_folder=settings.EMBEDDING_MODEL_CACHE_DIR
        )
        logger.info("Embedding model loaded successfully")

        # Initialize storage
        self.chunks: list[MarkdownChunk] = []
        self.embeddings: Optional[np.ndarray] = None

        # Compute current KB hash
        current_hash = self._compute_kb_hash()

        # Try to load from cache if KB unchanged
        if self._load_from_cache(current_hash):
            logger.info(f"Loaded {len(self.chunks)} chunks from cache")
        else:
            # Load and process knowledge base from scratch
            logger.info("Cache miss or invalid - regenerating embeddings...")
            self._load_knowledge_base()
            self._create_embeddings()
            self._save_to_cache(current_hash)
            logger.info(f"Knowledge base loaded: {len(self.chunks)} chunks")

    def get_all_sources(self) -> list[str]:
        """Get list of all source files in KB"""
        return list(set(chunk.source_file for chunk in self.chunks))

    def get_chunk_by_index(self, index: int) -> Optional[MarkdownChunk]:
        """Retrieve a specific chunk by index"""
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None

    def get_stats(self) -> dict:
        """Get knowledge base statistics"""
        return {
            "total_chunks": len(self.chunks),
            "total_sources": len(self.get_all_sources()),
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "sources": self.get_all_sources(),
            "model_details": self.model,
        }

    def _compute_kb_hash(self) -> str:
        """
        Compute a hash of the KB directory state.
        Uses file names, sizes, and modification times to detect changes.
        """
        if not self.kb_directory.exists():
            return ""

        hash_data = []
        md_files = sorted(self.kb_directory.glob("*.md"))

        for md_file in md_files:
            stat = md_file.stat()
            hash_data.append(f"{md_file.name}:{stat.st_size}:{stat.st_mtime}")

        # Also include chunk settings in hash (if settings change, re-embed)
        hash_data.append(f"chunk_size:{self.chunk_size}")
        hash_data.append(f"chunk_overlap:{self.chunk_overlap}")
        hash_data.append(f"model:{settings.EMBEDDING_MODEL}")

        hash_string = "|".join(hash_data)
        return hashlib.sha256(hash_string.encode()).hexdigest()

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
                chunk = MarkdownChunk(
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
        """Load all markdown files from the knowledge base directory"""
        if not self.kb_directory.exists():
            raise FileNotFoundError(
                f"Knowledge base directory not found: {self.kb_directory}")

        md_files = list(self.kb_directory.glob("*.md"))

        if not md_files:
            raise ValueError(f"No markdown files found in {self.kb_directory}")

        logger.info(f"Found {len(md_files)} markdown files")

        for md_file in md_files:
            self._process_markdown_file(md_file)

    def _process_markdown_file(self, file_path: Path) -> None:
        """
        Process a single markdown file into chunks.

        Chunking strategy:
        1. Parse markdown headers to identify sections
        2. Chunk by section boundaries (respects semantic structure)
        3. If section too large, split by paragraphs with overlap
        4. Preserve header context in each chunk
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse markdown into sections
        sections = self._parse_markdown_sections(content)

        # Process each section
        for section_index, (heading, section_content) in enumerate(sections):
            # Split large sections into smaller chunks
            section_chunks = self._chunk_section(section_content)

            for chunk_index, chunk_text in enumerate(section_chunks):
                # Add header context to chunk for better retrieval
                chunk_with_context = self._add_context(heading, chunk_text)

                chunk = MarkdownChunk(
                    content=chunk_with_context,
                    source_file=file_path.name,
                    heading=heading,
                    chunk_index=section_index *
                    len(section_chunks) + chunk_index
                )
                self.chunks.append(chunk)

    def _parse_markdown_sections(self, content: str) -> list[tuple[str, str]]:
        """
        Parse markdown into sections based on headers.

        Returns: list of (heading, content) tuples
        """
        # Split by headers (# ## ### etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'

        sections = []
        current_heading = "Introduction"
        current_content = []

        lines = content.split('\n')

        for line in lines:
            match = re.match(header_pattern, line, re.MULTILINE)

            if match:
                # Save previous section
                if current_content:
                    sections.append((
                        current_heading,
                        '\n'.join(current_content).strip()
                    ))

                # Start new section
                current_heading = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            sections.append((
                current_heading,
                '\n'.join(current_content).strip()
            ))

        return sections

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
        embeddings = self.model.encode(
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
        Semantic search for relevant chunks.

        Args:
            query: User's question
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of chunk dictionaries with similarity scores
        """
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

if __name__ == "__main__":
    KnowledgeBaseServiceMarkdown()