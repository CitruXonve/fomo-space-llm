"""
Text chunking strategies for the knowledge base: legacy paragraph/sentence splits
and semantic embedding-breakpoint splits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import settings


class ChunkingStrategy(str, Enum):
    LEGACY = "legacy"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class SemanticChunkSettings:
    breakpoint_percentile: float
    min_chunk_chars: int
    max_chunk_chars: int
    buffer_size: int

    @classmethod
    def from_settings(cls) -> SemanticChunkSettings:
        return cls(
            breakpoint_percentile=settings.SEMANTIC_BREAKPOINT_PERCENTILE,
            min_chunk_chars=settings.SEMANTIC_MIN_CHUNK_CHARS,
            max_chunk_chars=settings.SEMANTIC_MAX_CHUNK_CHARS,
            buffer_size=settings.SEMANTIC_BUFFER_SIZE,
        )


def chunking_strategy_from_settings(value: str | None = None) -> ChunkingStrategy:
    raw = (value or settings.CHUNKING_STRATEGY).lower().strip()
    if raw == ChunkingStrategy.SEMANTIC.value:
        return ChunkingStrategy.SEMANTIC
    return ChunkingStrategy.LEGACY


_BULLET_LINE = re.compile(r"^[\s]*[-*•]\s+.+", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_ABBREV_PROTECT = re.compile(
    r"\b(e\.g\.|i\.e\.|U\.S\.|etc\.|vs\.|Dr\.|Mr\.|Mrs\.|Ms\.)\b",
    re.IGNORECASE,
)


def normalize_section_text(text: str) -> str:
    """Collapse hard line breaks inside paragraphs while keeping block structure."""
    text = text.strip()
    if not text:
        return ""
    blocks = re.split(r"\n\s*\n", text)
    normalized_blocks = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if all(_BULLET_LINE.match(ln) or len(ln) < 80 for ln in lines):
            normalized_blocks.append("\n".join(lines))
        else:
            normalized_blocks.append(" ".join(lines))
    return "\n\n".join(normalized_blocks)


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into retrieval units: bullets as separate units, prose by sentence.
    """
    text = normalize_section_text(text)
    if not text:
        return []

    units: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) > 1 and all(
            _BULLET_LINE.match(ln) or (len(ln) < 80 and not ln.endswith("."))
            for ln in lines
        ):
            units.extend(lines)
            continue

        protected = _ABBREV_PROTECT.sub(
            lambda m: m.group(0).replace(".", "<DOT>"), block
        )
        parts = _SENTENCE_SPLIT.split(protected)
        for part in parts:
            restored = part.replace("<DOT>", ".").strip()
            if restored:
                units.append(restored)

    if not units and text.strip():
        units = [text.strip()]
    return units


def _recursive_fallback_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def legacy_chunk_section(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Paragraph/sentence split with overlap (former KnowledgeBaseService._chunk_section)."""
    if len(content) <= chunk_size:
        return [content]

    paragraphs = re.split(r"\n\s*\n", content)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_length = len(para)

        if para_length > chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            temp_chunk: list[str] = []
            temp_length = 0

            for sentence in sentences:
                if temp_length + len(sentence) > chunk_size and temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                    overlap_sentences = (
                        temp_chunk[-2:] if len(temp_chunk) >= 2 else temp_chunk
                    )
                    temp_chunk = list(overlap_sentences)
                    temp_length = sum(len(s) for s in temp_chunk)

                temp_chunk.append(sentence)
                temp_length += len(sentence)

            if temp_chunk:
                chunks.append(" ".join(temp_chunk))
            continue

        if current_length + para_length > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            if current_chunk:
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(para)
        current_length += para_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def _enforce_size_limits(
    chunks: list[str],
    semantic_settings: SemanticChunkSettings,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        if merged and len(merged[-1]) < semantic_settings.min_chunk_chars:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk.strip())

    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= semantic_settings.max_chunk_chars:
            final.append(chunk)
        else:
            final.extend(
                _recursive_fallback_split(
                    chunk,
                    semantic_settings.max_chunk_chars,
                    chunk_overlap,
                )
            )

    return [c for c in final if c.strip()]


def semantic_chunk(
    text: str,
    model: SentenceTransformer,
    semantic_settings: SemanticChunkSettings | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text at embedding-similarity breakpoints between sentence units.
    """
    semantic_settings = semantic_settings or SemanticChunkSettings.from_settings()
    chunk_size = chunk_size or settings.EMBEDDING_MODEL_CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.EMBEDDING_MODEL_CHUNK_OVERLAP

    text = normalize_section_text(text)
    if not text:
        return []

    if len(text) <= semantic_settings.min_chunk_chars:
        return [text]

    units = split_into_sentences(text)
    if not units:
        return [text]

    if len(units) == 1:
        if len(units[0]) <= semantic_settings.max_chunk_chars:
            return units
        return _recursive_fallback_split(
            units[0], semantic_settings.max_chunk_chars, chunk_overlap
        )

    # Buffer: compare embeddings of grouped windows when buffer_size > 1
    if semantic_settings.buffer_size > 1 and len(units) > semantic_settings.buffer_size:
        buffered_units: list[str] = []
        for i in range(0, len(units), semantic_settings.buffer_size):
            buffered_units.append(" ".join(units[i : i + semantic_settings.buffer_size]))
        compare_units = buffered_units
    else:
        compare_units = units

    embeddings = model.encode(compare_units, convert_to_numpy=True)
    if len(embeddings) == 1:
        combined = " ".join(units)
        if len(combined) <= semantic_settings.max_chunk_chars:
            return [combined]
        return _recursive_fallback_split(
            combined, semantic_settings.max_chunk_chars, chunk_overlap
        )

    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(
            cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1),
            )[0, 0]
        )
        similarities.append(sim)

    distances = [1.0 - s for s in similarities]
    threshold = float(
        np.percentile(distances, semantic_settings.breakpoint_percentile)
    )

    break_after_buffered: list[int] = []
    for i, dist in enumerate(distances):
        if dist >= threshold:
            break_after_buffered.append(i)

    # Map buffered breaks back to unit indices
    if semantic_settings.buffer_size > 1 and compare_units is not units:
        break_after_unit: list[int] = []
        for buf_idx in break_after_buffered:
            unit_idx = min(
                (buf_idx + 1) * semantic_settings.buffer_size - 1,
                len(units) - 1,
            )
            if not break_after_unit or break_after_unit[-1] != unit_idx:
                break_after_unit.append(unit_idx)
    else:
        break_after_unit = break_after_buffered

    chunks: list[str] = []
    start = 0
    for break_idx in break_after_unit:
        end = break_idx + 1
        chunk_text = " ".join(units[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start = end
    tail = " ".join(units[start:]).strip()
    if tail:
        chunks.append(tail)

    if not chunks:
        chunks = [" ".join(units)]

    return _enforce_size_limits(chunks, semantic_settings, chunk_size, chunk_overlap)


def chunk_section(
    content: str,
    strategy: ChunkingStrategy,
    *,
    model: SentenceTransformer,
    chunk_size: int,
    chunk_overlap: int,
    semantic_settings: SemanticChunkSettings | None = None,
) -> list[str]:
    """Chunk one section body using the selected strategy."""
    if not content or not content.strip():
        return []

    if strategy == ChunkingStrategy.SEMANTIC:
        return semantic_chunk(
            content,
            model,
            semantic_settings=semantic_settings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return legacy_chunk_section(content, chunk_size, chunk_overlap)
