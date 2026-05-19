#!/usr/bin/env python3
"""
A/B comparison: legacy vs semantic chunking stats and retrieval Hit@k.

Usage:
  poetry run python scripts/compare_chunking.py
  poetry run python scripts/compare_chunking.py --sources path/to/file.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.settings import settings
from src.service.chunking import (
    ChunkingStrategy,
    SemanticChunkSettings,
    chunk_section,
)
from src.service.file_parser import ParserFactory
SUPPORTED_SUFFIXES = {".md", ".pdf", ".txt"}
DEFAULT_QUERIES = _ROOT / "tests/fixtures/retrieval_queries.jsonl"


@dataclass
class ChunkRecord:
    content: str
    heading: str
    source_file: str


def _add_context(heading: str, content: str) -> str:
    if heading and heading.lower() != "introduction":
        return f"{heading}: {content}"
    return content


def collect_sources(sources_arg: str | None) -> list[Path]:
    if sources_arg:
        path = Path(sources_arg)
        if path.is_file():
            return [path]
        if path.is_dir():
            files = [
                f
                for f in sorted(path.rglob("*"))
                if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
            ]
            return files
        raise FileNotFoundError(f"Sources path not found: {path}")

    kb_dir = Path(settings.KB_DIRECTORY)
    if not kb_dir.exists():
        return []
    return [
        f
        for f in sorted(kb_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def build_chunks_for_strategy(
    strategy: ChunkingStrategy,
    source_files: list[Path],
    model: SentenceTransformer,
    semantic_settings: SemanticChunkSettings,
) -> list[ChunkRecord]:
    factory = ParserFactory()
    parse_args = {
        "prechunk": strategy == ChunkingStrategy.LEGACY,
        "chunk_size": settings.EMBEDDING_MODEL_CHUNK_SIZE,
        "chunk_overlap": settings.EMBEDDING_MODEL_CHUNK_OVERLAP,
    }
    records: list[ChunkRecord] = []

    for file_path in source_files:
        try:
            sections = factory.parse_file(file_path, args=parse_args)
        except Exception as exc:
            print(f"  skip {file_path.name}: {exc}", file=sys.stderr)
            continue

        for heading, body in sections:
            for chunk_text in chunk_section(
                body,
                strategy,
                model=model,
                chunk_size=settings.EMBEDDING_MODEL_CHUNK_SIZE,
                chunk_overlap=settings.EMBEDDING_MODEL_CHUNK_OVERLAP,
                semantic_settings=semantic_settings,
            ):
                records.append(
                    ChunkRecord(
                        content=_add_context(heading, chunk_text),
                        heading=heading,
                        source_file=file_path.name,
                    )
                )
    return records


def chunk_stats(records: list[ChunkRecord]) -> dict:
    lengths = [len(r.content) for r in records]
    if not lengths:
        return {
            "count": 0,
            "mean_chars": 0,
            "median_chars": 0,
            "p95_chars": 0,
            "tiny_count": 0,
            "empty_count": 0,
        }
    sorted_len = sorted(lengths)
    p95_idx = max(0, int(len(sorted_len) * 0.95) - 1)
    return {
        "count": len(records),
        "mean_chars": round(mean(lengths), 1),
        "median_chars": round(median(lengths), 1),
        "p95_chars": sorted_len[p95_idx],
        "tiny_count": sum(1 for n in lengths if n < settings.SEMANTIC_MIN_CHUNK_CHARS),
        "empty_count": sum(1 for n in lengths if n == 0),
    }


def load_queries(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Queries file not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def hit_at_k(
    query: str,
    records: list[ChunkRecord],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    k: int,
    expected_substrings: list[str],
    optional_heading_contains: str | None,
) -> dict:
    if not records or embeddings.size == 0:
        return {"hit_at_1": False, "hit_at_k": False, "mrr": 0.0, "top1_preview": ""}

    q_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    ranked = np.argsort(sims)[::-1][:k]

    def is_relevant(idx: int) -> bool:
        rec = records[idx]
        text_lower = rec.content.lower()
        if not all(s.lower() in text_lower for s in expected_substrings):
            return False
        if optional_heading_contains:
            return optional_heading_contains.lower() in rec.heading.lower()
        return True

    mrr = 0.0
    hit_at_1 = False
    hit_at_k = False
    for rank, idx in enumerate(ranked, start=1):
        if is_relevant(int(idx)):
            hit_at_k = True
            mrr = 1.0 / rank
            if rank == 1:
                hit_at_1 = True
            break

    top_idx = int(ranked[0])
    preview = records[top_idx].content[:200].replace("\n", " ")
    return {
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "mrr": round(mrr, 4),
        "top1_preview": preview,
    }


def evaluate_retrieval(
    queries: list[dict],
    records: list[ChunkRecord],
    model: SentenceTransformer,
    k: int,
) -> dict:
    if not records:
        return {"queries": [], "hit_at_1_rate": 0.0, "hit_at_k_rate": 0.0, "mean_mrr": 0.0}

    texts = [r.content for r in records]
    embeddings = model.encode(texts, convert_to_numpy=True)
    per_query = []
    for row in queries:
        result = hit_at_k(
            row["query"],
            records,
            embeddings,
            model,
            k,
            row.get("expected_substrings", []),
            row.get("optional_heading_contains"),
        )
        per_query.append({"query": row["query"], **result})

    n = len(per_query) or 1
    return {
        "queries": per_query,
        "hit_at_1_rate": round(sum(1 for q in per_query if q["hit_at_1"]) / n, 4),
        "hit_at_k_rate": round(sum(1 for q in per_query if q["hit_at_k"]) / n, 4),
        "mean_mrr": round(mean(q["mrr"] for q in per_query), 4),
    }


def print_strategy_report(
    name: str, records: list[ChunkRecord], retrieval: dict, k: int
) -> None:
    stats = chunk_stats(records)
    print(f"\n=== {name} ===")
    print(
        f"  chunks: {stats['count']}  "
        f"mean/median/p95 chars: {stats['mean_chars']}/{stats['median_chars']}/{stats['p95_chars']}  "
        f"tiny: {stats['tiny_count']}"
    )
    print(
        f"  Hit@1: {retrieval['hit_at_1_rate']:.2%}  "
        f"Hit@{k}: {retrieval['hit_at_k_rate']:.2%}  "
        f"MRR: {retrieval['mean_mrr']:.4f}"
    )
    samples = records[:3]
    for i, rec in enumerate(samples, 1):
        preview = rec.content[:120].replace("\n", " ")
        print(f"  sample[{i}]: {preview}...")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare legacy vs semantic chunking")
    parser.add_argument(
        "--sources",
        default=None,
        help="File or directory (default: all files in KB_DIRECTORY)",
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES),
        help="JSONL file with golden queries",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k for Hit@k")
    parser.add_argument(
        "--export",
        action="store_true",
        default=True,
        help="Write JSON report to EXPORT_DIRECTORY (default: on)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip writing JSON report",
    )
    args = parser.parse_args()

    source_files = collect_sources(args.sources)
    if not source_files:
        print("No source files found.", file=sys.stderr)
        return 1

    queries_path = Path(args.queries)
    try:
        queries = load_queries(queries_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Sources ({len(source_files)}):")
    for f in source_files:
        print(f"  - {f.name}")

    semantic_settings = SemanticChunkSettings.from_settings()
    model = SentenceTransformer(
        settings.EMBEDDING_MODEL,
        cache_folder=settings.EMBEDDING_MODEL_CACHE_DIR,
    )

    legacy_records = build_chunks_for_strategy(
        ChunkingStrategy.LEGACY, source_files, model, semantic_settings
    )
    semantic_records = build_chunks_for_strategy(
        ChunkingStrategy.SEMANTIC, source_files, model, semantic_settings
    )

    legacy_retrieval = evaluate_retrieval(queries, legacy_records, model, args.k)
    semantic_retrieval = evaluate_retrieval(queries, semantic_records, model, args.k)

    print_strategy_report("legacy", legacy_records, legacy_retrieval, args.k)
    print_strategy_report("semantic", semantic_records, semantic_retrieval, args.k)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": [str(p) for p in source_files],
        "k": args.k,
        "legacy": {
            "stats": chunk_stats(legacy_records),
            "retrieval": legacy_retrieval,
        },
        "semantic": {
            "stats": chunk_stats(semantic_records),
            "retrieval": semantic_retrieval,
        },
    }

    if args.export and not args.no_export:
        export_dir = Path(settings.EXPORT_DIRECTORY)
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = export_dir / f"compare_chunking_{stamp}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
