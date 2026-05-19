import unittest
from unittest.mock import MagicMock

import numpy as np

from src.config.settings import settings
from src.service.chunking import (
    ChunkingStrategy,
    SemanticChunkSettings,
    chunk_section,
    legacy_chunk_section,
    normalize_section_text,
    semantic_chunk,
    split_into_sentences,
)


class TestSplitIntoSentences(unittest.TestCase):
    def test_oltp_olap_preserved(self):
        text = "OLTP handles transactions. OLAP supports analytics."
        units = split_into_sentences(text)
        joined = " ".join(units)
        self.assertIn("OLTP", joined)
        self.assertIn("OLAP", joined)

    def test_bullet_lines(self):
        text = "- First point\n- Second point"
        units = split_into_sentences(text)
        self.assertGreaterEqual(len(units), 2)

    def test_url_in_sentence(self):
        text = "See https://example.com/docs for details."
        units = split_into_sentences(text)
        self.assertEqual(1, len(units))
        self.assertIn("https://example.com", units[0])


class TestLegacyChunkSection(unittest.TestCase):
    def test_short_section_unchanged(self):
        text = "Short section."
        chunks = legacy_chunk_section(text, chunk_size=500, chunk_overlap=100)
        self.assertEqual([text], chunks)


class TestSemanticChunk(unittest.TestCase):
    def _mock_model_with_shifts(self, dim: int = 8) -> MagicMock:
        """Embeddings that diverge after the second unit (topic shift)."""
        model = MagicMock()

        def encode(texts, convert_to_numpy=True):
            vectors = []
            for i, _ in enumerate(texts):
                v = np.zeros(dim, dtype=np.float32)
                v[0] = 1.0
                if i >= 2:
                    v[1] = 1.0
                vectors.append(v)
            return np.stack(vectors)

        model.encode.side_effect = encode
        return model

    def test_topic_shift_produces_multiple_chunks(self):
        text = (
            "Databases store application data. "
            "Indexes speed up queries. "
            "Load balancers distribute traffic. "
            "Health checks remove unhealthy nodes."
        )
        settings_obj = SemanticChunkSettings(
            breakpoint_percentile=50.0,
            min_chunk_chars=10,
            max_chunk_chars=1200,
            buffer_size=1,
        )
        chunks = semantic_chunk(text, self._mock_model_with_shifts(), settings_obj)
        self.assertGreaterEqual(len(chunks), 2)

    def test_max_chunk_chars_enforced(self):
        long_sentence = "word " * 400
        text = f"{long_sentence.strip()}. Another sentence here."
        model = MagicMock()
        model.encode.return_value = np.random.rand(2, 8).astype(np.float32)
        settings_obj = SemanticChunkSettings(
            breakpoint_percentile=99.0,
            min_chunk_chars=10,
            max_chunk_chars=200,
            buffer_size=1,
        )
        chunks = semantic_chunk(text, model, settings_obj)
        for chunk in chunks:
            self.assertLessEqual(
                len(chunk),
                settings_obj.max_chunk_chars + 50,
                msg=f"chunk too long ({len(chunk)} chars)",
            )


class TestChunkSectionFacade(unittest.TestCase):
    def test_legacy_strategy(self):
        text = ". ".join(["Legacy chunk sentence number %d" % i for i in range(30)])
        mock_model = MagicMock()
        chunks = chunk_section(
            text,
            ChunkingStrategy.LEGACY,
            model=mock_model,
            chunk_size=200,
            chunk_overlap=50,
        )
        self.assertGreater(len(chunks), 1)
        mock_model.encode.assert_not_called()

    def test_semantic_strategy_calls_encode(self):
        text = (
            "Databases persist application state. "
            "Replication improves read scalability. "
            "Load balancers spread incoming requests. "
            "Caches reduce database pressure."
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]],
            dtype=np.float32,
        )
        chunks = chunk_section(
            text,
            ChunkingStrategy.SEMANTIC,
            model=mock_model,
            chunk_size=settings.EMBEDDING_MODEL_CHUNK_SIZE,
            chunk_overlap=settings.EMBEDDING_MODEL_CHUNK_OVERLAP,
        )
        self.assertGreaterEqual(len(chunks), 1)
        mock_model.encode.assert_called()


class TestNormalizeSectionText(unittest.TestCase):
    def test_collapses_wrapped_lines(self):
        raw = "This is a long line that was\nwrapped in the PDF."
        normalized = normalize_section_text(raw)
        self.assertIn("wrapped in the PDF", normalized.replace("\n", " "))


if __name__ == "__main__":
    unittest.main()
