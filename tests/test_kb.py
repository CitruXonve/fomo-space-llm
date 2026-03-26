import json
import time
from src.config.settings import settings
from src.service.knowledge_base import KnowledgeBaseServiceMultiFormat
import unittest


class TestKBService(unittest.TestCase):
    kb_service = None  # Class-level attribute

    @classmethod
    def setUpClass(cls):
        """Run once before all tests in this class."""
        start_time = time.time()
        cls.kb_service = KnowledgeBaseServiceMultiFormat()
        cls.kb_service.get_all_sources()
        end_time = time.time()
        print(
            f"Time taken to initialize knowledge base with embeddings: {end_time - start_time} seconds")

    def test_get_stats(self):
        self.assertIsNotNone(self.kb_service.model)
        self.assertGreater(len(self.kb_service.chunks), 0)
        stats = self.kb_service.get_stats()
        self.assertGreater(stats["total_chunks"], 0)
        self.assertGreater(stats["total_sources"], 0)
        self.assertGreater(stats["embedding_dimensions"], 0)
        # Print stats
        print(
            f"Knowledge Base Stats: {json.dumps({k: v for k, v in stats.items() if k != 'model_details' and k != 'sources'}, indent=4)}")
        print(
            f"Model details: {stats['model_details']}")
        print(
            f"Example sources of Markdown: {[source for source in stats['sources'] if source.endswith('.md')][:3]}")
        print(
            f"Example sources of PDF: {[source for source in stats['sources'] if source.endswith('.pdf')][:3]}")

    def test_query_1(self):
        query = "What is OLTP/ OLAP and their use cases?"
        search_results = self.kb_service.search(query, top_k=2)
        self.assertGreater(len(search_results), 0)
        print(f"Search results for '{query}': {len(search_results)}")
        print(
            f"Example results: {json.dumps([result for result in search_results if result['similarity_score'] > settings.DEFAULT_SIMILARITY_THRESHOLD][:2], indent=4)}")

    def test_query_2(self):
        query = "How to prepare for a Java interview?"
        search_results = self.kb_service.search(query, top_k=8)
        self.assertGreater(len(search_results), 0)
        print(f"Search results for '{query}': {len(search_results)}")
        print(
            f"Example results: {json.dumps([result for result in search_results if result['similarity_score'] > settings.DEFAULT_SIMILARITY_THRESHOLD][:2], indent=4)}")


if __name__ == "__main__":
    unittest.main()
