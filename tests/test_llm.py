import json
import time
import asyncio
import unittest

from langchain.messages import AIMessage
from langchain_core.messages import HumanMessage
from src.service.llm_service import ClaudeLLMService, evaluate_confidence
from src.service.knowledge_base import KnowledgeBaseServiceMarkdown
from src.utility.content_helper import normalize_content
from src.utility.spinner import Spinner


class TestLLM(unittest.IsolatedAsyncioTestCase):
    llm_service = None  # Class-level attribute

    @classmethod
    def setUpClass(cls):
        """Run once before all tests in this class."""
        start_time = time.time()
        cls.llm_service = ClaudeLLMService(KnowledgeBaseServiceMarkdown())
        end_time = time.time()
        print(
            f"Time taken to initialize LLM service: {(end_time - start_time):.2f} seconds")
        print("Testing LLM context prompt...")

    async def _sent_user_message(self, user_message: str, start_time: float, chat_history: list[dict]):
        print(f"Sending user message: {user_message}")
        resp, context = await self.llm_service.generate_response(user_message, chat_history)
        assert resp is not None and len(resp) > 0
        chat_history.clear()
        chat_history.extend(resp)
        last_content = normalize_content(resp[-1].content)
        evaluation = evaluate_confidence(last_content, context)
        end_time = time.time()
        print(
            f"Time taken to generate response: {(end_time - start_time):.2f} seconds")
        print("LLM response length:", len(last_content))
        print(
            "LLM confidence:", json.dumps(evaluation, indent=4))
        return end_time

    async def test_claude_llm_context_prompt(self):
        chat_history = []
        start_time = time.time()
        end_time = await self._sent_user_message("What is multi-threading in Python?", start_time, chat_history)
        self.assertEqual(len(chat_history), 2)
        print(
            f"Time taken to send user message: {(end_time - start_time):.2f} seconds")
        print(
            f"Chat response: {len(chat_history)}", json.dumps({"role": chat_history[-1].type, "content": chat_history[-1].content}, indent=4))
        start_time = time.time()
        end_time = await self._sent_user_message("What is OLTP/ OLAP and their use cases?", start_time, chat_history)
        self.assertEqual(len(chat_history), 4)
        print(
            f"Time taken to send user message: {(end_time - start_time):.2f} seconds")
        print(
            f"Chat response: {len(chat_history)}", json.dumps({"role": chat_history[-1].type, "content": chat_history[-1].content}, indent=4))
        start_time = time.time()
        end_time = await self._sent_user_message("How to prepare for a Java interview?", start_time, chat_history)
        self.assertEqual(len(chat_history), 6)
        print(
            f"Time taken to send user message: {(end_time - start_time):.2f} seconds")
        print(
            f"Chat response: {len(chat_history)}", json.dumps({"role": chat_history[-1].type, "content": chat_history[-1].content}, indent=4))

    async def _stream_user_message(self, user_message: str, start_time: float, chat_history: list[dict]):
        chat_history.append(HumanMessage(content=user_message))
        print(f"Sending user message: {user_message}")
        spinner = Spinner()
        message_chunks = []
        async for message_chunk in self.llm_service.generate_stream_response(user_message, chat_history):
            spinner.spin(type_str="tokens")
            message_chunks.append(normalize_content(message_chunk.content))
        end_time = time.time()
        spinner.finish(
            message=f"Done streaming {len(message_chunks)} chunks. Total time taken: {(end_time - start_time):.2f} seconds")
        self.assertGreater(len(message_chunks), 1)
        combined = ''.join(message_chunks)
        self.assertGreater(len(combined), 10)
        print("Stream response:\n", combined)
        chat_history.append(AIMessage(content=combined))

        return end_time

    async def test_claude_llm_stream_prompt(self):
        chat_history = []
        start_time = time.time()
        end_time = await self._stream_user_message("What is multi-threading in Python?", start_time, chat_history)
        self.assertEqual(len(chat_history), 2)
        print(
            f"Time taken to stream response: {(end_time - start_time):.2f} seconds")

        start_time = time.time()
        end_time = await self._stream_user_message("What is OLTP/ OLAP and their use cases?", start_time, chat_history)
        self.assertEqual(len(chat_history), 4)
        print(
            f"Time taken to stream response: {(end_time - start_time):.2f} seconds")


if __name__ == "__main__":
    asyncio.run(unittest.main())
