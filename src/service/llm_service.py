import json
import logging
import re
from abc import ABC, abstractmethod
from typing import AsyncGenerator, TypedDict
from src.service.knowledge_base import KnowledgeBaseService
from src.config.settings import settings
from langchain.agents import create_agent
from langchain.messages import AnyMessage, HumanMessage, AIMessageChunk
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)


class LLMService(ABC):
    @abstractmethod
    def __init__(self, kb_service: KnowledgeBaseService):
        pass

    @abstractmethod
    async def generate_response(self, user_message: str, chat_history: list = []) -> tuple[list[AnyMessage, list[dict]]]:
        pass

    @abstractmethod
    async def generate_stream_response(self, user_message: str, chat_history: list = []) -> AsyncGenerator[AIMessageChunk, None]:
        pass


class PromptContext(TypedDict):
    """
    Context schema for the dynamic system prompt
    """
    kb_contexts: list[dict]


@dynamic_prompt
def build_prompt(request: ModelRequest) -> str:
    """
    Dynamic prompt middleware to build the system prompt based on the context
    """

    ROLE_DEFINITION = """
    You are a professional, tech lead with an industrial experience of senior-level (or above) software engineers for our virtual community.

    YOUR ROLE:
    You help community members, who are typically enthusiasts, job seekers, non-senior-level engineers by answering their questions using shared knowledge from our community, as well as from the external web search as supplemental resources. Your primary goal is to understand their pain points and provide relevant and meaningful guidance as a tech lead.
    Your responses should be ideally in the user's preferred programming language (if specified).
    You are also a helpful assistant for the user when they want to deep-dive into a specific topic, hear about the explanation of a concept, or broaden their knowledge.
    """

    TASK_DEFINITION = """
    - Professional yet conversational
    - Clear and easy to understand
    - Action-oriented (tell users what potential next steps/directions to take)
    - Empathetic to user concerns
    - Concise (avoid unnecessary elaboration)
    """

    FORMAT_INSTRUCTIONS = """
        - Beware of the HTML tags and markdown syntax in the response that need to be rendered correctly in the frontend.
    """
    # For example, a title/subtitle marked with leading one or more '#'-sign(s) should come with a <br/> tag in the end of the line.
    # Edge Case: No relevant KB context found
    if not request.runtime.context.get("kb_contexts"):
        return f"""
        {ROLE_DEFINITION}

        IMPORTANT CONTEXT:
        No relevant information was found in the community for the user's question.

        CRITICAL INSTRUCTIONS:
        1. Acknowledge that you don't have specific information about this topic in our community
        2. Be empathetic and professional
        3. Let the user know that additional handling is needed from external resources (e.g. web search)
        4. After searching the web, you should summarize the information briefly and make up to 3-5 recommendations with links to the external sources for the user to choose from as of the follow-up steps.
        5. The external sources from the same domain should be grouped or categorized together and listed as a single recommendation.

        RESPONSE GUIDELINES:
        {TASK_DEFINITION}

        FORMAT OF THE RESPONSE:
        {FORMAT_INSTRUCTIONS}

        TEMPLATE RESPONSE (it's ok if the actual response varies slightly from the template; no need to strictly follow the template):
        "I don't have specific information about inside our community. I'll refer to external sources for you. \n"
        """

    # Normal Case: KB context available - build structured prompt
    context_sections = []
    for index, result in enumerate(request.runtime.context.get("kb_contexts"), start=1):
        # Format each KB chunk with clear delineation
        context_sections.append(f"""
        <knowledge_source_{index}>
            <source_file>{result['source_file']}</source_file>
            <section_title>{result['heading']}</section_title>
            <relevance_score>{result['similarity_score']:.2f}</relevance_score>
            <content>
            {result['content']}
            </content>
        </knowledge_source_{index}>
        """)

    context_block = "\n".join(context_sections)

    # Build structured system prompt with context
    system_prompt = f"""
    {ROLE_DEFINITION}

    CRITICAL INSTRUCTIONS:
    1. Answer questions preferably using the information provided in the community sources below.
    2. If the community doesn't contain enough information to fully answer the question, be explicit about this before using the web_search tool.
    3. If the coding snippet is found in the community but only available in one programming language, you can rewrite the code in the user's preferred programming language.
    4. If you're uncertain or the information is incomplete, acknowledge this clearly before using the web_search tool.
    5. Be concise but complete - aim for 2-4 sentences unless more detail is clearly needed before using the web_search tool (without too much elaboration after searching the web).
    6. After searching the web, you should summarize the information briefly and make up to 3-5 recommendations with links to the external sources for the user to choose from as of the follow-up steps.
    7. The external sources from the same domain should be grouped or categorized together and listed as a single recommendation.
    8. Use a friendly, professional, and empathetic tone.

    COMMUNITY SOURCES:
    {context_block}

    RESPONSE STYLE:
    {TASK_DEFINITION}

    FORMAT OF THE RESPONSE:
    {FORMAT_INSTRUCTIONS}
    """
    return system_prompt


class ClaudeLLMService(LLMService):
    def __init__(self, kb_service: KnowledgeBaseService):
        self.kb_service = kb_service
        self.agent = create_agent(
            model=ChatAnthropic(
                model=settings.CLAUDE_MODEL,
                temperature=settings.CLAUDE_TEMPERATURE,
                max_tokens=settings.CLAUDE_MAX_TOKENS
            ),
            middleware=[build_prompt],
            context_schema=PromptContext,
            tools=[{
                "name": "web_search",
                "type": settings.CLAUDE_WEB_SEARCH_TOOL,
                "max_uses": 5
            }])

    async def generate_response(
        self,
        user_message: str,
        chat_history: list[AnyMessage] = []
    ) -> tuple[list[AnyMessage, list[dict]]]:
        # Retrieve relevant KB chunks
        context = self.kb_service.search(user_message)

        # Format messages for Claude
        formatted_messages: list[AnyMessage] = self._format_messages_for_claude(
            user_message, chat_history)

        # Call Claude API and get response
        new_messages: list[AnyMessage] = await self._call_claude(
            context,
            formatted_messages)

        return new_messages, context

    async def _call_claude(
        self,
        context: list[dict],
        messages: list[dict[str, str]]
    ) -> list[AnyMessage]:
        try:
            response = await self.agent.ainvoke({
                "messages": messages
            }, context={"kb_contexts": context})

            return response["messages"]
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return None

    async def generate_stream_response(
        self,
        user_message: str,
        chat_history: list[AnyMessage] = []
    ) -> AsyncGenerator[AIMessageChunk, None]:
        # Retrieve relevant KB chunks
        context = self.kb_service.search(user_message)

        # Format messages for Claude
        formatted_messages: list[AnyMessage] = self._format_messages_for_claude(
            user_message, chat_history)

        # Call Claude API and get response (message chunks)
        async for message_chunk in self._stream_from_claude(
                context,
                formatted_messages):
            yield message_chunk

    async def _stream_from_claude(self, context: list[dict], messages: list[dict[str, str]]) -> AsyncGenerator[AIMessageChunk, None]:
        try:
            async for token, _ in self.agent.astream({
                "messages": messages
            }, context={"kb_contexts": context}, stream_mode="messages"):
                yield token  # <class 'langchain_core.messages.ai.AIMessageChunk'>
        except Exception as e:
            logger.error(f"Error streaming from Claude API: {e}")

    def _format_messages_for_claude(
        self,
        user_message: str,
        chat_history: list[AnyMessage] = []
    ) -> list[AnyMessage]:
        # Add existing chat history and current user message
        return [*chat_history, HumanMessage(content=user_message)]


def evaluate_confidence(response: str, context: list[dict], confidence_threshold: float = 0.6) -> dict:
    confidence_score = 0.5  # Neutral starting point

    # Factor 1: KB Retrieval Quality (±0.4)
    if not context:
        confidence_score -= 0.4  # No context = low confidence
    else:
        # Average similarity of retrieved chunks
        avg_similarity = sum(r["similarity_score"]
                             for r in context) / len(context)
        best_similarity = max(r["similarity_score"] for r in context)

        # Scale: 0.3-0.9 similarity -> -0.2 to +0.4 confidence
        kb_confidence = (avg_similarity - 0.5) * 0.6 + \
            (best_similarity - 0.5) * 0.2
        confidence_score += kb_confidence

        logger.debug(
            f"KB quality: avg_sim={avg_similarity:.2f}, best_sim={best_similarity:.2f}, contrib={kb_confidence:.2f}")

    # Factor 2: Uncertainty Indicators (±0.3)
    uncertainty_phrases = [
        "i don't have",
        "i don't know",
        "not sure",
        "unclear",
        "can't find",
        "no information",
        "unable to",
        "don't have complete information",
        "don't have specific information",
        "i'll ask for additional support",
        "our team can provide",
        "limited information",
        "not certain"
    ]

    response_lower = response.lower()
    uncertainty_count = sum(
        1 for phrase in uncertainty_phrases if phrase in response_lower)

    if uncertainty_count > 0:
        uncertainty_penalty = min(0.3, uncertainty_count * 0.15)
        confidence_score -= uncertainty_penalty
        logger.debug(
            f"Uncertainty detected: {uncertainty_count} phrases, penalty={uncertainty_penalty:.2f}")

    # Factor 3: Response Completeness (±0.2)
    response_length = len(response)

    if response_length > 200:
        # Detailed response suggests confidence
        confidence_score += 0.15
    elif response_length > 100:
        confidence_score += 0.05
    elif response_length < 50:
        # Very short response might indicate uncertainty
        confidence_score -= 0.1

    # Check for actionable content (steps, instructions)
    actionable_indicators = [
        r'\d+\.',  # Numbered lists
        '- ',      # Bullet points
        'step',
        'first',
        'then',
        'click',
        'navigate',
        'go to'
    ]

    has_actionable = any(re.search(indicator, response_lower)
                         for indicator in actionable_indicators)
    if has_actionable:
        confidence_score += 0.05

    logger.debug(
        f"Response length: {response_length}, actionable: {has_actionable}")

    # Factor 4: Context Utilization (±0.1)
    if context:
        # Extract significant words from KB content
        kb_words = set()
        for result in context:
            # Words 5+ chars
            words = re.findall(r'\b\w{5,}\b', result['content'].lower())
            kb_words.update(words)

        # Extract words from response
        response_words = set(re.findall(r'\b\w{5,}\b', response_lower))

        # Calculate overlap
        overlap = len(kb_words & response_words)
        overlap_ratio = overlap / len(kb_words) if kb_words else 0

        if overlap > 8:
            confidence_score += 0.1
        elif overlap > 4:
            confidence_score += 0.05

        logger.debug(
            f"Context utilization: {overlap} shared words, ratio={overlap_ratio:.2f}")

    # Clamp to valid range
    final_score = max(0.0, min(1.0, confidence_score))

    logger.info(
        f"Confidence evaluation: {final_score:.2f} (threshold: {confidence_threshold})")

    return {
        "confidence": final_score,
        "threshold": confidence_threshold,
        "needs_attention": final_score < confidence_threshold
    }
