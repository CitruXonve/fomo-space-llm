from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.service.llm_service import ClaudeLLMService
from src.service.session_service import SessionService
from src.utility.content_helper import normalize_content

router = APIRouter(prefix="/api/chat", tags=["chat"])
stream_router = APIRouter(prefix="/api/chat-stream", tags=["chat-stream"])

# Dependency functions to get services from app.state


def get_llm_service(request: Request) -> ClaudeLLMService:
    """Dependency to get LLM service from app state."""
    return request.app.state.llm_service


def get_session_service(request: Request) -> SessionService:
    """Dependency to get session service from app state."""
    return request.app.state.session_service


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    request: ChatRequest,
    llm_service: ClaudeLLMService = Depends(get_llm_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    POST /api/chat
    Body: {
        "session_id": "uuid",
        "message": "How do I reset my password?",
    }

    Response: {
        "status": "success",
        "response": "To reset your password...",
    }
    """
    if not request.message or type(request.message) != str or len(request.message) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Message is required and must be a non-empty string")

    session_id = request.session_id
    if not session_id:
        session_id = session_service.create_session()

    chat_history = session_service.get_history(session_id)

    response, _ = await llm_service.generate_response(request.message, chat_history)

    if not response or type(response) != list or len(response) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No response from LLM")

    session_service.set_history(session_id, response)

    return {
        "status": "success",
        "session_id": session_id,
        "messages": response
    }


@stream_router.post("")
async def create_chat_stream(
    request: ChatRequest,
    llm_service: ClaudeLLMService = Depends(get_llm_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    POST /api/chat-stream
    Body: {
        "session_id": "uuid",
        "message": "How do I perform back-of-envelope estimation?",
    }

    Response: {
        "status": "success",
        "response": "To perform back-of-envelope estimation...",
    }
    """
    if not request.message or type(request.message) != str or len(request.message) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Message is required and must be a non-empty string")

    session_id = request.session_id
    if not session_id:
        session_id = session_service.create_session()

    chat_history = session_service.get_history(session_id)

    def _escape_html(content: str) -> str:
        """
        Escape HTML tags and markdown syntax in the content.
        """
        # return content.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;").replace("##", "<br/> ##").replace("\n", "  \n")
        return content.replace("##", "<br/> ##").replace("\n", "  \n")

    async def _stream_generator(request: ChatRequest, llm_service: ClaudeLLMService, chat_history):
        """
        An async generator that yields LLM tokens formatted as Server-Sent Events (SSE).
        """
        async for message_chunk in llm_service.generate_stream_response(request.message, chat_history):
            yield _escape_html(normalize_content(message_chunk.content))

    return StreamingResponse(
        _stream_generator(request, llm_service, chat_history),
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        media_type="text/event-stream",
        status_code=status.HTTP_200_OK
    )


@router.get("/{session_id}", status_code=status.HTTP_200_OK)
async def fetch_chat_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service)
):
    """
    GET /api/chat/{session_id}
    """
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Session ID is required")

    chat_history = session_service.get_history(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "messages": chat_history
    }
