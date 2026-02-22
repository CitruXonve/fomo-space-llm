"""
Compatibility patch for langchain-anthropic streaming.

The Anthropic SDK can return `container` and `context_management` as plain dicts
in some streaming scenarios (e.g. with extended thinking or web search tools),
but langchain-anthropic expects Pydantic models and calls `.model_dump()` on them.
This causes: AttributeError("'dict' object has no attribute 'model_dump'").

This patch adds a safe conversion that handles both dicts and Pydantic models.
"""

from typing import Any


def _to_json_dict(obj: Any) -> dict[str, Any] | None:
    """Convert Pydantic model or dict to JSON-serializable dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def apply_langchain_anthropic_patch() -> None:
    """Patch langchain_anthropic to handle dict container/context_management."""
    import langchain_anthropic.chat_models as chat_models

    original = chat_models._make_message_chunk_from_anthropic_event

    def _patched(event: Any, *, stream_usage: bool = True, coerce_content_to_string: bool = False, block_start_event: Any = None):  # noqa: E501
        try:
            return original(
                event,
                stream_usage=stream_usage,
                coerce_content_to_string=coerce_content_to_string,
                block_start_event=block_start_event,
            )
        except AttributeError as e:
            if "'dict' object has no attribute 'model_dump'" not in str(e):
                raise
            # Handle message_delta case where container/context_management are dicts
            if event.type != "message_delta" or not stream_usage:
                raise
            usage_metadata = chat_models._create_usage_metadata(event.usage)
            response_metadata = {
                "stop_reason": event.delta.stop_reason,
                "stop_sequence": event.delta.stop_sequence,
            }
            if context_management := getattr(event, "context_management", None):
                response_metadata["context_management"] = _to_json_dict(context_management)
            message_delta = getattr(event, "delta", None)
            if message_delta and (container := getattr(message_delta, "container", None)):
                response_metadata["container"] = _to_json_dict(container)
            message_chunk = chat_models.AIMessageChunk(
                content="" if coerce_content_to_string else [],
                usage_metadata=usage_metadata,
                response_metadata=response_metadata,
            )
            if message_chunk.response_metadata.get("stop_reason"):
                message_chunk.chunk_position = "last"
            message_chunk.response_metadata["model_provider"] = "anthropic"
            return message_chunk, block_start_event

    chat_models._make_message_chunk_from_anthropic_event = _patched
