def normalize_content(content) -> str:
    """
    Extract text from content, handling both str and list-of-blocks formats.
    LangChain AIMessageChunk.content can be a string or a list of content blocks
    (e.g. [{"type": "text", "text": "..."}]) when using tools or multi-modal output.
    """
    if isinstance(content, str):
        return f"&nbsp;{content}&nbsp;"
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", "") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content is not None else ""
