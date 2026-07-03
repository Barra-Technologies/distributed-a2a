from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_mcp_adapters.interceptors import (MCPToolCallRequest,
                                                 MCPToolCallResult)
from mcp.types import CallToolResult, TextContent

"""Key under ``CallToolResult.structuredContent`` where the interceptor stashes
any non-text MCP content blocks. Also the key under
``ToolMessage.artifact['structured_content']`` where downstream extraction
code (:func:`distributed_a2a.files.extract_file_parts`) reads them back."""
NON_TEXT_CONTENT_KEY = "non_text_content"


async def hide_binary_content_from_llm(
    request: MCPToolCallRequest,
    handler: Callable[
        [MCPToolCallRequest],
        Awaitable[MCPToolCallResult],  # pyright: ignore[reportInvalidTypeForm]
    ],
) -> MCPToolCallResult:  # pyright: ignore[reportInvalidTypeForm]
    result = await handler(request)
    if not isinstance(result, CallToolResult) or result.isError:
        return result

    text_blocks: list[TextContent] = []
    non_text_blocks: list[Any] = []
    for block in result.content:
        if isinstance(block, TextContent):
            text_blocks.append(block)
        else:
            non_text_blocks.append(block)

    merged_structured: dict[str, Any] = (
        dict(result.structuredContent) if result.structuredContent else {}
    )
    merged_structured[NON_TEXT_CONTENT_KEY] = non_text_blocks

    return result.model_copy(update={
        "content": text_blocks,
        "structuredContent": merged_structured,
    })
