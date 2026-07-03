from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import _convert_call_tool_result
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from mcp.types import (BlobResourceContents, CallToolResult, EmbeddedResource,
                       ImageContent, TextContent)
from pydantic import AnyUrl

from distributed_a2a.mcp_interceptors import (NON_TEXT_CONTENT_KEY,
                                              hide_binary_content_from_llm)


def _request() -> MCPToolCallRequest:
    return MCPToolCallRequest(
        name="render_file", args={}, server_name="test-server",
    )


def _embedded_docx(uri: str = "file://alice.docx", blob: str = "UEsDBAA=") -> EmbeddedResource:
    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl(uri),
            mimeType=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            blob=blob,
        ),
    )


def _make_handler(returning: Any) -> Any:
    async def _handler(_req: MCPToolCallRequest) -> Any:
        return returning
    return _handler


@pytest.mark.asyncio
async def test_text_only_result_is_passed_through() -> None:
    original = CallToolResult(
        content=[TextContent(type="text", text="just a summary")],
        structuredContent=None,
        isError=False,
    )
    result = await hide_binary_content_from_llm(_request(), _make_handler(original))
    assert isinstance(result, CallToolResult)
    assert result.content == original.content
    assert result.structuredContent == {NON_TEXT_CONTENT_KEY: []}


@pytest.mark.asyncio
async def test_mixed_result_moves_binary_into_structured_content() -> None:
    embedded = _embedded_docx()
    original = CallToolResult(
        content=[
            TextContent(type="text", text='{"filename": "alice.docx"}'),
            embedded,
        ],
        structuredContent=None,
        isError=False,
    )
    result = await hide_binary_content_from_llm(_request(), _make_handler(original))
    assert isinstance(result, CallToolResult)
    assert result is not original, "A mutated copy is expected, not the original object."
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == '{"filename": "alice.docx"}'
    assert result.structuredContent is not None
    # Non-text blocks are stored as JSON-safe dicts (not raw pydantic
    # instances) so downstream LangGraph msgpack checkpointing works.
    assert result.structuredContent[NON_TEXT_CONTENT_KEY] == [
        embedded.model_dump(mode="json"),
    ]


@pytest.mark.asyncio
async def test_binary_only_result_produces_empty_content_list() -> None:
    embedded = _embedded_docx()
    original = CallToolResult(
        content=[embedded],
        structuredContent=None,
        isError=False,
    )
    result = await hide_binary_content_from_llm(_request(), _make_handler(original))
    assert isinstance(result, CallToolResult)
    assert result.content == [], (
        "Binary-only tool output should leave content empty — the model "
        "receives no text, and the block is only reachable via artifact."
    )
    assert result.structuredContent == {
        NON_TEXT_CONTENT_KEY: [embedded.model_dump(mode="json")],
    }


@pytest.mark.asyncio
async def test_error_result_is_passed_through_unchanged() -> None:
    original = CallToolResult(
        content=[TextContent(type="text", text="boom")],
        structuredContent=None,
        isError=True,
    )
    result = await hide_binary_content_from_llm(_request(), _make_handler(original))
    assert result is original
    assert isinstance(result, CallToolResult) and result.isError is True


@pytest.mark.asyncio
async def test_error_result_with_binary_still_passes_through() -> None:
    embedded = _embedded_docx()
    original = CallToolResult(
        content=[TextContent(type="text", text="oops"), embedded],
        structuredContent=None,
        isError=True,
    )
    result = await hide_binary_content_from_llm(_request(), _make_handler(original))
    assert result is original


@pytest.mark.asyncio
async def test_preserves_existing_structured_content() -> None:
    embedded = _embedded_docx()
    original = CallToolResult(
        content=[TextContent(type="text", text="summary"), embedded],
        structuredContent={"foo": 1, "nested": {"bar": 2}},
        isError=False,
    )
    result = await hide_binary_content_from_llm(_request(), _make_handler(original))
    assert isinstance(result, CallToolResult)
    assert result.structuredContent == {
        "foo": 1,
        "nested": {"bar": 2},
        NON_TEXT_CONTENT_KEY: [embedded.model_dump(mode="json")],
    }
    assert original.structuredContent == {"foo": 1, "nested": {"bar": 2}}


@pytest.mark.asyncio
async def test_non_call_tool_result_is_passed_through() -> None:
    upstream = ToolMessage(content="upstream", tool_call_id="tc-1")
    result = await hide_binary_content_from_llm(_request(), _make_handler(upstream))
    assert result is upstream


@pytest.mark.asyncio
async def test_image_content_is_hidden() -> None:
    image = ImageContent(type="image", data="AA==", mimeType="image/png")
    original = CallToolResult(
        content=[TextContent(type="text", text="see below"), image],
        isError=False,
    )
    result = await hide_binary_content_from_llm(_request(), _make_handler(original))
    assert isinstance(result, CallToolResult)
    assert result.content == [TextContent(type="text", text="see below")]
    assert result.structuredContent == {
        NON_TEXT_CONTENT_KEY: [image.model_dump(mode="json")],
    }


def test_adapter_forwards_structured_content_into_tool_message_artifact() -> None:
    embedded = _embedded_docx()
    stashed = CallToolResult(
        content=[TextContent(type="text", text='{"filename": "file.docx"}')],
        structuredContent={NON_TEXT_CONTENT_KEY: [embedded]},
        isError=False,
    )
    content, artifact = _convert_call_tool_result(stashed)
    assert isinstance(content, list) and content
    # No file/image block leaked into the LLM-visible content list.
    for block in content:
        assert not isinstance(block, dict) or block.get("type") == "text"
    assert artifact is not None
    assert artifact["structured_content"] == {NON_TEXT_CONTENT_KEY: [embedded]}


def test_base_tool_invoke_sets_tool_call_id_when_content_is_not_tool_message() -> None:
    def fake_call(**_kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return (
            [{"type": "text", "text": "summary"}],
            {"structured_content": {NON_TEXT_CONTENT_KEY: ["placeholder"]}},
        )

    tool = StructuredTool.from_function(
        func=fake_call,
        name="render_file",
        description="stub",
        response_format="content_and_artifact",
    )
    tool_call = {
        "name": "render_file",
        "args": {},
        "id": "TCID_42",
        "type": "tool_call",
    }
    result = tool.invoke(tool_call)
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "TCID_42"
    assert result.artifact == {
        "structured_content": {NON_TEXT_CONTENT_KEY: ["placeholder"]},
    }


@pytest.mark.asyncio
async def test_tool_message_from_interceptor_is_msgpack_serializable() -> None:
    embedded = _embedded_docx()
    image = ImageContent(type="image", data="AA==", mimeType="image/png")
    original = CallToolResult(
        content=[
            TextContent(type="text", text='{"filename": "alice.docx"}'),
            embedded,
            image,
        ],
        isError=False,
    )
    stashed = await hide_binary_content_from_llm(
        _request(), _make_handler(original),
    )
    assert isinstance(stashed, CallToolResult)

    content, artifact = _convert_call_tool_result(stashed)
    tool_msg = ToolMessage(
        content=content,  # type: ignore[arg-type]
        tool_call_id="call-1",
        name="render_file",
        artifact=artifact,
    )

    serde = JsonPlusSerializer()
    kind, blob = serde.dumps_typed({"messages": [tool_msg]})
    assert kind == "msgpack"
    loaded = serde.loads_typed((kind, blob))
    assert isinstance(loaded, dict)
    loaded_messages = loaded["messages"]
    assert len(loaded_messages) == 1
    assert isinstance(loaded_messages[0], ToolMessage)
    assert loaded_messages[0].tool_call_id == "call-1"
    assert loaded_messages[0].artifact is not None
    non_text = loaded_messages[0].artifact["structured_content"][
        NON_TEXT_CONTENT_KEY
    ]
    assert len(non_text) == 2
    assert non_text[0]["type"] == "resource"
    assert non_text[1]["type"] == "image"
