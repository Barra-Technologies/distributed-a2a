from __future__ import annotations

import base64
from collections.abc import AsyncGenerator

import httpx
import pytest
from a2a.types import (AgentCapabilities, AgentCard, Artifact, FilePart,
                       FileWithBytes, FileWithUri, Message, Part, Task,
                       TaskState, TaskStatus, TextPart)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp.types import (BlobResourceContents, EmbeddedResource, ImageContent,
                       TextResourceContents)
from pydantic import AnyUrl

from distributed_a2a.client import AgentReply, RemoteAgentConnection
from distributed_a2a.executors import _extract_file_parts


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def test_extract_file_parts_picks_up_embedded_resource() -> None:
    docx_b64 = _b64(b"PK\x03\x04 fake docx bytes")
    tool_msg = ToolMessage(
        content="{\"filename\": \"cv-foo.docx\"}",
        tool_call_id="call-1",
        artifact=[
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=AnyUrl("cv://cv-foo.docx"),
                    mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    blob=docx_b64,
                ),
            ),
        ],
    )

    parts = _extract_file_parts([HumanMessage(content="hi"), tool_msg])

    assert len(parts) == 1
    name, file_part = parts[0]
    assert name == "cv-foo.docx"
    assert isinstance(file_part, FilePart)
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.name == "cv-foo.docx"
    assert file_part.file.mime_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert file_part.file.bytes == docx_b64


def test_extract_file_parts_handles_image_content() -> None:
    png_b64 = _b64(b"\x89PNG\r\n\x1a\n fake")
    tool_msg = ToolMessage(
        content="",
        tool_call_id="call-img",
        artifact=[
            ImageContent(type="image", data=png_b64, mimeType="image/png"),
        ],
    )

    parts = _extract_file_parts([tool_msg])

    assert len(parts) == 1
    name, file_part = parts[0]
    assert name == "image"
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.mime_type == "image/png"
    assert file_part.file.bytes == png_b64


def test_extract_file_parts_ignores_non_tool_messages_and_empty_artifacts() -> None:
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
        ToolMessage(content="just text", tool_call_id="c1"),
        ToolMessage(content="empty artifact", tool_call_id="c2", artifact=[]),
    ]
    assert _extract_file_parts(messages) == []


def test_extract_file_parts_skips_text_resource_contents() -> None:
    tool_msg = ToolMessage(
        content="",
        tool_call_id="call-x",
        artifact=[
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri=AnyUrl("text://note.txt"),
                    mimeType="text/plain",
                    text="some inline text",
                ),
            ),
        ],
    )
    assert _extract_file_parts([tool_msg]) == []


def test_extract_file_parts_falls_back_to_octet_stream_when_mime_missing() -> None:
    tool_msg = ToolMessage(
        content="",
        tool_call_id="call-x",
        artifact=[
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=AnyUrl("file:///tmp/x.bin"),
                    blob=_b64(b"x"),
                ),
            ),
        ],
    )
    parts = _extract_file_parts([tool_msg])
    assert len(parts) == 1
    name, file_part = parts[0]
    assert name == "x.bin"
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.mime_type == "application/octet-stream"


class _StubAgentClient:
    def __init__(self, task: Task):
        self._task = task

    async def send_message(self, _message: Message) -> AsyncGenerator[tuple[Task, None], None]:
        yield self._task, None


class _StubClientFactory:
    def __init__(self, _config: object, task: Task):
        self._task = task

    def create(self, _agent_card: AgentCard) -> _StubAgentClient:
        return _StubAgentClient(self._task)


def _patch_client_factory(monkeypatch: pytest.MonkeyPatch, task: Task) -> None:
    def _ctor(config: object) -> _StubClientFactory:
        return _StubClientFactory(config, task)
    monkeypatch.setattr("distributed_a2a.client.ClientFactory", _ctor)


def _agent_card() -> AgentCard:
    return AgentCard(
        name="stub",
        description="stub",
        url="http://127.0.0.1:0",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[],
        preferred_transport="JSONRPC",
    )


@pytest.mark.asyncio
async def test_remote_agent_connection_returns_text_and_files_in_agent_reply(
        monkeypatch: pytest.MonkeyPatch) -> None:
    docx_b64 = _b64(b"PK\x03\x04 docx-bytes")
    completed_task = Task(
        id="task-files",
        context_id="ctx-files",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[
            Artifact(
                artifact_id="cv-foo.docx",
                name="cv-foo.docx",
                parts=[Part(root=FilePart(file=FileWithBytes(
                    name="cv-foo.docx",
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    bytes=docx_b64,
                )))],
            ),
            Artifact(
                artifact_id="current_result",
                name="current_result",
                parts=[Part(root=TextPart(text="Here is your CV."))],
            ),
        ],
    )
    factory_task = completed_task
    _patch_client_factory(monkeypatch, factory_task)

    async with httpx.AsyncClient() as http_client:
        conn = RemoteAgentConnection(_agent_card(), http_client)
        reply = await conn.send_message("hello", "ctx-files")

    assert isinstance(reply, AgentReply)
    assert reply.text == "Here is your CV."
    assert len(reply.files) == 1
    f = reply.files[0]
    assert f.name == "cv-foo.docx"
    assert f.mime_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert f.bytes_b64 == docx_b64
    assert f.uri is None


@pytest.mark.asyncio
async def test_remote_agent_connection_handles_file_with_uri(
        monkeypatch: pytest.MonkeyPatch) -> None:
    completed_task = Task(
        id="task-uri",
        context_id="ctx-uri",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[
            Artifact(
                artifact_id="report",
                name="report",
                parts=[Part(root=FilePart(file=FileWithUri(
                    name="report.pdf",
                    mime_type="application/pdf",
                    uri="https://example.com/report.pdf",
                )))],
            ),
        ],
    )
    _patch_client_factory(monkeypatch, completed_task)

    async with httpx.AsyncClient() as http_client:
        conn = RemoteAgentConnection(_agent_card(), http_client)
        reply = await conn.send_message("hello", "ctx-uri")

    assert isinstance(reply, AgentReply)
    assert reply.text is None
    assert len(reply.files) == 1
    f = reply.files[0]
    assert f.name == "report.pdf"
    assert f.mime_type == "application/pdf"
    assert f.bytes_b64 == ""
    assert f.uri == "https://example.com/report.pdf"
