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
                       ResourceLink)
from pydantic import AnyUrl

from distributed_a2a.client import AgentReply, RemoteAgentConnection
from distributed_a2a.file_extractors import extract_file_parts
from distributed_a2a.mcp_interceptors import NON_TEXT_CONTENT_KEY

_DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def test_extract_file_parts_ignores_non_tool_messages_and_string_content() -> None:
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
        ToolMessage(content="just text", tool_call_id="c1"),
    ]
    parts, delivered = extract_file_parts(messages)
    assert parts == []
    assert delivered == []


def test_extract_file_parts_reads_langchain_file_content_block() -> None:
    docx_b64 = _b64(b"PK\x03\x04 real docx bytes")
    summary = (
        '{"filename": "cv-alice.docx", '
        '"mime_type": "application/vnd.openxmlformats-officedocument.'
        'wordprocessingml.document", "size_bytes": 4321}'
    )
    tool_msg = ToolMessage(
        content=[
            {"type": "text", "text": summary, "id": "lc_text_1"},
            {"type": "file",
             "base64": docx_b64,
             "mime_type": (
                 "application/vnd.openxmlformats-officedocument."
                 "wordprocessingml.document"
             ),
             "id": "lc_file_1"},
        ],
        tool_call_id="call-cv",
        artifact={"structured_content": {"filename": "cv-alice.docx"}},
    )

    parts, delivered = extract_file_parts([HumanMessage(content="hi"), tool_msg])

    assert delivered == [tool_msg]
    assert len(parts) == 1
    name, file_part = parts[0]
    assert name == "cv-alice.docx"
    assert isinstance(file_part, FilePart)
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.name == "cv-alice.docx"
    assert file_part.file.mime_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert file_part.file.bytes == docx_b64


def test_extract_file_parts_reads_langchain_image_content_block() -> None:
    png_b64 = _b64(b"\x89PNG\r\n\x1a\n fake")
    tool_msg = ToolMessage(
        content=[
            {"type": "image", "base64": png_b64,
             "mime_type": "image/png", "id": "lc_img_1"},
        ],
        tool_call_id="call-img",
    )

    parts, delivered = extract_file_parts([tool_msg])

    assert delivered == [tool_msg]
    assert len(parts) == 1
    name, file_part = parts[0]
    assert name.startswith("image")
    assert name.endswith(".png")
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.mime_type == "image/png"
    assert file_part.file.bytes == png_b64


def test_extract_file_parts_synthesises_name_without_filename_hint() -> None:
    docx_b64 = _b64(b"PK\x03\x04 bytes")
    tool_msg = ToolMessage(
        content=[
            {"type": "text", "text": "no json here", "id": "lc_text_1"},
            {"type": "file",
             "base64": docx_b64,
             "mime_type": (
                 "application/vnd.openxmlformats-officedocument."
                 "wordprocessingml.document"
             ),
             "id": "lc_file_1"},
        ],
        tool_call_id="call-cv",
    )

    parts, delivered = extract_file_parts([tool_msg])

    assert delivered == [tool_msg]
    assert len(parts) == 1
    name, file_part = parts[0]
    assert name.startswith("attachment")
    assert name.endswith(".docx")
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.bytes == docx_b64


def test_extract_file_parts_ignores_url_only_file_block() -> None:
    tool_msg = ToolMessage(
        content=[
            {"type": "file",
             "url": "https://example.com/report.pdf",
             "mime_type": "application/pdf",
             "id": "lc_file_1"},
        ],
        tool_call_id="call-url-only",
    )
    parts, delivered = extract_file_parts([tool_msg])
    assert parts == []
    assert delivered == []


def test_extract_file_parts_matches_multiple_filenames_by_order() -> None:
    """Two [text-summary, file] pairs in the same ToolMessage — each
    text-summary's ``filename`` is consumed by the immediately following
    file block."""
    b64_a = _b64(b"aaa")
    b64_b = _b64(b"bbb")
    tool_msg = ToolMessage(
        content=[
            {"type": "text", "text": '{"filename": "a.bin"}',
             "id": "lc_text_1"},
            {"type": "file", "base64": b64_a,
             "mime_type": "application/octet-stream", "id": "lc_file_1"},
            {"type": "text", "text": '{"filename": "b.bin"}',
             "id": "lc_text_2"},
            {"type": "file", "base64": b64_b,
             "mime_type": "application/octet-stream", "id": "lc_file_2"},
        ],
        tool_call_id="call-multi",
    )

    parts, delivered = extract_file_parts([tool_msg])

    assert delivered == [tool_msg]
    assert [name for name, _ in parts] == ["a.bin", "b.bin"]
    assert parts[0][1].file.bytes == b64_a  # type: ignore[union-attr]
    assert parts[1][1].file.bytes == b64_b  # type: ignore[union-attr]


def _interceptor_artifact(blocks: list[object], **extra: object) -> dict[str, object]:
    return {"structured_content": {NON_TEXT_CONTENT_KEY: blocks, **extra}}


def test_extract_file_parts_reads_interceptor_artifact_shape() -> None:
    docx_b64 = _b64(b"PK\x03\x04 real docx bytes")
    embedded = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("cv://cv-alice.docx"), mimeType=_DOCX_MIME, blob=docx_b64,
        ),
    )
    tool_msg = ToolMessage(
        content='{"filename": "cv-alice.docx"}',
        tool_call_id="call-cv",
        artifact=_interceptor_artifact([embedded]),
    )

    parts, delivered = extract_file_parts([tool_msg])

    assert delivered == [tool_msg]
    assert len(parts) == 1
    name, file_part = parts[0]
    assert name == "cv-alice.docx"
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.name == "cv-alice.docx"
    assert file_part.file.mime_type == _DOCX_MIME
    assert file_part.file.bytes == docx_b64


def test_extract_file_parts_prefers_interceptor_artifact_over_content_blocks() -> None:
    docx_b64 = _b64(b"PK\x03\x04 interceptor bytes")
    embedded = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("cv://cv-interceptor.docx"), mimeType=_DOCX_MIME, blob=docx_b64,
        ),
    )
    tool_msg = ToolMessage(
        content=[
            {"type": "text", "text": '{"filename": "cv-legacy.docx"}',
             "id": "lc_text_1"},
            {"type": "file",
             "base64": _b64(b"legacy fallback bytes"),
             "mime_type": _DOCX_MIME,
             "id": "lc_file_1"},
        ],
        tool_call_id="call-cv",
        artifact=_interceptor_artifact([embedded]),
    )

    parts, delivered = extract_file_parts([tool_msg])

    assert delivered == [tool_msg]
    assert len(parts) == 1
    name, file_part = parts[0]
    assert name == "cv-interceptor.docx"
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.bytes == docx_b64


def test_extract_file_parts_reads_image_content_from_interceptor_artifact() -> None:
    png_b64 = _b64(b"\x89PNG\r\n\x1a\n fake image bytes")
    image = ImageContent(type="image", data=png_b64, mimeType="image/png")
    tool_msg = ToolMessage(
        content="here is a chart",
        tool_call_id="call-img",
        artifact=_interceptor_artifact([image]),
    )

    parts, delivered = extract_file_parts([tool_msg])

    assert delivered == [tool_msg]
    assert len(parts) == 1
    name, file_part = parts[0]
    assert name.startswith("image")
    assert name.endswith(".png")
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.mime_type == "image/png"
    assert file_part.file.bytes == png_b64


def test_extract_file_parts_reads_resource_link_as_file_with_uri() -> None:
    link = ResourceLink(
        type="resource_link",
        uri=AnyUrl("https://example.com/reports/report.pdf"),
        name="report.pdf",
        mimeType="application/pdf",
    )
    tool_msg = ToolMessage(
        content="see attached report",
        tool_call_id="call-link",
        artifact=_interceptor_artifact([link]),
    )

    parts, delivered = extract_file_parts([tool_msg])

    assert delivered == [tool_msg]
    assert len(parts) == 1
    name, file_part = parts[0]
    assert name == "report.pdf"
    assert isinstance(file_part.file, FileWithUri)
    assert file_part.file.mime_type == "application/pdf"
    assert file_part.file.uri == "https://example.com/reports/report.pdf"


def test_extract_file_parts_reads_multiple_blocks_from_interceptor_artifact() -> None:
    b64_a = _b64(b"aaa docx")
    b64_b = _b64(b"bbb docx")
    a = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("cv://cv-a.docx"), mimeType=_DOCX_MIME, blob=b64_a,
        ),
    )
    b = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("cv://cv-b.docx"), mimeType=_DOCX_MIME, blob=b64_b,
        ),
    )
    tool_msg = ToolMessage(
        content='{"count": 2}',
        tool_call_id="call-multi",
        artifact=_interceptor_artifact([a, b]),
    )

    parts, delivered = extract_file_parts([tool_msg])

    assert delivered == [tool_msg]
    assert [name for name, _ in parts] == ["cv-a.docx", "cv-b.docx"]
    assert parts[0][1].file.bytes == b64_a  # type: ignore[union-attr]
    assert parts[1][1].file.bytes == b64_b  # type: ignore[union-attr]


def test_extract_file_parts_ignores_empty_interceptor_artifact() -> None:
    """An empty ``non_text_content`` list must not cause the extractor to
    fall through to the ``content`` path — that would double-extract files
    on any tool where the interceptor happened to filter everything out."""
    tool_msg = ToolMessage(
        content=[
            {"type": "file", "base64": _b64(b"leaked"),
             "mime_type": _DOCX_MIME, "id": "lc_file_1"},
        ],
        tool_call_id="call-mixed",
        artifact={"structured_content": {NON_TEXT_CONTENT_KEY: []}},
    )
    parts, delivered = extract_file_parts([tool_msg])
    assert delivered == [tool_msg]
    assert len(parts) == 1


def test_extract_file_parts_falls_back_to_content_when_artifact_has_no_key() -> None:
    docx_b64 = _b64(b"PK\x03\x04 bytes")
    tool_msg = ToolMessage(
        content=[
            {"type": "text", "text": '{"filename": "cv-fallback.docx"}',
             "id": "lc_text_1"},
            {"type": "file", "base64": docx_b64,
             "mime_type": _DOCX_MIME, "id": "lc_file_1"},
        ],
        tool_call_id="call-fallback",
        artifact={"structured_content": {"unrelated": {"foo": 1}}},
    )

    parts, delivered = extract_file_parts([tool_msg])

    assert delivered == [tool_msg]
    assert len(parts) == 1
    name, file_part = parts[0]
    assert name == "cv-fallback.docx"
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.bytes == docx_b64


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


@pytest.mark.asyncio
async def test_remote_agent_connection_gathers_multiple_files_in_order(
        monkeypatch: pytest.MonkeyPatch) -> None:
    b64_a = _b64(b"aaa")
    b64_b = _b64(b"bbb")
    completed_task = Task(
        id="task-multi",
        context_id="ctx-multi",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[
            Artifact(
                artifact_id="a.bin",
                name="a.bin",
                parts=[Part(root=FilePart(file=FileWithBytes(
                    name="a.bin",
                    mime_type="application/octet-stream",
                    bytes=b64_a,
                )))],
            ),
            Artifact(
                artifact_id="b.bin",
                name="b.bin",
                parts=[Part(root=FilePart(file=FileWithBytes(
                    name="b.bin",
                    mime_type="application/octet-stream",
                    bytes=b64_b,
                )))],
            ),
            Artifact(
                artifact_id="current_result",
                name="current_result",
                parts=[Part(root=TextPart(text="Two files ready."))],
            ),
        ],
    )
    _patch_client_factory(monkeypatch, completed_task)

    async with httpx.AsyncClient() as http_client:
        conn = RemoteAgentConnection(_agent_card(), http_client)
        reply = await conn.send_message("hello", "ctx-multi")

    assert isinstance(reply, AgentReply)
    assert reply.text == "Two files ready."
    assert [f.name for f in reply.files] == ["a.bin", "b.bin"]
    assert [f.mime_type for f in reply.files] == [
        "application/octet-stream", "application/octet-stream",
    ]
    assert [f.bytes_b64 for f in reply.files] == [b64_a, b64_b]
    assert all(f.uri is None for f in reply.files)

