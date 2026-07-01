from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import FilePart, FileWithBytes
from a2a.types import Message as A2AMessage
from a2a.types import (MessageSendParams, Part, Role, TaskArtifactUpdateEvent,
                       TaskState, TaskStatusUpdateEvent, TextPart)
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from distributed_a2a.agent import AgentInvocation, StringResponse
from distributed_a2a.executors import RoutingAgentExecutor


class _StubStatusAgent:
    def __init__(self, response: StringResponse, messages: list[BaseMessage]) -> None:
        self._response = response
        self._messages = messages

    async def __call__(self, message: str, context_id: str | None = None) -> AgentInvocation[StringResponse]:
        return AgentInvocation[StringResponse](
            structured=self._response,
            messages=self._messages,
        )


def _make_request_context() -> RequestContext:
    msg = A2AMessage(
        message_id="m-1",
        context_id="ctx-1",
        task_id="task-1",
        role=Role.user,
        parts=[Part(root=TextPart(text="render a CV please"))],
    )
    return RequestContext(
        request=MessageSendParams(message=msg),
        task_id="task-1",
        context_id="ctx-1",
    )


async def _drain_queue(queue: EventQueue) -> list[Any]:
    events: list[Any] = []
    while True:
        try:
            evt = await queue.dequeue_event(no_wait=True)
        except Exception:
            break
        events.append(evt)
        queue.task_done()
    return events


@pytest.mark.asyncio
async def test_executor_emits_no_file_event_when_no_artifacts() -> None:
    executor = RoutingAgentExecutor.__new__(RoutingAgentExecutor)
    executor.agent_config = SimpleNamespace(  # type: ignore[assignment]
        agent=SimpleNamespace(card=SimpleNamespace(name="plain-agent")),
    )
    executor.agent = _StubStatusAgent(  # type: ignore[assignment]
        StringResponse(status=TaskState.completed, response="No files here."),
        [HumanMessage(content="hi")],
    )

    async def _noop_reinit() -> None:
        return None

    executor.reinitialize_agent_with_tools = _noop_reinit  # type: ignore[method-assign]

    ctx = _make_request_context()
    queue = EventQueue()
    await executor.execute(ctx, queue)
    events = await _drain_queue(queue)

    artifact_events = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
    assert len(artifact_events) == 1
    assert artifact_events[0].artifact.name == "current_result"
    assert artifact_events[0].last_chunk is True


@pytest.mark.asyncio
async def test_executor_emits_file_part_from_langchain_content_block() -> None:
    docx_b64 = base64.b64encode(b"PK\x03\x04 fake docx").decode("ascii")
    summary_json = (
        '{"filename": "cv-bob.docx", '
        '"mime_type": "application/vnd.openxmlformats-officedocument.'
        'wordprocessingml.document"}'
    )
    tool_msg = ToolMessage(
        content=[
            {"type": "text", "text": summary_json, "id": "lc_text_1"},
            {"type": "file",
             "base64": docx_b64,
             "mime_type": (
                 "application/vnd.openxmlformats-officedocument."
                 "wordprocessingml.document"
             ),
             "id": "lc_file_1"},
        ],
        tool_call_id="call-cv",
        artifact={"structured_content": {"filename": "cv-bob.docx"}},
    )

    executor = RoutingAgentExecutor.__new__(RoutingAgentExecutor)
    executor.agent_config = SimpleNamespace(  # type: ignore[assignment]
        agent=SimpleNamespace(card=SimpleNamespace(name="cv-agent")),
    )
    executor.agent = _StubStatusAgent(  # type: ignore[assignment]
        StringResponse(status=TaskState.completed,
                       response="Here is your CV."),
        [HumanMessage(content="render a CV please"), tool_msg],
    )

    async def _noop_reinit() -> None:
        return None

    executor.reinitialize_agent_with_tools = _noop_reinit  # type: ignore[method-assign]

    ctx = _make_request_context()
    queue = EventQueue()
    await executor.execute(ctx, queue)
    events = await _drain_queue(queue)

    artifact_events = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
    assert len(artifact_events) == 2, (
        "Expected the executor to emit a file artifact followed by the text "
        "artifact when the tool response uses the production LangChain "
        "content-block shape."
    )
    file_event, text_event = artifact_events
    assert file_event.last_chunk is False
    assert file_event.artifact.name == "cv-bob.docx"
    file_part = file_event.artifact.parts[0].root
    assert isinstance(file_part, FilePart)
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.name == "cv-bob.docx"
    assert file_part.file.bytes == docx_b64

    assert text_event.last_chunk is True
    assert text_event.artifact.name == "current_result"


@pytest.mark.asyncio
async def test_executor_emits_one_file_event_per_file_block() -> None:
    b64_a = base64.b64encode(b"aaa docx bytes").decode("ascii")
    b64_b = base64.b64encode(b"bbb docx bytes").decode("ascii")
    docx_mime = (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    )
    tool_msg = ToolMessage(
        content=[
            {"type": "text", "text": '{"filename": "cv-a.docx"}', "id": "t1"},
            {"type": "file", "base64": b64_a, "mime_type": docx_mime, "id": "f1"},
            {"type": "text", "text": '{"filename": "cv-b.docx"}', "id": "t2"},
            {"type": "file", "base64": b64_b, "mime_type": docx_mime, "id": "f2"},
        ],
        tool_call_id="call-multi",
    )

    executor = RoutingAgentExecutor.__new__(RoutingAgentExecutor)
    executor.agent_config = SimpleNamespace(  # type: ignore[assignment]
        agent=SimpleNamespace(card=SimpleNamespace(name="cv-agent")),
    )
    executor.agent = _StubStatusAgent(  # type: ignore[assignment]
        StringResponse(status=TaskState.completed,
                       response="Here are your two CVs."),
        [HumanMessage(content="render two CVs"), tool_msg],
    )

    async def _noop_reinit() -> None:
        return None

    executor.reinitialize_agent_with_tools = _noop_reinit  # type: ignore[method-assign]

    ctx = _make_request_context()
    queue = EventQueue()
    await executor.execute(ctx, queue)
    events = await _drain_queue(queue)

    artifact_events = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
    assert len(artifact_events) == 3, (
        "Expected 2 file artifacts (last_chunk=False) + 1 text artifact "
        f"(last_chunk=True). Got: "
        f"{[(e.artifact.name, e.last_chunk) for e in artifact_events]}"
    )
    file_a_event, file_b_event, text_event = artifact_events

    assert file_a_event.last_chunk is False
    assert file_a_event.artifact.name == "cv-a.docx"
    file_a_part = file_a_event.artifact.parts[0].root
    assert isinstance(file_a_part, FilePart)
    assert isinstance(file_a_part.file, FileWithBytes)
    assert file_a_part.file.name == "cv-a.docx"
    assert file_a_part.file.bytes == b64_a

    assert file_b_event.last_chunk is False
    assert file_b_event.artifact.name == "cv-b.docx"
    file_b_part = file_b_event.artifact.parts[0].root
    assert isinstance(file_b_part, FilePart)
    assert isinstance(file_b_part.file, FileWithBytes)
    assert file_b_part.file.name == "cv-b.docx"
    assert file_b_part.file.bytes == b64_b

    assert text_event.last_chunk is True
    assert text_event.artifact.name == "current_result"

    final_status = [e for e in events
                    if isinstance(e, TaskStatusUpdateEvent) and e.final]
    assert len(final_status) == 1
    assert final_status[0].status.state == TaskState.completed
