from __future__ import annotations

import base64
from typing import Any

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import FilePart, FileWithBytes
from a2a.types import Message as A2AMessage
from a2a.types import (MessageSendParams, Part, Role, TaskArtifactUpdateEvent,
                       TaskState, TaskStatusUpdateEvent, TextPart)
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from mcp.types import BlobResourceContents, EmbeddedResource
from pydantic import AnyUrl

from distributed_a2a.agent import AgentInvocation, StringResponse
from distributed_a2a.executors import RoutingAgentExecutor


class _StubStatusAgent:
    """Minimal stand-in for ``StatusAgent`` used in the executor."""

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


def _make_tool_message_with_docx(b64: str) -> ToolMessage:
    return ToolMessage(
        content='{"filename": "cv-foo.docx"}',
        tool_call_id="call-1",
        artifact=[
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=AnyUrl("cv://cv-foo.docx"),
                    mimeType=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    blob=b64,
                ),
            ),
        ],
    )


async def _drain_queue(queue: EventQueue) -> list[Any]:
    """Dequeue every event currently in ``queue`` without blocking.

    Uses ``no_wait=True`` and stops on the first empty/closed exception. We
    intentionally avoid ``await queue.close()`` because the graceful close
    waits for ``queue.join()`` — which requires the consumer to call
    ``task_done()`` for every enqueued item — and would otherwise deadlock
    the test once we've finished dequeuing.
    """
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
async def test_executor_emits_file_part_event_for_embedded_resource(
        monkeypatch: pytest.MonkeyPatch) -> None:
    docx_b64 = base64.b64encode(b"PK\x03\x04 fake docx").decode("ascii")
    tool_msg = _make_tool_message_with_docx(docx_b64)

    # Build the executor without invoking its real __init__ (which requires
    # API keys + registry network calls). We only need ``execute`` to use:
    #   - self.agent (stubbed)
    #   - self.reinitialize_agent_with_tools (stubbed no-op)
    #   - self.agent_config.agent.card.name (stubbed via SimpleNamespace)
    from types import SimpleNamespace

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

    # Expected sequence: working status, file artifact, text artifact, final status.
    artifact_events = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
    status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]

    assert len(artifact_events) == 2, (
        f"Expected 2 artifact events (file + text), got: "
        f"{[(e.artifact.name, e.last_chunk) for e in artifact_events]}"
    )

    file_event, text_event = artifact_events
    # 1) file artifact carries a FilePart, is NOT the final chunk
    assert file_event.last_chunk is False
    assert file_event.artifact.name == "cv-foo.docx"
    assert len(file_event.artifact.parts) == 1
    file_part = file_event.artifact.parts[0].root
    assert isinstance(file_part, FilePart)
    assert isinstance(file_part.file, FileWithBytes)
    assert file_part.file.name == "cv-foo.docx"
    assert file_part.file.bytes == docx_b64

    # 2) text artifact carries the LLM-visible summary, IS the final chunk
    assert text_event.last_chunk is True
    assert text_event.artifact.name == "current_result"
    text_part = text_event.artifact.parts[0].root
    assert isinstance(text_part, TextPart)
    assert text_part.text == "*cv-agent*: Here is your CV."

    # 3) Final status event is `completed`
    final_status = [e for e in status_events if e.final]
    assert len(final_status) == 1
    assert final_status[0].status.state == TaskState.completed


@pytest.mark.asyncio
async def test_executor_emits_no_file_event_when_no_artifacts() -> None:
    from types import SimpleNamespace

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
