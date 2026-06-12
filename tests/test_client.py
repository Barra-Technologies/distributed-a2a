import httpx
import pytest
from a2a.types import (AgentCapabilities, AgentCard, Message, Part, Role, Task,
                       TaskState, TaskStatus, TextPart)

from distributed_a2a.client import RemoteAgentConnection


class _StubAgentClient:
    def __init__(self, task: Task):
        self._task = task

    async def send_message(self, _message: Message):
        yield self._task, None


class _StubClientFactory:
    def __init__(self, _config: object, task: Task):
        self._task = task

    def create(self, _agent_card: AgentCard) -> _StubAgentClient:
        return _StubAgentClient(self._task)


@pytest.mark.asyncio
async def test_remote_agent_connection_raises_failed_task_message(monkeypatch: pytest.MonkeyPatch) -> None:
    failed_task = Task(
        id="task-1",
        context_id="context-1",
        status=TaskStatus(
            state=TaskState.failed,
            message=Message(
                message_id="message-1",
                role=Role.agent,
                parts=[Part(root=TextPart(text="remote agent failed: tool timeout"))],
            ),
        ),
    )

    def _stub_client_factory_ctor(config: object) -> _StubClientFactory:
        return _StubClientFactory(config, failed_task)

    monkeypatch.setattr("distributed_a2a.client.ClientFactory", _stub_client_factory_ctor)

    agent_card = AgentCard(
        name="stub-agent",
        description="stub",
        url="http://127.0.0.1:9999",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[],
        preferred_transport="JSONRPC",
    )

    async with httpx.AsyncClient() as http_client:
        connection = RemoteAgentConnection(agent_card, http_client)

        # When/Then: the client surfaces the remote task failure message
        with pytest.raises(Exception, match="remote agent failed: tool timeout"):
            await connection.send_message("Hello", "context-1")
