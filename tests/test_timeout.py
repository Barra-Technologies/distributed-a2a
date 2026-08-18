import asyncio
import time
from typing import Any
from uuid import uuid4

import httpx
import pytest
from a2a.types import (AgentCapabilities, AgentCard, Artifact, Message, Part,
                       Role, Task, TaskState, TaskStatus, TextPart)

from distributed_a2a.client import (A2ATimeoutError, AgentReply,
                                    RemoteAgentConnection, RoutingA2AClient)


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


def _task(state: TaskState, with_result: bool = False) -> Task:
    artifacts: list[Artifact] = []
    if with_result:
        artifacts = [
            Artifact(
                artifact_id="a1",
                name="current_result",
                parts=[Part(root=TextPart(text="all done"))],
            )
        ]
    return Task(
        id="task-x",
        context_id="ctx-x",
        status=TaskStatus(state=state),
        artifacts=artifacts,
    )


def _working_task() -> Task:
    return _task(TaskState.working)


class _AlwaysWorkingClient:
    async def send_message(self, _message: Message) -> Any:
        async def _gen() -> Any:
            yield _working_task(), None
        async for item in _gen():
            yield item

    async def get_task(self, _params: Any) -> Task:
        return _working_task()


class _AlwaysWorkingFactory:
    def __init__(self, config: object) -> None:
        self._config = config

    def create(self, _card: AgentCard) -> _AlwaysWorkingClient:
        return _AlwaysWorkingClient()


@pytest.mark.asyncio
async def test_send_message_raises_a2a_timeout_error_after_max_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("distributed_a2a.client.ClientFactory", _AlwaysWorkingFactory)

    async with httpx.AsyncClient() as http_client:
        connection = RemoteAgentConnection(
            _agent_card(), http_client, max_polls=3, poll_interval=0.0
        )
        with pytest.raises(A2ATimeoutError) as excinfo:
            await connection.send_message("hi", "ctx-x")

    err = excinfo.value
    assert err.attempts == 3
    assert err.last_task_state == TaskState.working
    assert err.target_url == "http://127.0.0.1:0"


class _ScriptedClient:
    """Returns ``working`` on send, then a scripted state sequence from ``get_task``."""

    def __init__(self, get_states: list[TaskState]):
        self._get_states = list(get_states)
        self.get_task_calls = 0

    async def send_message(self, _message: Message) -> Any:
        async def _gen() -> Any:
            yield _working_task(), None
        async for item in _gen():
            yield item

    async def get_task(self, _params: Any) -> Task:
        self.get_task_calls += 1
        state = self._get_states.pop(0)
        return _task(state, with_result=(state == TaskState.completed))


@pytest.mark.asyncio
async def test_send_message_honors_completion_on_final_allowed_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = _ScriptedClient([TaskState.working, TaskState.working, TaskState.completed])

    class _Factory:
        def __init__(self, config: object) -> None: ...
        def create(self, _card: AgentCard) -> _ScriptedClient:
            return scripted

    monkeypatch.setattr("distributed_a2a.client.ClientFactory", _Factory)

    async with httpx.AsyncClient() as http_client:
        connection = RemoteAgentConnection(
            _agent_card(), http_client, max_polls=3, poll_interval=0.0
        )
        result = await connection.send_message("hi", "ctx-x")

    assert isinstance(result, AgentReply)
    assert result.text == "all done"
    assert result.files == []
    assert scripted.get_task_calls == 3
    assert scripted._get_states == []


@pytest.mark.asyncio
async def test_send_message_returns_immediately_if_already_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Immediate:
        async def send_message(self, _message: Message) -> Any:
            async def _gen() -> Any:
                yield _task(TaskState.completed, with_result=True), None
            async for item in _gen():
                yield item

        async def get_task(self, _params: Any) -> Task:
            raise AssertionError("get_task should not be called when already done")

    class _Factory:
        def __init__(self, config: object) -> None: ...
        def create(self, _card: AgentCard) -> _Immediate:
            return _Immediate()

    monkeypatch.setattr("distributed_a2a.client.ClientFactory", _Factory)

    async with httpx.AsyncClient() as http_client:
        connection = RemoteAgentConnection(
            _agent_card(), http_client, max_polls=3, poll_interval=0.0
        )
        result = await connection.send_message("hi", "ctx-x")

    assert isinstance(result, AgentReply)
    assert result.text == "all done"
    assert result.files == []


def test_a2a_timeout_error_message_includes_diagnostics() -> None:
    err = A2ATimeoutError("http://x", attempts=5, elapsed_seconds=1.25,
                          last_task_state=TaskState.working)
    text = str(err)
    assert "http://x" in text
    assert "attempts=5" in text
    assert "elapsed_seconds=1.25" in text
    assert "TaskState.working" in text or "working" in text


@pytest.mark.asyncio
async def test_routing_a2a_client_propagates_poll_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Capturing:
        def __init__(self, agent_card: AgentCard, _client: object, *,
                     max_polls: int, poll_interval: float) -> None:
            captured["max_polls"] = max_polls
            captured["poll_interval"] = poll_interval
            self.agent_card = agent_card

        async def send_message(self, _message: str, _context_id: str) -> AgentReply:
            return AgentReply(text="done")

    monkeypatch.setattr("distributed_a2a.client.RemoteAgentConnection", _Capturing)

    client = RoutingA2AClient(initial_url="http://router", max_polls=7,
                              poll_interval=0.25)
    client.current_card = _agent_card()

    async def _fetch() -> None:
        client.current_card = _agent_card()

    monkeypatch.setattr(client, "fetch_initial_card", _fetch)

    result = await client.send_message("hi", context_id="ctx-1")
    assert isinstance(result, AgentReply)
    assert result.text == "done"
    assert result.files == []
    assert captured == {"max_polls": 7, "poll_interval": 0.25}
