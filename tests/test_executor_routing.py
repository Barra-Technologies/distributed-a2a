from __future__ import annotations

import json
from typing import Any

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.types import (Message, MessageSendParams, Part, Role, TaskState,
                       TextPart)

from distributed_a2a.agent import AgentInvocation, RoutingResponse
from distributed_a2a.executors import (RoutingFailed,
                                       _route_request_to_matching_agent)


class _StubRoutingAgent:
    def __init__(self, response: RoutingResponse) -> None:
        self._response = response

    async def __call__(
        self,
        message: str,
        context_id: str | None = None,
    ) -> AgentInvocation[RoutingResponse]:
        return AgentInvocation[RoutingResponse](
            structured=self._response,
            messages=[],
        )


class _StubRegistry:
    def __init__(self, agent_card: dict[str, Any] | None) -> None:
        self._agent_card = agent_card

    async def get_agent_card(self, agent_name: str) -> dict[str, Any] | None:
        return self._agent_card


def _ctx() -> RequestContext:
    msg = Message(
        message_id="m-1",
        context_id="ctx-1",
        task_id="task-1",
        role=Role.user,
        parts=[Part(root=TextPart(text="hello"))],
    )
    return RequestContext(
        request=MessageSendParams(message=msg),
        task_id="task-1",
        context_id="ctx-1",
    )


@pytest.mark.asyncio
async def test_route_returns_target_agent_artifact() -> None:
    response = RoutingResponse(
        status=TaskState.completed,
        agent_name="weather-agent",
    )
    routing_agent = _StubRoutingAgent(response)
    registry = _StubRegistry(
        {"name": "weather-agent", "url": "http://weather"}
    )

    artifact = await _route_request_to_matching_agent(
        routing_agent,
        registry,
        _ctx(),
    )

    assert artifact.name == "target_agent"
    assert artifact.parts is not None
    text = artifact.parts[0].root
    assert isinstance(text, TextPart)
    assert json.loads(text.text)["name"] == "weather-agent"


@pytest.mark.asyncio
async def test_route_returns_fallback_message_when_no_agent_name() -> None:
    fallback = (
        "DISCLAIMER: I am not a specialized agent and will answer "
        "to the best of my knowledge"
    )
    response = RoutingResponse(
        status=TaskState.completed,
        agent_name=None,
        message=fallback,
    )
    routing_agent = _StubRoutingAgent(response)
    registry = _StubRegistry(None)

    artifact = await _route_request_to_matching_agent(
        routing_agent,
        registry,
        _ctx(),
    )

    assert artifact.name == "current_result"
    assert artifact.parts is not None
    text = artifact.parts[0].root
    assert isinstance(text, TextPart)
    assert text.text == fallback


@pytest.mark.asyncio
async def test_route_raises_when_failed_without_route_target() -> None:
    response = RoutingResponse(
        status=TaskState.failed,
        agent_name=None,
        message="cannot route",
    )
    routing_agent = _StubRoutingAgent(response)
    registry = _StubRegistry(None)

    with pytest.raises(RoutingFailed, match="cannot route"):
        await _route_request_to_matching_agent(routing_agent, registry, _ctx())
