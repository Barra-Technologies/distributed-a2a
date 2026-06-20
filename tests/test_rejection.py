import pytest
from a2a.types import AgentCapabilities, AgentCard, TaskState

from distributed_a2a.client import RoutingA2AClient


def build_card(name: str, url: str) -> AgentCard:
    return AgentCard(
        name=name,
        description=f"{name} description",
        url=url,
        version="1.0.0",
        default_input_modes=["text", "text/plaintext"],
        default_output_modes=["text", "text/plaintext"],
        skills=[],
        preferred_transport="HTTP+JSON",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
    )


@pytest.mark.asyncio
async def test_rejection_triggers_automated_rerouting(monkeypatch: pytest.MonkeyPatch) -> None:
    router_card = build_card("Router", "http://router")
    rejecting_card = build_card("rejecting-agent", "http://rejecting")
    success_card = build_card("success-agent", "http://success")

    client = RoutingA2AClient(initial_url="http://router")
    client.current_card = router_card

    async def fake_fetch_initial_card() -> None:
        client.current_card = router_card

    monkeypatch.setattr(client, "fetch_initial_card", fake_fetch_initial_card)

    router_messages: list[str] = []

    class FakeRemoteAgentConnection:
        def __init__(self, agent_card: AgentCard, _client: object, **_kwargs: object) -> None:
            self.agent_card = agent_card

        async def send_message(self, message_to_send: str, _context_id: str) -> str | AgentCard | TaskState:
            if self.agent_card.name == "Router":
                router_messages.append(message_to_send)
                if "Please exclude the following agents from routing: rejecting-agent" in message_to_send:
                    return success_card
                return rejecting_card

            if self.agent_card.name == "rejecting-agent":
                return TaskState.rejected

            if self.agent_card.name == "success-agent":
                return "final answer"

            raise AssertionError(f"Unexpected agent {self.agent_card.name}")

    monkeypatch.setattr("distributed_a2a.client.RemoteAgentConnection", FakeRemoteAgentConnection)

    result = await client.send_message("Hello", context_id="ctx-1")

    assert result == "final answer"
    assert len(router_messages) == 2
    assert "Please exclude the following agents from routing" not in router_messages[0]
    assert "Please exclude the following agents from routing: rejecting-agent" in router_messages[1]


@pytest.mark.asyncio
async def test_rejected_agents_reset_between_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    router_card = build_card("Router", "http://router")
    rejecting_card = build_card("rejecting-agent", "http://rejecting")
    success_card = build_card("success-agent", "http://success")

    client = RoutingA2AClient(initial_url="http://router")
    client.current_card = router_card

    async def fake_fetch_initial_card() -> None:
        client.current_card = router_card

    monkeypatch.setattr(client, "fetch_initial_card", fake_fetch_initial_card)

    step = {"idx": 0}

    def next_step() -> int:
        current = step["idx"]
        step["idx"] += 1
        return current

    class FakeRemoteAgentConnection:
        def __init__(self, agent_card: AgentCard, _client: object, **_kwargs: object) -> None:
            self.agent_card = agent_card

        async def send_message(self, message_to_send: str, _context_id: str) -> str | AgentCard | TaskState:
            idx = next_step()
            match idx:
                case 0:
                    assert self.agent_card.name == "Router"
                    assert "Please exclude the following agents from routing" not in message_to_send
                    return rejecting_card
                case 1:
                    assert self.agent_card.name == "rejecting-agent"
                    return TaskState.rejected
                case 2:
                    assert self.agent_card.name == "Router"
                    assert "Please exclude the following agents from routing: rejecting-agent" in message_to_send
                    return success_card
                case 3:
                    assert self.agent_card.name == "success-agent"
                    return "first response"
                case 4:
                    assert self.agent_card.name == "success-agent"
                    return TaskState.rejected
                case 5:
                    assert self.agent_card.name == "Router"
                    assert "Please exclude the following agents from routing: success-agent" in message_to_send
                    assert "rejecting-agent" not in message_to_send
                    return rejecting_card
                case 6:
                    assert self.agent_card.name == "rejecting-agent"
                    return "second response"
                case _:
                    raise AssertionError(f"Unexpected step {idx}")

    monkeypatch.setattr("distributed_a2a.client.RemoteAgentConnection", FakeRemoteAgentConnection)

    first = await client.send_message("First", context_id="ctx-1")
    second = await client.send_message("Second", context_id="ctx-2")

    assert first == "first response"
    assert second == "second response"


@pytest.mark.asyncio
async def test_fails_when_router_returns_already_rejected_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    router_card = build_card("Router", "http://router")
    rejecting_card = build_card("rejecting-agent", "http://rejecting")

    client = RoutingA2AClient(initial_url="http://router")
    client.current_card = router_card

    async def fake_fetch_initial_card() -> None:
        client.current_card = router_card

    monkeypatch.setattr(client, "fetch_initial_card", fake_fetch_initial_card)

    step = {"idx": 0}

    class FakeRemoteAgentConnection:
        def __init__(self, agent_card: AgentCard, _client: object, **_kwargs: object) -> None:
            self.agent_card = agent_card

        async def send_message(self, message_to_send: str, _context_id: str) -> str | AgentCard | TaskState:
            idx = step["idx"]
            step["idx"] += 1
            if idx == 0:
                assert self.agent_card.name == "Router"
                return rejecting_card
            if idx == 1:
                assert self.agent_card.name == "rejecting-agent"
                return TaskState.rejected
            if idx == 2:
                assert self.agent_card.name == "Router"
                assert "Please exclude the following agents from routing: rejecting-agent" in message_to_send
                return rejecting_card
            raise AssertionError(f"Unexpected step {idx}")

    monkeypatch.setattr("distributed_a2a.client.RemoteAgentConnection", FakeRemoteAgentConnection)

    with pytest.raises(Exception, match="already rejected but was redirected to again"):
        await client.send_message("Hello", context_id="ctx-1")
