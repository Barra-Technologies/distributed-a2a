import threading
from http.server import HTTPServer
from typing import Generator

import pytest
import uvicorn
from a2a.types import TaskState

from distributed_a2a.client import RoutingA2AClient
from distributed_a2a.registry_server.bootstrap import load_registry
from distributed_a2a.registry_server.in_memory_registry_storage import (
    InMemoryAgentRegistry, InMemoryMcpRegistry)
from tests.fake_agent import FakeAgent, _free_port, wait_for_http
from tests.fake_llm import get_llm_handler

FINAL_RESPONSE = "Hello! This is a mock response from the fake OpenAI server."

@pytest.fixture(scope="module")
def fake_completed_llm() -> Generator[str, None, None]:
    for url in fake_llm_server(TaskState.completed, FINAL_RESPONSE):
        yield url


def fake_llm_server(state: TaskState, response: str) -> Generator[str, None, None]:
    port = _free_port()
    # noinspection PyTypeChecker
    server = HTTPServer(('127.0.0.1', port), get_llm_handler(state, response))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # The stdlib HTTPServer binds before serve_forever, so the socket is
    # already accepting connections; a single readiness probe is enough.
    wait_for_http(f"http://127.0.0.1:{port}/", timeout=5.0)
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def fake_registry_server() -> Generator[str, None, None]:
    port = _free_port()
    agent_registry = InMemoryAgentRegistry()
    mcp_registry = InMemoryMcpRegistry()
    app = load_registry(agent_registry, mcp_registry)

    config = uvicorn.Config(app, host="127.0.0.1", port=port)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_for_http(f"http://127.0.0.1:{port}/health", timeout=10.0)

    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.mark.asyncio
async def test_app_completed_path(fake_registry_server: str, fake_completed_llm: str) -> None:
    # Given
    with FakeAgent(fake_registry_server, fake_completed_llm, "test-agent") as agent:
        # When
        client = RoutingA2AClient(initial_url=f"http://127.0.0.1:{agent.app_port}/{agent.name}")
        reply = await client.send_message(message="Hello", context_id="test-context")

        # Then: Check the response
        assert reply.text is not None
        assert "This is a mock response from the fake OpenAI server." in reply.text
        assert reply.files == []


@pytest.mark.asyncio
async def test_app_redirect_path(fake_registry_server: str, fake_completed_llm: str) -> None:
    # Given
    with FakeAgent(fake_registry_server, fake_completed_llm, "second-agent") as second_agent:
        # use the agent card of the second agent as the response message of the first agent
        for llm_url in fake_llm_server(TaskState.rejected, second_agent.name):
            with FakeAgent(fake_registry_server, llm_url, "redirect-agent") as first_agent:
                client = RoutingA2AClient(initial_url=f"http://127.0.0.1:{first_agent.app_port}")

                # When
                reply = await client.send_message(message="Hello", context_id="test-context")

                # Then
                assert reply.text is not None
                assert FINAL_RESPONSE in reply.text
