import random
import threading
import time
from http.server import HTTPServer
from typing import Generator

import pytest
import uvicorn
from a2a.types import TaskState

from distributed_a2a.client import RoutingA2AClient
from distributed_a2a.registry_server.bootstrap import load_registry
from distributed_a2a.registry_server.in_memory_registry_storage import InMemoryAgentRegistry, InMemoryMcpRegistry
from tests.fake_agent import TestAgent
from tests.fake_llm import get_llm_handler

FINAL_RESPONSE = "Hello! This is a mock response from the fake OpenAI server."

@pytest.fixture(scope="module")
def fake_completed_llm() -> Generator[str]:
    for url in fake_llm_server(TaskState.completed, FINAL_RESPONSE):
        yield url


def fake_llm_server(state: TaskState, response: str) -> Generator[str]:
    port = random.randint(10000, 60000)
    # noinspection PyTypeChecker
    server = HTTPServer(('127.0.0.1', port), get_llm_handler(state, response))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(1)
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def fake_registry_server():
    port = 8082
    agent_registry = InMemoryAgentRegistry()
    mcp_registry = InMemoryMcpRegistry()
    app = load_registry(agent_registry, mcp_registry)

    config = uvicorn.Config(app, host="127.0.0.1", port=port)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1)

    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.mark.asyncio
async def test_app_completed_path(fake_registry_server, fake_completed_llm):
    # Given
    with TestAgent(fake_registry_server, fake_completed_llm, "test-agent") as agent:
        # When
        client = RoutingA2AClient(initial_url=f"http://127.0.0.1:{agent.app_port}/{agent.name}")
        response = await client.send_message(message="Hello", context_id="test-context")

        # Then
        assert FINAL_RESPONSE in response


@pytest.mark.asyncio
async def test_app_redirect_path(fake_registry_server, fake_completed_llm):
    # Given
    with TestAgent(fake_registry_server, fake_completed_llm, "second-agent") as second_agent:
        # use the agent card of the second agent as the response message of the first agent
        card_response: str = second_agent.get_agent_card().model_dump_json()
        for llm_url in fake_llm_server(TaskState.rejected, card_response):
            with TestAgent(fake_registry_server, llm_url, "redirect-agent") as first_agent:
                client = RoutingA2AClient(initial_url=f"http://127.0.0.1:{first_agent.app_port}/{first_agent.name}")

                # When
                response = await client.send_message(message="Hello", context_id="test-context")

                # Then
                assert FINAL_RESPONSE in response
