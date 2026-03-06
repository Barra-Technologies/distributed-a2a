import random
import threading
import time
import os
import json
from typing import Generator, Any, Callable
from http.server import HTTPServer, BaseHTTPRequestHandler
import pytest
import uvicorn
from a2a.types import TaskState
from distributed_a2a.client import RoutingA2AClient
from distributed_a2a.registry_server.bootstrap import load_registry
from distributed_a2a.registry_server.in_memory_registry_storage import InMemoryAgentRegistry, InMemoryMcpRegistry
from tests.fake_agent import FakeAgent
from distributed_a2a.router import load_router
from distributed_a2a.model import RouterConfig, RouterItem, CardConfig, LLMConfig, RegistryConfig, RegistryItemConfig
import httpx

FINAL_RESPONSE = "Hello! This is a mock response from the second agent."

@pytest.fixture
def captured_requests() -> list[dict[str, Any]]:
    return []

@pytest.fixture
def llm_responses() -> list[tuple[TaskState, str]]:
    return []

@pytest.fixture
def fake_llm_server(llm_responses: list[tuple[TaskState, str]], captured_requests: list[dict[str, Any]]) -> Generator[str, None, None]:
    port = random.randint(10000, 60000)
    
    class StatefulHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            if self.path == '/v1/chat/completions':
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                request_body = json.loads(body)
                captured_requests.append(request_body)
                
                if llm_responses:
                    status, message = llm_responses.pop(0)
                else:
                    status, message = TaskState.completed, "No more responses planned"

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()

                arguments = {"status": status.name}
                # noinspection PyTypeChecker
                requested_tools: list[str] = [tool['function']['name'] for tool in request_body.get('tools', [])]
                
                tool_name: str
                if "RoutingResponse" in requested_tools:
                    arguments["agent_name"] = message
                    tool_name = "RoutingResponse"
                elif "StringResponse" in requested_tools:
                    arguments["response"] = message
                    tool_name = "StringResponse"
                else:
                    # Default to StringResponse
                    arguments["response"] = message
                    tool_name = "StringResponse"

                response = {
                    "id": "chatcmpl-mock123",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(arguments)
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer(('127.0.0.1', port), StatefulHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()
    thread.join(timeout=2)

@pytest.fixture(scope="module")
def fake_registry_server() -> Generator[str, None, None]:
    port = 8083
    agent_registry = InMemoryAgentRegistry()
    mcp_registry = InMemoryMcpRegistry()
    app = load_registry(agent_registry, mcp_registry)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
    start_time = time.time()
    while time.time() - start_time < 5:
        try:
            with httpx.Client() as client:
                if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                    break
        except:
            pass
        time.sleep(0.1)

    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=2)

@pytest.fixture
def router_server_factory(fake_registry_server: str, fake_llm_server: str) -> Generator[Callable[[int], str], None, None]:
    servers = []
    def _run_router(port: int) -> str:
        router_config = RouterConfig(
            router=RouterItem(
                registry=RegistryConfig(agent=RegistryItemConfig(url=fake_registry_server)),
                card=CardConfig(name="Router", description="Router Agent", version="1.0.0", url=f"http://127.0.0.1:{port}", skills=[]),
                llm=LLMConfig(base_url=fake_llm_server, model="foo", api_key_env="FAKE_API_KEY")
            )
        )
        os.environ["FAKE_API_KEY"] = "fake-key"
        app = load_router(router_config=router_config)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        servers.append((server, thread))
        
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                with httpx.Client() as client:
                    if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                        break
            except:
                pass
            time.sleep(0.1)
        return f"http://127.0.0.1:{port}"

    yield _run_router

    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=2)

@pytest.mark.asyncio
async def test_rejection_full_flow(fake_registry_server: str, fake_llm_server: str, llm_responses: list[tuple[TaskState, str]], router_server_factory: Callable[[int], str]) -> None:
    with FakeAgent(registry_url=fake_registry_server, llm_url=fake_llm_server, name="rejecting-agent", routing=False) as rejecting_agent:
        with FakeAgent(fake_registry_server, fake_llm_server, "success-agent") as success_agent:
            router_port = random.randint(10000, 60000)
            router_url = router_server_factory(router_port)
            
            llm_responses.extend([
                (TaskState.completed, "rejecting-agent"),
                (TaskState.rejected, "I cannot handle this"),
                (TaskState.completed, "success-agent"),
                (TaskState.completed, FINAL_RESPONSE)
            ])
            
            client = RoutingA2AClient(initial_url=router_url)
            result = await client.send_message("Hello", context_id="test-context")
            assert FINAL_RESPONSE in result

@pytest.mark.asyncio
async def test_rejection_reset_between_calls(fake_registry_server: str, fake_llm_server: str, llm_responses: list[tuple[TaskState, str]], captured_requests: list[Any], router_server_factory: Callable[[int], str]) -> None:
    with FakeAgent(registry_url=fake_registry_server, llm_url=fake_llm_server, name="rejecting-agent", routing=False) as rejecting_agent:
        with FakeAgent(fake_registry_server, fake_llm_server, "success-agent") as success_agent:
            router_port = random.randint(10000, 60000)
            router_url = router_server_factory(router_port)
            
            llm_responses.extend([
                (TaskState.completed, "rejecting-agent"),
                (TaskState.rejected, "I cannot handle this"),
                (TaskState.completed, "success-agent"),
                (TaskState.completed, "First response")
            ])
            
            client = RoutingA2AClient(initial_url=router_url)
            await client.send_message("First call", context_id="ctx1")
            
            llm_responses.extend([
                (TaskState.completed, "success-agent"),
                (TaskState.completed, "Second response")
            ])
            
            await client.send_message("Second call", context_id="ctx2")
