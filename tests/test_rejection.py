import random
import threading
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import pytest
import uvicorn
import json
from a2a.types import TaskState, AgentCard
from distributed_a2a.client import RoutingA2AClient
from distributed_a2a.registry_server.bootstrap import load_registry
from distributed_a2a.registry_server.in_memory_registry_storage import InMemoryAgentRegistry, InMemoryMcpRegistry
from stubs import NonRoutingAgent
from tests.fake_agent import FakeAgent
from distributed_a2a.agent import RoutingResponse, StringResponse
from distributed_a2a.router import load_router
from distributed_a2a.model import RouterConfig, RouterItem, CardConfig, LLMConfig, RegistryConfig, RegistryItemConfig

FINAL_RESPONSE = "Hello! This is a mock response from the second agent."

def fake_llm_server_stateful(responses: list[tuple[TaskState, str]], captured_requests: list):
    port = random.randint(10000, 60000)
    
    class StatefulHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/v1/chat/completions':
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                request_body = json.loads(body)
                captured_requests.append(request_body)
                
                # Get response based on call count
                if responses:
                    status, message = responses.pop(0)
                else:
                    status, message = TaskState.completed, "No more responses planned"

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()

                arguments = {"status": status.name}
                # noinspection PyTypeChecker
                requested_tools: list[str] = [tool['function']['name'] for tool in request_body.get('tools', [])]
                
                if RoutingResponse.__name__ in requested_tools:
                    arguments["agent_card"] = message
                    tool: str = RoutingResponse.__name__
                elif StringResponse.__name__ in requested_tools:
                    arguments["response"] = message
                    tool: str = StringResponse.__name__
                else:
                    # Default to StringResponse if not specified, though it should be one of them
                    arguments["response"] = message
                    tool: str = StringResponse.__name__

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
                                    "name": tool,
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
    time.sleep(1)
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()
    thread.join(timeout=5)

@pytest.fixture(scope="module")
def fake_registry_server():
    port = 8083
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
async def test_rejection_full_flow(fake_registry_server):
    # This test verifies that RoutingA2AClient handles agent rejection by asking the router for another agent.
    
    captured_requests = []
    
    # We need a sequence of responses for the stateful LLM server:
    # 1. Router: Return RejectingAgent card
    # 2. RejectingAgent: Return rejected status
    # 3. Router: Return SuccessAgent card
    # 4. SuccessAgent: Return success message
    
    # These will be created inside the loop
    responses = []
    
    for llm_url in fake_llm_server_stateful(responses, captured_requests):
        with FakeAgent(registry_url=fake_registry_server, llm_url=llm_url, name="rejecting-agent", routing=False) as rejecting_agent:
            rejecting_agent.executor_overwrite = NonRoutingAgent(rejecting_agent.config)
            with FakeAgent(fake_registry_server, llm_url, "success-agent") as success_agent:
                
                # Setup Router
                router_port = random.randint(10000, 60000)
                router_config = RouterConfig(
                    router=RouterItem(
                        registry=RegistryConfig(
                            agent=RegistryItemConfig(url=fake_registry_server)
                        ),
                        card=CardConfig(
                            name="Router",
                            description="Router Agent",
                            version="1.0.0",
                            url=f"http://127.0.0.1:{router_port}",
                            skills=[]
                        ),
                        llm=LLMConfig(
                            base_url=llm_url,
                            model="foo",
                            api_key_env="FAKE_API_KEY"
                        )
                    )
                )
                os.environ["FAKE_API_KEY"] = "fake-key"
                router_app = load_router(router_config=router_config)
                router_uvicorn_config = uvicorn.Config(router_app, host="127.0.0.1", port=router_port)
                router_server = uvicorn.Server(router_uvicorn_config)
                router_thread = threading.Thread(target=router_server.run, daemon=True)
                router_thread.start()
                time.sleep(2)
                
                try:
                    # Now prepare the responses for the LLM
                    # 1. Router receives initial request and returns rejecting-agent card
                    responses.append((TaskState.completed, rejecting_agent.get_agent_card().model_dump_json()))
                    # 2. RejectingAgent receives request and rejects
                    responses.append((TaskState.rejected, "I cannot handle this"))
                    # 3. Router receives retry request (with exclude_agents=['rejecting-agent']) and returns success-agent card
                    responses.append((TaskState.completed, success_agent.get_agent_card().model_dump_json()))
                    # 4. SuccessAgent receives request and completes
                    responses.append((TaskState.completed, FINAL_RESPONSE))
                    
                    client = RoutingA2AClient(initial_url=f"http://127.0.0.1:{router_port}")
                    
                    result = await client.send_message("Hello", context_id="test-context")

                    assert result == FINAL_RESPONSE
                    assert "rejecting-agent" in client.rejected_agents
                    
                finally:
                    router_server.should_exit = True
                    router_thread.join(timeout=5)
