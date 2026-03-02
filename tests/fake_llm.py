import json
from http.server import BaseHTTPRequestHandler
from socketserver import BaseRequestHandler

from a2a.types import TaskState

from distributed_a2a.agent import RoutingResponse, StringResponse


def get_llm_handler(status: TaskState, message: str) -> type[BaseRequestHandler]:
    class FakeOpenAIHandler(BaseHTTPRequestHandler):

        # noinspection PyPep8Naming
        def do_POST(self):
            if self.path == '/v1/chat/completions':
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    raise ValueError("No request body provided")
                request_body: dict = json.loads(self.rfile.read(content_length).decode('utf-8'))

                # noinspection PyTypeChecker
                requested_tools: list[str] = [tool['function']['name'] for tool in request_body['tools']]
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()

                arguments = {
                    "status": status.name,
                }
                if RoutingResponse.__name__ in requested_tools:
                    arguments["agent_card"] = message
                    tool: str = RoutingResponse.__name__
                elif StringResponse.__name__ in requested_tools:
                    arguments["response"] = message
                    tool: str = StringResponse.__name__
                else:
                    raise ValueError(f"Unknown tools requested: {requested_tools}")

                response = {
                    "id": "chatcmpl-mock123",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "foo",
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
                        "logprobs": None,
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 12,
                        "total_tokens": 22
                    }
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
    return FakeOpenAIHandler