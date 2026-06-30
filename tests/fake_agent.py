import os
import socket
import threading
import time
from typing import Any

import httpx
import uvicorn
from a2a.types import AgentCard

from distributed_a2a.model import (AgentConfig, AgentItem, CardConfig,
                                   LLMConfig, RegistryConfig,
                                   RegistryItemConfig)
from distributed_a2a.server import get_agent_card, load_app

API_KEY_ENV_VAR = "FAKE_API_KEY"
os.environ["FAKE_API_KEY"] = "fake-key"


def _free_port() -> int:
    """Return a free local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_http(url: str, timeout: float = 10.0, interval: float = 0.05) -> None:
    """Poll ``url`` until it returns any HTTP response or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=1.0)
            return
        except Exception as e:  # noqa: BLE001 - waiting on server startup
            last_exc = e
            time.sleep(interval)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s: {last_exc}")


class FakeAgent:

    def __init__(self, registry_url: str, llm_url: str, name: str) -> None:
        self._registry_url = registry_url
        self._llm_url = llm_url
        self.name = name
        self.app_port = _free_port()
        self.config = AgentConfig(
            agent=AgentItem(
                registry=RegistryConfig(
                    agent=RegistryItemConfig(url=self._registry_url),
                    mcp=RegistryItemConfig(url=self._registry_url)
                ),
                card=CardConfig(
                    name=self.name,
                    description="A test agent",
                    version="1.0.0",
                    url=f"http://127.0.0.1:{self.app_port}",
                    skills=[]
                ),
                llm=LLMConfig(
                    base_url=self._llm_url,
                    model="foo",
                    api_key_env=API_KEY_ENV_VAR
                ),
                system_prompt="You are a test agent."
            )
        )

    def get_agent_card(self) -> AgentCard:
        return get_agent_card(self.config)

    def __enter__(self) -> "FakeAgent":
        app = load_app(self.config)

        # Start the app server in a separate thread
        app_config = uvicorn.Config(app, host="127.0.0.1", port=self.app_port)
        self._app_server = uvicorn.Server(app_config)
        self._app_thread = threading.Thread(target=self._app_server.run, daemon=True)
        self._app_thread.start()
        # The A2A app exposes the agent card at /{slug}/.well-known/agent-card.json.
        # Probe the root URL: any HTTP response (200/404) signals the server is up.
        wait_for_http(f"http://127.0.0.1:{self.app_port}/", timeout=10.0)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._app_server.should_exit = True
        self._app_thread.join(timeout=5)
