import os
import random
import threading
import time
from typing import Any, Optional

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.types import AgentCard

from distributed_a2a.model import AgentConfig, AgentItem, RegistryConfig, RegistryItemConfig, LLMConfig, CardConfig
from distributed_a2a.server import load_app, get_agent_card
from stubs import NonRoutingAgent

API_KEY_ENV_VAR = "FAKE_API_KEY"
os.environ["FAKE_API_KEY"] = "fake-key"


class FakeAgent:
    executor_overwrite: Optional[AgentExecutor] = None

    def __init__(self, registry_url: str, llm_url: str, name: str, routing: bool = True) -> None:
        self._registry_url = registry_url
        self._llm_url = llm_url
        self.name = name
        self.app_port = random.randint(10000, 60000)
        self.routing = routing
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

    def __enter__(self) -> FakeAgent:
        executor_overwrite  = NonRoutingAgent(agent_config=self.config) if not self.routing else None
        app = load_app(agent_config=self.config, executor_overwrite=executor_overwrite)

        # Start the app server in a separate thread
        app_config = uvicorn.Config(app, host="127.0.0.1", port=self.app_port)
        self._app_server = uvicorn.Server(app_config)
        self._app_thread = threading.Thread(target=self._app_server.run, daemon=True)
        self._app_thread.start()
        time.sleep(2)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._app_server.should_exit = True
        self._app_thread.join(timeout=5)
