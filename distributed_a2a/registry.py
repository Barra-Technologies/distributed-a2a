"""Registry lookup for agents and MCP servers."""
import asyncio
import os
import logging
from typing import Callable, Any, cast
import httpx
from langchain_core.tools import StructuredTool
from a2a.types import AgentCard

from .config import settings

logger = logging.getLogger(__name__)

if settings.httpx_logging:
    logging.getLogger("httpx").setLevel(logging.DEBUG)


async def registry_heart_beat(name: str, registry: 'AgentRegistryLookup', agent_card: AgentCard, interval_sec: int,
                              get_expire_at: Callable[[], int]) -> None:
    """Periodically sends heartbeats to the registry to keep the agent registration alive.

    Args:
        name: Name of the agent.
        registry: The agent registry lookup instance.
        agent_card: The agent's card information.
        interval_sec: The interval in seconds between heartbeats.
        get_expire_at: The next expiration timestamp.
    """
    registry.put_agent_card(name=name, agent_card=agent_card.model_dump(), expire_at=get_expire_at())
    while True:
        try:
            registry.patch_agent_expiry(name=name, expire_at=get_expire_at())
        except Exception as e:
            logger.error(f"Failed to send heart beat to registry: {e}")
        await asyncio.sleep(interval_sec)


class AgentRegistryLookup:
    """Client for looking up agent information in the registry."""

    def __init__(self, registry_url: str, req_opts: dict[str,str] = {}):
        """Initializes the AgentRegistryLookup client.

        Args:
            registry_url: The base URL of the registry service.
            req_opts: Optional dictionary of HTTP headers for requests.
        """
        if req_opts is None:
            req_opts = {}
        self.registry_url = registry_url
        self.client = httpx.Client(headers=req_opts, timeout=30)

    def get_agent_cards(self) -> list[dict[str, Any]]:
        """Retrieves all registered agent cards.

        Returns:
            A list of agent cards as dictionaries.
        """
        response = self.client.get(url=f"{self.registry_url}/agent-cards")
        response.raise_for_status()
        return cast(list[dict[str, Any]], response.json())

    def get_agent_card(self, name: str) -> dict[str, Any] | None:
        """Retrieves a specific agent card by name.

        Args:
            name: The name of the agent.

        Returns:
            The agent card as a dictionary, or None if not found.
        """
        response = self.client.get(url=f"{self.registry_url}/agent-card/{name}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return cast(dict[str, Any] | None, response.json())

    def put_agent_card(self, name: str, agent_card: dict[str, Any], expire_at: int) -> None:
        """Registers or updates an agent card in the registry.

        Args:
            name: The name of the agent.
            agent_card: The agent card dictionary.
            expire_at: Expiration timestamp for the registration.
        """
        response = self.client.put(
            url=f"{self.registry_url}/agent-card/{name}",
            params={"expire_at": str(expire_at)},
            json=agent_card,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during put_agent_card: {e}")
            if response.text:
                logger.error(f"Response content: {response.text}")
            raise

    def patch_agent_expiry(self, name: str, expire_at: int) -> None:
        """Updates the expiration timestamp for an agent registration (heartbeat).

        Args:
            name: The name of the agent.
            expire_at: The new expiration timestamp.
        """
        response = self.client.patch(
            url=f"{self.registry_url}/agent-card/{name}/heartbeat",
            params={"expire_at": str(expire_at)},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during patch_agent_expiry: {e} with response: {response.text if response.text else '<empty>'}")
            raise e

    def as_tool(self) -> StructuredTool:
        """Wraps the agent card lookup as a LangChain StructuredTool.

        Returns:
            A StructuredTool for looking up agent cards.
        """
        return StructuredTool.from_function(func=lambda: self.get_agent_cards(), name="agent_card_lookup",
                                            description="Gets all available agent cards")


class McpRegistryLookup:
    """Client for looking up MCP server information in the registry."""

    def __init__(self, registry_url: str, req_opts: dict[str, str] = {}):
        """Initializes the McpRegistryLookup client.

        Args:
            registry_url: The base URL of the registry service.
            req_opts: Optional dictionary of HTTP headers for requests.
        """
        if req_opts is None:
            req_opts = {}
        self.registry_url = registry_url
        self.client = httpx.Client(timeout=30, headers=req_opts)

    def get_mcp_tool_for_agent(self, agent_name: str) -> list[dict[str, Any]]:
        """Retrieves MCP servers associated with a specific agent.

        Args:
            agent_name: The name of the agent.

        Returns:
            A list of MCP server definitions.
        """
        response = self.client.get(url=f"{self.registry_url}/mcp/agent/{agent_name}/servers")
        response.raise_for_status()
        return cast(list[dict[str, Any]], response.json())