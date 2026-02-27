"""Storage abstractions for agent and MCP registries."""
from abc import ABC, abstractmethod
from typing import Any
from .model import McpServer

class AgentRegistryLookup(ABC):
    @abstractmethod
    def get_agent_cards(self) -> list[dict[str, Any]]:
        """Retrieves all registered agent cards.

        Returns:
            A list of agent cards as dictionaries.
        """
        pass

    @abstractmethod
    def get_agent_card(self, name: str) -> str | None:
        """Retrieves a specific agent card by name.

        Args:
            name: The name of the agent.

        Returns:
            The agent card (JSON string), or None if not found.
        """
        pass

    @abstractmethod
    def put_agent_card(self, name: str, card: str, expire_at: str) -> None:
        """Registers or updates an agent card.

        Args:
            name: The name of the agent.
            card: The agent card (JSON string).
            expire_at: Expiration timestamp for the registration.
        """
        pass

    @abstractmethod
    def update_agent_expiry(self, name: str, expire_at: str) -> None:
        """Updates the expiration timestamp for an agent registration.

        Args:
            name: The name of the agent.
            expire_at: The new expiration timestamp.
        """
        pass


class McpRegistryLookup(ABC):
    @abstractmethod
    def get_mcp_servers(self) -> list[McpServer]:
        """Retrieves all registered MCP servers.

         Returns:
             A list of McpServer instances.
         """
        pass

    @abstractmethod
    def get_mcp_server(self, name: str) -> McpServer | None:
        """Retrieves a specific MCP server by name.

        Args:
            name: The name of the MCP server.

        Returns:
            The McpServer instance, or None if not found.
        """
        pass

    @abstractmethod
    def put_mcp_server(self, server: McpServer, allowed_agents: set[str] | None = None) -> None:
        """Registers or updates an MCP server.

        Args:
            server: The McpServer instance to register.
            allowed_agents: Optional set of agent names allowed to access this server.
        """
        pass

    @abstractmethod
    def get_allowed_agents(self, server_name: str) -> set[str]:
        """Registers or updates an MCP server.

        Args:
            server: The McpServer instance to register.
            allowed_agents: Optional set of agent names allowed to access this server.
        """
        pass

    @abstractmethod
    def enable_mcp_server_for_agent(self, server_name: str, agent_name: str) -> None:
        """Grants an agent access to a specific MCP server.

        Args:
            server_name: The name of the MCP server.
            agent_name: The name of the agent to authorize.
        """
        pass

    @abstractmethod
    def disable_mcp_server_for_agent(self, server_name: str, agent_name: str) -> None:
        """Revokes an agent's access to a specific MCP server.

        Args:
            server_name: The name of the MCP server.
            agent_name: The name of the agent to deauthorize.
        """
        pass

    @abstractmethod
    def get_mcp_server_for_agent(self, agent_name: str) -> list[McpServer]:
        """Retrieves all MCP servers that a specific agent is authorized to access.

        Args:
            agent_name: The name of the agent.

        Returns:
            A list of McpServer instances.
        """
        pass
