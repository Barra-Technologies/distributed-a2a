"""In-memory storage implementation for agent and MCP registries."""
import json
from typing import Any
from .model import McpServer
from .storage import McpRegistryLookup, AgentRegistryLookup


class InMemoryMcpRegistry(McpRegistryLookup):
    """In-memory implementation of the MCP server registry."""

    def __init__(self) -> None:
        self._servers: dict[str, McpServer] = {}
        self._allowed_agents: dict[str, set[str]] = {}

    def get_mcp_servers(self) -> list[McpServer]:
        """Retrieves all registered MCP servers."""
        return list(self._servers.values())

    def get_mcp_server(self, name: str) -> McpServer | None:
        """Retrieves a specific MCP server by name."""
        return self._servers.get(name)

    def put_mcp_server(self, server: McpServer, allowed_agents: set[str] | None = None) -> None:
        """Registers or updates an MCP server."""
        self._servers[server.name] = server
        if allowed_agents is not None:
            self._allowed_agents[server.name] = set(allowed_agents)
        elif server.name not in self._allowed_agents:
            self._allowed_agents[server.name] = set()

    def get_allowed_agents(self, server_name: str) -> set[str]:
        """Retrieves the set of agent names allowed to access a specific MCP server."""
        return self._allowed_agents.get(server_name, set())

    def enable_mcp_server_for_agent(self, server_name: str, agent_name: str) -> None:
        """Grants an agent access to a specific MCP server."""
        if server_name not in self._allowed_agents:
            self._allowed_agents[server_name] = set()
        self._allowed_agents[server_name].add(agent_name)

    def disable_mcp_server_for_agent(self, server_name: str, agent_name: str) -> None:
        """Revokes an agent's access to a specific MCP server."""
        if server_name in self._allowed_agents:
            self._allowed_agents[server_name].discard(agent_name)

    def get_mcp_server_for_agent(self, agent_name: str) -> list[McpServer]:
        """Retrieves all MCP servers that a specific agent is authorized to access."""
        authorized_servers = []
        for server_name, agents in self._allowed_agents.items():
            if agent_name in agents:
                server = self._servers.get(server_name)
                if server:
                    authorized_servers.append(server)
        return authorized_servers


class InMemoryAgentRegistry(AgentRegistryLookup):
    """In-memory implementation of the agent registry."""

    def __init__(self) -> None:
        self._agents: dict[str, dict[str, Any]] = {}

    def get_agent_cards(self) -> list[dict[str, Any]]:
        """Retrieves all registered agent cards."""
        return [json.loads(agent["card"]) for agent in self._agents.values()]

    def get_agent_card(self, name: str) -> str | None:
        """Retrieves a specific agent card by name."""
        agent_data = self._agents.get(name)
        return agent_data["card"] if agent_data else None

    def put_agent_card(self, name: str, card: str, expire_at: str) -> None:
        """Registers or updates an agent card."""
        self._agents[name] = {
            "card": card,
            "expire_at": expire_at
        }

    def update_agent_expiry(self, name: str, expire_at: str) -> None:
        """Updates the expiration timestamp for an agent registration."""
        if name in self._agents:
            self._agents[name]["expire_at"] = expire_at
