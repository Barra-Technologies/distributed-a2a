"""DynamoDB implementation of the registry storage."""
import json
import logging

from typing import Any, cast
import boto3
from boto3.dynamodb.conditions import Attr

from .model import McpServer
from .storage import AgentRegistryLookup, McpRegistryLookup

MCP_SERVER_COLUMN = "server"
ALLOWED_AGENTS_FIELD = "allowed-agents"


class DynamoDbAgentRegistryLookup(AgentRegistryLookup):
    """DynamoDB-backed storage for agent registry."""
    def __init__(self, agent_card_table: str) -> None:
        """Initializes the DynamoDB agent registry lookup.

          Args:
              agent_card_tabel: The name of the DynamoDB table for agent cards.
          """
        dynamo = boto3.resource("dynamodb", region_name="eu-central-1")
        self.table = dynamo.Table(agent_card_table)

    def get_agent_cards(self) -> list[dict[str, Any]]:
        """Initializes the DynamoDB agent registry lookup.

        Args:
            agent_card_tabel: The name of the DynamoDB table for agent cards.
        """
        items = self.table.scan().get("Items", [])
        cards: list[dict[str, Any]] = [json.loads(it["card"]) for it in items]
        return cards

    def get_agent_card(self, name: str) -> str | None:
        """Retrieves a specific agent card from DynamoDB by name.

             Args:
                 name: The name of the agent.

             Returns:
                 The agent card (JSON string), or None if not found.
             """
        response = self.table.get_item(Key={"id": name})
        item = response.get("Item")
        if item:
            return cast(str, item.get("card"))
        return None

    def put_agent_card(self, name: str, card: str, expire_at: str) -> None:
        """Registers or updates an agent card in DynamoDB.

        Args:
            name: The name of the agent.
            card: The agent card (JSON string).
            expire_at: Expiration timestamp for the registration.
        """
        self.table.put_item(Item={"id": name, "card": card, "expireAt": expire_at})

    def update_agent_expiry(self, name: str, expire_at: str) -> None:
        """Updates the expiration timestamp for an agent registration in DynamoDB.

        Args:
            name: The name of the agent.
            expire_at: The new expiration timestamp.
        """
        self.table.update_item(
            Key={"id": name},
            UpdateExpression="SET expireAt = :val",
            ExpressionAttributeValues={":val": expire_at}
        )


class DynamoDbMcpRegistryLookup(McpRegistryLookup):
    """DynamoDB-backed storage for MCP server registry."""
    def __init__(self, mcp_table: str = "mcp-servers") -> None:
        """Initializes the DynamoDB MCP registry lookup.

         Args:
             mcp_table: The name of the DynamoDB table for MCP servers.
         """
        dynamo = boto3.resource("dynamodb", region_name="eu-central-1")
        self.table = dynamo.Table(mcp_table)

    def get_mcp_servers(self) -> list[McpServer]:
        """Retrieves all registered MCP servers from DynamoDB.

        Returns:
            A list of McpServer instances.
        """
        items = self.table.scan().get("Items", [])
        servers: list[McpServer] = [McpServer.model_validate_json(cast(str, it[MCP_SERVER_COLUMN])) for it in items]
        return servers

    def get_mcp_server(self, name: str) -> McpServer | None:
        """Retrieves a specific MCP server from DynamoDB by name.

        Args:
            name: The name of the MCP server.

        Returns:
            The McpServer instance, or None if not found.
        """
        response = self.table.get_item(Key={"id": name})
        item: dict[str, str | set[str]] = response.get("Item")
        if item:
            return McpServer.model_validate_json(item.get(MCP_SERVER_COLUMN))
        return None

    def put_mcp_server(self, server: McpServer, allowed_agents: set[str] = None) -> None:
        """Registers or updates an MCP server in DynamoDB.

         Args:
             server: The McpServer instance to register.
             allowed_agents: Optional set of agent names allowed to access this server.
         """
        item: dict[str, Any] = {
            "id": server.name,
            MCP_SERVER_COLUMN: server.model_dump_json(),
        }
        if allowed_agents:
            item[ALLOWED_AGENTS_FIELD] = set(allowed_agents)

        self.table.put_item(Item=item)

    def get_allowed_agents(self, server_name: str) -> set[str]:
        """Retrieves the set of agent names allowed to access a specific MCP server from DynamoDB.

        Args:
            server_name: The name of the MCP server.

        Returns:
            A set of allowed agent names.
        """
        response = self.table.get_item(Key={"id": server_name})
        item: dict[str, Any] | None = response.get("Item")
        logging.info(item)
        if item and ALLOWED_AGENTS_FIELD in item:
            return item.get(ALLOWED_AGENTS_FIELD)
        return set()

    ## TODO Cross check if agent exists
    def enable_mcp_server_for_agent(self, server_name: str, agent_name: str) -> None:
        """Grants an agent access to a specific MCP server in DynamoDB.

        Args:
            server_name: The name of the MCP server.
            agent_name: The name of the agent to authorize.

        Raises:
            Exception: If the MCP server is not found.
        """
        response = self.table.get_item(Key={"id": server_name})
        item: dict[str, str | set[str]] = response.get("Item")
        if not item:
            raise Exception(f"MCP server '{server_name}' not found")

        server = McpServer.model_validate_json(item.get(MCP_SERVER_COLUMN))
        allowed_agents_raw = item.get(ALLOWED_AGENTS_FIELD)
        allowed_agents: set[str] = allowed_agents_raw if allowed_agents_raw is not None else set()

        if agent_name not in allowed_agents:
            allowed_agents.add(agent_name)
            logging.info(f"put mcp server {server_name} with allowed agents {allowed_agents}")
            self.put_mcp_server(server=server, allowed_agents=allowed_agents)

    def disable_mcp_server_for_agent(self, server_name: str, agent_name: str) -> None:
        """Revokes an agent's access to a specific MCP server in DynamoDB.

        Args:
            server_name: The name of the MCP server.
            agent_name: The name of the agent to deauthorize.

        Raises:
            Exception: If the MCP server is not found.
        """
        response = self.table.get_item(Key={"id": server_name})
        item: dict[str, str | set[str]] = response.get("Item")
        if not item:
            raise Exception(f"MCP server '{server_name}' not found")

        server = McpServer.model_validate_json(item.get(MCP_SERVER_COLUMN))
        allowed_agents_raw = item.get(ALLOWED_AGENTS_FIELD)
        allowed_agents: set[str] | None = allowed_agents_raw
        if allowed_agents and agent_name in allowed_agents:
            allowed_agents.remove(agent_name)
            self.put_mcp_server(server=server, allowed_agents=allowed_agents)

    def get_mcp_server_for_agent(self, agent_name: str) -> list[McpServer]:
        """Retrieves all MCP servers that a specific agent is authorized to access from DynamoDB.

        Args:
            agent_name: The name of the agent.

        Returns:
            A list of McpServer instances.
        """
        response = self.table.scan(
            FilterExpression=Attr(ALLOWED_AGENTS_FIELD).contains(agent_name)
        )
        items = response.get("Items", [])
        return [McpServer.model_validate_json(cast(str, item.get(MCP_SERVER_COLUMN))) for item in items]
