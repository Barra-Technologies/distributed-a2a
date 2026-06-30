import json
import logging
from typing import Any, cast

from .model import McpServer
from .storage import AgentRegistryLookup, McpRegistryLookup

MCP_SERVER_COLUMN = "server"
ALLOWED_AGENTS_FIELD = "allowed-agents"


def _dynamo_resource(region_name: str = "eu-central-1") -> Any:
    """Lazily import ``boto3`` so consumers that never use DynamoDB don't pay
    the import cost (boto3 pulls in botocore which is heavy).
    """
    import boto3  # noqa: WPS433 - intentional local import
    return boto3.resource("dynamodb", region_name=region_name)


class DynamoDbAgentRegistryLookup(AgentRegistryLookup):
    """DynamoDB-backed storage for agent registry."""
    def __init__(self, agent_card_table: str) -> None:
        """Initializes the DynamoDB agent registry lookup.

          Args:
              agent_card_table: The name of the DynamoDB table for agent cards.
          """
        dynamo = _dynamo_resource()
        self.table = dynamo.Table(agent_card_table)

    def get_agent_cards(self) -> list[dict[str, Any]]:
        """Retrieves all registered agent cards from DynamoDB.

        Returns:
            A list of agent cards as dictionaries.
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

    def put_agent_card(self, name: str, card: str, expire_at: int) -> None:
        """Registers or updates an agent card in DynamoDB.

        Args:
            name: The name of the agent.
            card: The agent card (JSON string).
            expire_at: Unix-epoch expiration timestamp (seconds). Stored as a
                Number so DynamoDB TTL can evict expired entries.
        """
        self.table.put_item(Item={"id": name, "card": card, "expireAt": expire_at})

    def update_agent_expiry(self, name: str, expire_at: int) -> None:
        """Updates the expiration timestamp for an agent registration in DynamoDB.

        Args:
            name: The name of the agent.
            expire_at: The new Unix-epoch expiration timestamp (seconds).
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
        dynamo = _dynamo_resource()
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
        item: dict[str, Any] | None = response.get("Item")
        if item:
            server_json = item.get(MCP_SERVER_COLUMN)
            if isinstance(server_json, str):
                return McpServer.model_validate_json(server_json)
        return None

    def put_mcp_server(self, server: McpServer, allowed_agents: set[str] | None = None) -> None:
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
            agents = item.get(ALLOWED_AGENTS_FIELD)
            if isinstance(agents, set):
                return cast(set[str], agents)
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
        item: dict[str, Any] | None = response.get("Item")
        if not item:
            raise Exception(f"MCP server '{server_name}' not found")

        server_json = item.get(MCP_SERVER_COLUMN)
        if not isinstance(server_json, str):
            raise Exception(f"Invalid server data for '{server_name}'")
        server = McpServer.model_validate_json(server_json)
        allowed_agents_raw = item.get(ALLOWED_AGENTS_FIELD)
        allowed_agents: set[str] = cast(set[str], allowed_agents_raw) if isinstance(allowed_agents_raw, set) else set()

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
        item: dict[str, Any] | None = response.get("Item")
        if not item:
            raise Exception(f"MCP server '{server_name}' not found")

        server_json = item.get(MCP_SERVER_COLUMN)
        if not isinstance(server_json, str):
            raise Exception(f"Invalid server data for '{server_name}'")
        server = McpServer.model_validate_json(server_json)
        allowed_agents_raw = item.get(ALLOWED_AGENTS_FIELD)
        allowed_agents: set[str] | None = cast(set[str], allowed_agents_raw) if isinstance(allowed_agents_raw, set) else None
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
        from boto3.dynamodb.conditions import \
            Attr  # local import: see _dynamo_resource
        response = self.table.scan(
            FilterExpression=Attr(ALLOWED_AGENTS_FIELD).contains(agent_name)
        )
        items = response.get("Items", [])
        return [McpServer.model_validate_json(cast(str, item.get(MCP_SERVER_COLUMN))) for item in items]
