"""Bootstrap logic for the registry server FastAPI application."""
import json
from typing import Any, cast
from fastapi import FastAPI, APIRouter, HTTPException

from .storage import AgentRegistryLookup, McpRegistryLookup
from .dynamo_db import DynamoDbAgentRegistryLookup, DynamoDbMcpRegistryLookup
from .model import McpServer

def load_registry(agent_registry: AgentRegistryLookup, mcp_registry: McpRegistryLookup) -> FastAPI:
    """Bootstraps the registry server FastAPI application.

    Args:
        agent_registry: The agent registry storage implementation.
        mcp_registry: The MCP registry storage implementation.

    Returns:
        A configured FastAPI application instance.
    """
    app = FastAPI()

    # Agent Registry Endpoints
    agent_router = APIRouter()

    @agent_router.put("/agent-card/{name}")
    def put_agent_card(name: str, agent_card: dict[str, Any], expire_at: str) -> None:
        """Endpoint to register or update an agent card."""
        agent_registry.put_agent_card(name=name, card=json.dumps(agent_card), expire_at=expire_at)

    @agent_router.get("/agent-card/{name}")
    def get_agent_card(name: str) -> dict[str, Any]:
        card_str = agent_registry.get_agent_card(name=name)
        """Endpoint to retrieve a specific agent card."""

        if card_str:
            return cast(dict[str, Any], json.loads(card_str))
        raise HTTPException(status_code=404, detail="Agent card not found")

    @agent_router.get("/agent-cards")
    def get_agent_cards() -> list[dict[str, Any]]:
        """Endpoint to retrieve all agent cards."""
        return agent_registry.get_agent_cards()

    @agent_router.patch("/agent-card/{name}/heartbeat")
    def patch_agent_heartbeat(name: str, expire_at: str) -> None:
        """Endpoint to update the heartbeat/expiration for an agent."""
        agent_registry.update_agent_expiry(name=name, expire_at=expire_at)

    # MCP Registry Endpoints
    mcp_router = APIRouter()

    @mcp_router.put("/mcp/server")
    def put_mcp_server(server: McpServer) -> None:
        """Endpoint to register or update an MCP server."""
        mcp_registry.put_mcp_server(server=server)

    @mcp_router.get("/mcp/server/{name}")
    def get_mcp_server(name: str) -> McpServer:
        """Endpoint to retrieve a specific MCP server."""
        server = mcp_registry.get_mcp_server(name=name)
        if server:
            return server
        raise HTTPException(status_code=404, detail="MCP Server not found")

    @mcp_router.get("/mcp/servers")
    def get_mcp_servers() -> list[McpServer]:
        """Endpoint to retrieve all MCP servers."""
        return mcp_registry.get_mcp_servers()

    @mcp_router.put("/mcp/{name}/agent/{agent_name}")
    def enable_mcp_server_for_agent(name: str, agent_name: str) -> None:
        """Endpoint to authorize an agent for an MCP server."""
        try:
            mcp_registry.enable_mcp_server_for_agent(server_name=name, agent_name=agent_name)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @mcp_router.delete("/mcp/{name}/agent/{agent_name}")
    def disable_mcp_server_for_agent(name: str, agent_name: str) -> None:
        """Endpoint to deauthorize an agent for an MCP server."""
        try:
            mcp_registry.disable_mcp_server_for_agent(server_name=name, agent_name=agent_name)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @mcp_router.get("/mcp/{name}/agent")
    def get_allowed_agents(name: str) -> set[str]:
        """Endpoint to retrieve all authorized agents for an MCP server."""
        return mcp_registry.get_allowed_agents(server_name=name)

    @mcp_router.get("/mcp/agent/{agent_name}/servers")
    def get_mcp_server_for_agent(agent_name: str) -> list[McpServer]:
        """Endpoint to retrieve all MCP servers authorized for a specific agent."""
        return mcp_registry.get_mcp_server_for_agent(agent_name=agent_name)

    app.include_router(agent_router)
    app.include_router(mcp_router)

    @app.get("/health")
    def health_check() -> dict[str, Any]:
        """Health check endpoint."""
        return {"status": "OK"}

    return app
