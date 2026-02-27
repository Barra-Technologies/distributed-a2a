"""Data models for the registry server."""
from pydantic import BaseModel, Field


class McpServer(BaseModel):
    """Data model for an MCP server definition."""
    name: str = Field(description="Name of the tool")
    url: str = Field(description="URL of the tool")
    protocol: str = Field(description="Protocol used by the tool e.g. http, https")
    description: str = Field(description="Description of the tool")
