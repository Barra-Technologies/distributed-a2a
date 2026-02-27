import json
import os
from typing import Optional, Dict, Any

class Settings:
    """Central configuration for environment variables."""

    @property
    def api_root_path(self) -> Optional[str]:
        return os.getenv("API_ROOT_PATH")

    @property
    def httpx_logging(self) -> bool:
        return os.getenv("HTTPX_LOGGING", "false").lower() == "true"

    @property
    def registry_auth_headers(self) -> Dict[str, str]:
        auth_headers_str = os.getenv("REGISTRY_AUTH_HEADERS")
        headers = {}
        if auth_headers_str:
            try:
                headers = json.loads(auth_headers_str)
            except json.JSONDecodeError:
                headers = {}

        return headers

    @property
    def mcp_auth_headers(self) -> Dict[str, str]:
        mcp_auth_headers_str = os.getenv("MCP_AUTH_HEADER")
        headers = {}
        if mcp_auth_headers_str:
            try:
                headers = json.loads(mcp_auth_headers_str)
            except json.JSONDecodeError:
                headers = {}

        return headers

    def get_mcp_auth_headers(self, service_name: str) -> Dict[str, str]:
        env_var_name = f"MCP_AUTH_HEADER_{service_name.upper().replace('-', '_')}"
        mcp_auth_headers_str = os.getenv(env_var_name)
        if not mcp_auth_headers_str:
            mcp_auth_headers_str = os.getenv("MCP_AUTH_HEADER")

        headers = {}
        if mcp_auth_headers_str:
            try:
                headers = json.loads(mcp_auth_headers_str)
            except json.JSONDecodeError:
                headers = {}

        return headers

    def get_env_var(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(name, default)

settings = Settings()
