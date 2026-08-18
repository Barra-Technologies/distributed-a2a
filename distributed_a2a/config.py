import json
import os
from typing import Dict


class Settings:
    """Central configuration for environment variables."""

    @property
    def api_root_path(self) -> str | None:
        return os.getenv("API_ROOT_PATH")

    @property
    def httpx_logging(self) -> bool:
        return os.getenv("HTTPX_LOGGING", "false").lower() == "true"

    @property
    def registry_auth_headers(self) -> Dict[str, str]:
        return _parse_json_env("REGISTRY_AUTH_HEADERS")

    def get_mcp_auth_headers(self, service_name: str) -> Dict[str, str]:
        env_var_name = f"MCP_AUTH_HEADER_{service_name.upper().replace('-', '_')}"
        headers = _parse_json_env(env_var_name)
        return headers or _parse_json_env("MCP_AUTH_HEADER")

    def get_env_var(self, name: str, default: str | None = None) -> str | None:
        return os.getenv(name, default)

    @property
    def context_edit_trigger_tokens(self) -> int:
        """Token threshold that triggers ``ClearToolUsesEdit``.

        Set well below the ``langchain`` default of 100_000 so that older
        tool outputs are dropped from the message history before the
        LangGraph checkpoint reaches DynamoDB's 400 KB item-size limit.
        Override with ``CONTEXT_EDIT_TRIGGER_TOKENS``.
        """
        return int(os.getenv("CONTEXT_EDIT_TRIGGER_TOKENS", "30000"))

    @property
    def context_edit_keep_tool_uses(self) -> int:
        """How many recent tool results ``ClearToolUsesEdit`` preserves.

        Override with ``CONTEXT_EDIT_KEEP_TOOL_USES``.
        """
        return int(os.getenv("CONTEXT_EDIT_KEEP_TOOL_USES", "3"))


def _parse_json_env(name: str) -> Dict[str, str]:
    raw = os.getenv(name)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


settings = Settings()
