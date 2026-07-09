from typing import Any, List

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

FILE_PROMPT_SCHEME = "file://"


class SkillConfig(BaseModel):
    id: str = Field(description="The id of the skill e.g. weather")
    name: str = Field(description="The name of the skill e.g. weather")
    description: str = Field(description="A short description of the skill")
    tags: List[str] = Field(description="The tags associated with the skill")
    examples: List[str] = Field(description="Examples of how to use the skill", default_factory=list)


class RegistryItemConfig(BaseModel):
    url: str = Field(description="The url of the registry")


class RegistryConfig(BaseModel):
    agent: RegistryItemConfig = Field(description="The agent registry configuration")
    mcp: RegistryItemConfig | None = Field(description="The mcp registry configuration", default=None)


class LLMConfig(BaseModel):
    base_url: str = Field(description="The base url of the LLM provider")
    model: str = Field(description="The model to use for the LLM e.g. gpt-3.5-turbo")
    api_key_env: str = Field(description="The environment variable containing the api key for the LLM provider")
    reasoning_effort: str | None = Field(
        description="The reasoning effort to use for the LLM e.g. high. Only set this for reasoning-capable models; leave unset for models like gpt-4 / gpt-3.5 that reject the parameter.",
        default=None,
    )


class CardConfig(BaseModel):
    name: str = Field(description="The name of the agent" )
    description: str = Field(description="A short description of the agent")
    version: str = Field(description="The version of the agent")
    default_input_modes: List[str] = Field(description="The default input modes supported by the agent", default_factory=lambda: ["text", "text/plaintext"])
    default_output_modes: List[str] = Field(description="The default output modes supported by the agent", default_factory=lambda: ["text", "text/plaintext"])
    preferred_transport_protocol: str = Field(description="The preferred transport protocol for the agent", default="HTTP+JSON")
    url: str = Field(description="The url of the agent")
    skills: List[SkillConfig] = Field(description="The skills supported by the agent", default_factory=list)


class AgentItem(BaseModel):
    registry: RegistryConfig | None = Field(description="The registry configuration node", default=None)
    card: CardConfig = Field(description="The agent card configuration node")
    llm: LLMConfig = Field(description="The LLM configuration node")
    system_prompt: str = Field(
        description=(
            "The system prompt to use for the LLM. To load it from a file "
            "instead, prefix the path with the explicit `file://` scheme "
            "(e.g. `file:///etc/prompts/agent.md` or `file://./prompt.md`)."
        )
    )
    advertise_routing_skill: bool = Field(
        default=False,
        description=(
            "When True, the agent advertises a `routing` skill on its "
            "AgentCard in addition to the configured skills. Off by default: "
            "specialized agents should advertise only the capabilities they "
            "actually provide, and the implicit reroute-on-rejection happens "
            "inside the executor regardless of this flag."
        ),
    )

    def __init__(self, /, **data: Any) -> None:
        prompt = data.get('system_prompt')
        if isinstance(prompt, str) and prompt.startswith(FILE_PROMPT_SCHEME):
            path = prompt[len(FILE_PROMPT_SCHEME):]
            with open(path, "r", encoding="utf-8") as f:
                data['system_prompt'] = f.read()

        super().__init__(**data)


class AgentConfig(BaseModel):
    agent: AgentItem = Field(description="The agent configuration node")


class RouterItem(BaseModel):
    registry: RegistryConfig | None = Field(description="The registry configuration node", default=None)
    card: CardConfig = Field(description="The router card configuration node")
    llm: LLMConfig = Field(description="The LLM configuration node")


class RouterConfig(BaseModel):
    router: RouterItem = Field(description="The router configuration node")


def get_model(
        api_key: str,
        model: str,
        base_url: str,
        reasoning_effort: str | None = None) -> BaseChatModel:
    return ChatOpenAI(
        model_name=model,
        openai_api_base=base_url,
        openai_api_key=SecretStr(api_key),
        reasoning_effort=reasoning_effort,
    )
