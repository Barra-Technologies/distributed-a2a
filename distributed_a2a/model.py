import os
from typing import List, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ConfigDict

class SkillConfig(BaseModel):
    id: str = Field(description="The id of the skill e.g. weather")
    name: str = Field(description="The name of the skill e.g. weather")
    description: str = Field(description="A short description of the skill")
    tags: List[str] = Field(description="The tags associated with the skill")


class RegistryItemConfig(BaseModel):
    url: str = Field(description="The url of the registry")


class RegistryConfig(BaseModel):
    agent: RegistryItemConfig = Field(description="The agent registry configuration")
    mcp: Optional[RegistryItemConfig] = Field(description="The mcp registry configuration", default=None)


class LLMConfig(BaseModel):
    base_url: str = Field(description="The base url of the LLM provider")
    model: str = Field(description="The model to use for the LLM e.g. gpt-3.5-turbo")
    api_key_env: str = Field(description="The environment variable containing the api key for the LLM provider")
    reasoning_effort: str  = Field(description="The reasoning effort to use for the LLM e.g. high", default="high")


class CardConfig(BaseModel):
    name: str = Field(description="The name of the agent" )
    description: str = Field(description="A short description of the agent")
    version: str = Field(description="The version of the agent")
    default_input_modes: List[str]  = Field(description="The default input modes supported by the agent", default=["text","text/plaintext"])
    default_output_modes: List[str] = Field(description="The default output modes supported by the agent", default=["text","text/plaintext"])
    preferred_transport_protocol: str = Field(description="The preferred transport protocol for the agent", default="HTTP+JSON")
    url: str = Field(description="The url of the agent")
    skills: List[SkillConfig] = Field(description="The skills supported by the agent", default=[])


class AgentItem(BaseModel):
    registry: Optional[RegistryConfig] = Field(description="The registry configuration node", default=None)
    card: CardConfig = Field(description="The agent card configuration node")
    llm: LLMConfig = Field(description="The LLM configuration node")
    system_prompt: str = Field(description="The system prompt to use for the LLM or a path to a file containing the system prompt")

    def __init__(self, /, **data: Any) -> None:
        prompt_or_path= data['system_prompt']
        if os.path.exists(prompt_or_path):
            with open(prompt_or_path, "r", encoding="utf-8") as f:
                data['system_prompt'] =  f.read()

        super().__init__(**data)


class AgentConfig(BaseModel):
    agent: AgentItem = Field(description="The agent configuration node")


class RouterItem(BaseModel):
    registry: Optional[RegistryConfig] = Field(description="The registry configuration node", default=None)
    card: CardConfig = Field(description="The router card configuration node")
    llm: LLMConfig = Field(description="The LLM configuration node")


class RouterConfig(BaseModel):
    router: RouterItem = Field(description="The router configuration node")

def get_model(api_key: str, model: str, base_url: str, reasoning_effort: str) -> BaseChatModel:
    return ChatOpenAI(
        model_name=model,
        openai_api_base=base_url,
        openai_api_key=lambda: api_key,
        reasoning_effort=reasoning_effort
    )
