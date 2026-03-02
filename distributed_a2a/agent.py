import logging
from typing import Any, Literal, Optional, Callable, Sequence

from a2a.types import TaskState
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from .model import get_model, LLMConfig


class AgentResponse(BaseModel):
    status: Literal[TaskState.rejected, TaskState.completed, TaskState.failed] = Field(
        description=(
            f'You should select status as {TaskState.rejected} for requests that fall outside your area of expertise.'
            f'You should select status as {TaskState.completed} if the request is fully addressed and no further input is needed. '
            f'You should select status as {TaskState.input_required} if you need more information from the user or are asking a clarifying question. '
            f'You should select status as {TaskState.failed} if an error occurred or the request cannot be fulfilled.'
        )
    )


class RoutingResponse(AgentResponse):
    agent_name: str = Field(description="The agent_name of the agent to be returned to the user")


class StringResponse(AgentResponse):
    response: str = Field(description="The main response to be returned to the user")


class StatusAgent[ResponseT: AgentResponse]:

    def __init__(self, llm_config: LLMConfig, name: str, system_prompt: str, api_key: str, is_routing: bool,
                 tools: Sequence[BaseTool | Callable | dict[str, Any]],
                 checkpointer: Optional[BaseCheckpointSaver[Any]] = None):
        response_format: type[AgentResponse]
        if is_routing:
            response_format = RoutingResponse
        else:
            response_format = StringResponse

        saver = checkpointer
        if saver is None:
            try:
                saver = MemorySaver()
            except Exception as e:
                logging.warning(f"Failed to initialize DynamoDBSaver: {e}. Falling back to no checkpointer.")
                saver = None

        self.agent = create_agent(
            get_model(api_key=api_key,
                      model=llm_config.model,
                      base_url=llm_config.base_url,
                      reasoning_effort=llm_config.reasoning_effort),
            tools=tools,
            checkpointer=saver,
            system_prompt=system_prompt,
            response_format=response_format,
            name=name
        )

    async def __call__(self, message: str, context_id: Optional[str] = None) -> ResponseT:
        config: RunnableConfig = RunnableConfig(configurable={'thread_id': context_id})
        response = await self.agent.ainvoke(LangGraphMessage(message), config)  # type: ignore[arg-type]
        logging.info("agent response: %s", response)
        return response['structured_response']  # type: ignore[no-any-return]


class LangGraphMessage(BaseModel):
    messages: list[tuple[Literal['user'], str]]

    def __init__(self, messages: str):
        super().__init__(messages=[("user", messages)])
