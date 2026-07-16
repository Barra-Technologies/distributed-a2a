import logging
from typing import Any, ClassVar, Literal, cast

from a2a.types import TaskState
from langchain.agents import create_agent
from langchain.agents.middleware import (ClearToolUsesEdit,
                                         ContextEditingMiddleware)
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from .config import settings
from .model import LLMConfig, get_model


class AgentResponse(BaseModel):
    status: Literal[
        TaskState.rejected,
        TaskState.completed,
        TaskState.failed,
        TaskState.input_required,
    ] = Field(
        description=(
            f'You should select status as {TaskState.rejected} for requests that fall outside your area of expertise.'
            f'You should select status as {TaskState.completed} if the request is fully addressed and no further input is needed. '
            f'You should select status as {TaskState.input_required} if you need more information from the user or are asking a clarifying question. '
            f'You should select status as {TaskState.failed} if an error occurred or the request cannot be fulfilled.'
        )
    )


class RoutingResponse(AgentResponse):
    agent_name: str | None = Field(default=None, description="The agent_name of the agent to be routed to")
    message: str | None = Field(default=None, description="The answer, answered by the routing agent")


class StringResponse(AgentResponse):
    response: str = Field(description="The main response to be returned to the user")


class AgentInvocation[ResponseT: AgentResponse](BaseModel):
    """Structured response plus the raw message list.

    ``messages`` lets callers extract ``ToolMessage.artifact`` payloads (e.g.
    binary files from MCP tools) without routing them through the LLM context.
    Artifacts already delivered on a previous turn are flagged in-place on
    the ``ToolMessage.artifact`` dict (see `.file_extractors`) so they
    are skipped by downstream extraction and are not re-sent to the client.
    """

    structured: ResponseT
    messages: list[BaseMessage]
    model_config = {"arbitrary_types_allowed": True}


class StatusAgent[ResponseT: AgentResponse]:
    RESPONSE_FORMAT: ClassVar[type[AgentResponse]]

    def __init__(self,
                 llm_config: LLMConfig,
                 name: str,
                 system_prompt: str,
                 api_key: str,
                 tools: list[BaseTool],
                 checkpointer: BaseCheckpointSaver[Any] | None = None):

        saver = checkpointer
        if saver is None:
            try:
                saver = MemorySaver()
            except Exception as e:
                logging.warning(f"Failed to initialize MemorySaver: {e}. Falling back to no checkpointer.")
                saver = None

        middleware = [
            ContextEditingMiddleware(edits=[
                ClearToolUsesEdit(
                    trigger=settings.context_edit_trigger_tokens,
                    keep=settings.context_edit_keep_tool_uses,
                ),
            ]),
        ]

        self.agent = create_agent(
            get_model(api_key=api_key,
                      model=llm_config.model,
                      base_url=llm_config.base_url,
                      reasoning_effort=llm_config.reasoning_effort),
            tools=tools,
            middleware=middleware,
            checkpointer=saver,
            system_prompt=system_prompt,
            response_format=self.RESPONSE_FORMAT,
            name=name
        )

    async def __call__(self,
                       message: str,
                       context_id: str | None = None) -> AgentInvocation[ResponseT]:
        config: RunnableConfig = RunnableConfig(
            configurable={'thread_id': context_id}
        )
        response = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config
        )
        logging.info("agent response: %s", response)
        return AgentInvocation[ResponseT](
            structured=cast(ResponseT, response['structured_response']),
            messages=list(response.get('messages', [])),
        )

    async def persist_delivered_messages(self,
                                         context_id: str | None,
                                         messages: list[BaseMessage]) -> None:
        """Write ``messages`` back into the checkpoint so the delivered
        flag set by :func:`.file_extractors.extract_file_parts` survives to
        the next turn and prevents artifact re-delivery.
        """
        if not messages or context_id is None:
            return
        config: RunnableConfig = RunnableConfig(
            configurable={'thread_id': context_id}
        )
        try:
            await self.agent.aupdate_state(config, {"messages": messages})
        except Exception:
            logging.warning(
                "Failed to persist delivered-artifact flag for context %s",
                context_id,
                exc_info=True,
            )


class RoutingAgent(StatusAgent[RoutingResponse]):
    RESPONSE_FORMAT: ClassVar[type[RoutingResponse]] = RoutingResponse


class SpecializedAgent(StatusAgent[StringResponse]):
    RESPONSE_FORMAT: ClassVar[type[StringResponse]] = StringResponse
