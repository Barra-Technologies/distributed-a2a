import json
import logging
import os
from logging import Logger
from typing import Optional, Any, Sequence, Callable

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskStatusUpdateEvent, TaskStatus, TaskState, TaskArtifactUpdateEvent, Artifact, AgentCard
from a2a.utils import new_text_artifact
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver

from .agent import StatusAgent, RoutingResponse, StringResponse
from .config import settings
from .model import AgentConfig, RouterConfig
from .registry import McpRegistryLookup, AgentRegistryLookup

logger: Logger = logging.getLogger(__name__)

ROUTING_SYSTEM_PROMPT = """
# Role: Multi-agent Router

You are a helpful routing assistant which routes user requests to specialized remote agents in  a multi-agent setup.

## Core capability & Task
Your main task is to:
1. look up available agents via the provided registry tool
2. select the best matching agent for the user query.
3. return the agent_name for the selected agent.

## Rules
- Return only the agent_name as a string.
- If the user query is relevant to multiple agents, return the agent_name of the agent with the highest match.
- If the user query is not relevant to any agent, return the agent_name of the generic agent.
- If no generic agent is available, return an error message.
"""


class RoutingAgentExecutor(AgentExecutor):

    def __init__(self, agent_config: AgentConfig, agent_registry: AgentRegistryLookup, tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
                 routing_checkpointer: Optional[BaseCheckpointSaver[Any]] = None,
                 specialized_checkpointer: Optional[BaseCheckpointSaver[Any]] = None):
        super().__init__()
        api_key = settings.get_env_var(agent_config.agent.llm.api_key_env)
        if api_key is None:
            raise ValueError("No API key found for LLM.")

        self.auth_headers = settings.registry_auth_headers

        if not self.auth_headers.get("x-api-key"):
            logger.warning("No A2A API key found for registry communication")

        self.mcp_registry = McpRegistryLookup(
            registry_url=agent_config.agent.registry.mcp.url if agent_config.agent.registry and agent_config.agent.registry.mcp else "",
            req_opts={
                **settings.registry_auth_headers,
                "Accept": "application/json"
            })
        self.agent_config = agent_config
        self.registered_tools: dict[str, Any] = {}
        self.api_key = api_key
        agent_tools = [] if tools is None else tools
        self.llm_tools = [{ llm_tool: {} } for llm_tool in agent_config.agent.llm_tools] if agent_config.agent.llm_tools else []

        agent_tools.append(self.llm_tools)
        logger.info(f"Agent tools: {agent_tools}")


        self.agent = StatusAgent[StringResponse](
            llm_config=agent_config.agent.llm,
            system_prompt=agent_config.agent.system_prompt,
            name=agent_config.agent.card.name,
            api_key=api_key,
            is_routing=False,
            tools=agent_tools,
            checkpointer=specialized_checkpointer
        )
        self.agent_registry = agent_registry
        self.routing_agent = StatusAgent[RoutingResponse](
            llm_config=agent_config.agent.llm,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            name="Router",
            api_key=api_key,
            is_routing=True,
            tools=[agent_registry.as_tool()],
            checkpointer=routing_checkpointer

        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.context_id is None or context.task_id is None:
            raise ValueError("Context ID and Task ID must be provided.")

        # set status to processing
        await event_queue.enqueue_event(TaskStatusUpdateEvent(status=TaskStatus(state=TaskState.working),
                                                              final=False,
                                                              context_id=context.context_id,
                                                              task_id=context.task_id))
        await self.reinitialize_agent_with_tools()
        agent_response: StringResponse = await self.agent(message=context.get_user_input(),
                                                          context_id=context.context_id)

        artifact: Artifact
        if agent_response.status == TaskState.rejected:
            artifact = await _route_request_to_matching_agent(self.routing_agent, self.agent_registry, context)
        else:
            logger.info(f"Request with id {context.context_id} was successfully processed by agent.")
            artifact = new_text_artifact(name='current_result', description='Result of request to agent.',
                                         text=f"*{self.agent_config.agent.card.name}*: {agent_response.response}")

        # publish actual result
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(append=False,
                                                                context_id=context.context_id,
                                                                task_id=context.task_id,
                                                                last_chunk=True,
                                                                artifact=artifact))
        # set and publish the final status
        await event_queue.enqueue_event(TaskStatusUpdateEvent(status=TaskStatus(
            state=TaskState(agent_response.status)),
            final=True,
            context_id=context.context_id,
            task_id=context.task_id))

    async def reinitialize_agent_with_tools(self) -> None:
        mcp_server_raw = self.mcp_registry.get_mcp_tool_for_agent(self.agent_config.agent.card.name)
        if not mcp_server_raw:
            # no mcp tool found no need to reinitialize Agent
            return

        logger.info(f"Agent {self.agent_config.agent.card.name} has access to the following tools: {mcp_server_raw}")
        mcp_servers = {tool["name"]: {"url": tool["url"], "transport": tool["protocol"],
                                "headers": settings.get_mcp_auth_headers(tool["name"])} for tool in mcp_server_raw}
        mcp_client = MultiServerMCPClient(mcp_servers)  # type: ignore[arg-type]
        agent_tools: list[BaseTool | dict[str, Any]] = await mcp_client.get_tools()
        agent_tools.append(self.llm_tools)

        self.agent = StatusAgent[StringResponse](
            llm_config=self.agent_config.agent.llm,
            system_prompt=self.agent_config.agent.system_prompt,
            name=self.agent_config.agent.card.name,
            api_key=self.api_key,
            is_routing=False,
            tools=agent_tools,
        )


class RoutingExecutor(AgentExecutor):
    def __init__(self, router_config: RouterConfig, agent_registry: AgentRegistryLookup) -> None:
        super().__init__()
        api_key = settings.get_env_var(router_config.router.llm.api_key_env)
        if api_key is None:
            raise ValueError("No API key found for LLM.")

        self.agent_registry = agent_registry

        self.routing_agent = StatusAgent[RoutingResponse](
            llm_config=router_config.router.llm,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            name=router_config.router.card.name,
            api_key=api_key,
            is_routing=True,
            tools=[agent_registry.as_tool() ]
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.context_id is None or context.task_id is None:
            raise ValueError("Context ID and Task ID must be provided.")

        await event_queue.enqueue_event(TaskStatusUpdateEvent(status=TaskStatus(state=TaskState.working),
                                                              final=False,
                                                              context_id=context.context_id,
                                                              task_id=context.task_id))

        artifact = await _route_request_to_matching_agent(self.routing_agent, self.agent_registry, context)

        # publish actual result
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(append=False,
                                                                context_id=context.context_id,
                                                                task_id=context.task_id,
                                                                last_chunk=True,
                                                                artifact=artifact))
        # set and publish the final status
        await event_queue.enqueue_event(TaskStatusUpdateEvent(status=TaskStatus(
            state=TaskState.completed),
            final=True,
            context_id=context.context_id,
            task_id=context.task_id))


async def _route_request_to_matching_agent(routing_agent: StatusAgent[RoutingResponse], agent_registry: AgentRegistryLookup, context):
    routing_agent_response: RoutingResponse = await routing_agent(message=context.get_user_input(),
                                                                  context_id=context.context_id)
    agent_name: str = routing_agent_response.agent_name
    if agent_name is None:
        logger.error(f"Failed to identify agent name")
        raise
    logger.info(f"Request with id {context.context_id} got rejected and will be rerouted to a '{agent_name}'.")
    agent_card: dict[str, Any] = agent_registry.get_agent_card(agent_name)
    if agent_card is None:
        logger.error(f"agent not found  for name: {agent_name}")
        raise
    logger.info(f"Routing agent response for request with id {context.context_id}: {agent_card}")
    artifact = new_text_artifact(name='target_agent', description='New target agent for request.',
                                 text=json.dumps(agent_card))
    return artifact
