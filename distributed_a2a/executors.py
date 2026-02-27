import json
import logging
import os
from logging import Logger
from typing import Optional, Any

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
from .registry import McpRegistryLookup

logger: Logger = logging.getLogger(__name__)

ROUTING_SYSTEM_PROMPT = """
You are a helpful routing assistant which routes user requests to specialized remote agents. Your main task is to:
1. look up available agents via their A2A agent cards
2. select the best matching agent for the user query, in case no agent matches use the generic agent.
3. return the matching agent card for that agent.
"""


class RoutingAgentExecutor(AgentExecutor):

    def __init__(self, agent_config: AgentConfig, routing_tool: BaseTool, tools: list[BaseTool] | None = None,
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
        self.agent = StatusAgent[StringResponse](
            llm_config=agent_config.agent.llm,
            system_prompt=agent_config.agent.system_prompt,
            name=agent_config.agent.card.name,
            api_key=api_key,
            is_routing=False,
            tools=[] if tools is None else tools,
            checkpointer=specialized_checkpointer
        )
        self.routing_agent = StatusAgent[RoutingResponse](
            llm_config=agent_config.agent.llm,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            name="Router",
            api_key=api_key,
            is_routing=True,
            tools=[routing_tool],
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
            routing_agent_response: RoutingResponse = await self.routing_agent(message=context.get_user_input(),
                                                                               context_id=context.context_id)
            agent_card = routing_agent_response.agent_card
            if isinstance(agent_card, str):
                try:
                    logger.info(f"Routing agent response for request with id {context.context_id}: {agent_card}")
                    agent_card_dict = json.loads(agent_card)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse agent_card as JSON: {agent_card}")
                    raise
            else:
                logger.info(f"Routing agent response for request with id {context.context_id}: {agent_card}")
                agent_card_dict = agent_card

            agent_name: str = agent_card_dict["name"]
            logger.info(f"Request with id {context.context_id} got rejected and will be rerouted to a '{agent_name}'.",
                        extra={"card": routing_agent_response.agent_card})
            artifact = new_text_artifact(name='target_agent', description='New target agent for request.',
                                         text=json.dumps(agent_card_dict) if isinstance(agent_card_dict, dict) else str(
                                             agent_card))
        else:
            logger.info(f"Request with id {context.context_id} was successfully processed by agent.")
            artifact = new_text_artifact(name='current_result', description='Result of request to agent.',
                                         text=agent_response.response)

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
        tools = {tool["name"]: {"url": tool["url"], "transport": tool["protocol"],
                                "headers": settings.get_mcp_auth_headers(tool["name"])} for tool in mcp_server_raw}
        mcp_client = MultiServerMCPClient(tools)  # type: ignore[arg-type]
        mcp_tools = await mcp_client.get_tools()

        self.agent = StatusAgent[StringResponse](
            llm_config=self.agent_config.agent.llm,
            system_prompt=self.agent_config.agent.system_prompt,
            name=self.agent_config.agent.card.name,
            api_key=self.api_key,
            is_routing=False,
            tools=mcp_tools,
        )


class RoutingExecutor(AgentExecutor):
    def __init__(self, router_config: RouterConfig, routing_tool: BaseTool) -> None:
        super().__init__()
        api_key = settings.get_env_var(router_config.router.llm.api_key_env)
        if api_key is None:
            raise ValueError("No API key found for LLM.")

        self.routing_agent = StatusAgent[RoutingResponse](
            llm_config=router_config.router.llm,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            name=router_config.router.card.name,
            api_key=api_key,
            is_routing=True,
            tools=[routing_tool]
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

        agent_response: RoutingResponse = await self.routing_agent(message=context.get_user_input(),
                                                                   context_id=context.context_id)
        logger.info(f"Routing agent response for request with id {context.context_id}: {agent_response}")
        agent_card = agent_response.agent_card
        error_artifact: Artifact | None = None
        if isinstance(agent_card, str):
            try:
                agent_card_dict = json.loads(agent_card)
                logger.info(f"Routing agent response for request with id {context.context_id}: {agent_card_dict}")
            except json.JSONDecodeError:
                # If it's not JSON, maybe it's already the name or something else?
                # But AgentCard expects a dict if we use model_validate
                logger.warning(f"Failed to parse agent_card as JSON: {agent_card}")
                artifact_text = agent_card if agent_card and isinstance(agent_card,
                                                                        str) else "No agent found to handle the request."
                error_artifact = new_text_artifact(name='current_result',
                                                   description='Error message from routing agent.',
                                                   text=artifact_text)

        else:
            agent_card_dict = agent_card

        if error_artifact:
            artifact = error_artifact
        else:
            artifact = new_text_artifact(name='target_agent', description='New target agent for request.',
                                         text=json.dumps(agent_card_dict) if isinstance(agent_card_dict, dict) else str(
                                             agent_card))

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
