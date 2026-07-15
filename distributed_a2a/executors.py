import json
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (Artifact, FilePart, Part, TaskArtifactUpdateEvent,
                       TaskState, TaskStatus, TaskStatusUpdateEvent)
from a2a.utils import new_text_artifact
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver

from .agent import (AgentInvocation, RoutingAgent, RoutingResponse,
                    SpecializedAgent, StringResponse)
from .config import settings
from .file_extractors import extract_file_parts
from .mcp_interceptors import hide_binary_content_from_llm
from .model import AgentConfig, RouterConfig
from .registry import AgentRegistryLookupClient, McpRegistryLookup

logger = logging.getLogger(__name__)


class RoutingFailed(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


ROUTING_SYSTEM_PROMPT = """
# Role: Multi-agent Router
You are a helpful routing assistant which routes user requests to specialized remote agents in  a multi-agent setup.

## Core capability & Task
Your main task is to:
1. look up available agents via the provided registry tool
2. select the best matching agent for the user query.
3. return the agent_name for the selected agent or the answer if no matching agent is found.

## Rules
- Return only the agent_name as a string.
- If the user query is relevant to multiple agents, return the agent_name of the agent with the highest match.
- If the user provides a list of rejected or excluded agents, do not route to any agent in that list. Use the `exclude_agents` parameter in the `agent_lookup` tool.
- If the user query is not relevant to any agent, try to answer it yourself starting with a disclaimer that states "DISCLAIMER: I am not a specialized agent and will answer to the best of my knowledge" plus a short description of which skills the specialized remote agents have
"""

GENERAL_SYSTEM_PROMPT = """
You are an agent in a distributed multi-agent platform in which different specialized agents are deployed in order to answer different user queries.
Therefore you should only reply to user queries pertaining to the scope explicitly mentioned in your role description below.
Queries not pertaining to your main scope should be rejected such that another agent which is better suited can handle them.
See below for your role and scope:

"""


class RoutingAgentExecutor(AgentExecutor):

    def __init__(self, agent_config: AgentConfig,
                 agent_registry: AgentRegistryLookupClient,
                 tools: list[BaseTool] | None = None,
                 routing_checkpointer: BaseCheckpointSaver[Any] | None = None,
                 specialized_checkpointer: BaseCheckpointSaver[Any] | None = None):
        super().__init__()
        api_key = settings.get_env_var(agent_config.agent.llm.api_key_env)
        if api_key is None:
            raise ValueError("No API key found for LLM.")

        self.auth_headers = settings.registry_auth_headers

        if not self.auth_headers.get("x-api-key"):
            logger.warning("No A2A API key found for registry communication")

        registry_url = mcp.url if (mcp := agent_config.agent.registry and agent_config.agent.registry.mcp) else ""
        self.mcp_registry = McpRegistryLookup(
            registry_url=registry_url,
            req_opts={
                **settings.registry_auth_headers,
                "Accept": "application/json"
            })
        self.agent_config = agent_config
        self.registered_tools: dict[str, Any] = {}
        self.api_key = api_key
        self.specialized_checkpointer = specialized_checkpointer
        self.agent_registry = agent_registry
        self.agent = SpecializedAgent(
            llm_config=agent_config.agent.llm,
            system_prompt=GENERAL_SYSTEM_PROMPT + agent_config.agent.system_prompt,
            name=agent_config.agent.card.name,
            api_key=api_key,
            tools=[] if tools is None else tools,
            checkpointer=specialized_checkpointer
        )
        self.routing_agent = RoutingAgent(
            llm_config=agent_config.agent.llm,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            name="Router",
            api_key=api_key,
            tools=[agent_registry.as_tool()],
            checkpointer=routing_checkpointer

        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Close any owned async resources (the MCP registry client)."""
        try:
            await self.mcp_registry.aclose()
        except Exception:
            logger.warning("Failed to close MCP registry client", exc_info=True)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.context_id is None or context.task_id is None:
            raise ValueError("Context ID and Task ID must be provided.")

        try:
            await _emit_status(event_queue, context, TaskState.working, final=False)
            mcp_client = await self._build_mcp_client()
            if mcp_client is not None:
                self._update_agent_with_tools(await mcp_client.get_tools())
            invocation = await self.agent(message=context.get_user_input(),
                                          context_id=context.context_id)
            artifact, final_state, file_parts, delivered_sources = await self._build_result(invocation, context)
            for name, file_part in file_parts:
                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    append=False,
                    last_chunk=False,
                    context_id=context.context_id,
                    task_id=context.task_id,
                    artifact=Artifact(
                        artifact_id=name,
                        name=name,
                        parts=[Part(root=file_part)],
                    ),
                ))
            await _emit_artifact(event_queue, context, artifact)
            await _emit_status(event_queue, context, final_state, final=True)
            if delivered_sources:
                await self.agent.persist_delivered_messages(
                    context.context_id, delivered_sources,
                )
        except RoutingFailed as e:
            logger.error(f"Routing failed for context {context.context_id}: {e.message}")
            await _fail_task(event_queue, context, name='routing_error',
                             description='Error message for routing failure.',
                             text=e.message)
        except Exception as e:
            logger.error(f"Error executing agent task for context {context.context_id}: {e}", )
            await _fail_task(event_queue, context, name='current_result',
                             description='Unexpected error while executing the agent task.',
                             text=f"*{self.agent_config.agent.card.name}* failed to process the request: {e}")

    async def _build_result(self, invocation: AgentInvocation[StringResponse],
                            context: RequestContext,
                            ) -> tuple[Artifact, TaskState, list[tuple[str, FilePart]], list[Any]]:
        """Build the terminal artifact and state; on ``rejected``, reroute and report ``completed``."""
        agent_response = invocation.structured
        if agent_response.status == TaskState.rejected:
            artifact = await _route_request_to_matching_agent(self.routing_agent, self.agent_registry, context)
            return artifact, TaskState.completed, [], []
        logger.info(f"Request with id {context.context_id} was successfully processed by agent.")
        file_parts, delivered_sources = extract_file_parts(invocation.messages)
        artifact = new_text_artifact(
            name='current_result',
            description='Result of request to agent.',
            text=f"*{self.agent_config.agent.card.name}*: {agent_response.response}"
        )
        return artifact, TaskState(agent_response.status), file_parts, delivered_sources

    async def _build_mcp_client(self) -> MultiServerMCPClient | None:
        mcp_server_raw = await self.mcp_registry.get_mcp_tool_for_agent(self.agent_config.agent.card.name)
        if not mcp_server_raw:
            return None
        logger.info(f"Agent {self.agent_config.agent.card.name} has access to the following tools: {mcp_server_raw}")
        mcp_servers: dict[str, Any] = {
            tool["name"]: {
                "url": tool["url"],
                "transport": tool["protocol"],
                "headers": settings.get_mcp_auth_headers(tool["name"])
            }
            for tool in mcp_server_raw
        }
        return MultiServerMCPClient(
            connections=mcp_servers,
            tool_interceptors=[hide_binary_content_from_llm],
        )

    def _update_agent_with_tools(self, mcp_tools: list[Any]) -> None:
        self.agent = SpecializedAgent(
            llm_config=self.agent_config.agent.llm,
            system_prompt=GENERAL_SYSTEM_PROMPT + self.agent_config.agent.system_prompt,
            name=self.agent_config.agent.card.name,
            api_key=self.api_key,
            tools=mcp_tools,
            checkpointer=self.specialized_checkpointer,
        )


class RoutingExecutor(AgentExecutor):
    def __init__(self, router_config: RouterConfig, agent_registry: AgentRegistryLookupClient) -> None:
        super().__init__()
        api_key = settings.get_env_var(router_config.router.llm.api_key_env)
        if api_key is None:
            raise ValueError("No API key found for LLM.")
        self.agent_registry = agent_registry
        self.routing_agent = RoutingAgent(
            llm_config=router_config.router.llm,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            name=router_config.router.card.name,
            api_key=api_key,
            tools=[agent_registry.as_tool()]
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.context_id is None or context.task_id is None:
            raise ValueError("Context ID and Task ID must be provided.")

        try:
            await _emit_status(event_queue, context, TaskState.working, final=False)
            artifact = await _route_request_to_matching_agent(self.routing_agent, self.agent_registry, context)
            await _emit_artifact(event_queue, context, artifact)
            await _emit_status(event_queue, context, TaskState.completed, final=True)
        except RoutingFailed as e:
            logger.error(f"Routing failed for context {context.context_id}: {e.message}")
            await _fail_task(event_queue, context, name='routing_error',
                             description='Error message for routing failure.',
                             text=e.message)
        except Exception as e:
            logger.error(f"Error executing agent task for context {context.context_id}: {e}")
            await _fail_task(event_queue, context, name='routing_error',
                             description='Unexpected error while executing the routing task.',
                             text=f"Routing failed: {e}")


async def _route_request_to_matching_agent(routing_agent: RoutingAgent,
                                           agent_registry: AgentRegistryLookupClient,
                                           context: RequestContext) -> Artifact:
    invocation = await routing_agent(message=context.get_user_input(),
                                     context_id=context.context_id)
    routing_agent_response: RoutingResponse = invocation.structured
    agent_name: str | None = routing_agent_response.agent_name
    logger.info(f"routing response received: {routing_agent_response}")
    if agent_name is None:
        raise RoutingFailed(message=routing_agent_response.message if routing_agent_response.message else str(routing_agent_response))
    logger.info(f"Request with id {context.context_id} got rejected and will be rerouted to a '{agent_name}'.")
    agent_card: dict[str, Any] | None = await agent_registry.get_agent_card(agent_name)
    if agent_card is None:
        raise RoutingFailed(message=routing_agent_response.message if routing_agent_response.message else str(routing_agent_response))
    logger.info(f"Routing agent response for request with id {context.context_id}: {agent_card}")
    artifact = new_text_artifact(name='target_agent', description='New target agent for request.',
                                 text=json.dumps(agent_card))
    return artifact


async def _emit_status(event_queue: EventQueue,
                       context: RequestContext,
                       state: TaskState,
                       *,
                       final: bool) -> None:
    """Publish a ``TaskStatusUpdateEvent`` for the current task."""
    assert context.context_id is not None and context.task_id is not None
    await event_queue.enqueue_event(TaskStatusUpdateEvent(
        status=TaskStatus(state=state),
        final=final,
        context_id=context.context_id,
        task_id=context.task_id,
    ))


async def _emit_artifact(event_queue: EventQueue,
                         context: RequestContext,
                         artifact: Artifact) -> None:
    """Publish a single, terminal ``TaskArtifactUpdateEvent``."""
    assert context.context_id is not None and context.task_id is not None
    await event_queue.enqueue_event(TaskArtifactUpdateEvent(
        append=False,
        context_id=context.context_id,
        task_id=context.task_id,
        last_chunk=True,
        artifact=artifact,
    ))


async def _fail_task(event_queue: EventQueue,
                     context: RequestContext,
                     *,
                     name: str,
                     description: str,
                     text: str) -> None:
    """Emit an error artifact followed by a final ``failed`` status."""
    await _emit_artifact(event_queue, context, new_text_artifact(
        name=name,
        description=description,
        text=text,
    ))
    await _emit_status(event_queue, context, TaskState.failed, final=True)
