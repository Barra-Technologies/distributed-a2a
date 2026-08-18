import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from a2a.server.apps import A2ARESTFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver

from .config import settings
from .executors import RoutingAgentExecutor
from .model import AgentConfig
from .registry import AgentRegistryLookupClient, registry_heart_beat

CAPABILITIES = AgentCapabilities(streaming=False, push_notifications=False)

HEART_BEAT_INTERVAL_SEC = 5
MAX_HEART_BEAT_MISSES = 3


def get_expire_at() -> int:
    return int(time.time() + MAX_HEART_BEAT_MISSES * HEART_BEAT_INTERVAL_SEC)


def _name_to_slug(name: str) -> str:
    """Normalize an agent/router name into a URL path segment."""
    return name.replace(" ", "_").lower()


def get_agent_card(agent_config: AgentConfig) -> AgentCard:
    config_card = agent_config.agent.card
    skills = [AgentSkill(
        id=skill.id,
        name=skill.name,
        description=skill.description,
        tags=skill.tags,
        examples=skill.examples) for skill in config_card.skills]
    if agent_config.agent.advertise_routing_skill:
        skills.append(AgentSkill(
            id='routing',
            name='Agent routing',
            description='Identifies the most suitable agent for the given task and returns the agent card',
            tags=['agent', 'routing']
        ))
    agent_card = AgentCard(
        name=config_card.name,
        description=config_card.description,
        url=config_card.url,
        version=config_card.version,
        default_input_modes=config_card.default_input_modes,
        default_output_modes=config_card.default_output_modes,
        skills=skills,
        preferred_transport=config_card.preferred_transport_protocol,
        capabilities=CAPABILITIES
    )
    return agent_card


def load_app(agent_config: AgentConfig,
             routing_checkpointer: BaseCheckpointSaver[Any] | None = None,
             specialized_checkpointer: BaseCheckpointSaver[Any] | None = None) -> FastAPI:
    agent_card = get_agent_card(agent_config)
    req_opts = settings.registry_auth_headers

    agent_registry_url = ""
    if agent_config.agent.registry and agent_config.agent.registry.agent:
        agent_registry_url = agent_config.agent.registry.agent.url
    agent_registry = AgentRegistryLookupClient(agent_registry_url, req_opts=req_opts)
    executor = RoutingAgentExecutor(agent_config=agent_config,
                                    agent_registry=agent_registry,
                                    routing_checkpointer=routing_checkpointer,
                                    specialized_checkpointer=specialized_checkpointer)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, Any]:
        heartbeat_task = asyncio.create_task(
            registry_heart_beat(
                name=agent_card.name,
                registry=agent_registry,
                agent_card=agent_card,
                interval_sec=HEART_BEAT_INTERVAL_SEC,
                get_expire_at=get_expire_at,
            )
        )
        try:
            yield
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await agent_registry.aclose()
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to close agent registry client", exc_info=True
                )
            try:
                await executor.aclose()
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to close executor resources", exc_info=True
                )

    root_path = settings.api_root_path or f"/{_name_to_slug(agent_config.agent.card.name)}"
    if root_path == "/":
        root_path = ""

    return A2ARESTFastAPIApplication(
        agent_card=agent_card,
        http_handler=DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore()  # TODO: replace with dynamodb store

        )).build(title=agent_card.name, lifespan=lifespan, root_path=root_path)
