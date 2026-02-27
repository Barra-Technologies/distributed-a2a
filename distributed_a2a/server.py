import os
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import boto3
from a2a.server.apps import A2ARESTFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, \
    AgentCapabilities, AgentCard
from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver

from .executors import RoutingAgentExecutor
from .config import settings
from .model import AgentConfig
from .registry import registry_heart_beat, AgentRegistryLookup

CAPABILITIES = AgentCapabilities(streaming=False, push_notifications=False)

HEART_BEAT_INTERVAL_SEC = 5
MAX_HEART_BEAT_MISSES = 3


def get_expire_at() -> int:
    return int(time.time() + MAX_HEART_BEAT_MISSES * HEART_BEAT_INTERVAL_SEC)


async def heart_beat(name: str, agent_card_table: str, agent_card: AgentCard) -> None:
    table = boto3.resource("dynamodb", region_name="eu-central-1").Table(agent_card_table)
    table.put_item(Item={"id": name, "card": agent_card.model_dump_json(), "expireAt": get_expire_at()})
    while True:
        await asyncio.sleep(HEART_BEAT_INTERVAL_SEC)
        table.update_item(
            Key={"id": name},
            UpdateExpression="SET expireAt = :expire_at",
            ExpressionAttributeValues={":expire_at": get_expire_at()}
        )


def load_app(agent_config: AgentConfig, routing_checkpointer: Optional[BaseCheckpointSaver[Any]] = None,
             specialized_checkpointer: Optional[BaseCheckpointSaver[Any]] = None) -> FastAPI:
    skills = [AgentSkill(
        id=skill.id,
        name=skill.name,
        description=skill.description,
        tags=skill.tags,
        examples=skill.examples)
        for skill in agent_config.agent.card.skills]
    skills.append(AgentSkill(
        id='routing',
        name='Agent routing',
        description='Identifies the most suitable agent for the given task and returns the agent card',
        tags=['agent', 'routing']
    ))

    agent_card = AgentCard(
        name=agent_config.agent.card.name,
        description=agent_config.agent.card.description,
        url=agent_config.agent.card.url,
        version=agent_config.agent.card.version,
        default_input_modes=agent_config.agent.card.default_input_modes,
        default_output_modes=agent_config.agent.card.default_output_modes,
        skills=skills,
        preferred_transport=agent_config.agent.card.preferred_transport_protocol,
        capabilities=CAPABILITIES
    )
    req_opts = settings.registry_auth_headers

    agent_registry_url = ""
    if agent_config.agent.registry and agent_config.agent.registry.agent:
        agent_registry_url = agent_config.agent.registry.agent.url
    agent_registry = AgentRegistryLookup(agent_registry_url, req_opts=req_opts)
    executor = RoutingAgentExecutor(agent_config=agent_config,
                                    routing_tool=agent_registry.as_tool(),
                                    routing_checkpointer=routing_checkpointer,
                                    specialized_checkpointer=specialized_checkpointer)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, Any]:
        asyncio.create_task(registry_heart_beat(name=agent_card.name, registry=agent_registry,
                                                agent_card=agent_card, interval_sec=HEART_BEAT_INTERVAL_SEC,
                                                get_expire_at=get_expire_at))

        yield

    root_path = settings.api_root_path or f"/{agent_config.agent.card.name}"
    if root_path == "/":
        root_path = ""

    return A2ARESTFastAPIApplication(
        agent_card=agent_card,
        http_handler=DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore()  # TODO replace with dynamodb store

        )).build(title=agent_card.name, lifespan=lifespan, root_path=root_path)
