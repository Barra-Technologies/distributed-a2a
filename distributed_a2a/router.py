from a2a.server.apps import A2ARESTFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill
from fastapi import FastAPI

from .config import settings
from .executors import RoutingExecutor
from .model import RouterConfig
from .registry import AgentRegistryLookupClient
from .server import CAPABILITIES, _name_to_slug


def load_router(router_config: RouterConfig) -> FastAPI:
    agent_card = AgentCard(
        name="Router",
        description="Agent to redirect to the best matching agent based on the agent card",
        url=router_config.router.card.url,
        version="1.0.0",
        default_input_modes=router_config.router.card.default_input_modes,
        default_output_modes=router_config.router.card.default_output_modes,
        skills=[AgentSkill(
            id='routing',
            name='Agent routing',
            description='Identifies the most suitable agent for the given task and returns the agent card',
            tags=['agent', 'routing']
        )],
        preferred_transport=router_config.router.card.preferred_transport_protocol,
        capabilities=CAPABILITIES
    )
    req_opts = settings.registry_auth_headers

    agent_registry_url = ""
    if router_config.router.registry and router_config.router.registry.agent:
        agent_registry_url = router_config.router.registry.agent.url

    executor = RoutingExecutor(
        router_config=router_config,
        agent_registry=AgentRegistryLookupClient(
            agent_registry_url,
            req_opts=req_opts)
    )


    root_path = settings.api_root_path or f"/{_name_to_slug(router_config.router.card.name)}"
    if root_path == "/":
        root_path = ""

    return A2ARESTFastAPIApplication(
        agent_card=agent_card,
        http_handler=DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore()  # TODO replace with dynamodb store

        )).build(title=agent_card.name,
                 root_path=root_path)
