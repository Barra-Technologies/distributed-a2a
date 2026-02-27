import json
import time
from uuid import uuid4

import httpx
from a2a.client import ClientConfig, ClientFactory, A2ACardResolver, ClientEvent
from a2a.client import create_text_message_object
from a2a.types import (
    AgentCard,
    Message, TaskQueryParams, Task, Artifact, Part, TextPart
)
from a2a.types import TaskState


class RemoteAgentConnection:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, client: httpx.AsyncClient):
        if agent_card.preferred_transport is None:
            raise ValueError("Agent card preferred transport must be provided.")
        if agent_card.capabilities.streaming is None:
            raise ValueError("Agent card streaming capability must be provided.")

        client_config = ClientConfig(
            httpx_client=client,
            supported_transports=[agent_card.preferred_transport],
            streaming=agent_card.capabilities.streaming,
            polling=True
        )
        client_factory = ClientFactory(config=client_config)
        self.agent_client = client_factory.create(agent_card)

    async def _send_message_to_agent(self, message_request: Message) -> Task:

        responses: list[ClientEvent] = []
        async for response in self.agent_client.send_message(message_request):
            if isinstance(response, tuple):
                responses.append(response)

        task_response: Task | None = None
        match responses:
            case [(task, _)]:
                task_response = task
            case _:
                raise Exception("Wrong response format")
        return task_response

    async def _get_task(self, task_id: str) -> Task:
        query_params: TaskQueryParams = TaskQueryParams(id=task_id)
        response: Task = await self.agent_client.get_task(query_params)
        return response

    async def send_message(self, message_to_send: str, context_id: str, task_id: None | str = None,
                           count: int = 0) -> str | AgentCard :
        message: Message = create_text_message_object(content=message_to_send)
        message.message_id = str(uuid4())
        message.context_id = context_id

        response: Task
        if task_id is None:
            response = await self._send_message_to_agent(message)
        else:
            response = await self._get_task(task_id)

        task_state = response.status.state
        if task_state == TaskState.working or task_state == TaskState.submitted:
            if count < 50:
                time.sleep(round(pow(1.2,count))) ##Do exponential backoff number^count
                return await self.send_message(message_to_send, context_id, response.id, count + 1)
            else:
                raise Exception("Timeout waiting for agent to respond")

        if task_state == TaskState.failed:
            raise Exception("A2ATaskFailed")
        elif task_state == TaskState.auth_required:
            raise Exception("A2ATaskAuthRequired")

        match response.artifacts:
            case [Artifact(name='target_agent', parts=[Part(root=TextPart(text=agent_card))])]:
                return AgentCard(**json.loads(agent_card))
            case [Artifact(name='current_result', parts=[Part(root=TextPart(text=result))])]:
                return result
            case _:
                raise Exception("Wrong response format")


MAX_RECURSION_DEPTH = 10


class RoutingA2AClient:
    def __init__(self, initial_url: str, opts: dict[str,str] | None = None):
        self.initial_url = initial_url
        self.client = httpx.AsyncClient(headers=opts)
        self.current_card: AgentCard | None = None

    async def fetch_initial_card(self) -> None:
        card_resolver = A2ACardResolver(
            self.client, self.initial_url
        )
        self.current_card = (
            await card_resolver.get_agent_card()
        )

    async def send_message(self, message: str, context_id: str, depth: int = 0) -> str:
        if depth > MAX_RECURSION_DEPTH:
            raise Exception("Maximum recursion depth exceeded. This is likely due to an infinite loop in your agent.")
        if self.current_card is None:
            await self.fetch_initial_card()

        if self.current_card is None:
            raise ValueError("Failed to fetch current agent card.")

        agent_connection = RemoteAgentConnection(self.current_card, self.client)

        agent_response: str | AgentCard  = await agent_connection.send_message(message, context_id)
        if isinstance(agent_response, AgentCard):
            self.current_card = agent_response
            return await self.send_message(message, context_id, depth + 1)

        return agent_response
