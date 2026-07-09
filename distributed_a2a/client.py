import asyncio
import json
import time
from dataclasses import dataclass, field
from uuid import uuid4

import httpx
from a2a.client import (A2ACardResolver, ClientConfig, ClientEvent,
                        ClientFactory, create_text_message_object)
from a2a.types import (AgentCard, FilePart, FileWithBytes, FileWithUri,
                       Message, Part, Task, TaskQueryParams, TaskState,
                       TextPart)

DEFAULT_MAX_POLLS = 50
DEFAULT_POLL_INTERVAL = 1.0


@dataclass
class FileRef:
    """A file payload received as part of an A2A agent reply.

    Exactly one of ``bytes_b64`` (for ``FileWithBytes``) or ``uri`` (for
    ``FileWithUri``) is populated. ``bytes_b64`` is the raw base64 string
    delivered over the wire by the A2A SDK — the caller is responsible for
    decoding before forwarding the bytes.
    """

    name: str
    mime_type: str
    bytes_b64: str = ""
    uri: str | None = None


@dataclass
class AgentReply:
    """Structured reply from a routing-aware agent.

    Carries the user-visible text (if any) plus zero-or-more files that the
    agent emitted out-of-band as ``FilePart`` artifacts.
    """

    text: str | None = None
    files: list[FileRef] = field(default_factory=list)


class A2ATimeoutError(Exception):
    """Raised when polling a remote agent task exceeds max_polls."""

    def __init__(self, target_url: str, attempts: int, elapsed_seconds: float,
                 last_task_state: TaskState | None) -> None:
        self.target_url = target_url
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds
        self.last_task_state = last_task_state
        super().__init__(
            f"Timed out waiting for agent at {target_url}: "
            f"attempts={attempts}, elapsed_seconds={elapsed_seconds:.2f}, "
            f"last_task_state={last_task_state}"
        )


class A2AProtocolError(Exception):
    """Base class for protocol-level failures talking to a remote agent."""


class A2AAuthRequiredError(A2AProtocolError):
    """The remote agent reported that authentication is required."""

    def __init__(self, target_url: str) -> None:
        self.target_url = target_url
        super().__init__(f"Remote agent at {target_url} requires authentication")


class A2ARemoteTaskError(A2AProtocolError):
    """The remote agent task ended in a ``failed`` state."""

    def __init__(self, target_url: str, message: str) -> None:
        self.target_url = target_url
        self.message = message
        super().__init__(message)


class A2AEmptyResponseError(A2AProtocolError):
    """The remote agent returned no task response at all."""

    def __init__(self, target_url: str) -> None:
        self.target_url = target_url
        super().__init__(f"No task response received from agent at {target_url}")


class A2AUnexpectedResponseError(A2AProtocolError):
    """The remote response did not match any known artifact / state shape."""

    def __init__(self, target_url: str, task_state: TaskState | None,
                 artifact_names: list[str]) -> None:
        self.target_url = target_url
        self.task_state = task_state
        self.artifact_names = artifact_names
        super().__init__(
            f"Wrong response format from agent at {target_url}: "
            f"task state={task_state}, artifact_names={artifact_names}"
        )


class RemoteAgentConnection:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, client: httpx.AsyncClient,
                 max_polls: int = DEFAULT_MAX_POLLS,
                 poll_interval: float = DEFAULT_POLL_INTERVAL):
        if agent_card.preferred_transport is None:
            raise ValueError("Agent card preferred transport must be provided.")
        if agent_card.capabilities.streaming is None:
            raise ValueError("Agent card streaming capability must be provided.")

        self.agent_card = agent_card
        self.max_polls = max_polls
        self.poll_interval = poll_interval

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

        if not responses:
            raise A2AEmptyResponseError(self.agent_card.url)
        task_response, _ = responses[-1]
        return task_response

    async def _get_task(self, task_id: str) -> Task:
        query_params: TaskQueryParams = TaskQueryParams(id=task_id)
        response: Task = await self.agent_client.get_task(query_params)
        return response

    async def send_message(self,
                           message_to_send: str,
                           context_id: str,
                           task_id: None | str = None) -> AgentReply | AgentCard | TaskState:
        message: Message = create_text_message_object(content=message_to_send)
        message.message_id = str(uuid4())
        message.context_id = context_id

        started_at = time.monotonic()
        response: Task
        if task_id is None:
            response = await self._send_message_to_agent(message)
        else:
            response = await self._get_task(task_id)

        attempts = 1
        while response.status.state in (TaskState.working, TaskState.submitted):
            if attempts > self.max_polls:
                raise A2ATimeoutError(
                    target_url=self.agent_card.url,
                    attempts=attempts - 1,
                    elapsed_seconds=time.monotonic() - started_at,
                    last_task_state=response.status.state,
                )
            await asyncio.sleep(self.poll_interval * pow(1.05, attempts))
            response = await self._get_task(response.id)
            attempts += 1

        task_state = response.status.state
        if task_state == TaskState.auth_required:
            raise A2AAuthRequiredError(self.agent_card.url)

        if task_state == TaskState.failed:
            error_msg = "Agent task failed"
            if response.status.message:
                for part in response.status.message.parts or []:
                    root = getattr(part, 'root', None)
                    if root is not None and isinstance(root, TextPart):
                        error_msg = root.text
                        break
            raise A2ARemoteTaskError(self.agent_card.url, error_msg)

        for artifact in response.artifacts or []:
            match artifact.name, artifact.parts:
                case 'routing_error', [Part(root=TextPart(text=error_msg)), *_]:
                    return AgentReply(text=error_msg)
                case 'rejected', [Part(root=TextPart()), *_]:
                    return TaskState.rejected
                case 'target_agent', [Part(root=TextPart(text=agent_card_str)), *_]:
                    return AgentCard(**json.loads(agent_card_str))

        text_out: str | None = None
        files_out: list[FileRef] = []
        for artifact in response.artifacts or []:
            for part in artifact.parts or []:
                root = getattr(part, "root", None)
                if isinstance(root, TextPart) and artifact.name == "current_result":
                    text_out = root.text
                elif isinstance(root, FilePart):
                    root_file = root.file
                    if isinstance(root_file, FileWithBytes):
                        files_out.append(FileRef(
                            name=root_file.name or artifact.name or "file.bin",
                            mime_type=root_file.mime_type or "application/octet-stream",
                            bytes_b64=root_file.bytes,
                        ))
                    elif isinstance(root_file, FileWithUri):
                        files_out.append(FileRef(
                            name=root_file.name or artifact.name or "file.bin",
                            mime_type=root_file.mime_type or "application/octet-stream",
                            uri=root_file.uri,
                        ))

        if text_out is not None or files_out:
            return AgentReply(text=text_out, files=files_out)

        if task_state == TaskState.rejected:
            return TaskState.rejected

        artifact_names = [getattr(a, 'name', type(a).__name__) for a in (response.artifacts or [])]
        raise A2AUnexpectedResponseError(self.agent_card.url, task_state, artifact_names)


MAX_RECURSION_DEPTH = 10


class RoutingA2AClient:
    def __init__(self, initial_url: str, opts: dict[str, str] | None = None,
                 max_polls: int = DEFAULT_MAX_POLLS,
                 poll_interval: float = DEFAULT_POLL_INTERVAL):
        self.initial_url = initial_url
        self.client = httpx.AsyncClient(headers=opts)
        self.current_card: AgentCard | None = None
        self.max_polls = max_polls
        self.poll_interval = poll_interval

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "RoutingA2AClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    async def fetch_initial_card(self) -> None:
        card_resolver = A2ACardResolver(
            self.client, self.initial_url
        )
        self.current_card = (
            await card_resolver.get_agent_card()
        )

    async def send_message(self, message: str,
                           context_id: str,
                           depth: int = 0,
                           rejected_agents: list[str] | None = None) -> AgentReply:
        if depth > MAX_RECURSION_DEPTH:
            raise A2AProtocolError(
                "Maximum recursion depth exceeded. This is likely due to an infinite loop in your agent."
            )

        if rejected_agents is None:
            rejected_agents = []

        if self.current_card is None:
            await self.fetch_initial_card()

        if self.current_card is None:
            raise ValueError("Failed to fetch current agent card.")

        current_depth = depth
        while True:
            if current_depth > MAX_RECURSION_DEPTH:
                raise A2AProtocolError(
                    "Maximum recursion depth exceeded. This is likely due to an infinite loop in your agent."
                )

            agent_connection = RemoteAgentConnection(
                self.current_card, self.client,
                max_polls=self.max_polls, poll_interval=self.poll_interval,
            )
            message_to_send = message
            if rejected_agents:
                excluded_names = ", ".join(sorted(set(rejected_agents)))
                rejection_msg = f"Please exclude the following agents from routing: {excluded_names}"
                if rejection_msg not in message:
                    message_to_send = f"{message}\n\n{rejection_msg}"

            agent_response: AgentReply | AgentCard | TaskState = await agent_connection.send_message(message_to_send, context_id)

            if isinstance(agent_response, AgentCard):
                if agent_response.url == self.current_card.url:
                    raise A2ARemoteTaskError(self.current_card.url, "Agent redirected to itself.")
                if agent_response.name in rejected_agents:
                    raise A2ARemoteTaskError(
                        agent_response.url,
                        f"Agent {agent_response.name} was already rejected but was redirected to again.",
                    )
                self.current_card = agent_response
                current_depth += 1
                continue

            if isinstance(agent_response, TaskState):
                if agent_response == TaskState.rejected:
                    if self.current_card.name in rejected_agents:
                        raise A2ARemoteTaskError(
                            self.current_card.url,
                            f"Agent {self.current_card.name} rejected the request again after being already in the rejected list.",
                        )
                    rejected_agents.append(self.current_card.name)
                    await self.fetch_initial_card()
                    if self.current_card is None:
                        raise ValueError("Failed to refetch initial agent card after rejection.")
                    current_depth += 1
                    continue
                raise A2AUnexpectedResponseError(
                    self.current_card.url, agent_response, artifact_names=[]
                )

            return agent_response
