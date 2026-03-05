import json
import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskStatusUpdateEvent, TaskStatus, TaskState, TaskArtifactUpdateEvent
from a2a.utils import new_text_artifact
from distributed_a2a.agent import StatusAgent, StringResponse
from distributed_a2a.config import settings
from distributed_a2a.model import AgentConfig

logger = logging.getLogger(__name__)


class NonRoutingAgent(AgentExecutor):
    def __init__(self, agent_config: AgentConfig):
        super().__init__()
        api_key = settings.get_env_var(agent_config.agent.llm.api_key_env)
        if api_key is None:
            raise ValueError("No API key found for LLM.")

        self.agent_name = agent_config.agent.card.name
        self.agent = StatusAgent[StringResponse](
            llm_config=agent_config.agent.llm,
            system_prompt=agent_config.agent.system_prompt,
            name=agent_config.agent.card.name,
            api_key=api_key,
            is_routing=False,
            tools=[]
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.context_id is None or context.task_id is None:
            raise ValueError("Context ID and Task ID must be provided.")

        try:
            # set status to processing
            await event_queue.enqueue_event(TaskStatusUpdateEvent(status=TaskStatus(state=TaskState.working),
                                                                  final=False,
                                                                  context_id=context.context_id,
                                                                  task_id=context.task_id))

            agent_response: StringResponse = await self.agent(message=context.get_user_input(),
                                                              context_id=context.context_id)

            logger.info(f"Request with id {context.context_id} was processed by agent with status {agent_response.status}")
            if agent_response.status == TaskState.rejected:
                logger.info(f"Agent {self.agent_name} rejected the request.")
                artifact = new_text_artifact(name='rejected', description='Agent rejected the request.',
                                             text=agent_response.response if hasattr(agent_response, 'response') else "Agent rejected the request.")
                state = TaskState.completed
            elif hasattr(agent_response, 'agent_card') and agent_response.agent_card:
                logger.info(f"Agent {self.agent_name} redirected to {agent_response.agent_card}")
                artifact = new_text_artifact(name='target_agent', description='New target agent for request.',
                                             text=agent_response.agent_card if isinstance(agent_response.agent_card, str) else json.dumps(agent_response.agent_card))
                state = TaskState.completed
            else:
                artifact = new_text_artifact(name='current_result', description='Result of request to agent.',
                                             text=agent_response.response if hasattr(agent_response, 'response') else str(agent_response))
                state = TaskState(agent_response.status)

            # publish actual result
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(append=False,
                                                                    context_id=context.context_id,
                                                                    task_id=context.task_id,
                                                                    last_chunk=True,
                                                                    artifact=artifact))
            # set and publish the final status
            await event_queue.enqueue_event(TaskStatusUpdateEvent(status=TaskStatus(
                state=state),
                final=True,
                context_id=context.context_id,
                task_id=context.task_id))
        except Exception as e:
            logger.error(f"Error executing agent task for context {context.context_id}: {e}", )
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                status=TaskStatus(state=TaskState.failed),
                final=True,
                context_id=context.context_id,
                task_id=context.task_id))