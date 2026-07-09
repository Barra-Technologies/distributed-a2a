from distributed_a2a.model import (AgentConfig, AgentItem, CardConfig,
                                   LLMConfig, SkillConfig)
from distributed_a2a.server import get_agent_card


def _agent_config(advertise_routing_skill: bool, with_skills: bool = False) -> AgentConfig:
    skills = (
        [SkillConfig(id="weather", name="weather", description="weather skill", tags=["w"])]
        if with_skills
        else []
    )
    return AgentConfig(
        agent=AgentItem(
            card=CardConfig(
                name="test-agent",
                description="d",
                version="1.0.0",
                url="http://example.com",
                skills=skills,
            ),
            llm=LLMConfig(base_url="http://llm", model="m", api_key_env="X"),
            system_prompt="prompt",
            advertise_routing_skill=advertise_routing_skill,
        )
    )


def test_routing_skill_not_advertised_by_default() -> None:
    card = get_agent_card(_agent_config(advertise_routing_skill=False, with_skills=True))
    assert [s.id for s in card.skills] == ["weather"]


def test_no_skills_when_none_configured_and_routing_off() -> None:
    card = get_agent_card(_agent_config(advertise_routing_skill=False, with_skills=False))
    assert card.skills == []


def test_routing_skill_appended_when_opted_in() -> None:
    card = get_agent_card(_agent_config(advertise_routing_skill=True, with_skills=True))
    ids = [s.id for s in card.skills]
    assert ids == ["weather", "routing"]
