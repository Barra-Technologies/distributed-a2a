import json
from typing import Any

from distributed_a2a.registry_server.in_memory_registry_storage import \
    InMemoryAgentRegistry


def _card(name: str) -> str:
    return json.dumps({"name": name, "skills": []})


def test_default_does_not_enforce_ttl() -> None:
    clock = {"now": 1_000.0}
    registry = InMemoryAgentRegistry(time_func=lambda: clock["now"])
    registry.put_agent_card("a", _card("a"), expire_at=1_500)

    clock["now"] = 9_999.0

    assert registry.get_agent_card("a") is not None
    names = [c["name"] for c in registry.get_agent_cards()]
    assert names == ["a"]


def test_enforce_ttl_hides_expired_entries() -> None:
    clock = {"now": 1_000.0}
    registry = InMemoryAgentRegistry(enforce_ttl=True, time_func=lambda: clock["now"])
    registry.put_agent_card("fresh", _card("fresh"), expire_at=2_000)
    registry.put_agent_card("stale", _card("stale"), expire_at=1_100)

    clock["now"] = 1_500.0

    assert registry.get_agent_card("stale") is None
    assert registry.get_agent_card("fresh") is not None
    cards: list[dict[str, Any]] = registry.get_agent_cards()
    assert [c["name"] for c in cards] == ["fresh"]
