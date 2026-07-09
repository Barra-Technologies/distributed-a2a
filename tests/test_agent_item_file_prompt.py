from pathlib import Path

import pytest

from distributed_a2a.model import AgentItem, CardConfig, LLMConfig


def _kwargs(prompt: str) -> dict[str, object]:
    return {
        "card": CardConfig(
            name="test",
            description="d",
            version="1.0.0",
            url="http://example.com",
        ),
        "llm": LLMConfig(
            base_url="http://llm",
            model="m",
            api_key_env="X",
        ),
        "system_prompt": prompt,
    }


def test_literal_prompt_is_preserved_verbatim() -> None:
    item = AgentItem(**_kwargs("You are a helpful agent."))
    assert item.system_prompt == "You are a helpful agent."


def test_path_like_literal_is_not_loaded_as_file(tmp_path: Path) -> None:
    # Even if a file with that *name* exists, plain strings must be preserved.
    pseudo = tmp_path / "looks_like_a_path.txt"
    pseudo.write_text("LOADED FROM FILE", encoding="utf-8")

    item = AgentItem(**_kwargs(str(pseudo)))
    assert item.system_prompt == str(pseudo)
    assert "LOADED FROM FILE" not in item.system_prompt


def test_file_scheme_loads_prompt_from_disk(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("hello from file", encoding="utf-8")

    item = AgentItem(**_kwargs(f"file://{prompt_file}"))
    assert item.system_prompt == "hello from file"


def test_file_scheme_with_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.md"
    with pytest.raises(FileNotFoundError):
        AgentItem(**_kwargs(f"file://{missing}"))
