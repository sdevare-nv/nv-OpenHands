import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import openhands.agenthub.opencode_agent.opencode_agent as opencode_agent_module
from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent
from openhands.core.config import AgentConfig


UPSTREAM_DEFAULT_PROMPT_SHA256 = (
    "962fbf3cb3ec659c9a5244425ee2e7bb141ad4428f489a630a7738566880dc6a"
)
PROMPT_DIR = Path(opencode_agent_module.__file__).with_name("prompts")
PROMPT_PATH = PROMPT_DIR / "system_prompt.j2"


def make_agent(config: AgentConfig, model: str = "unknown") -> OpenCodeAgent:
    agent = OpenCodeAgent.__new__(OpenCodeAgent)
    agent.config = config
    agent.llm = SimpleNamespace(config=SimpleNamespace(model=model))
    agent._prompt_manager = None
    return agent


def test_system_prompt_is_exact_pinned_upstream_default() -> None:
    assert hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest() == (
        UPSTREAM_DEFAULT_PROMPT_SHA256
    )


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5-codex",
        "gpt-4.1",
        "claude-sonnet-4",
        "gemini-2.5-pro",
        "kimi-k2.5",
        "unknown-model",
    ],
)
@pytest.mark.parametrize("enable_plan_mode", [False, True])
def test_every_model_uses_only_upstream_default_prompt(
    model: str, enable_plan_mode: bool
) -> None:
    agent = make_agent(
        AgentConfig(enable_plan_mode=enable_plan_mode),
        model=model,
    )

    prompt_manager = agent.prompt_manager
    rendered = prompt_manager.get_system_message()

    assert prompt_manager.system_template.name == "system_prompt.j2"
    assert rendered == PROMPT_PATH.read_text().strip()
    assert "task_tracker" not in rendered
    assert "finish: Signal task completion" not in rendered


def test_explicit_system_prompt_path_keeps_existing_override_behavior(
    tmp_path: Path,
) -> None:
    override = tmp_path / "custom-system.j2"
    override.write_text("custom system prompt")
    agent = make_agent(AgentConfig(system_prompt_path=str(override)))

    prompt_manager = agent.prompt_manager

    assert prompt_manager.system_template.name == "system_prompt_long_horizon.j2"
    assert prompt_manager.get_system_message().startswith("custom system prompt")


def test_explicit_long_horizon_prompt_path_still_wins(tmp_path: Path) -> None:
    override = tmp_path / "custom-long-horizon.j2"
    override.write_text("custom long-horizon prompt")
    agent = make_agent(AgentConfig(system_prompt_long_horizon_path=str(override)))

    assert agent.prompt_manager.get_system_message() == "custom long-horizon prompt"


def test_custom_prompt_directory_keeps_existing_resolution(tmp_path: Path) -> None:
    for filename, content in {
        "system_prompt.j2": "custom base prompt",
        "system_prompt_long_horizon.j2": "custom long-horizon prompt",
        "user_prompt.j2": "",
        "additional_info.j2": "",
        "microagent_info.j2": "",
    }.items():
        (tmp_path / filename).write_text(content)
    agent = make_agent(AgentConfig(custom_prompt_dir=str(tmp_path)))

    assert agent.prompt_manager.get_system_message() == "custom long-horizon prompt"


def test_explicit_bundled_prompt_filename_still_wins() -> None:
    agent = make_agent(
        AgentConfig(system_prompt_filename="system_prompt_tech_philosophy.j2")
    )

    assert (
        agent.prompt_manager.system_template.name == "system_prompt_tech_philosophy.j2"
    )
