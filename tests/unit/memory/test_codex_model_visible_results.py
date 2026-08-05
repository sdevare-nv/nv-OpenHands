"""Exact model-visible result tests for Codex tool calls."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from openhands.agenthub.codex_agent.tool_output import truncate_function_output
from openhands.core.config.agent_config import AgentConfig
from openhands.core.message import Message, TextContent
from openhands.events.observation import (
    CmdOutputMetadata,
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.events.observation.codex import (
    CodexApplyPatchObservation,
    CodexUpdatePlanObservation,
)
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.events.tool import ToolCallMetadata
from openhands.llm.tool_names import (
    CODEX_APPLY_PATCH_TOOL_NAME,
    CODEX_READ_FILE_TOOL_NAME,
    CODEX_SHELL_COMMAND_TOOL_NAME,
    CODEX_UPDATE_PLAN_TOOL_NAME,
)
from openhands.memory.conversation_memory import ConversationMemory
from openhands.utils.prompt import PromptManager


def _make_memory(*, include_turns_remaining_reminder: bool) -> ConversationMemory:
    config = AgentConfig(
        include_turns_remaining_reminder=include_turns_remaining_reminder,
    )
    return ConversationMemory(config, MagicMock(spec=PromptManager))


def _tool_call_metadata(
    *,
    tool_call_id: str = "call-1",
    function_name: str,
    tool_result_format: str | None = None,
    model: str = "mock-model",
) -> ToolCallMetadata:
    model_response = {
        "id": "response-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            }
        ],
        "created": 0,
        "model": model,
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }
    kwargs = {}
    if tool_result_format is not None:
        kwargs["tool_result_format"] = tool_result_format
    return ToolCallMetadata(
        tool_call_id=tool_call_id,
        function_name=function_name,
        model_response=model_response,
        total_calls_in_response=1,
        **kwargs,
    )


def _as_tool_message(
    memory: ConversationMemory,
    observation: Observation,
    *,
    metadata: ToolCallMetadata,
    max_message_chars: int | None,
) -> Message:
    observation.tool_call_metadata = metadata
    messages_by_call_id: dict[str, Message] = {}

    returned_messages = memory._process_observation(
        obs=observation,
        tool_call_id_to_message=messages_by_call_id,
        max_message_chars=max_message_chars,
    )

    assert returned_messages == []
    assert metadata.tool_call_id is not None
    return messages_by_call_id[metadata.tool_call_id]


def _cmd_observation() -> Observation:
    return CmdOutputObservation(
        content="stdout line\nstderr line\n",
        command="example-command",
        metadata=CmdOutputMetadata(
            exit_code=9,
            prefix="generic prefix\n",
            suffix="\ngeneric suffix",
            working_dir="/workspace/project",
            py_interpreter_path="/usr/bin/python",
        ),
    )


def _error_observation() -> Observation:
    return ErrorObservation(content="Codex read failed exactly.")


def _apply_patch_observation() -> Observation:
    return CodexApplyPatchObservation(
        content="Patch applied successfully. Changed files:\n  M example.py",
        files_changed=["example.py"],
        success=True,
    )


def _update_plan_observation() -> Observation:
    return CodexUpdatePlanObservation(
        content="Plan updated",
        plan=[{"step": "Inspect", "status": "completed"}],
        success=True,
    )


@pytest.mark.parametrize(
    "observation_factory,function_name",
    [
        (_cmd_observation, CODEX_SHELL_COMMAND_TOOL_NAME),
        (_error_observation, CODEX_READ_FILE_TOOL_NAME),
        (_apply_patch_observation, CODEX_APPLY_PATCH_TOOL_NAME),
        (_update_plan_observation, CODEX_UPDATE_PLAN_TOOL_NAME),
    ],
    ids=["cmd", "error", "apply-patch", "update-plan"],
)
def test_codex_tool_result_is_exact_model_visible_content(
    observation_factory: Callable[[], Observation],
    function_name: str,
) -> None:
    """Codex content bypasses generic wrappers, truncation, and reminders."""
    memory = _make_memory(include_turns_remaining_reminder=True)
    observation = observation_factory()
    setattr(observation, "_turns_left", 2)
    metadata = _tool_call_metadata(
        function_name=function_name,
        tool_result_format="codex",
    )

    message = _as_tool_message(
        memory,
        observation,
        metadata=metadata,
        max_message_chars=4,
    )

    assert message.role == "tool"
    assert message.tool_call_id == "call-1"
    assert message.name == function_name
    assert message.content == [TextContent(text=observation.content)]


def test_codex_marker_survives_event_serialization_round_trip() -> None:
    observation = ErrorObservation(content="exact Codex failure body")
    observation.tool_call_metadata = _tool_call_metadata(
        function_name=CODEX_READ_FILE_TOOL_NAME,
        tool_result_format="codex",
    )

    serialized = event_to_dict(observation)
    restored = event_from_dict(serialized)

    assert serialized["tool_call_metadata"]["tool_result_format"] == "codex"
    assert restored.tool_call_metadata is not None
    assert restored.tool_call_metadata.tool_result_format == "codex"


@pytest.mark.parametrize(
    ("observation", "model", "source"),
    [
        (
            ErrorObservation(content="x" * 12_001),
            "gpt-5.2",
            "x" * 12_001,
        ),
        (
            CodexApplyPatchObservation(
                content="y" * 48_001,
                files_changed=[],
                success=False,
            ),
            "gpt-5.6-sol",
            "y" * 48_001,
        ),
    ],
    ids=["byte-policy-error", "token-policy-apply-failure"],
)
def test_codex_failure_gets_exact_history_budget_once(
    observation: Observation,
    model: str,
    source: str,
) -> None:
    memory = _make_memory(include_turns_remaining_reminder=True)
    metadata = _tool_call_metadata(
        function_name=CODEX_APPLY_PATCH_TOOL_NAME,
        tool_result_format="codex",
        model=model,
    )

    message = _as_tool_message(
        memory,
        observation,
        metadata=metadata,
        max_message_chars=1,
    )
    expected = truncate_function_output(source, model_name=model)

    assert message.content == [TextContent(text=expected)]
    assert expected.count(" truncated…") == 1
    assert "[... Observation truncated due to length ...]" not in expected


def test_unmarked_cmd_result_keeps_generic_formatting() -> None:
    memory = _make_memory(include_turns_remaining_reminder=False)
    observation = CmdOutputObservation(
        content="command output",
        command="example-command",
        command_id=17,
        exit_code=9,
    )

    message = _as_tool_message(
        memory,
        observation,
        metadata=_tool_call_metadata(
            function_name=CODEX_SHELL_COMMAND_TOOL_NAME,
        ),
        max_message_chars=None,
    )

    assert message.content == [
        TextContent(
            text="command output\n[Command finished with exit code 9]",
        )
    ]
