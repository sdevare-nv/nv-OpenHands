"""Exact model-visible result tests for OpenCode tool calls."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from openhands.core.config.agent_config import AgentConfig
from openhands.core.message import Message, TextContent
from openhands.events.event import FileEditSource
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileEditObservation,
    FileWriteObservation,
    Observation,
    TodoReadObservation,
    TodoWriteObservation,
)
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.events.tool import ToolCallMetadata
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
    function_name: str = "test_tool",
    tool_result_format: str | None = None,
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
        "model": "mock-model",
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
        command_id=17,
        exit_code=9,
    )


def _write_observation() -> Observation:
    return FileWriteObservation(
        content="Wrote file successfully.",
        path="/workspace/example.py",
    )


def _edit_observation() -> Observation:
    return FileEditObservation(
        content="Edit applied successfully.",
        path="/workspace/example.py",
        prev_exist=True,
        old_content="before",
        new_content="after",
    )


def _error_observation() -> Observation:
    return ErrorObservation(
        content="Could not find oldString in the file.",
    )


def _todo_read_observation() -> Observation:
    return TodoReadObservation(
        content='[{"content": "inspect", "status": "pending"}]',
        todos=[{"content": "inspect", "status": "pending"}],
    )


def _todo_write_observation() -> Observation:
    return TodoWriteObservation(
        content=(
            '[\n  {\n    "content": "inspect",\n    "status": "completed"\n  }\n]'
        ),
        todos=[{"content": "inspect", "status": "completed"}],
    )


@pytest.mark.parametrize(
    "observation_factory,function_name",
    [
        (_cmd_observation, "bash"),
        (_write_observation, "write"),
        (_edit_observation, "edit"),
        (_error_observation, "edit"),
        (_todo_read_observation, "todo_read"),
        (_todo_write_observation, "todo_write"),
    ],
    ids=["cmd", "write", "edit", "error", "todo-read", "todo-write"],
)
def test_opencode_tool_result_is_exact_model_visible_content(
    observation_factory: Callable[[], Observation],
    function_name: str,
) -> None:
    """OpenCode content bypasses generic truncation, wrappers, and reminders."""
    memory = _make_memory(include_turns_remaining_reminder=True)
    observation = observation_factory()
    setattr(observation, "_turns_left", 2)
    metadata = _tool_call_metadata(
        function_name=function_name,
        tool_result_format="opencode",
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


def test_opencode_marker_survives_event_serialization_round_trip() -> None:
    observation = ErrorObservation(content="exact failure body")
    observation.tool_call_metadata = _tool_call_metadata(
        function_name="edit",
        tool_result_format="opencode",
    )

    serialized = event_to_dict(observation)
    restored = event_from_dict(serialized)

    assert serialized["tool_call_metadata"]["tool_result_format"] == "opencode"
    assert restored.tool_call_metadata is not None
    assert (
        getattr(restored.tool_call_metadata, "tool_result_format", None) == "opencode"
    )


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
        metadata=_tool_call_metadata(function_name="execute_bash"),
        max_message_chars=None,
    )

    assert message.content == [
        TextContent(
            text="command output\n[Command finished with exit code 9]",
        )
    ]


def test_unmarked_write_result_keeps_generic_formatting() -> None:
    memory = _make_memory(include_turns_remaining_reminder=False)
    observation = FileWriteObservation(
        content="write details",
        path="/workspace/example.py",
    )

    message = _as_tool_message(
        memory,
        observation,
        metadata=_tool_call_metadata(function_name="write_file"),
        max_message_chars=None,
    )

    assert message.content == [
        TextContent(
            text=("File written successfully: /workspace/example.py\nwrite details"),
        )
    ]


def test_unmarked_edit_result_keeps_generic_formatting() -> None:
    memory = _make_memory(include_turns_remaining_reminder=False)
    observation = FileEditObservation(
        content="runtime edit details",
        path="/workspace/example.py",
        prev_exist=True,
        old_content="before",
        new_content="after",
        impl_source=FileEditSource.LLM_BASED_EDIT,
    )

    message = _as_tool_message(
        memory,
        observation,
        metadata=_tool_call_metadata(function_name="str_replace_editor"),
        max_message_chars=None,
    )

    assert message.content == [
        TextContent(
            text=(
                "[Existing file /workspace/example.py is edited with 1 changes.]\n"
                "[begin of edit 1 / 1]\n"
                "(content before edit)\n"
                "-1|before\n"
                "(content after edit)\n"
                "+1|after\n"
                "[end of edit 1 / 1]\n"
            ),
        )
    ]


def test_unmarked_error_result_keeps_generic_formatting() -> None:
    memory = _make_memory(include_turns_remaining_reminder=False)
    observation = ErrorObservation(content="failure details")

    message = _as_tool_message(
        memory,
        observation,
        metadata=_tool_call_metadata(function_name="str_replace_editor"),
        max_message_chars=None,
    )

    assert message.content == [
        TextContent(
            text=("failure details\n[Error occurred in processing last action]"),
        )
    ]
