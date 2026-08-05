"""End-to-end runtime formatting tests for Codex ``shell_command`` bodies."""

import asyncio
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.agenthub.codex_agent.tool_output import format_shell_output
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.events.observation.commands import CmdOutputMetadata
from openhands.events.tool import ToolCallMetadata
from openhands.llm.tool_names import CODEX_SHELL_COMMAND_TOOL_NAME


def _metadata(*, model: str = "gpt-5.6-sol") -> ToolCallMetadata:
    return ToolCallMetadata(
        tool_call_id="call-1",
        function_name=CODEX_SHELL_COMMAND_TOOL_NAME,
        model_response={
            "id": "response-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                    }
                }
            ],
            "created": 0,
            "model": model,
            "object": "chat.completion",
        },
        total_calls_in_response=1,
        tool_result_format="codex",
    )


@pytest.fixture(scope="module")
def action_server() -> ModuleType:
    from openhands.runtime import action_execution_server

    return action_execution_server


@pytest.fixture
def executor(action_server: ModuleType):
    action_executor = object.__new__(action_server.ActionExecutor)
    action_executor.bash_session = SimpleNamespace(
        cwd="/workspace",
        execute=MagicMock(name="bash_execute"),
    )
    return action_executor


def _run_shell(
    action_server: ModuleType,
    executor,
    action: CmdRunAction,
    raw_observation,
    *,
    started_at: float = 10.0,
    finished_at: float = 11.25,
):
    execute = AsyncMock(return_value=raw_observation)
    monotonic = MagicMock(side_effect=[started_at, finished_at])
    with (
        patch.object(action_server, "call_sync_from_async", execute),
        patch.object(
            action_server,
            "time",
            SimpleNamespace(monotonic=monotonic),
        ),
    ):
        result = asyncio.run(executor.run(action))

    execute.assert_awaited_once_with(executor.bash_session.execute, action)
    assert monotonic.call_count == 2
    return result


def test_codex_shell_runtime_formats_exact_body_and_ignores_oh_annotations(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="example-command")
    action.tool_call_metadata = _metadata()
    raw = CmdOutputObservation(
        content="  leading\ntrailing  \n",
        command=action.command,
        metadata=CmdOutputMetadata(
            exit_code=7,
            prefix="[OpenHands prefix]\n",
            suffix="\n[The command completed with exit code 7.]",
        ),
        max_content_size=None,
    )

    result = _run_shell(action_server, executor, action, raw)

    assert result is raw
    assert result.content == (
        "Exit code: 7\nWall time: 1.3 seconds\nOutput:\n  leading\ntrailing  \n"
    )


def test_codex_shell_runtime_formats_timeout_exactly(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="sleep 2")
    action.set_hard_timeout(2.0)
    action.tool_call_metadata = _metadata(model="gpt-5.2")
    raw = CmdOutputObservation(
        content="partial",
        command=action.command,
        metadata=CmdOutputMetadata(
            suffix=(
                "\n[The command timed out after 2.0 seconds. "
                "The command is still running.]"
            )
        ),
        max_content_size=None,
    )

    result = _run_shell(
        action_server,
        executor,
        action,
        raw,
        finished_at=12.0406,
    )

    assert result.content == (
        "Exit code: 124\n"
        "Wall time: 2 seconds\n"
        "Output:\n"
        "command timed out after 2040 milliseconds\n"
        "partial"
    )


@pytest.mark.parametrize(
    ("model", "raw_output"),
    [
        ("gpt-5.2", "b" * 10_001),
        ("gpt-5.6-sol", "t" * 40_001),
    ],
    ids=("byte-policy", "token-policy"),
)
def test_codex_shell_runtime_uses_model_truncation_policy(
    action_server: ModuleType,
    executor,
    model: str,
    raw_output: str,
) -> None:
    action = CmdRunAction(command="large-output")
    action.tool_call_metadata = _metadata(model=model)
    raw = CmdOutputObservation(
        content=raw_output,
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=0),
        max_content_size=None,
    )

    result = _run_shell(
        action_server,
        executor,
        action,
        raw,
        finished_at=10.0,
    )

    assert result.content == format_shell_output(
        raw_output,
        exit_code=0,
        duration_seconds=0,
        model_name=model,
    )
    assert "[... Observation truncated due to length ...]" not in result.content


def test_codex_shell_runtime_preserves_empty_output(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="true")
    action.tool_call_metadata = _metadata()
    raw = CmdOutputObservation(
        content="",
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=0),
        max_content_size=None,
    )

    result = _run_shell(
        action_server,
        executor,
        action,
        raw,
        finished_at=10.0,
    )

    assert result.content == "Exit code: 0\nWall time: 0 seconds\nOutput:\n"


def test_codex_shell_runtime_leaves_execution_errors_unchanged(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="rejected")
    action.tool_call_metadata = _metadata()
    raw = ErrorObservation("execution error: denied")

    result = _run_shell(action_server, executor, action, raw)

    assert result is raw
    assert result.content == "execution error: denied"


def test_unmarked_shell_runtime_keeps_generic_observation(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="legacy-command")
    raw = CmdOutputObservation(
        content="legacy output",
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=3),
    )

    result = _run_shell(action_server, executor, action, raw)

    assert result is raw
    assert result.content == "legacy output"


def test_codex_raw_shell_collection_preserves_output_whitespace() -> None:
    from openhands.runtime.utils.bash import BashSession, _remove_command_prefix

    assert (
        _remove_command_prefix(
            "example-command\r\n  leading\ntrailing  \n",
            "example-command",
            preserve_output_whitespace=True,
        )
        == "  leading\ntrailing  \n"
    )


def test_codex_tmux_capture_requests_and_preserves_trailing_spaces() -> None:
    from openhands.runtime.utils.bash import BashSession

    session = object.__new__(BashSession)
    session.pane = MagicMock()
    session.pane.cmd.return_value.stdout = ["alpha  ", "beta "]

    assert session._get_pane_content(preserve_trailing=True) == "alpha  \nbeta "
    session.pane.cmd.assert_called_once_with(
        "capture-pane",
        "-J",
        "-N",
        "-pS",
        "-",
    )

    session.pane.cmd.reset_mock()
    session.pane.cmd.return_value.stdout = ["alpha  ", "beta "]
    assert session._get_pane_content() == "alpha\nbeta"
    session.pane.cmd.assert_called_once_with(
        "capture-pane",
        "-J",
        "-pS",
        "-",
    )

    session = object.__new__(BashSession)
    session.prev_output = ""
    metadata = CmdOutputMetadata()
    assert (
        session._get_command_output(
            "example-command",
            "example-command\n  leading\ntrailing  \n",
            metadata,
            preserve_trailing=True,
        )
        == "  leading\ntrailing  \n"
    )
