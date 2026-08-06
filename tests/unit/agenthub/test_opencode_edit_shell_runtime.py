"""Runtime integration tests for OpenCode edit and shell result bodies."""

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.events.action import CmdRunAction, FileEditAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileEditObservation,
)
from openhands.events.observation.commands import CmdOutputMetadata
from openhands.events.tool import ToolCallMetadata


def _tool_call_metadata(
    *,
    function_name: str,
    tool_result_format: str | None = "opencode",
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
                            "id": "call-1",
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
    return ToolCallMetadata(
        tool_call_id="call-1",
        function_name=function_name,
        model_response=model_response,
        total_calls_in_response=1,
        tool_result_format=tool_result_format,
    )


@pytest.fixture(scope="module")
def action_server() -> ModuleType:
    from openhands.runtime import action_execution_server

    return action_execution_server


@pytest.fixture
def executor(action_server: ModuleType, tmp_path: Path):
    """Build a real ActionExecutor instance without starting runtime services."""
    action_executor = object.__new__(action_server.ActionExecutor)
    action_executor._initial_cwd = str(tmp_path)
    action_executor.bash_session = SimpleNamespace(
        cwd=str(tmp_path),
        execute=MagicMock(name="bash_execute"),
    )
    action_executor.file_editor = MagicMock(
        side_effect=AssertionError("marked edits must bypass the legacy editor"),
    )
    return action_executor


@pytest.fixture
def saved_shell_output_dir(
    action_server: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[Path]:
    """Keep saved shell output scoped to the test and remove it at teardown."""
    output_dir = tmp_path / "saved-shell-output"
    output_dir.mkdir()
    real_named_temporary_file = tempfile.NamedTemporaryFile

    def named_temporary_file(*args, **kwargs):
        kwargs["dir"] = output_dir
        return real_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(
        action_server,
        "tempfile",
        SimpleNamespace(NamedTemporaryFile=named_temporary_file),
    )
    yield output_dir

    for output_path in output_dir.glob("opencode-tool-*"):
        if output_path.is_file():
            output_path.unlink()


def _marked_edit_action(
    path: str,
    *,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> FileEditAction:
    action = FileEditAction(
        path=path,
        command="str_replace",
        old_str=old_string,
        new_str=new_string,
        replace_all=replace_all,
    )
    action.tool_call_metadata = _tool_call_metadata(function_name="edit")
    return action


def _marked_shell_action(
    command: str = "example-command",
    *,
    timeout: float | None = None,
) -> CmdRunAction:
    action = CmdRunAction(command=command)
    if timeout is not None:
        action.set_hard_timeout(timeout)
    action.tool_call_metadata = _tool_call_metadata(function_name="bash")
    return action


def _run_shell(
    action_server: ModuleType,
    executor,
    action: CmdRunAction,
    raw_observation: CmdOutputObservation,
):
    call = AsyncMock(return_value=raw_observation)
    with patch.object(action_server, "call_sync_from_async", call):
        result = asyncio.run(executor.run(action))

    call.assert_awaited_once_with(executor.bash_session.execute, action)
    return result


def test_marked_edit_returns_concise_success_and_edits_real_file(
    executor,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before\n", encoding="utf-8")
    action = _marked_edit_action(
        "target.txt",
        old_string="before",
        new_string="after",
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, FileEditObservation)
    assert observation.content == "Edit applied successfully."
    assert target.read_text(encoding="utf-8") == "after\n"
    executor.file_editor.assert_not_called()


def test_marked_edit_rejects_identical_strings(executor) -> None:
    action = _marked_edit_action(
        "missing.txt",
        old_string="same",
        new_string="same",
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, ErrorObservation)
    assert (
        observation.content
        == "No changes to apply: oldString and newString are identical."
    )


def test_marked_edit_reports_missing_file(executor, tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"
    action = _marked_edit_action(
        "missing.txt",
        old_string="before",
        new_string="after",
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, ErrorObservation)
    assert observation.content == f"File {target} not found"


def test_marked_edit_rejects_directory(executor, tmp_path: Path) -> None:
    target = tmp_path / "directory"
    target.mkdir()
    action = _marked_edit_action(
        "directory",
        old_string="before",
        new_string="after",
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, ErrorObservation)
    assert observation.content == f"Path is a directory, not a file: {target}"


def test_marked_edit_rejects_empty_old_string_for_existing_file(
    executor,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("existing", encoding="utf-8")
    action = _marked_edit_action(
        "target.txt",
        old_string="",
        new_string="replacement",
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, ErrorObservation)
    assert observation.content == (
        "oldString cannot be empty when editing an existing file. "
        "Provide the exact text to replace, or use write for an intentional "
        "full-file replacement."
    )
    assert target.read_text(encoding="utf-8") == "existing"


def test_marked_edit_reports_no_match(executor, tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("actual text\n", encoding="utf-8")
    action = _marked_edit_action(
        "target.txt",
        old_string="absent text",
        new_string="replacement",
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, ErrorObservation)
    assert observation.content == (
        "Could not find oldString in the file. It must match exactly, "
        "including whitespace, indentation, and line endings."
    )
    assert target.read_text(encoding="utf-8") == "actual text\n"


def test_marked_edit_reports_multiple_matches(executor, tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("repeat\nrepeat\n", encoding="utf-8")
    action = _marked_edit_action(
        "target.txt",
        old_string="repeat",
        new_string="replacement",
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, ErrorObservation)
    assert observation.content == (
        "Found multiple matches for oldString. Provide more surrounding context "
        "to make the match unique."
    )
    assert target.read_text(encoding="utf-8") == "repeat\nrepeat\n"


def test_marked_edit_does_not_replace_disproportionate_anchor_block(
    executor,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    original = "start\nmatch1\nmatch2\nextra1\nextra2\nextra3\nend"
    target.write_text(original, encoding="utf-8")
    action = _marked_edit_action(
        "target.txt",
        old_string="start\nmatch1\nmatch2\nend",
        new_string="replacement",
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, ErrorObservation)
    assert observation.content == (
        "Could not find oldString in the file. It must match exactly, "
        "including whitespace, indentation, and line endings."
    )
    assert target.read_text(encoding="utf-8") == original


def test_marked_edit_unescapes_double_backslash_once(
    executor,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("prefix x\\n y suffix", encoding="utf-8")
    action = _marked_edit_action(
        "target.txt",
        old_string="x\\\\n y",
        new_string="done",
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, FileEditObservation)
    assert observation.content == "Edit applied successfully."
    assert target.read_text(encoding="utf-8") == "prefix done suffix"


def test_marked_edit_replace_all_preserves_crlf(
    executor,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"header\r\nneedle\r\nneedle\r\nfooter\r\n")
    action = _marked_edit_action(
        "target.txt",
        old_string="needle\n",
        new_string="changed\n",
        replace_all=True,
    )

    observation = asyncio.run(executor.edit(action))

    assert isinstance(observation, FileEditObservation)
    assert observation.content == "Edit applied successfully."
    assert target.read_bytes() == b"header\r\nchanged\r\nchanged\r\nfooter\r\n"


def test_marked_shell_returns_raw_output_without_exit_wrapper(
    action_server: ModuleType,
    executor,
) -> None:
    action = _marked_shell_action()
    raw_observation = CmdOutputObservation(
        content="normal output\n",
        command=action.command,
        metadata=CmdOutputMetadata(
            exit_code=7,
            suffix="\n[The command completed with exit code 7.]",
        ),
        max_content_size=None,
    )

    observation = _run_shell(action_server, executor, action, raw_observation)

    assert observation is raw_observation
    assert observation.content == "normal output\n"


def test_marked_shell_formats_empty_output(
    action_server: ModuleType,
    executor,
) -> None:
    action = _marked_shell_action()
    raw_observation = CmdOutputObservation(
        content="",
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=0),
        max_content_size=None,
    )

    observation = _run_shell(action_server, executor, action, raw_observation)

    assert observation.content == "(no output)"


def test_marked_shell_formats_hard_timeout_in_milliseconds(
    action_server: ModuleType,
    executor,
) -> None:
    action = _marked_shell_action(command="sleep 10", timeout=1.25)
    raw_observation = CmdOutputObservation(
        content="partial output",
        command=action.command,
        metadata=CmdOutputMetadata(
            suffix=(
                "\n[The command timed out after 1.25 seconds. "
                "The command is still running.]"
            ),
        ),
        max_content_size=None,
    )

    observation = _run_shell(action_server, executor, action, raw_observation)

    assert observation.content == (
        "partial output\n\n"
        "<shell_metadata>\n"
        "shell tool terminated command after exceeding timeout 1250 ms. "
        "If this command is expected to take longer and is not waiting for "
        "interactive input, retry with a larger timeout value in milliseconds.\n"
        "</shell_metadata>"
    )


def test_marked_shell_truncates_tail_and_saves_full_output(
    action_server: ModuleType,
    executor,
    saved_shell_output_dir: Path,
) -> None:
    lines = [f"{line_number:04d}:" + "x" * 20 for line_number in range(2_101)]
    raw_output = "\n".join(lines)
    assert len(raw_output.encode("utf-8")) > 50 * 1024
    assert len(lines) > 2_000
    action = _marked_shell_action(command="large-output")
    raw_observation = CmdOutputObservation(
        content=raw_output,
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=0),
        max_content_size=None,
    )

    observation = _run_shell(action_server, executor, action, raw_observation)

    saved_path_prefix = "...output truncated...\n\nFull output saved to: "
    assert observation.content.startswith(saved_path_prefix)
    path_end = observation.content.index("\n\n", len(saved_path_prefix))
    saved_path = Path(observation.content[len(saved_path_prefix) : path_end])
    expected_tail = "\n".join(lines[-1_969:])
    assert saved_path.parent.resolve() == saved_shell_output_dir.resolve()
    assert saved_path.name.startswith("opencode-tool-")
    assert saved_path.read_text(encoding="utf-8") == raw_output
    assert observation.content == (
        f"{saved_path_prefix}{saved_path}\n\n{expected_tail}"
    )


def test_unmarked_shell_keeps_generic_observation(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="legacy-command")
    action.tool_call_metadata = _tool_call_metadata(
        function_name="execute_bash",
        tool_result_format=None,
    )
    raw_observation = CmdOutputObservation(
        content="legacy output",
        command=action.command,
        metadata=CmdOutputMetadata(
            exit_code=3,
            suffix="\n[The command completed with exit code 3.]",
        ),
    )

    observation = _run_shell(action_server, executor, action, raw_observation)

    assert observation is raw_observation
    assert observation.content == "legacy output"
    assert observation.to_agent_observation() == (
        "legacy output\n"
        "[The command completed with exit code 3.]\n"
        "[Command finished with exit code 3]"
    )


def test_marked_shell_prefix_removal_preserves_output_whitespace() -> None:
    from openhands.runtime.utils.bash import _remove_command_prefix

    command = "printf '  x  \\n\\n'"
    captured = command + "\n  x  \n\n"

    assert (
        _remove_command_prefix(
            captured,
            command,
            preserve_output_whitespace=True,
        )
        == "  x  \n\n"
    )
