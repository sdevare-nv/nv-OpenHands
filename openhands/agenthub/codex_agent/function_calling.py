"""Function calling implementation for Codex agent."""

import json
import math
import os

from litellm import ModelResponse

from openhands.agenthub.codeact_agent.function_calling import (
    combine_thought,
    set_security_risk,
)
from openhands.agenthub.codex_agent.tools.apply_patch import ApplyPatchTool
from openhands.agenthub.codex_agent.tools.grep_files import GrepFilesTool
from openhands.agenthub.codex_agent.tools.list_dir import ListDirTool
from openhands.agenthub.codex_agent.tools.read_file import ReadFileTool
from openhands.agenthub.codex_agent.tools.shell_command import ShellCommandTool
from openhands.agenthub.codex_agent.tools.update_plan import UpdatePlanTool
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
    CmdRunAction,
    FunctionCallNotExistsAction,
    ValidationFailureAction,
)
from openhands.events.action.codex import (
    CodexApplyPatchAction,
    CodexGrepFilesAction,
    CodexListDirAction,
    CodexReadFileAction,
    CodexUpdatePlanAction,
)
from openhands.events.action.mcp import MCPAction
from openhands.events.tool import ToolCallMetadata

_CODEX_FORMATTED_TOOL_NAMES = frozenset(
    {
        ShellCommandTool['function']['name'],
        ReadFileTool['function']['name'],
        ListDirTool['function']['name'],
        GrepFilesTool['function']['name'],
        ApplyPatchTool['function']['name'],
        UpdatePlanTool['function']['name'],
    }
)


def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    """Convert LLM response to OpenHands actions for Codex agent."""
    actions: list[Action] = []
    assert len(response.choices) == 1, "Only one choice is supported for now"
    choice = response.choices[0]
    assistant_msg = choice.message

    if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
        # Extract thought from content
        thought = ""
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg["type"] == "text":
                    thought += msg["text"]

        # Process each tool call
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            logger.debug(f'Tool call in codex function_calling.py: {tool_call}')

            try:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.decoder.JSONDecodeError as e:
                    raise FunctionCallValidationError(
                        f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                    ) from e
                if not isinstance(arguments, dict):
                    raise FunctionCallValidationError(
                        'Tool call arguments must be a JSON object'
                    )

                # ================================================
                # Shell Command
                # ================================================
                if tool_call.function.name == ShellCommandTool['function']['name']:
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {tool_call.function.name}'
                        )
                    command_arg = arguments['command']
                    if not isinstance(command_arg, str) or '\0' in command_arg:
                        raise FunctionCallValidationError(
                            f"Invalid value passed to 'command' argument: {command_arg!r}"
                        )

                    is_input_arg = arguments.get('is_input', 'false')
                    if not isinstance(is_input_arg, str) or is_input_arg not in {
                        'true',
                        'false',
                    }:
                        raise FunctionCallValidationError(
                            f"Invalid value passed to 'is_input' argument: {is_input_arg!r}"
                        )

                    workdir = arguments.get('workdir')
                    if workdir is not None and (
                        not isinstance(workdir, str)
                        or not workdir.strip()
                        or '\0' in workdir
                        or not os.path.isabs(workdir)
                    ):
                        raise FunctionCallValidationError(
                            f"Invalid value passed to 'workdir' argument: {workdir!r}"
                        )

                    login = arguments.get('login')
                    if login is not None and not isinstance(login, bool):
                        raise FunctionCallValidationError(
                            f"Invalid value passed to 'login' argument: {login!r}"
                        )
                    if is_input_arg == 'true' and (
                        'workdir' in arguments or 'login' in arguments
                    ):
                        raise FunctionCallValidationError(
                            "'workdir' and 'login' cannot be used when 'is_input' is true"
                        )

                    action = CmdRunAction(
                        command=command_arg,
                        is_input=is_input_arg == 'true',
                        cwd=workdir,
                        login=login,
                    )
                    if 'timeout_ms' in arguments:
                        timeout_ms = arguments['timeout_ms']
                        timeout_is_invalid = (
                            isinstance(timeout_ms, bool)
                            or not isinstance(timeout_ms, (int, float))
                            or timeout_ms <= 0
                            or (
                                isinstance(timeout_ms, float)
                                and not math.isfinite(timeout_ms)
                            )
                        )
                        if timeout_is_invalid:
                            raise FunctionCallValidationError(
                                f"Invalid value passed to 'timeout_ms' argument: {timeout_ms!r}"
                            )
                        timeout_s = (
                            600
                            if timeout_ms >= 600_000
                            else timeout_ms / 1000.0
                        )
                        if timeout_s <= 0:
                            raise FunctionCallValidationError(
                                f"Invalid value passed to 'timeout_ms' argument: {timeout_ms!r}"
                            )
                        action.set_hard_timeout(timeout_s)
                    set_security_risk(action, arguments)

                # ================================================
                # Read File
                # ================================================
                elif tool_call.function.name == ReadFileTool['function']['name']:
                    if 'file_path' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "file_path" in tool call {tool_call.function.name}'
                        )
                    action = CodexReadFileAction(
                        file_path=arguments['file_path'],
                        offset=arguments.get('offset', 1),
                        limit=arguments.get('limit', 2000),
                        mode=arguments.get('mode', 'slice'),
                        indentation=arguments.get('indentation', {}),
                    )

                # ================================================
                # List Dir
                # ================================================
                elif tool_call.function.name == ListDirTool['function']['name']:
                    if 'dir_path' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "dir_path" in tool call {tool_call.function.name}'
                        )
                    action = CodexListDirAction(
                        dir_path=arguments['dir_path'],
                        offset=arguments.get('offset', 1),
                        limit=arguments.get('limit', 25),
                        depth=arguments.get('depth', 2),
                    )

                # ================================================
                # Grep Files
                # ================================================
                elif tool_call.function.name == GrepFilesTool['function']['name']:
                    if 'pattern' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "pattern" in tool call {tool_call.function.name}'
                        )
                    action = CodexGrepFilesAction(
                        pattern=arguments['pattern'],
                        include=arguments.get('include', ''),
                        path=arguments.get('path', ''),
                        limit=arguments.get('limit', 100),
                    )

                # ================================================
                # Apply Patch
                # ================================================
                elif tool_call.function.name == ApplyPatchTool['function']['name']:
                    if 'input' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "input" in tool call {tool_call.function.name}'
                        )
                    action = CodexApplyPatchAction(
                        patch=arguments['input'],
                    )

                # ================================================
                # Update Plan
                # ================================================
                elif tool_call.function.name == UpdatePlanTool['function']['name']:
                    if 'plan' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "plan" in tool call {tool_call.function.name}'
                        )
                    action = CodexUpdatePlanAction(
                        plan=arguments['plan'],
                        explanation=arguments.get('explanation', ''),
                    )

                # ================================================
                # MCP
                # ================================================
                elif mcp_tool_names and tool_call.function.name in mcp_tool_names:
                    action = MCPAction(
                        name=tool_call.function.name,
                        arguments=arguments,
                    )

                else:
                    raise FunctionCallNotExistsError(
                        f'Tool {tool_call.function.name} is not registered. '
                        f'(arguments: {arguments}). '
                        f'Please check the tool name and retry with an existing tool.'
                    )

            except FunctionCallValidationError as e:
                # Convert validation errors to ValidationFailureAction
                action = ValidationFailureAction(
                    function_name=tool_call.function.name,
                    error_message=str(e),
                    thought=thought if i == 0 else '',
                )

            except FunctionCallNotExistsError as e:
                # Send error as a user message while preserving the assistant message
                action = FunctionCallNotExistsAction(
                    function_name=tool_call.function.name,
                    error_message=str(e),
                    thought=thought if i == 0 else '',
                )

            # Add thought to first action
            if i == 0 and not isinstance(action, (ValidationFailureAction, FunctionCallNotExistsAction)):
                action = combine_thought(action, thought)

            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                model_response=response,
                total_calls_in_response=len(assistant_msg.tool_calls),
                tool_result_format=(
                    'codex'
                    if tool_call.function.name in _CODEX_FORMATTED_TOOL_NAMES
                    else None
                ),
            )
            actions.append(action)
    else:
        final_thought = str(assistant_msg.content) if assistant_msg.content else ""
        finish_action = AgentFinishAction(
            final_thought=final_thought,
            thought=final_thought,
        )
        finish_action.tool_call_metadata = ToolCallMetadata(
            model_response=response,
            total_calls_in_response=0,
        )
        actions.append(finish_action)

    # Add response id to actions
    for action in actions:
        action.response_id = response.id

    assert len(actions) >= 1
    return actions
