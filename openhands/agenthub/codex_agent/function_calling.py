"""Function calling implementation for Codex agent."""

import json

from litellm import ModelResponse

from openhands.agenthub.codeact_agent.function_calling import (
    combine_thought,
    set_security_risk,
)
from openhands.agenthub.codex_agent.tools.apply_patch import ApplyPatchTool
from openhands.agenthub.codex_agent.tools.finish import FinishTool
from openhands.agenthub.codex_agent.tools.grep_files import GrepFilesTool
from openhands.agenthub.codex_agent.tools.list_dir import ListDirTool
from openhands.agenthub.codex_agent.tools.read_file import ReadFileTool
from openhands.agenthub.codex_agent.tools.shell_command import ShellCommandTool
from openhands.agenthub.codex_agent.tools.update_plan import UpdatePlanTool
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
    LLMContextWindowExceedError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
    CmdRunAction,
    MessageAction,
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


def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    """Convert LLM response to OpenHands actions for Codex agent."""
    actions: list[Action] = []
    assert len(response.choices) == 1, "Only one choice is supported for now"
    choice = response.choices[0]
    assistant_msg = choice.message

    # Check if both content and tool_calls are None
    has_content = assistant_msg.content is not None
    has_tool_calls = hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls

    if not has_content and not has_tool_calls:
        raise LLMContextWindowExceedError(
            "LLM returned empty response with no content and no tool calls. "
            "This indicates the context length limit has been exceeded."
        )

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

                # ================================================
                # Shell Command
                # ================================================
                if tool_call.function.name == ShellCommandTool['function']['name']:
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {tool_call.function.name}'
                        )

                    is_input = arguments.get('is_input', 'false') == 'true'
                    action = CmdRunAction(
                        command=arguments['command'],
                        is_input=is_input,
                    )
                    if 'timeout_ms' in arguments:
                        try:
                            timeout_s = min(float(arguments['timeout_ms']) / 1000.0, 600)
                            action.set_hard_timeout(timeout_s)
                        except ValueError as e:
                            raise FunctionCallValidationError(
                                f"Invalid value passed to 'timeout_ms' argument: {arguments['timeout_ms']}"
                            ) from e
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
                # Finish
                # ================================================
                elif tool_call.function.name == FinishTool['function']['name']:
                    action = AgentFinishAction(
                        final_thought=arguments.get('message', ''),
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
                action = MessageAction(
                    content=str(e),
                    wait_for_response=False,
                )

            # Add thought to first action
            if i == 0 and not isinstance(action, (ValidationFailureAction, MessageAction)):
                action = combine_thought(action, thought)

            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                model_response=response,
                total_calls_in_response=len(assistant_msg.tool_calls),
            )
            actions.append(action)
    else:
        message_action = MessageAction(
            content=str(assistant_msg.content) if assistant_msg.content else "",
            wait_for_response=True,
        )
        message_action.tool_call_metadata = ToolCallMetadata(
            model_response=response,
            total_calls_in_response=0,
        )
        actions.append(message_action)

    # Add response id to actions
    for action in actions:
        action.response_id = response.id

    assert len(actions) >= 1
    return actions
