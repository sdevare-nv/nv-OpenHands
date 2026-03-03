"""Function calling implementation for OpenCode agent."""

import json

from litellm import ModelResponse

from openhands.agenthub.codeact_agent.function_calling import (
    combine_thought,
    set_security_risk,
)
from openhands.agenthub.codeact_agent.tools import create_cmd_run_tool
from openhands.agenthub.opencode_agent.tools.apply_patch import APPLY_PATCH_TOOL_NAME
from openhands.agenthub.opencode_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.opencode_agent.tools.edit import EditTool
from openhands.agenthub.opencode_agent.tools.finish import FinishTool
from openhands.agenthub.opencode_agent.tools.glob import GlobTool
from openhands.agenthub.opencode_agent.tools.grep import GrepTool
from openhands.agenthub.opencode_agent.tools.list_dir import ListDirTool
from openhands.agenthub.opencode_agent.tools.question import QUESTION_TOOL_NAME
from openhands.agenthub.opencode_agent.tools.read import ReadTool
from openhands.agenthub.opencode_agent.tools.think import ThinkTool
from openhands.agenthub.opencode_agent.tools.todo import (
    TODO_READ_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
)
from openhands.agenthub.opencode_agent.tools.write import WriteTool
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
    LLMContextWindowExceedError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
    AgentThinkAction,
    ApplyPatchAction,
    CmdRunAction,
    FileEditAction,
    GlobAction,
    GrepAction,
    ListDirAction,
    MessageAction,
    OpenCodeReadAction,
    OpenCodeWriteAction,
    QuestionAction,
    TodoReadAction,
    TodoWriteAction,
    ValidationFailureAction,
)
from openhands.events.action.mcp import MCPAction
from openhands.events.event import FileEditSource
from openhands.events.tool import ToolCallMetadata


def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    """Convert LLM response to OpenHands actions for OpenCode agent."""
    actions: list[Action] = []
    assert len(response.choices) == 1, "Only one choice is supported for now"
    choice = response.choices[0]
    assistant_msg = choice.message

    # Check if both content and tool_calls are None
    has_content = assistant_msg.content is not None
    has_tool_calls = hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls

    if not has_content and not has_tool_calls:
        raise LLMContextWindowExceedError(
            "LLM returned empty response with no content and no tool calls. This indicates the context length limit has been exceeded."
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
            logger.debug(f'Tool call in opencode function_calling.py: {tool_call}')

            try:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.decoder.JSONDecodeError as e:
                    raise FunctionCallValidationError(
                        f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                    ) from e

                # ================================================
                # Bash/CmdRun
                # ================================================
                if tool_call.function.name == create_cmd_run_tool()['function']['name']:
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {tool_call.function.name}'
                        )
                    is_input = arguments.get('is_input', 'false') == 'true'
                    action = CmdRunAction(command=arguments['command'], is_input=is_input)
                    if 'timeout' in arguments:
                        try:
                            action.set_hard_timeout(min(float(arguments['timeout']), 600))
                        except ValueError as e:
                            raise FunctionCallValidationError(
                                f"Invalid float passed to 'timeout' argument: {arguments['timeout']}"
                            ) from e
                    set_security_risk(action, arguments)

                # ================================================
                # Read
                # ================================================
                elif tool_call.function.name == ReadTool["function"]["name"]:
                    if "file_path" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "file_path" in tool call {tool_call.function.name}'
                        )
                    action = OpenCodeReadAction(
                        path=arguments["file_path"],
                        offset=arguments.get("offset", 0),
                        limit=arguments.get("limit", 2000),
                    )

                # ================================================
                # Write
                # ================================================
                elif tool_call.function.name == WriteTool["function"]["name"]:
                    if "file_path" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "file_path" in tool call {tool_call.function.name}'
                        )
                    if "content" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "content" in tool call {tool_call.function.name}'
                        )
                    action = OpenCodeWriteAction(
                        path=arguments["file_path"],
                        content=arguments["content"],
                    )

                # ================================================
                # Edit
                # ================================================
                elif tool_call.function.name == EditTool["function"]["name"]:
                    if "file_path" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "file_path" in tool call {tool_call.function.name}'
                        )
                    if "old_string" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "old_string" in tool call {tool_call.function.name}'
                        )
                    if "new_string" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "new_string" in tool call {tool_call.function.name}'
                        )
                    action = FileEditAction(
                        path=arguments["file_path"],
                        command="str_replace",
                        old_str=arguments["old_string"],
                        new_str=arguments["new_string"],
                        impl_source=FileEditSource.OH_ACI,
                    )

                # ================================================
                # Glob
                # ================================================
                elif tool_call.function.name == GlobTool["function"]["name"]:
                    if "pattern" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "pattern" in tool call {tool_call.function.name}'
                        )
                    action = GlobAction(
                        pattern=arguments["pattern"],
                        path=arguments.get("path", "."),
                    )

                # ================================================
                # Grep
                # ================================================
                elif tool_call.function.name == GrepTool["function"]["name"]:
                    if "pattern" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "pattern" in tool call {tool_call.function.name}'
                        )
                    action = GrepAction(
                        pattern=arguments["pattern"],
                        path=arguments.get("path", "."),
                        include=arguments.get("include", ""),
                    )

                # ================================================
                # ListDir
                # ================================================
                elif tool_call.function.name == ListDirTool["function"]["name"]:
                    action = ListDirAction(
                        path=arguments.get("path", "."),
                        ignore=arguments.get("ignore", []),
                    )

                # ================================================
                # Question
                # ================================================
                elif tool_call.function.name == QUESTION_TOOL_NAME:
                    if "questions" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "questions" in tool call {tool_call.function.name}'
                        )
                    action = QuestionAction(
                        questions=arguments["questions"],
                    )

                # ================================================
                # ApplyPatch
                # ================================================
                elif tool_call.function.name == APPLY_PATCH_TOOL_NAME:
                    if "patchText" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "patchText" in tool call {tool_call.function.name}'
                        )
                    action = ApplyPatchAction(
                        patchText=arguments["patchText"],
                    )

                # ================================================
                # TodoRead
                # ================================================
                elif tool_call.function.name == TODO_READ_TOOL_NAME:
                    action = TodoReadAction()

                # ================================================
                # TodoWrite
                # ================================================
                elif tool_call.function.name == TODO_WRITE_TOOL_NAME:
                    if "todos" not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "todos" in tool call {tool_call.function.name}'
                        )
                    action = TodoWriteAction(
                        todos=arguments["todos"],
                    )

                # ================================================
                # Think
                # ================================================
                elif tool_call.function.name == ThinkTool['function']['name']:
                    action = AgentThinkAction(thought=arguments.get('thought', ''))

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
                        f'Tool {tool_call.function.name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool.'
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
