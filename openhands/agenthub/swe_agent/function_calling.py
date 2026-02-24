"""Function calling implementation for SWE-Agent.

Maps SWE-Agent tool calls (bash, str_replace_editor, submit) to OpenHands actions.
"""

import json

from litellm import ModelResponse

from openhands.agenthub.codeact_agent.function_calling import combine_thought
from openhands.agenthub.swe_agent.tools.bash import SWE_AGENT_BASH_TOOL_NAME
from openhands.agenthub.swe_agent.tools.str_replace_editor import (
    SWE_AGENT_STR_REPLACE_EDITOR_TOOL_NAME,
    StrReplaceEditorTool,
)
from openhands.agenthub.swe_agent.tools.submit import SWE_AGENT_SUBMIT_TOOL_NAME
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
    FileEditAction,
    FileReadAction,
    MessageAction,
    ValidationFailureAction,
)
from openhands.events.action.mcp import MCPAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.tool import ToolCallMetadata


def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1
    choice = response.choices[0]
    assistant_msg = choice.message

    has_content = assistant_msg.content is not None
    has_tool_calls = hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls

    if not has_content and not has_tool_calls:
        raise LLMContextWindowExceedError(
            'LLM returned empty response with no content and no tool calls.'
        )

    if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
        # Extract thought from content
        thought = ''
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg['type'] == 'text':
                    thought += msg['text']

        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            logger.debug(f'SWE-Agent tool call: {tool_call}')

            try:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.decoder.JSONDecodeError as e:
                    raise FunctionCallValidationError(
                        f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                    ) from e

                # ================================================
                # Bash tool
                # ================================================
                if tool_call.function.name == SWE_AGENT_BASH_TOOL_NAME:
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {tool_call.function.name}'
                        )
                    action = CmdRunAction(command=arguments['command'])

                # ================================================
                # str_replace_editor tool
                # ================================================
                elif tool_call.function.name == SWE_AGENT_STR_REPLACE_EDITOR_TOOL_NAME:
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {tool_call.function.name}'
                        )
                    if 'path' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "path" in tool call {tool_call.function.name}'
                        )

                    path = arguments['path']
                    command = arguments['command']
                    other_kwargs = {
                        k: v
                        for k, v in arguments.items()
                        if k not in ['command', 'path']
                    }

                    if command == 'view':
                        action = FileReadAction(
                            path=path,
                            impl_source=FileReadSource.OH_ACI,
                            view_range=other_kwargs.get('view_range', None),
                        )
                    else:
                        if 'view_range' in other_kwargs:
                            other_kwargs.pop('view_range')

                        # Filter to valid str_replace_editor params
                        valid_params = set(
                            StrReplaceEditorTool['function']['parameters'][
                                'properties'
                            ].keys()
                        )
                        valid_kwargs_for_editor = {
                            k: v
                            for k, v in other_kwargs.items()
                            if k in valid_params
                        }

                        action = FileEditAction(
                            path=path,
                            command=command,
                            impl_source=FileEditSource.OH_ACI,
                            **valid_kwargs_for_editor,
                        )

                # ================================================
                # Submit tool
                # ================================================
                elif tool_call.function.name == SWE_AGENT_SUBMIT_TOOL_NAME:
                    action = AgentFinishAction()

                # ================================================
                # MCP tools
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
                action = ValidationFailureAction(
                    function_name=tool_call.function.name,
                    error_message=str(e),
                    thought=thought if i == 0 else '',
                )

            except FunctionCallNotExistsError as e:
                action = MessageAction(
                    content=str(e),
                    wait_for_response=False,
                )

            # Add thought to first action
            if i == 0 and not isinstance(
                action, (ValidationFailureAction, MessageAction)
            ):
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
            content=str(assistant_msg.content) if assistant_msg.content else '',
            wait_for_response=True,
        )
        message_action.tool_call_metadata = ToolCallMetadata(
            model_response=response,
            total_calls_in_response=0,
        )
        actions.append(message_action)

    for action in actions:
        action.response_id = response.id

    assert len(actions) >= 1
    return actions
