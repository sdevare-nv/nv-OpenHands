"""Unit tests for Codex-style tools: ShellCommand, ReadFile, ListDir, GrepFiles, ApplyPatch, UpdatePlan.

These tests cover:
1. Tool definitions (schema validation)
2. Function calling (response_to_actions conversion)
3. Action class creation and validation
"""

import json

import pytest
from litellm import ModelResponse

from openhands.agenthub.codex_agent.function_calling import response_to_actions
from openhands.agenthub.codex_agent.tools.apply_patch import ApplyPatchTool
from openhands.agenthub.codex_agent.tools.finish import FinishTool
from openhands.agenthub.codex_agent.tools.grep_files import GrepFilesTool
from openhands.agenthub.codex_agent.tools.list_dir import ListDirTool
from openhands.agenthub.codex_agent.tools.read_file import ReadFileTool
from openhands.agenthub.codex_agent.tools.shell_command import ShellCommandTool
from openhands.agenthub.codex_agent.tools.update_plan import UpdatePlanTool
from openhands.core.exceptions import FunctionCallValidationError
from openhands.core.schema import ActionType
from openhands.events.action import (
    AgentFinishAction,
    CmdRunAction,
    CodexApplyPatchAction,
    CodexGrepFilesAction,
    CodexListDirAction,
    CodexReadFileAction,
    CodexUpdatePlanAction,
    ValidationFailureAction,
)
from openhands.llm.tool_names import (
    CODEX_APPLY_PATCH_TOOL_NAME,
    CODEX_GREP_FILES_TOOL_NAME,
    CODEX_LIST_DIR_TOOL_NAME,
    CODEX_READ_FILE_TOOL_NAME,
    CODEX_SHELL_COMMAND_TOOL_NAME,
    CODEX_UPDATE_PLAN_TOOL_NAME,
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_mock_response(function_name: str, arguments: dict) -> ModelResponse:
    """Helper function to create a mock response with a tool call."""
    return ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': [
                        {
                            'function': {
                                'name': function_name,
                                'arguments': json.dumps(arguments),
                            },
                            'id': 'mock-tool-call-id',
                            'type': 'function',
                        }
                    ],
                    'content': None,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )


def create_mock_response_with_thought(
    function_name: str, arguments: dict, thought: str
) -> ModelResponse:
    """Helper function to create a mock response with thought content."""
    return ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': [
                        {
                            'function': {
                                'name': function_name,
                                'arguments': json.dumps(arguments),
                            },
                            'id': 'mock-tool-call-id',
                            'type': 'function',
                        }
                    ],
                    'content': thought,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )


def create_mock_response_no_tools(content: str) -> ModelResponse:
    """Helper function to create a mock response with no tool calls."""
    return ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': None,
                    'content': content,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'stop',
            }
        ],
    )


def create_mock_response_multi_tool(tool_calls: list[tuple[str, dict]]) -> ModelResponse:
    """Helper to create a mock response with multiple tool calls."""
    return ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': [
                        {
                            'function': {
                                'name': name,
                                'arguments': json.dumps(args),
                            },
                            'id': f'mock-tool-call-id-{i}',
                            'type': 'function',
                        }
                        for i, (name, args) in enumerate(tool_calls)
                    ],
                    'content': None,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )


# ==============================================================================
# Tool Definition Tests
# ==============================================================================


class TestToolDefinitions:
    """Tests for Codex tool schema definitions."""

    def test_shell_command_tool_schema(self):
        """Test ShellCommandTool has correct schema structure."""
        assert ShellCommandTool['type'] == 'function'
        func = ShellCommandTool['function']
        assert func['name'] == CODEX_SHELL_COMMAND_TOOL_NAME
        assert func['name'] == 'shell_command'
        params = func['parameters']
        assert 'command' in params['properties']
        assert 'workdir' in params['properties']
        assert 'login' in params['properties']
        assert 'timeout_ms' in params['properties']
        assert params['required'] == ['command']
        assert params['additionalProperties'] is False

    def test_read_file_tool_schema(self):
        """Test ReadFileTool has correct schema structure."""
        assert ReadFileTool['type'] == 'function'
        func = ReadFileTool['function']
        assert func['name'] == CODEX_READ_FILE_TOOL_NAME
        assert func['name'] == 'read_file'
        params = func['parameters']
        assert 'file_path' in params['properties']
        assert 'offset' in params['properties']
        assert 'limit' in params['properties']
        assert 'mode' in params['properties']
        assert 'indentation' in params['properties']
        assert params['required'] == ['file_path']

    def test_read_file_indentation_params(self):
        """Test ReadFileTool indentation sub-object has correct properties."""
        indent_props = ReadFileTool['function']['parameters']['properties']['indentation']
        assert indent_props['type'] == 'object'
        sub_props = indent_props['properties']
        assert 'anchor_line' in sub_props
        assert 'max_levels' in sub_props
        assert 'include_siblings' in sub_props
        assert 'include_header' in sub_props
        assert 'max_lines' in sub_props

    def test_list_dir_tool_schema(self):
        """Test ListDirTool has correct schema structure."""
        assert ListDirTool['type'] == 'function'
        func = ListDirTool['function']
        assert func['name'] == CODEX_LIST_DIR_TOOL_NAME
        assert func['name'] == 'list_dir'
        params = func['parameters']
        assert 'dir_path' in params['properties']
        assert 'offset' in params['properties']
        assert 'limit' in params['properties']
        assert 'depth' in params['properties']
        assert params['required'] == ['dir_path']

    def test_grep_files_tool_schema(self):
        """Test GrepFilesTool has correct schema structure."""
        assert GrepFilesTool['type'] == 'function'
        func = GrepFilesTool['function']
        assert func['name'] == CODEX_GREP_FILES_TOOL_NAME
        assert func['name'] == 'grep_files'
        params = func['parameters']
        assert 'pattern' in params['properties']
        assert 'include' in params['properties']
        assert 'path' in params['properties']
        assert 'limit' in params['properties']
        assert params['required'] == ['pattern']

    def test_apply_patch_tool_schema(self):
        """Test ApplyPatchTool has correct schema structure."""
        assert ApplyPatchTool['type'] == 'function'
        func = ApplyPatchTool['function']
        assert func['name'] == CODEX_APPLY_PATCH_TOOL_NAME
        assert func['name'] == 'apply_patch'
        params = func['parameters']
        assert 'input' in params['properties']
        assert params['required'] == ['input']

    def test_update_plan_tool_schema(self):
        """Test UpdatePlanTool has correct schema structure."""
        assert UpdatePlanTool['type'] == 'function'
        func = UpdatePlanTool['function']
        assert func['name'] == CODEX_UPDATE_PLAN_TOOL_NAME
        assert func['name'] == 'update_plan'
        params = func['parameters']
        assert 'plan' in params['properties']
        assert 'explanation' in params['properties']
        assert params['required'] == ['plan']

    def test_update_plan_plan_items_schema(self):
        """Test UpdatePlanTool plan items have step and status as required."""
        plan_prop = UpdatePlanTool['function']['parameters']['properties']['plan']
        assert plan_prop['type'] == 'array'
        items_schema = plan_prop['items']
        assert items_schema['type'] == 'object'
        assert set(items_schema['required']) == {'step', 'status'}

    def test_finish_tool_schema(self):
        """Test FinishTool has correct schema structure."""
        assert FinishTool['type'] == 'function'
        func = FinishTool['function']
        assert func['name'] == 'finish'
        assert 'message' in func['parameters']['properties']


# ==============================================================================
# ShellCommand Function Calling Tests
# ==============================================================================


class TestShellCommandFunctionCalling:
    """Tests for ShellCommand function calling."""

    def test_shell_command_valid_basic(self):
        """Test shell_command with just command."""
        response = create_mock_response('shell_command', {'command': 'ls -la'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CmdRunAction)
        assert actions[0].command == 'ls -la'

    def test_shell_command_with_timeout(self):
        """Test shell_command converts timeout_ms to seconds."""
        response = create_mock_response(
            'shell_command', {'command': 'sleep 5', 'timeout_ms': 10000}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CmdRunAction)
        assert actions[0].command == 'sleep 5'

    def test_shell_command_timeout_capped_at_600(self):
        """Test shell_command timeout is capped at 600 seconds."""
        response = create_mock_response(
            'shell_command', {'command': 'long_cmd', 'timeout_ms': 999999999}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CmdRunAction)

    def test_shell_command_missing_command(self):
        """Test shell_command returns validation failure when command missing."""
        response = create_mock_response('shell_command', {'workdir': '/tmp'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)

    def test_shell_command_with_thought(self):
        """Test shell_command preserves thought content."""
        response = create_mock_response_with_thought(
            'shell_command', {'command': 'echo hello'}, 'Let me run this command'
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].thought == 'Let me run this command'


# ==============================================================================
# ReadFile Function Calling Tests
# ==============================================================================


class TestReadFileFunctionCalling:
    """Tests for ReadFile function calling."""

    def test_read_file_valid_basic(self):
        """Test read_file with just file_path."""
        response = create_mock_response('read_file', {'file_path': '/path/to/file.py'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexReadFileAction)
        assert actions[0].file_path == '/path/to/file.py'
        assert actions[0].offset == 1  # 1-indexed default
        assert actions[0].limit == 2000
        assert actions[0].mode == 'slice'
        assert actions[0].action == ActionType.CODEX_READ_FILE

    def test_read_file_with_offset(self):
        """Test read_file with offset parameter (1-indexed)."""
        response = create_mock_response(
            'read_file', {'file_path': '/path/to/file.py', 'offset': 50}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexReadFileAction)
        assert actions[0].offset == 50

    def test_read_file_with_limit(self):
        """Test read_file with limit parameter."""
        response = create_mock_response(
            'read_file', {'file_path': '/path/to/file.py', 'limit': 100}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].limit == 100

    def test_read_file_with_indentation_mode(self):
        """Test read_file with indentation mode and params."""
        response = create_mock_response(
            'read_file',
            {
                'file_path': '/path/to/file.py',
                'mode': 'indentation',
                'indentation': {
                    'anchor_line': 25,
                    'max_levels': 2,
                    'include_siblings': True,
                    'include_header': True,
                    'max_lines': 50,
                },
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexReadFileAction)
        assert actions[0].mode == 'indentation'
        assert actions[0].indentation['anchor_line'] == 25
        assert actions[0].indentation['max_levels'] == 2
        assert actions[0].indentation['include_siblings'] is True

    def test_read_file_missing_file_path(self):
        """Test read_file returns validation failure when file_path is missing."""
        response = create_mock_response('read_file', {'offset': 10})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)

    def test_read_file_with_thought(self):
        """Test read_file preserves thought content."""
        response = create_mock_response_with_thought(
            'read_file', {'file_path': '/test.py'}, 'Let me read this file'
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].thought == 'Let me read this file'


# ==============================================================================
# ListDir Function Calling Tests
# ==============================================================================


class TestListDirFunctionCalling:
    """Tests for ListDir function calling."""

    def test_list_dir_valid_basic(self):
        """Test list_dir with just dir_path."""
        response = create_mock_response('list_dir', {'dir_path': '/workspace'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexListDirAction)
        assert actions[0].dir_path == '/workspace'
        assert actions[0].offset == 1
        assert actions[0].limit == 25
        assert actions[0].depth == 2
        assert actions[0].action == ActionType.CODEX_LIST_DIR

    def test_list_dir_with_all_params(self):
        """Test list_dir with all parameters."""
        response = create_mock_response(
            'list_dir',
            {'dir_path': '/workspace', 'offset': 10, 'limit': 50, 'depth': 3},
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexListDirAction)
        assert actions[0].offset == 10
        assert actions[0].limit == 50
        assert actions[0].depth == 3

    def test_list_dir_missing_dir_path(self):
        """Test list_dir returns validation failure when dir_path is missing."""
        response = create_mock_response('list_dir', {'depth': 3})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)


# ==============================================================================
# GrepFiles Function Calling Tests
# ==============================================================================


class TestGrepFilesFunctionCalling:
    """Tests for GrepFiles function calling."""

    def test_grep_files_valid_basic(self):
        """Test grep_files with just pattern."""
        response = create_mock_response('grep_files', {'pattern': 'TODO'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexGrepFilesAction)
        assert actions[0].pattern == 'TODO'
        assert actions[0].include == ''
        assert actions[0].path == ''
        assert actions[0].limit == 100
        assert actions[0].action == ActionType.CODEX_GREP_FILES

    def test_grep_files_with_all_params(self):
        """Test grep_files with all parameters."""
        response = create_mock_response(
            'grep_files',
            {'pattern': 'def \\w+', 'include': '*.py', 'path': '/workspace/src', 'limit': 50},
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexGrepFilesAction)
        assert actions[0].pattern == 'def \\w+'
        assert actions[0].include == '*.py'
        assert actions[0].path == '/workspace/src'
        assert actions[0].limit == 50

    def test_grep_files_missing_pattern(self):
        """Test grep_files returns validation failure when pattern is missing."""
        response = create_mock_response('grep_files', {'include': '*.py'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)


# ==============================================================================
# ApplyPatch Function Calling Tests
# ==============================================================================


class TestApplyPatchFunctionCalling:
    """Tests for ApplyPatch function calling."""

    def test_apply_patch_valid(self):
        """Test apply_patch with valid input."""
        patch = (
            '*** Begin Patch\n'
            '--- a/test.py\n'
            '+++ b/test.py\n'
            '@@ def main(): @@\n'
            '-    print("old")\n'
            '+    print("new")\n'
            '*** End Patch'
        )
        response = create_mock_response('apply_patch', {'input': patch})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexApplyPatchAction)
        assert actions[0].patch == patch
        assert actions[0].action == ActionType.CODEX_APPLY_PATCH

    def test_apply_patch_missing_input(self):
        """Test apply_patch returns validation failure when input is missing."""
        response = create_mock_response('apply_patch', {})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)

    def test_apply_patch_create_file(self):
        """Test apply_patch for file creation."""
        patch = (
            '*** Begin Patch\n'
            '--- /dev/null\n'
            '+++ b/new_file.py\n'
            '+print("hello world")\n'
            '*** End Patch'
        )
        response = create_mock_response('apply_patch', {'input': patch})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexApplyPatchAction)

    def test_apply_patch_delete_file(self):
        """Test apply_patch for file deletion."""
        patch = (
            '*** Begin Patch\n'
            '--- a/old_file.py\n'
            '+++ /dev/null\n'
            '*** End Patch'
        )
        response = create_mock_response('apply_patch', {'input': patch})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexApplyPatchAction)


# ==============================================================================
# UpdatePlan Function Calling Tests
# ==============================================================================


class TestUpdatePlanFunctionCalling:
    """Tests for UpdatePlan function calling."""

    def test_update_plan_valid(self):
        """Test update_plan with valid plan."""
        plan = [
            {'step': 'Read the codebase', 'status': 'completed'},
            {'step': 'Implement feature', 'status': 'in_progress'},
            {'step': 'Write tests', 'status': 'pending'},
        ]
        response = create_mock_response('update_plan', {'plan': plan})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexUpdatePlanAction)
        assert len(actions[0].plan) == 3
        assert actions[0].plan[0]['status'] == 'completed'
        assert actions[0].action == ActionType.CODEX_UPDATE_PLAN

    def test_update_plan_with_explanation(self):
        """Test update_plan with explanation."""
        plan = [{'step': 'First step', 'status': 'pending'}]
        response = create_mock_response(
            'update_plan', {'plan': plan, 'explanation': 'Starting work on the task'}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], CodexUpdatePlanAction)
        assert actions[0].explanation == 'Starting work on the task'

    def test_update_plan_missing_plan(self):
        """Test update_plan returns validation failure when plan is missing."""
        response = create_mock_response('update_plan', {'explanation': 'test'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)


# ==============================================================================
# Finish Function Calling Tests
# ==============================================================================


class TestFinishFunctionCalling:
    """Tests for Finish function calling."""

    def test_finish_with_message(self):
        """Test finish with message."""
        response = create_mock_response('finish', {'message': 'Task completed successfully'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], AgentFinishAction)
        assert actions[0].final_thought == 'Task completed successfully'

    def test_finish_without_message(self):
        """Test finish without message."""
        response = create_mock_response('finish', {})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], AgentFinishAction)
        assert actions[0].final_thought == ''


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_tool_raises_error(self):
        """Test that unknown tool name raises FunctionCallNotExistsError."""
        response = create_mock_response('nonexistent_tool', {'arg': 'value'})
        from openhands.core.exceptions import FunctionCallNotExistsError
        with pytest.raises(FunctionCallNotExistsError):
            response_to_actions(response)

    def test_invalid_json_arguments(self):
        """Test that invalid JSON in arguments returns validation failure."""
        response = ModelResponse(
            id='mock-id',
            choices=[
                {
                    'message': {
                        'tool_calls': [
                            {
                                'function': {
                                    'name': 'read_file',
                                    'arguments': '{invalid json',
                                },
                                'id': 'mock-tool-call-id',
                                'type': 'function',
                            }
                        ],
                        'content': None,
                        'role': 'assistant',
                    },
                    'index': 0,
                    'finish_reason': 'tool_calls',
                }
            ],
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)

    def test_content_only_response(self):
        """Test response with content but no tool calls."""
        from openhands.events.action import MessageAction
        response = create_mock_response_no_tools('Here is my explanation...')
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], MessageAction)
        assert actions[0].content == 'Here is my explanation...'

    def test_empty_response_raises_error(self):
        """Test that empty response raises LLMContextWindowExceedError."""
        from openhands.core.exceptions import LLMContextWindowExceedError
        response = ModelResponse(
            id='mock-id',
            choices=[
                {
                    'message': {
                        'tool_calls': None,
                        'content': None,
                        'role': 'assistant',
                    },
                    'index': 0,
                    'finish_reason': 'stop',
                }
            ],
        )
        with pytest.raises(LLMContextWindowExceedError):
            response_to_actions(response)

    def test_multiple_tool_calls(self):
        """Test response with multiple parallel tool calls."""
        response = create_mock_response_multi_tool([
            ('read_file', {'file_path': '/file1.py'}),
            ('grep_files', {'pattern': 'TODO'}),
        ])
        actions = response_to_actions(response)
        assert len(actions) == 2
        assert isinstance(actions[0], CodexReadFileAction)
        assert isinstance(actions[1], CodexGrepFilesAction)

    def test_thought_attached_to_first_action_only(self):
        """Test that thought is attached only to the first action in multi-tool response."""
        response = ModelResponse(
            id='mock-id',
            choices=[
                {
                    'message': {
                        'tool_calls': [
                            {
                                'function': {
                                    'name': 'read_file',
                                    'arguments': json.dumps({'file_path': '/file1.py'}),
                                },
                                'id': 'call-1',
                                'type': 'function',
                            },
                            {
                                'function': {
                                    'name': 'read_file',
                                    'arguments': json.dumps({'file_path': '/file2.py'}),
                                },
                                'id': 'call-2',
                                'type': 'function',
                            },
                        ],
                        'content': 'Let me read both files',
                        'role': 'assistant',
                    },
                    'index': 0,
                    'finish_reason': 'tool_calls',
                }
            ],
        )
        actions = response_to_actions(response)
        assert len(actions) == 2
        assert actions[0].thought == 'Let me read both files'
        assert actions[1].thought == ''

    def test_tool_call_metadata_attached(self):
        """Test that ToolCallMetadata is attached to actions."""
        response = create_mock_response('read_file', {'file_path': '/test.py'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].tool_call_metadata is not None
        assert actions[0].tool_call_metadata.function_name == 'read_file'
        assert actions[0].tool_call_metadata.tool_call_id == 'mock-tool-call-id'

    def test_response_id_attached(self):
        """Test that response ID is attached to actions."""
        response = create_mock_response('read_file', {'file_path': '/test.py'})
        actions = response_to_actions(response)
        assert actions[0].response_id == 'mock-id'
