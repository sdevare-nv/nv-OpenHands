"""Unit tests for OpenCode-style tools: Read, Write, Edit, Glob, Grep, ListDir.

These tests cover:
1. Tool definitions (schema validation)
2. Function calling (response_to_actions conversion)
3. Action class creation and validation
"""

import json
from unittest.mock import patch

import pytest
from litellm import ModelResponse

from openhands.agenthub.codeact_agent.function_calling import response_to_actions
from openhands.agenthub.codeact_agent.tools import (
    EditTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadTool,
    WriteTool,
)
from openhands.core.exceptions import FunctionCallValidationError
from openhands.core.schema import ActionType
from openhands.events.action import (
    FileEditAction,
    GlobAction,
    GrepAction,
    ListDirAction,
    OpenCodeReadAction,
    OpenCodeWriteAction,
)
from openhands.events.event import FileEditSource
from openhands.llm.tool_names import (
    EDIT_TOOL_NAME,
    GLOB_TOOL_NAME,
    GREP_TOOL_NAME,
    LIST_DIR_TOOL_NAME,
    READ_TOOL_NAME,
    WRITE_TOOL_NAME,
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


# ==============================================================================
# Tool Definition Tests
# ==============================================================================


class TestToolDefinitions:
    """Tests for tool schema definitions."""

    def test_read_tool_schema(self):
        """Test ReadTool has correct schema structure."""
        assert ReadTool['type'] == 'function'
        func = ReadTool['function']
        assert func['name'] == READ_TOOL_NAME
        assert 'parameters' in func
        params = func['parameters']
        assert 'file_path' in params['properties']
        assert 'offset' in params['properties']
        assert 'limit' in params['properties']
        assert params['required'] == ['file_path']

    def test_write_tool_schema(self):
        """Test WriteTool has correct schema structure."""
        assert WriteTool['type'] == 'function'
        func = WriteTool['function']
        assert func['name'] == WRITE_TOOL_NAME
        assert 'parameters' in func
        params = func['parameters']
        assert 'file_path' in params['properties']
        assert 'content' in params['properties']
        assert set(params['required']) == {'file_path', 'content'}

    def test_edit_tool_schema(self):
        """Test EditTool has correct schema structure."""
        assert EditTool['type'] == 'function'
        func = EditTool['function']
        assert func['name'] == EDIT_TOOL_NAME
        assert 'parameters' in func
        params = func['parameters']
        assert 'file_path' in params['properties']
        assert 'old_string' in params['properties']
        assert 'new_string' in params['properties']
        assert set(params['required']) == {'file_path', 'old_string', 'new_string'}

    def test_glob_tool_schema(self):
        """Test GlobTool has correct schema structure."""
        assert GlobTool['type'] == 'function'
        func = GlobTool['function']
        assert func['name'] == GLOB_TOOL_NAME
        assert 'parameters' in func
        params = func['parameters']
        assert 'pattern' in params['properties']
        assert 'path' in params['properties']
        assert params['required'] == ['pattern']

    def test_grep_tool_schema(self):
        """Test GrepTool has correct schema structure."""
        assert GrepTool['type'] == 'function'
        func = GrepTool['function']
        assert func['name'] == GREP_TOOL_NAME
        assert 'parameters' in func
        params = func['parameters']
        assert 'pattern' in params['properties']
        assert 'path' in params['properties']
        assert 'include' in params['properties']
        assert params['required'] == ['pattern']

    def test_list_dir_tool_schema(self):
        """Test ListDirTool has correct schema structure."""
        assert ListDirTool['type'] == 'function'
        func = ListDirTool['function']
        assert func['name'] == LIST_DIR_TOOL_NAME
        assert 'parameters' in func
        params = func['parameters']
        assert 'path' in params['properties']
        assert 'ignore' in params['properties']
        # path is optional, so required should be empty or not include path
        assert 'required' not in params or 'path' not in params.get('required', [])


# ==============================================================================
# ReadTool Function Calling Tests
# ==============================================================================


class TestReadToolFunctionCalling:
    """Tests for ReadTool function calling."""

    def test_read_tool_valid_basic(self):
        """Test ReadTool with just file_path."""
        response = create_mock_response(READ_TOOL_NAME, {'file_path': '/path/to/file.py'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], OpenCodeReadAction)
        assert actions[0].path == '/path/to/file.py'
        assert actions[0].offset == 0  # default
        assert actions[0].limit == 2000  # default
        assert actions[0].action == ActionType.OPENCODE_READ

    def test_read_tool_with_offset(self):
        """Test ReadTool with offset parameter."""
        response = create_mock_response(
            READ_TOOL_NAME, {'file_path': '/path/to/file.py', 'offset': 100}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], OpenCodeReadAction)
        assert actions[0].offset == 100
        assert actions[0].limit == 2000

    def test_read_tool_with_limit(self):
        """Test ReadTool with limit parameter."""
        response = create_mock_response(
            READ_TOOL_NAME, {'file_path': '/path/to/file.py', 'limit': 500}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], OpenCodeReadAction)
        assert actions[0].offset == 0
        assert actions[0].limit == 500

    def test_read_tool_with_all_params(self):
        """Test ReadTool with all parameters."""
        response = create_mock_response(
            READ_TOOL_NAME, {'file_path': '/path/to/file.py', 'offset': 50, 'limit': 100}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], OpenCodeReadAction)
        assert actions[0].path == '/path/to/file.py'
        assert actions[0].offset == 50
        assert actions[0].limit == 100

    def test_read_tool_missing_file_path(self):
        """Test ReadTool raises error when file_path is missing."""
        response = create_mock_response(READ_TOOL_NAME, {'offset': 10})
        with pytest.raises(FunctionCallValidationError) as exc_info:
            response_to_actions(response)
        assert 'Missing required argument "file_path"' in str(exc_info.value)

    def test_read_tool_with_thought(self):
        """Test ReadTool preserves thought content."""
        response = create_mock_response_with_thought(
            READ_TOOL_NAME, {'file_path': '/test.py'}, 'Let me read this file'
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].thought == 'Let me read this file'


# ==============================================================================
# WriteTool Function Calling Tests
# ==============================================================================


class TestWriteToolFunctionCalling:
    """Tests for WriteTool function calling."""

    def test_write_tool_valid(self):
        """Test WriteTool with valid arguments."""
        response = create_mock_response(
            WRITE_TOOL_NAME,
            {'file_path': '/path/to/file.py', 'content': 'print("hello world")'},
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], OpenCodeWriteAction)
        assert actions[0].path == '/path/to/file.py'
        assert actions[0].content == 'print("hello world")'
        assert actions[0].action == ActionType.OPENCODE_WRITE

    def test_write_tool_multiline_content(self):
        """Test WriteTool with multiline content."""
        content = 'def hello():\n    print("hello")\n\nif __name__ == "__main__":\n    hello()'
        response = create_mock_response(
            WRITE_TOOL_NAME, {'file_path': '/path/to/file.py', 'content': content}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], OpenCodeWriteAction)
        assert actions[0].content == content

    def test_write_tool_empty_content(self):
        """Test WriteTool with empty content (valid case for creating empty file)."""
        response = create_mock_response(
            WRITE_TOOL_NAME, {'file_path': '/path/to/file.py', 'content': ''}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], OpenCodeWriteAction)
        assert actions[0].content == ''

    def test_write_tool_missing_file_path(self):
        """Test WriteTool raises error when file_path is missing."""
        response = create_mock_response(WRITE_TOOL_NAME, {'content': 'some content'})
        with pytest.raises(FunctionCallValidationError) as exc_info:
            response_to_actions(response)
        assert 'Missing required argument "file_path"' in str(exc_info.value)

    def test_write_tool_missing_content(self):
        """Test WriteTool raises error when content is missing."""
        response = create_mock_response(WRITE_TOOL_NAME, {'file_path': '/path/to/file.py'})
        with pytest.raises(FunctionCallValidationError) as exc_info:
            response_to_actions(response)
        assert 'Missing required argument "content"' in str(exc_info.value)

    def test_write_tool_special_characters(self):
        """Test WriteTool handles special characters in content."""
        content = 'print("Special: \\n\\t\\"escaped\\" \'quotes\'")'
        response = create_mock_response(
            WRITE_TOOL_NAME, {'file_path': '/path/to/file.py', 'content': content}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].content == content


# ==============================================================================
# EditTool Function Calling Tests
# ==============================================================================


class TestEditToolFunctionCalling:
    """Tests for EditTool function calling."""

    def test_edit_tool_valid(self):
        """Test EditTool with valid arguments."""
        response = create_mock_response(
            EDIT_TOOL_NAME,
            {
                'file_path': '/path/to/file.py',
                'old_string': 'def foo():',
                'new_string': 'def bar():',
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], FileEditAction)
        assert actions[0].path == '/path/to/file.py'
        assert actions[0].old_str == 'def foo():'
        assert actions[0].new_str == 'def bar():'
        assert actions[0].command == 'str_replace'
        assert actions[0].impl_source == FileEditSource.OH_ACI

    def test_edit_tool_multiline_replacement(self):
        """Test EditTool with multiline strings."""
        old = 'def foo():\n    pass'
        new = 'def foo():\n    return 42'
        response = create_mock_response(
            EDIT_TOOL_NAME,
            {'file_path': '/path/to/file.py', 'old_string': old, 'new_string': new},
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].old_str == old
        assert actions[0].new_str == new

    def test_edit_tool_empty_new_string(self):
        """Test EditTool with empty new_string (deletion)."""
        response = create_mock_response(
            EDIT_TOOL_NAME,
            {
                'file_path': '/path/to/file.py',
                'old_string': 'remove this',
                'new_string': '',
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].new_str == ''

    def test_edit_tool_missing_file_path(self):
        """Test EditTool raises error when file_path is missing."""
        response = create_mock_response(
            EDIT_TOOL_NAME, {'old_string': 'old', 'new_string': 'new'}
        )
        with pytest.raises(FunctionCallValidationError) as exc_info:
            response_to_actions(response)
        assert 'Missing required argument "file_path"' in str(exc_info.value)

    def test_edit_tool_missing_old_string(self):
        """Test EditTool raises error when old_string is missing."""
        response = create_mock_response(
            EDIT_TOOL_NAME, {'file_path': '/path/to/file.py', 'new_string': 'new'}
        )
        with pytest.raises(FunctionCallValidationError) as exc_info:
            response_to_actions(response)
        assert 'Missing required argument "old_string"' in str(exc_info.value)

    def test_edit_tool_missing_new_string(self):
        """Test EditTool raises error when new_string is missing."""
        response = create_mock_response(
            EDIT_TOOL_NAME, {'file_path': '/path/to/file.py', 'old_string': 'old'}
        )
        with pytest.raises(FunctionCallValidationError) as exc_info:
            response_to_actions(response)
        assert 'Missing required argument "new_string"' in str(exc_info.value)


# ==============================================================================
# GlobTool Function Calling Tests
# ==============================================================================


class TestGlobToolFunctionCalling:
    """Tests for GlobTool function calling."""

    def test_glob_tool_valid_basic(self):
        """Test GlobTool with just pattern."""
        response = create_mock_response(GLOB_TOOL_NAME, {'pattern': '*.py'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], GlobAction)
        assert actions[0].pattern == '*.py'
        assert actions[0].path == '.'  # default
        assert actions[0].action == ActionType.GLOB

    def test_glob_tool_with_path(self):
        """Test GlobTool with path parameter."""
        response = create_mock_response(
            GLOB_TOOL_NAME, {'pattern': '*.ts', 'path': '/project/src'}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], GlobAction)
        assert actions[0].pattern == '*.ts'
        assert actions[0].path == '/project/src'

    def test_glob_tool_recursive_pattern(self):
        """Test GlobTool with recursive pattern."""
        response = create_mock_response(GLOB_TOOL_NAME, {'pattern': '**/*.test.js'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].pattern == '**/*.test.js'

    def test_glob_tool_missing_pattern(self):
        """Test GlobTool raises error when pattern is missing."""
        response = create_mock_response(GLOB_TOOL_NAME, {'path': '/some/path'})
        with pytest.raises(FunctionCallValidationError) as exc_info:
            response_to_actions(response)
        assert 'Missing required argument "pattern"' in str(exc_info.value)

    def test_glob_tool_complex_patterns(self):
        """Test GlobTool with various glob patterns."""
        patterns = [
            '*.{js,ts}',
            '[!_]*.py',
            '**/*_test.go',
            '*.py[cod]',
        ]
        for pattern in patterns:
            response = create_mock_response(GLOB_TOOL_NAME, {'pattern': pattern})
            actions = response_to_actions(response)
            assert len(actions) == 1
            assert actions[0].pattern == pattern


# ==============================================================================
# GrepTool Function Calling Tests
# ==============================================================================


class TestGrepToolFunctionCalling:
    """Tests for GrepTool function calling."""

    def test_grep_tool_valid_basic(self):
        """Test GrepTool with just pattern."""
        response = create_mock_response(GREP_TOOL_NAME, {'pattern': 'TODO'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], GrepAction)
        assert actions[0].pattern == 'TODO'
        assert actions[0].path == '.'  # default
        assert actions[0].include == ''  # default
        assert actions[0].action == ActionType.GREP

    def test_grep_tool_with_path(self):
        """Test GrepTool with path parameter."""
        response = create_mock_response(
            GREP_TOOL_NAME, {'pattern': 'import', 'path': '/project/src'}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], GrepAction)
        assert actions[0].path == '/project/src'

    def test_grep_tool_with_include(self):
        """Test GrepTool with include filter."""
        response = create_mock_response(
            GREP_TOOL_NAME, {'pattern': 'class.*Handler', 'include': '*.py'}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], GrepAction)
        assert actions[0].include == '*.py'

    def test_grep_tool_with_all_params(self):
        """Test GrepTool with all parameters."""
        response = create_mock_response(
            GREP_TOOL_NAME,
            {'pattern': 'function', 'path': '/app', 'include': '*.{js,ts}'},
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].pattern == 'function'
        assert actions[0].path == '/app'
        assert actions[0].include == '*.{js,ts}'

    def test_grep_tool_missing_pattern(self):
        """Test GrepTool raises error when pattern is missing."""
        response = create_mock_response(GREP_TOOL_NAME, {'path': '/some/path'})
        with pytest.raises(FunctionCallValidationError) as exc_info:
            response_to_actions(response)
        assert 'Missing required argument "pattern"' in str(exc_info.value)

    def test_grep_tool_regex_pattern(self):
        """Test GrepTool with regex patterns."""
        patterns = [
            r'def \w+\(',
            r'^import',
            r'[A-Z][a-z]+Error',
            r'\d{4}-\d{2}-\d{2}',
        ]
        for pattern in patterns:
            response = create_mock_response(GREP_TOOL_NAME, {'pattern': pattern})
            actions = response_to_actions(response)
            assert len(actions) == 1
            assert actions[0].pattern == pattern


# ==============================================================================
# ListDirTool Function Calling Tests
# ==============================================================================


class TestListDirToolFunctionCalling:
    """Tests for ListDirTool function calling."""

    def test_list_dir_tool_default(self):
        """Test ListDirTool with defaults."""
        response = create_mock_response(LIST_DIR_TOOL_NAME, {})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ListDirAction)
        assert actions[0].path == '.'  # default
        assert actions[0].ignore == []  # default
        assert actions[0].action == ActionType.LIST_DIR

    def test_list_dir_tool_with_path(self):
        """Test ListDirTool with path parameter."""
        response = create_mock_response(LIST_DIR_TOOL_NAME, {'path': '/project/src'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ListDirAction)
        assert actions[0].path == '/project/src'

    def test_list_dir_tool_with_ignore(self):
        """Test ListDirTool with ignore patterns."""
        response = create_mock_response(
            LIST_DIR_TOOL_NAME, {'ignore': ['*.log', 'tmp', '*.bak']}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ListDirAction)
        assert actions[0].ignore == ['*.log', 'tmp', '*.bak']

    def test_list_dir_tool_with_all_params(self):
        """Test ListDirTool with all parameters."""
        response = create_mock_response(
            LIST_DIR_TOOL_NAME, {'path': '/app', 'ignore': ['node_modules', 'dist']}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].path == '/app'
        assert actions[0].ignore == ['node_modules', 'dist']

    def test_list_dir_default_ignores(self):
        """Test ListDirAction has correct default ignore patterns."""
        action = ListDirAction()
        expected_defaults = [
            'node_modules',
            '__pycache__',
            '.git',
            'dist',
            'build',
            'target',
            'vendor',
            '.venv',
            'venv',
            '.cache',
        ]
        assert action.DEFAULT_IGNORES == expected_defaults
        assert action.all_ignores == expected_defaults

    def test_list_dir_combined_ignores(self):
        """Test ListDirAction combines default and custom ignores."""
        action = ListDirAction(ignore=['custom', '*.tmp'])
        assert 'node_modules' in action.all_ignores
        assert 'custom' in action.all_ignores
        assert '*.tmp' in action.all_ignores


# ==============================================================================
# Action Class Tests
# ==============================================================================


class TestOpenCodeReadAction:
    """Tests for OpenCodeReadAction class."""

    def test_action_creation(self):
        """Test basic action creation."""
        action = OpenCodeReadAction(path='/test.py')
        assert action.path == '/test.py'
        assert action.offset == 0
        assert action.limit == 2000
        assert action.action == ActionType.OPENCODE_READ
        assert action.runnable is True

    def test_action_with_params(self):
        """Test action creation with parameters."""
        action = OpenCodeReadAction(path='/test.py', offset=100, limit=500)
        assert action.offset == 100
        assert action.limit == 500

    def test_action_message(self):
        """Test action message property."""
        action = OpenCodeReadAction(path='/test.py')
        assert action.message == 'Reading file: /test.py'

    def test_action_message_with_offset(self):
        """Test action message with offset."""
        action = OpenCodeReadAction(path='/test.py', offset=50)
        assert action.message == 'Reading file: /test.py (from line 51)'


class TestOpenCodeWriteAction:
    """Tests for OpenCodeWriteAction class."""

    def test_action_creation(self):
        """Test basic action creation."""
        action = OpenCodeWriteAction(path='/test.py', content='print("hi")')
        assert action.path == '/test.py'
        assert action.content == 'print("hi")'
        assert action.action == ActionType.OPENCODE_WRITE
        assert action.runnable is True

    def test_action_message(self):
        """Test action message property."""
        action = OpenCodeWriteAction(path='/test.py', content='')
        assert action.message == 'Writing file: /test.py'


class TestGlobAction:
    """Tests for GlobAction class."""

    def test_action_creation(self):
        """Test basic action creation."""
        action = GlobAction(pattern='*.py')
        assert action.pattern == '*.py'
        assert action.path == '.'
        assert action.action == ActionType.GLOB
        assert action.runnable is True

    def test_action_with_path(self):
        """Test action creation with path."""
        action = GlobAction(pattern='*.ts', path='/project')
        assert action.path == '/project'

    def test_action_message(self):
        """Test action message property."""
        action = GlobAction(pattern='**/*.py')
        assert action.message == 'Searching for files matching: **/*.py'


class TestGrepAction:
    """Tests for GrepAction class."""

    def test_action_creation(self):
        """Test basic action creation."""
        action = GrepAction(pattern='TODO')
        assert action.pattern == 'TODO'
        assert action.path == '.'
        assert action.include == ''
        assert action.action == ActionType.GREP
        assert action.runnable is True

    def test_action_with_params(self):
        """Test action creation with all params."""
        action = GrepAction(pattern='import', path='/src', include='*.py')
        assert action.path == '/src'
        assert action.include == '*.py'

    def test_action_message(self):
        """Test action message property."""
        action = GrepAction(pattern='class.*Test')
        assert action.message == 'Searching for pattern: class.*Test'


class TestListDirAction:
    """Tests for ListDirAction class."""

    def test_action_creation(self):
        """Test basic action creation."""
        action = ListDirAction()
        assert action.path == '.'
        assert action.ignore == []
        assert action.action == ActionType.LIST_DIR
        assert action.runnable is True

    def test_action_with_params(self):
        """Test action creation with parameters."""
        action = ListDirAction(path='/project', ignore=['*.log'])
        assert action.path == '/project'
        assert action.ignore == ['*.log']

    def test_action_message(self):
        """Test action message property."""
        action = ListDirAction(path='/project')
        assert action.message == 'Listing directory: /project'

    def test_action_message_default(self):
        """Test action message with default path."""
        action = ListDirAction()
        # Message contains the path, which is '.'
        assert '.' in action.message or 'directory' in action.message


# ==============================================================================
# Edge Cases and Error Handling Tests
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_invalid_json_arguments(self):
        """Test handling of invalid JSON in arguments."""
        response = ModelResponse(
            id='mock-id',
            choices=[
                {
                    'message': {
                        'tool_calls': [
                            {
                                'function': {
                                    'name': READ_TOOL_NAME,
                                    'arguments': 'not valid json',
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
        with pytest.raises(FunctionCallValidationError) as exc_info:
            response_to_actions(response)
        assert 'Failed to parse tool call arguments' in str(exc_info.value)

    def test_unicode_in_paths(self):
        """Test handling of Unicode characters in paths."""
        response = create_mock_response(READ_TOOL_NAME, {'file_path': '/путь/к/файлу.py'})
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].path == '/путь/к/файлу.py'

    def test_unicode_in_content(self):
        """Test handling of Unicode characters in content."""
        content = '# 你好世界\nprint("こんにちは")'
        response = create_mock_response(
            WRITE_TOOL_NAME, {'file_path': '/test.py', 'content': content}
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].content == content

    def test_whitespace_handling_in_edit(self):
        """Test whitespace preservation in edit operations."""
        old = '    def foo():\n        pass'
        new = '    def foo():\n        return None'
        response = create_mock_response(
            EDIT_TOOL_NAME,
            {'file_path': '/test.py', 'old_string': old, 'new_string': new},
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert actions[0].old_str == old
        assert actions[0].new_str == new

    def test_empty_path_glob(self):
        """Test empty path defaults to current directory."""
        response = create_mock_response(GLOB_TOOL_NAME, {'pattern': '*.py', 'path': ''})
        actions = response_to_actions(response)
        assert len(actions) == 1
        # Empty string should work (will be interpreted as current dir)
        assert actions[0].path == ''

    def test_multiple_tool_calls(self):
        """Test handling of multiple tool calls in single response."""
        response = ModelResponse(
            id='mock-id',
            choices=[
                {
                    'message': {
                        'tool_calls': [
                            {
                                'function': {
                                    'name': READ_TOOL_NAME,
                                    'arguments': json.dumps(
                                        {'file_path': '/file1.py'}
                                    ),
                                },
                                'id': 'call-1',
                                'type': 'function',
                            },
                            {
                                'function': {
                                    'name': READ_TOOL_NAME,
                                    'arguments': json.dumps(
                                        {'file_path': '/file2.py'}
                                    ),
                                },
                                'id': 'call-2',
                                'type': 'function',
                            },
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
        assert len(actions) == 2
        assert actions[0].path == '/file1.py'
        assert actions[1].path == '/file2.py'
