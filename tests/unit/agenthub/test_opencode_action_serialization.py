"""Unit tests for OpenCode action serialization and deserialization.

Tests that OpenCode actions can be properly serialized to dict and
deserialized back to action objects for network transport.
"""

import pytest

from openhands.core.schema import ActionType
from openhands.events.action import (
    GlobAction,
    GrepAction,
    ListDirAction,
    OpenCodeReadAction,
    OpenCodeWriteAction,
)
from openhands.events.serialization import event_from_dict, event_to_dict


# ==============================================================================
# OpenCodeReadAction Serialization Tests
# ==============================================================================


class TestOpenCodeReadActionSerialization:
    """Tests for OpenCodeReadAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        action = OpenCodeReadAction(path='/test/file.py')
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.OPENCODE_READ
        assert serialized['args']['path'] == '/test/file.py'
        assert serialized['args']['offset'] == 0
        assert serialized['args']['limit'] == 2000

    def test_serialize_with_params(self):
        """Test serialization with all parameters."""
        action = OpenCodeReadAction(
            path='/test/file.py',
            offset=100,
            limit=500,
            thought='Reading file to understand implementation'
        )
        serialized = event_to_dict(action)

        assert serialized['args']['path'] == '/test/file.py'
        assert serialized['args']['offset'] == 100
        assert serialized['args']['limit'] == 500
        assert serialized['args']['thought'] == 'Reading file to understand implementation'

    def test_deserialize_basic(self):
        """Test basic deserialization."""
        data = {
            'id': 1,
            'action': ActionType.OPENCODE_READ,
            'args': {
                'path': '/test/file.py',
                'offset': 0,
                'limit': 2000,
                'thought': '',
            }
        }
        action = event_from_dict(data)

        assert isinstance(action, OpenCodeReadAction)
        assert action.path == '/test/file.py'
        assert action.offset == 0
        assert action.limit == 2000

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = OpenCodeReadAction(
            path='/path/to/file.txt',
            offset=50,
            limit=100,
            thought='Test thought'
        )

        serialized = event_to_dict(original)
        deserialized = event_from_dict(serialized)

        assert isinstance(deserialized, OpenCodeReadAction)
        assert deserialized.path == original.path
        assert deserialized.offset == original.offset
        assert deserialized.limit == original.limit
        assert deserialized.thought == original.thought


# ==============================================================================
# OpenCodeWriteAction Serialization Tests
# ==============================================================================


class TestOpenCodeWriteActionSerialization:
    """Tests for OpenCodeWriteAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        action = OpenCodeWriteAction(
            path='/test/file.py',
            content='print("hello")'
        )
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.OPENCODE_WRITE
        assert serialized['args']['path'] == '/test/file.py'
        assert serialized['args']['content'] == 'print("hello")'

    def test_serialize_multiline_content(self):
        """Test serialization with multiline content."""
        content = 'def foo():\n    return 42\n\nfoo()'
        action = OpenCodeWriteAction(
            path='/test/file.py',
            content=content
        )
        serialized = event_to_dict(action)

        assert serialized['args']['content'] == content

    def test_serialize_special_characters(self):
        """Test serialization with special characters."""
        content = 'print("Special: \\n\\t\\"quotes\\"")'
        action = OpenCodeWriteAction(
            path='/test/file.py',
            content=content
        )
        serialized = event_to_dict(action)
        deserialized = event_from_dict(serialized)

        assert deserialized.content == content

    def test_deserialize_basic(self):
        """Test basic deserialization."""
        data = {
            'id': 1,
            'action': ActionType.OPENCODE_WRITE,
            'args': {
                'path': '/test/file.py',
                'content': 'test content',
                'thought': '',
            }
        }
        action = event_from_dict(data)

        assert isinstance(action, OpenCodeWriteAction)
        assert action.path == '/test/file.py'
        assert action.content == 'test content'

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = OpenCodeWriteAction(
            path='/path/to/file.py',
            content='def test():\n    pass',
            thought='Creating test file'
        )

        serialized = event_to_dict(original)
        deserialized = event_from_dict(serialized)

        assert isinstance(deserialized, OpenCodeWriteAction)
        assert deserialized.path == original.path
        assert deserialized.content == original.content
        assert deserialized.thought == original.thought


# ==============================================================================
# GlobAction Serialization Tests
# ==============================================================================


class TestGlobActionSerialization:
    """Tests for GlobAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        action = GlobAction(pattern='*.py')
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.GLOB
        assert serialized['args']['pattern'] == '*.py'
        assert serialized['args']['path'] == '.'

    def test_serialize_with_path(self):
        """Test serialization with path."""
        action = GlobAction(pattern='**/*.ts', path='/project/src')
        serialized = event_to_dict(action)

        assert serialized['args']['pattern'] == '**/*.ts'
        assert serialized['args']['path'] == '/project/src'

    def test_deserialize_basic(self):
        """Test basic deserialization."""
        data = {
            'id': 1,
            'action': ActionType.GLOB,
            'args': {
                'pattern': '*.js',
                'path': '/app',
                'thought': '',
            }
        }
        action = event_from_dict(data)

        assert isinstance(action, GlobAction)
        assert action.pattern == '*.js'
        assert action.path == '/app'

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = GlobAction(
            pattern='**/*.{js,ts}',
            path='/workspace',
            thought='Finding all JS/TS files'
        )

        serialized = event_to_dict(original)
        deserialized = event_from_dict(serialized)

        assert isinstance(deserialized, GlobAction)
        assert deserialized.pattern == original.pattern
        assert deserialized.path == original.path


# ==============================================================================
# GrepAction Serialization Tests
# ==============================================================================


class TestGrepActionSerialization:
    """Tests for GrepAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        action = GrepAction(pattern='TODO')
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.GREP
        assert serialized['args']['pattern'] == 'TODO'
        assert serialized['args']['path'] == '.'
        assert serialized['args']['include'] == ''

    def test_serialize_with_all_params(self):
        """Test serialization with all parameters."""
        action = GrepAction(
            pattern='class.*Handler',
            path='/src',
            include='*.py'
        )
        serialized = event_to_dict(action)

        assert serialized['args']['pattern'] == 'class.*Handler'
        assert serialized['args']['path'] == '/src'
        assert serialized['args']['include'] == '*.py'

    def test_deserialize_basic(self):
        """Test basic deserialization."""
        data = {
            'id': 1,
            'action': ActionType.GREP,
            'args': {
                'pattern': 'import',
                'path': '/lib',
                'include': '*.ts',
                'thought': '',
            }
        }
        action = event_from_dict(data)

        assert isinstance(action, GrepAction)
        assert action.pattern == 'import'
        assert action.path == '/lib'
        assert action.include == '*.ts'

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = GrepAction(
            pattern=r'def \w+\(',
            path='/project',
            include='*.py',
            thought='Finding function definitions'
        )

        serialized = event_to_dict(original)
        deserialized = event_from_dict(serialized)

        assert isinstance(deserialized, GrepAction)
        assert deserialized.pattern == original.pattern
        assert deserialized.path == original.path
        assert deserialized.include == original.include


# ==============================================================================
# ListDirAction Serialization Tests
# ==============================================================================


class TestListDirActionSerialization:
    """Tests for ListDirAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        action = ListDirAction()
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.LIST_DIR
        assert serialized['args']['path'] == '.'
        assert serialized['args']['ignore'] == []

    def test_serialize_with_params(self):
        """Test serialization with parameters."""
        action = ListDirAction(
            path='/project',
            ignore=['*.log', 'tmp', '.cache']
        )
        serialized = event_to_dict(action)

        assert serialized['args']['path'] == '/project'
        assert serialized['args']['ignore'] == ['*.log', 'tmp', '.cache']

    def test_deserialize_basic(self):
        """Test basic deserialization."""
        data = {
            'id': 1,
            'action': ActionType.LIST_DIR,
            'args': {
                'path': '/workspace',
                'ignore': ['node_modules'],
                'thought': '',
            }
        }
        action = event_from_dict(data)

        assert isinstance(action, ListDirAction)
        assert action.path == '/workspace'
        assert action.ignore == ['node_modules']

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = ListDirAction(
            path='/app',
            ignore=['dist', 'build'],
            thought='Listing project files'
        )

        serialized = event_to_dict(original)
        deserialized = event_from_dict(serialized)

        assert isinstance(deserialized, ListDirAction)
        assert deserialized.path == original.path
        assert deserialized.ignore == original.ignore

    def test_all_ignores_not_serialized(self):
        """Test that all_ignores property is not serialized (it's computed)."""
        action = ListDirAction(ignore=['custom'])
        serialized = event_to_dict(action)

        # all_ignores should not be in serialized args
        assert 'all_ignores' not in serialized['args']
        # But ignore should be
        assert serialized['args']['ignore'] == ['custom']


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestSerializationEdgeCases:
    """Tests for serialization edge cases."""

    def test_unicode_in_path(self):
        """Test Unicode characters in paths."""
        action = OpenCodeReadAction(path='/путь/文件/αρχείο.py')
        serialized = event_to_dict(action)
        deserialized = event_from_dict(serialized)

        assert deserialized.path == '/путь/文件/αρχείο.py'

    def test_unicode_in_content(self):
        """Test Unicode characters in content."""
        content = '# 日本語コメント\nprint("你好世界")'
        action = OpenCodeWriteAction(path='/test.py', content=content)
        serialized = event_to_dict(action)
        deserialized = event_from_dict(serialized)

        assert deserialized.content == content

    def test_empty_string_handling(self):
        """Test empty strings are preserved."""
        action = OpenCodeWriteAction(path='/empty.txt', content='')
        serialized = event_to_dict(action)
        deserialized = event_from_dict(serialized)

        assert deserialized.content == ''

    def test_newlines_preserved(self):
        """Test newlines in content are preserved."""
        content = 'line1\nline2\r\nline3\n'
        action = OpenCodeWriteAction(path='/test.txt', content=content)
        serialized = event_to_dict(action)
        deserialized = event_from_dict(serialized)

        assert deserialized.content == content

    def test_regex_pattern_preserved(self):
        """Test regex patterns are preserved correctly."""
        pattern = r'^import\s+(\w+)\s+from\s+["\'](.+)["\']'
        action = GrepAction(pattern=pattern)
        serialized = event_to_dict(action)
        deserialized = event_from_dict(serialized)

        assert deserialized.pattern == pattern

    def test_special_glob_patterns(self):
        """Test special glob patterns are preserved."""
        patterns = [
            '*.{js,ts,jsx,tsx}',
            '[!_]*.py',
            '**/*[0-9]*.log',
            '**/test_*.py',
        ]
        for pattern in patterns:
            action = GlobAction(pattern=pattern)
            serialized = event_to_dict(action)
            deserialized = event_from_dict(serialized)

            assert deserialized.pattern == pattern, f"Failed for pattern: {pattern}"

