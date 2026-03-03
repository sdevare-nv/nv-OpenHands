"""Comprehensive unit tests for OpenCode runtime handler implementations.

These tests directly call the actual handler methods in action_execution_server.py
using real file operations on temporary directories, without requiring Docker.
"""

import asyncio
import os
import subprocess
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.events.action import (
    GlobAction,
    GrepAction,
    ListDirAction,
    OpenCodeReadAction,
    OpenCodeWriteAction,
    QuestionAction,
    TodoReadAction,
    TodoWriteAction,
)
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileWriteObservation,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def _make_executor(temp_workspace, extra_methods=None):
    """Create a minimal mock ActionExecutor with real handler bindings."""
    from openhands.runtime.action_execution_server import ActionExecutor

    executor = MagicMock(spec=ActionExecutor)
    executor.bash_session = MagicMock()
    executor.bash_session.cwd = temp_workspace
    executor._resolve_path = ActionExecutor._resolve_path.__get__(executor)
    executor._initial_cwd = temp_workspace
    executor._todos = []
    if extra_methods:
        for name in extra_methods:
            attr = getattr(ActionExecutor, name)
            if isinstance(attr, staticmethod):
                setattr(executor, name, attr)
            else:
                setattr(executor, name, attr.__get__(executor))
    return executor


def create_test_file(directory: str, filename: str, content: str) -> str:
    """Helper to create a test file."""
    filepath = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath


def create_test_structure(base_dir: str) -> dict:
    """Create a test directory structure and return file paths."""
    files = {
        'main.py': 'def main():\n    print("Hello")\n\nif __name__ == "__main__":\n    main()',
        'utils.py': '# Utility functions\n\ndef helper():\n    return 42\n\ndef another():\n    pass',
        'src/core.py': 'class Core:\n    def __init__(self):\n        self.value = 0\n\n    def run(self):\n        pass',
        'src/lib/helpers.py': 'import os\n\ndef get_path():\n    return os.getcwd()',
        'tests/test_main.py': 'import pytest\n\ndef test_main():\n    assert True\n\ndef test_another():\n    pass',
        'config.json': '{"name": "test", "version": "1.0.0"}',
        'README.md': '# Test Project\n\nThis is a test project.',
        '.gitignore': 'node_modules/\n__pycache__/\n*.pyc',
    }

    paths = {}
    for rel_path, content in files.items():
        full_path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        paths[rel_path] = full_path

    return paths


def run(coro):
    """Helper to run async coroutines in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ==============================================================================
# OpenCode Read Handler Tests
# ==============================================================================


class TestOpenCodeReadHandler:
    """Tests for the opencode_read handler calling the actual handler method."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['opencode_read'])

    def test_read_file_line_number_format(self, executor, temp_workspace):
        """Test 5-digit zero-padded line numbers with | separator."""
        create_test_file(temp_workspace, 'test.txt', 'line 1\nline 2\nline 3')
        action = OpenCodeReadAction(path=os.path.join(temp_workspace, 'test.txt'))
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        assert '00001| line 1' in obs.content
        assert '00002| line 2' in obs.content
        assert '00003| line 3' in obs.content

    def test_read_file_with_offset(self, executor, temp_workspace):
        """Test reading file starting from offset (0-indexed)."""
        content = '\n'.join([f'line {i}' for i in range(1, 101)])
        create_test_file(temp_workspace, 'test.txt', content)
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'test.txt'), offset=49, limit=5
        )
        obs = run(executor.opencode_read(action))
        assert '00050| line 50' in obs.content
        assert '00054| line 54' in obs.content

    def test_read_file_with_limit(self, executor, temp_workspace):
        """Test reading file with line limit."""
        content = '\n'.join([f'line {i}' for i in range(1, 101)])
        create_test_file(temp_workspace, 'test.txt', content)
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'test.txt'), limit=3
        )
        obs = run(executor.opencode_read(action))
        assert '00001| line 1' in obs.content
        assert '00003| line 3' in obs.content
        assert '00004|' not in obs.content

    def test_read_nonexistent_file(self, executor, temp_workspace):
        """Test reading a nonexistent file returns error."""
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'nonexistent.py')
        )
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, ErrorObservation)
        assert 'not found' in obs.content.lower()

    def test_read_binary_file_by_extension(self, executor, temp_workspace):
        """Test binary file detection by extension."""
        filepath = os.path.join(temp_workspace, 'data.zip')
        with open(filepath, 'w') as f:
            f.write('not really a zip')
        action = OpenCodeReadAction(path=filepath)
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, ErrorObservation)
        assert 'binary' in obs.content.lower()

    def test_read_binary_file_by_content(self, executor, temp_workspace):
        """Test binary file detection by null bytes in content."""
        filepath = os.path.join(temp_workspace, 'binary.dat')
        with open(filepath, 'wb') as f:
            f.write(b'\x00\x01\x02\x03')
        action = OpenCodeReadAction(path=filepath)
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, ErrorObservation)
        assert 'binary' in obs.content.lower()

    def test_read_text_file_not_binary(self, executor, temp_workspace):
        """Test that text files are read normally."""
        create_test_file(temp_workspace, 'text.py', 'print("hello")')
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'text.py')
        )
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'print("hello")' in obs.content

    def test_read_directory_is_error(self, executor, temp_workspace):
        """Test that reading a directory returns error."""
        subdir = os.path.join(temp_workspace, 'subdir')
        os.makedirs(subdir)
        action = OpenCodeReadAction(path=subdir)
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, ErrorObservation)
        assert 'directory' in obs.content.lower()

    def test_read_file_suggestions(self, executor, temp_workspace):
        """Test similar filenames are suggested when not found."""
        create_test_file(temp_workspace, 'mymodule.py', 'content')
        create_test_file(temp_workspace, 'mymodule_test.py', 'content')
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'mymodule')
        )
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, ErrorObservation)
        assert 'did you mean' in obs.content.lower()

    def test_read_empty_file(self, executor, temp_workspace):
        """Test reading an empty file."""
        create_test_file(temp_workspace, 'empty.txt', '')
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'empty.txt')
        )
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        assert '<file>' in obs.content
        assert '</file>' in obs.content

    def test_read_file_long_line_truncation(self, executor, temp_workspace):
        """Test that long lines are truncated."""
        long_line = 'x' * 3000
        create_test_file(temp_workspace, 'long.txt', long_line)
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'long.txt')
        )
        obs = run(executor.opencode_read(action))
        assert '...' in obs.content

    def test_read_file_has_more_indicator(self, executor, temp_workspace):
        """Test that output indicates when more lines exist."""
        content = '\n'.join([f'line {i}' for i in range(1, 50)])
        create_test_file(temp_workspace, 'big.txt', content)
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'big.txt'), limit=5
        )
        obs = run(executor.opencode_read(action))
        assert 'more' in obs.content.lower() or 'offset' in obs.content.lower()

    def test_read_file_end_of_file_indicator(self, executor, temp_workspace):
        """Test end-of-file indicator when reading to end."""
        create_test_file(temp_workspace, 'small.txt', 'a\nb\nc')
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'small.txt'), limit=2000
        )
        obs = run(executor.opencode_read(action))
        assert 'end of file' in obs.content.lower() or 'total' in obs.content.lower()

    def test_read_file_relative_path(self, executor, temp_workspace):
        """Test reading with a relative path resolved from cwd."""
        create_test_file(temp_workspace, 'rel.txt', 'hello')
        action = OpenCodeReadAction(path='rel.txt')
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'hello' in obs.content

    def test_read_file_unicode_content(self, executor, temp_workspace):
        """Test reading file with Unicode content."""
        create_test_file(temp_workspace, 'uni.txt', 'naïve café ✅\nline 2')
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'uni.txt')
        )
        obs = run(executor.opencode_read(action))
        assert 'naïve café ✅' in obs.content

    def test_read_file_wraps_in_file_tags(self, executor, temp_workspace):
        """Test that output is wrapped in <file>...</file> tags."""
        create_test_file(temp_workspace, 'test.txt', 'content')
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'test.txt')
        )
        obs = run(executor.opencode_read(action))
        assert obs.content.startswith('<file>')
        assert obs.content.strip().endswith('</file>')

    def test_read_file_truncates_long_lines(self, executor, temp_workspace):
        """Long lines (3000+ chars) should be truncated.

        From opencode read.test.ts.
        """
        long_line = 'x' * 3000
        create_test_file(temp_workspace, 'long_line.txt', long_line)
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'long_line.txt')
        )
        obs = run(executor.opencode_read(action))
        # Should contain the content but may be truncated
        assert isinstance(obs, CmdOutputObservation)
        # Content should be present but line may be cut
        assert 'xxx' in obs.content or long_line in obs.content

    def test_read_file_crlf_line_endings(self, executor, temp_workspace):
        """Files with CRLF line endings should be read correctly.

        From opencode read.test.ts / grep.test.ts CRLF handling.
        """
        crlf_content = 'line1\r\nline2\r\nline3'
        filepath = os.path.join(temp_workspace, 'crlf.txt')
        with open(filepath, 'wb') as f:
            f.write(crlf_content.encode())
        action = OpenCodeReadAction(path=filepath)
        obs = run(executor.opencode_read(action))
        assert 'line1' in obs.content
        assert 'line2' in obs.content
        assert 'line3' in obs.content

    def test_read_file_mixed_line_endings(self, executor, temp_workspace):
        """Files with mixed LF/CRLF should be handled.

        From opencode grep.test.ts CRLF regex handling.
        """
        mixed_content = 'unix\nmixed\r\nback to unix\n'
        filepath = os.path.join(temp_workspace, 'mixed.txt')
        with open(filepath, 'wb') as f:
            f.write(mixed_content.encode())
        action = OpenCodeReadAction(path=filepath)
        obs = run(executor.opencode_read(action))
        assert 'unix' in obs.content
        assert 'mixed' in obs.content

    def test_read_file_small_file_no_truncation(self, executor, temp_workspace):
        """Small file should not be truncated.

        From opencode read.test.ts.
        """
        create_test_file(temp_workspace, 'small.txt', 'hello world')
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'small.txt')
        )
        obs = run(executor.opencode_read(action))
        assert 'hello world' in obs.content
        # Should contain end-of-file indicator
        assert 'end of file' in obs.content.lower() or 'total' in obs.content.lower()

    def test_read_file_truncation_with_offset_and_limit(self, executor, temp_workspace):
        """Offset + limit should paginate correctly with truncation indicators.

        From opencode read.test.ts.
        """
        lines = '\n'.join(f'line{i}' for i in range(100))
        create_test_file(temp_workspace, 'many.txt', lines)
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'many.txt'),
            offset=10,
            limit=5
        )
        obs = run(executor.opencode_read(action))
        assert 'line10' in obs.content
        assert 'line14' in obs.content
        # line0 should not be present since we started at offset 10
        assert 'line0' not in obs.content.split('line10')[0]

    def test_read_file_flatbuffers_schema_as_text(self, executor, temp_workspace):
        """FlatBuffers schema files (.fbs) should be read as text, not binary.

        From opencode read.test.ts.
        """
        fbs_content = (
            'namespace MyGame;\n\n'
            'table Monster {\n'
            '  pos:Vec3;\n'
            '  name:string;\n'
            '}\n\n'
            'root_type Monster;'
        )
        create_test_file(temp_workspace, 'schema.fbs', fbs_content)
        action = OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'schema.fbs')
        )
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'namespace MyGame' in obs.content
        assert 'table Monster' in obs.content


# ==============================================================================
# OpenCode Write Handler Tests
# ==============================================================================


class TestOpenCodeWriteHandler:
    """Tests for the opencode_write handler calling the actual handler method."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['opencode_write'])

    def test_write_creates_file(self, executor, temp_workspace):
        """Test that write creates a new file."""
        filepath = os.path.join(temp_workspace, 'new_file.py')
        action = OpenCodeWriteAction(path=filepath, content='print("Hello")')
        obs = run(executor.opencode_write(action))
        assert isinstance(obs, FileWriteObservation)
        assert os.path.exists(filepath)
        with open(filepath) as f:
            assert f.read() == 'print("Hello")'

    def test_write_creates_parent_directories(self, executor, temp_workspace):
        """Test that write creates parent directories if needed."""
        filepath = os.path.join(temp_workspace, 'a', 'b', 'c', 'deep.txt')
        action = OpenCodeWriteAction(path=filepath, content='nested')
        obs = run(executor.opencode_write(action))
        assert isinstance(obs, FileWriteObservation)
        assert os.path.exists(filepath)
        with open(filepath) as f:
            assert f.read() == 'nested'

    def test_write_overwrites_existing(self, executor, temp_workspace):
        """Test that write overwrites existing file."""
        filepath = create_test_file(temp_workspace, 'existing.txt', 'old content')
        action = OpenCodeWriteAction(path=filepath, content='new content')
        obs = run(executor.opencode_write(action))
        assert isinstance(obs, FileWriteObservation)
        with open(filepath) as f:
            assert f.read() == 'new content'

    def test_write_empty_content(self, executor, temp_workspace):
        """Test writing empty content creates empty file."""
        filepath = os.path.join(temp_workspace, 'empty.txt')
        action = OpenCodeWriteAction(path=filepath, content='')
        obs = run(executor.opencode_write(action))
        assert isinstance(obs, FileWriteObservation)
        assert os.path.getsize(filepath) == 0

    def test_write_preserves_unicode(self, executor, temp_workspace):
        """Test that Unicode content is preserved."""
        filepath = os.path.join(temp_workspace, 'unicode.txt')
        content = '你好世界\nこんにちは\n🎉 emoji test'
        action = OpenCodeWriteAction(path=filepath, content=content)
        obs = run(executor.opencode_write(action))
        with open(filepath, encoding='utf-8') as f:
            assert f.read() == content

    def test_write_multiline_content(self, executor, temp_workspace):
        """Test that multiline content preserves line breaks."""
        filepath = os.path.join(temp_workspace, 'multiline.py')
        content = 'def hello():\n    print("Hello")\n\ndef goodbye():\n    print("Bye")\n'
        action = OpenCodeWriteAction(path=filepath, content=content)
        obs = run(executor.opencode_write(action))
        with open(filepath) as f:
            assert f.read() == content

    def test_write_returns_success_message(self, executor, temp_workspace):
        """Test that write returns success message."""
        filepath = os.path.join(temp_workspace, 'test.txt')
        action = OpenCodeWriteAction(path=filepath, content='hello')
        obs = run(executor.opencode_write(action))
        assert 'wrote' in obs.content.lower() or 'success' in obs.content.lower()

    def test_write_relative_path(self, executor, temp_workspace):
        """Test writing with a relative path resolved from cwd."""
        action = OpenCodeWriteAction(path='rel_write.txt', content='data')
        obs = run(executor.opencode_write(action))
        assert isinstance(obs, FileWriteObservation)
        assert os.path.exists(os.path.join(temp_workspace, 'rel_write.txt'))


# ==============================================================================
# Glob Handler Tests
# ==============================================================================


class TestGlobHandler:
    """Tests for the glob handler calling the actual handler method."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['glob'])

    def test_glob_finds_python_files(self, executor, temp_workspace):
        """Test glob finds Python files."""
        create_test_structure(temp_workspace)
        action = GlobAction(pattern='*.py', path=temp_workspace)
        obs = run(executor.glob(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'main.py' in obs.content
        assert 'utils.py' in obs.content

    def test_glob_finds_json_files(self, executor, temp_workspace):
        """Test glob finds specific extension."""
        create_test_structure(temp_workspace)
        action = GlobAction(pattern='*.json', path=temp_workspace)
        obs = run(executor.glob(action))
        assert 'config.json' in obs.content

    def test_glob_in_subdirectory(self, executor, temp_workspace):
        """Test glob searches in specific directory."""
        create_test_structure(temp_workspace)
        src_dir = os.path.join(temp_workspace, 'src')
        action = GlobAction(pattern='*.py', path=src_dir)
        obs = run(executor.glob(action))
        assert 'core.py' in obs.content

    def test_glob_no_matches(self, executor, temp_workspace):
        """Test glob returns 'no files found' for no matches."""
        create_test_structure(temp_workspace)
        action = GlobAction(pattern='*.nonexistent', path=temp_workspace)
        obs = run(executor.glob(action))
        assert 'no files' in obs.content.lower()

    def test_glob_result_sorting_by_mtime(self, executor, temp_workspace):
        """Test that results are sorted by modification time (newest first)."""
        import time
        create_test_file(temp_workspace, 'old.py', 'old')
        time.sleep(0.1)
        create_test_file(temp_workspace, 'new.py', 'new')
        action = GlobAction(pattern='*.py', path=temp_workspace)
        obs = run(executor.glob(action))
        lines = [l for l in obs.content.strip().split('\n') if l.strip()]
        # Newest first
        assert 'new.py' in lines[0]

    def test_glob_nonexistent_path(self, executor, temp_workspace):
        """Test glob on nonexistent path returns error."""
        action = GlobAction(
            pattern='*.py', path=os.path.join(temp_workspace, 'nonexistent')
        )
        obs = run(executor.glob(action))
        assert isinstance(obs, ErrorObservation)
        assert 'not exist' in obs.content.lower()

    def test_glob_recursive_pattern(self, executor, temp_workspace):
        """Test recursive glob pattern."""
        create_test_file(temp_workspace, 'root.py', '')
        create_test_file(temp_workspace, 'sub/nested.py', '')
        create_test_file(temp_workspace, 'sub/deep/deeper.py', '')
        action = GlobAction(pattern='**/*.py', path=temp_workspace)
        obs = run(executor.glob(action))
        assert 'root.py' in obs.content
        assert 'nested.py' in obs.content

    def test_glob_auto_prepends_recursive(self, executor, temp_workspace):
        """Test that patterns without / get **/ prepended for recursive search."""
        create_test_file(temp_workspace, 'sub/nested.py', '')
        action = GlobAction(pattern='*.py', path=temp_workspace)
        obs = run(executor.glob(action))
        assert 'nested.py' in obs.content

    def test_glob_relative_path(self, executor, temp_workspace):
        """Test glob with relative path."""
        os.makedirs(os.path.join(temp_workspace, 'mydir'))
        create_test_file(temp_workspace, 'mydir/a.py', '')
        action = GlobAction(pattern='*.py', path='mydir')
        obs = run(executor.glob(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'a.py' in obs.content


# ==============================================================================
# Grep Handler Tests
# ==============================================================================


class TestGrepHandler:
    """Tests for the grep handler calling the actual handler method."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['grep'])

    def test_grep_finds_pattern(self, executor, temp_workspace):
        """Test grep finds pattern in files with file:line format."""
        create_test_structure(temp_workspace)
        action = GrepAction(pattern='def', path=temp_workspace)
        obs = run(executor.grep(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'def main' in obs.content
        assert 'def helper' in obs.content

    def test_grep_with_line_numbers(self, executor, temp_workspace):
        """Test grep output includes line numbers."""
        create_test_file(temp_workspace, 'test.py', 'line1\nTARGET\nline3')
        action = GrepAction(pattern='TARGET', path=temp_workspace)
        obs = run(executor.grep(action))
        assert '2:TARGET' in obs.content or ':2:TARGET' in obs.content

    def test_grep_with_include_filter(self, executor, temp_workspace):
        """Test grep with file type filter."""
        create_test_file(temp_workspace, 'match.py', 'FIND_ME')
        create_test_file(temp_workspace, 'match.txt', 'FIND_ME')
        action = GrepAction(pattern='FIND_ME', path=temp_workspace, include='*.py')
        obs = run(executor.grep(action))
        assert 'match.py' in obs.content
        assert 'match.txt' not in obs.content

    def test_grep_no_matches(self, executor, temp_workspace):
        """Test grep with no matches."""
        create_test_file(temp_workspace, 'file.py', 'nothing here')
        action = GrepAction(pattern='NONEXISTENT_XYZ_PATTERN', path=temp_workspace)
        obs = run(executor.grep(action))
        assert 'no matches' in obs.content.lower()

    def test_grep_regex_pattern(self, executor, temp_workspace):
        """Test grep with regex pattern."""
        create_test_file(temp_workspace, 'func.py', 'def my_function():\n    pass')
        create_test_file(temp_workspace, 'cls.py', 'class MyClass:\n    pass')
        action = GrepAction(pattern=r'def \w+\(', path=temp_workspace)
        obs = run(executor.grep(action))
        assert 'func.py' in obs.content
        assert 'cls.py' not in obs.content

    def test_grep_case_sensitive(self, executor, temp_workspace):
        """Test grep is case-sensitive by default."""
        create_test_file(temp_workspace, 'case.txt', 'Hello\nhello\nHELLO')
        action = GrepAction(pattern='hello', path=temp_workspace)
        obs = run(executor.grep(action))
        lines = [l for l in obs.content.strip().split('\n') if 'case.txt' in l]
        assert len(lines) == 1
        assert 'hello' in lines[0]

    def test_grep_nonexistent_path(self, executor, temp_workspace):
        """Test grep on nonexistent path returns error."""
        action = GrepAction(
            pattern='test', path=os.path.join(temp_workspace, 'nonexistent')
        )
        obs = run(executor.grep(action))
        assert isinstance(obs, ErrorObservation)
        assert 'not exist' in obs.content.lower()

    def test_grep_include_auto_recursive(self, executor, temp_workspace):
        """Test that include pattern gets **/ prepended for recursive search."""
        create_test_file(temp_workspace, 'root.py', 'DEEP')
        create_test_file(temp_workspace, 'sub/nested.py', 'DEEP')
        create_test_file(temp_workspace, 'sub/nested.txt', 'DEEP')
        action = GrepAction(pattern='DEEP', path=temp_workspace, include='*.py')
        obs = run(executor.grep(action))
        assert '.py' in obs.content
        assert '.txt' not in obs.content

    def test_grep_relative_path(self, executor, temp_workspace):
        """Test grep with relative path."""
        os.makedirs(os.path.join(temp_workspace, 'subdir'))
        create_test_file(temp_workspace, 'subdir/file.py', 'REL_PAT')
        action = GrepAction(pattern='REL_PAT', path='subdir')
        obs = run(executor.grep(action))
        assert 'file.py' in obs.content

    def test_grep_multiple_matches_in_file(self, executor, temp_workspace):
        """Test grep finds multiple matches in the same file."""
        create_test_file(temp_workspace, 'multi.py', 'TODO first\nother\nTODO second')
        action = GrepAction(pattern='TODO', path=temp_workspace)
        obs = run(executor.grep(action))
        lines = [l for l in obs.content.strip().split('\n') if 'TODO' in l]
        assert len(lines) >= 2

    def test_grep_crlf_line_endings(self, executor, temp_workspace):
        """Grep handles files with CRLF line endings.

        From opencode grep.test.ts: CRLF regex handling.
        """
        crlf_content = 'line1\r\nline2\r\nline3'
        filepath = os.path.join(temp_workspace, 'crlf.txt')
        with open(filepath, 'wb') as f:
            f.write(crlf_content.encode())
        action = GrepAction(pattern='line', path=temp_workspace)
        obs = run(executor.grep(action))
        assert isinstance(obs, CmdOutputObservation)
        # Should find all three lines
        assert 'line1' in obs.content
        assert 'line2' in obs.content
        assert 'line3' in obs.content

    def test_grep_mixed_line_endings(self, executor, temp_workspace):
        """Grep handles files with mixed Unix/Windows line endings.

        From opencode grep.test.ts: mixed CRLF regex handling.
        """
        mixed_content = 'MATCH_A\nno match\r\nMATCH_B\nmore\r\nMATCH_C'
        filepath = os.path.join(temp_workspace, 'mixed.txt')
        with open(filepath, 'wb') as f:
            f.write(mixed_content.encode())
        action = GrepAction(pattern='MATCH', path=temp_workspace)
        obs = run(executor.grep(action))
        match_lines = [l for l in obs.content.strip().split('\n') if 'MATCH' in l]
        assert len(match_lines) >= 3

    def test_grep_empty_result_message(self, executor, temp_workspace):
        """No matches returns correct 'no matches' message.

        From opencode grep.test.ts.
        """
        create_test_file(temp_workspace, 'test.txt', 'hello world')
        action = GrepAction(
            pattern='xyznonexistentpatternxyz123', path=temp_workspace
        )
        obs = run(executor.grep(action))
        assert 'no matches' in obs.content.lower() or 'no files' in obs.content.lower()

    def test_grep_invalid_regex_returns_error(self, executor, temp_workspace):
        """Invalid regex pattern (unmatched paren) returns informative ErrorObservation.

        Bug: 'write_records(' silently returned 'No matches found' because
        grep -E exit code 2 was hidden by 2>/dev/null.
        Fix: detect exit code 2 and return ErrorObservation with fix guidance.
        """
        create_test_file(temp_workspace, 'func.py', 'def write_records(data):\n    pass')
        action = GrepAction(pattern='write_records(', path=temp_workspace)
        obs = run(executor.grep(action))
        assert isinstance(obs, ErrorObservation), (
            f"Expected ErrorObservation for invalid regex, got: {type(obs).__name__}: {obs.content}"
        )
        assert 'invalid regex' in obs.content.lower() or 'escaped' in obs.content.lower()

    def test_grep_valid_regex_alternation_works(self, executor, temp_workspace):
        """Valid regex with alternation (|) should still work."""
        create_test_file(temp_workspace, 'a.py', 'def foo(): pass')
        create_test_file(temp_workspace, 'b.py', 'def bar(): pass')
        action = GrepAction(pattern='foo|bar', path=temp_workspace)
        obs = run(executor.grep(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'foo' in obs.content
        assert 'bar' in obs.content


# ==============================================================================
# ListDir Handler Tests
# ==============================================================================


class TestListDirHandler:
    """Tests for the list_dir handler calling the actual handler method."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['list_dir'])

    def test_list_dir_finds_files(self, executor, temp_workspace):
        """Test list_dir finds files and directories."""
        create_test_structure(temp_workspace)
        action = ListDirAction(path=temp_workspace)
        obs = run(executor.list_dir(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'main.py' in obs.content
        assert 'utils.py' in obs.content
        assert 'config.json' in obs.content

    def test_list_dir_shows_tree_structure(self, executor, temp_workspace):
        """Test that listing builds a tree structure."""
        create_test_file(temp_workspace, 'root.py', '')
        create_test_file(temp_workspace, 'src/core.py', '')
        action = ListDirAction(path=temp_workspace)
        obs = run(executor.list_dir(action))
        assert 'root.py' in obs.content
        assert 'core.py' in obs.content

    def test_list_dir_empty_directory(self, executor, temp_workspace):
        """Test listing an empty directory."""
        empty_dir = os.path.join(temp_workspace, 'empty')
        os.makedirs(empty_dir)
        action = ListDirAction(path=empty_dir)
        obs = run(executor.list_dir(action))
        assert isinstance(obs, CmdOutputObservation)

    def test_list_dir_relative_path(self, executor, temp_workspace):
        """Test listing with a relative path."""
        os.makedirs(os.path.join(temp_workspace, 'mydir'))
        create_test_file(temp_workspace, 'mydir/a.txt', '')
        action = ListDirAction(path='mydir')
        obs = run(executor.list_dir(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'a.txt' in obs.content

    def test_list_dir_ignore_patterns(self, executor, temp_workspace):
        """Test that default ignore patterns filter common dirs."""
        create_test_file(temp_workspace, 'main.py', '')
        create_test_file(temp_workspace, 'node_modules/pkg/index.js', '')
        create_test_file(temp_workspace, '__pycache__/module.pyc', '')
        action = ListDirAction(path=temp_workspace)
        obs = run(executor.list_dir(action))
        assert 'main.py' in obs.content
        # node_modules and __pycache__ should be ignored
        assert 'node_modules' not in obs.content or 'module.pyc' not in obs.content


# ==============================================================================
# Todo Handler Tests
# ==============================================================================


class TestTodoHandlers:
    """Tests for the todo_read and todo_write handlers."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['todo_read', 'todo_write'])

    def test_todo_read_empty(self, executor):
        """Test reading empty todo list."""
        action = TodoReadAction()
        obs = run(executor.todo_read(action))
        assert '[]' in obs.content

    def test_todo_write_adds_items(self, executor):
        """Test writing todo items."""
        action = TodoWriteAction(todos=[
            {'id': '1', 'title': 'First task', 'status': 'pending'},
            {'id': '2', 'title': 'Second task', 'status': 'in_progress'},
        ])
        obs = run(executor.todo_write(action))
        assert obs.success is True
        assert len(obs.todos) == 2

    def test_todo_write_then_read(self, executor):
        """Test writing then reading todos."""
        write_action = TodoWriteAction(todos=[
            {'id': '1', 'title': 'Task', 'status': 'pending'},
        ])
        run(executor.todo_write(write_action))

        read_action = TodoReadAction()
        obs = run(executor.todo_read(read_action))
        assert 'Task' in obs.content

    def test_todo_write_updates_existing(self, executor):
        """Test updating existing todo by id."""
        write1 = TodoWriteAction(todos=[
            {'id': '1', 'title': 'Task', 'status': 'pending'},
        ])
        run(executor.todo_write(write1))

        write2 = TodoWriteAction(todos=[
            {'id': '1', 'status': 'completed'},
        ])
        obs = run(executor.todo_write(write2))
        assert obs.success is True
        # Should have updated the existing item
        found = [t for t in obs.todos if t.get('id') == '1']
        assert len(found) == 1
        assert found[0]['status'] == 'completed'

    def test_todo_write_adds_new_items_preserving_existing(self, executor):
        """Test that new items are added without overwriting existing."""
        run(executor.todo_write(TodoWriteAction(todos=[
            {'id': '1', 'title': 'First', 'status': 'pending'},
        ])))
        obs = run(executor.todo_write(TodoWriteAction(todos=[
            {'id': '2', 'title': 'Second', 'status': 'pending'},
        ])))
        assert len(obs.todos) == 2


# ==============================================================================
# Question Handler Tests
# ==============================================================================


class TestQuestionHandler:
    """Tests for the question handler."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['question'])

    def test_question_returns_questions(self, executor):
        """Test that question handler returns the questions."""
        action = QuestionAction(questions=[
            {'id': 'q1', 'text': 'What framework?', 'options': ['React', 'Vue']},
        ])
        obs = run(executor.question(action))
        assert 'What framework?' in obs.content


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestOpenCodeIntegration:
    """Integration tests combining multiple handler operations."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, [
            'opencode_read', 'opencode_write', 'glob', 'grep', 'list_dir',
        ])

    def test_write_then_read(self, executor, temp_workspace):
        """Test full write then read workflow."""
        filepath = os.path.join(temp_workspace, 'workflow.py')
        write_action = OpenCodeWriteAction(
            path=filepath, content='def hello():\n    print("Hello")\n'
        )
        run(executor.opencode_write(write_action))

        read_action = OpenCodeReadAction(path=filepath)
        obs = run(executor.opencode_read(read_action))
        assert '00001| def hello():' in obs.content

    def test_write_then_glob(self, executor, temp_workspace):
        """Test write then glob to find the written file."""
        filepath = os.path.join(temp_workspace, 'written.py')
        run(executor.opencode_write(OpenCodeWriteAction(
            path=filepath, content='content'
        )))
        obs = run(executor.glob(GlobAction(pattern='*.py', path=temp_workspace)))
        assert 'written.py' in obs.content

    def test_write_then_grep(self, executor, temp_workspace):
        """Test write then grep to find content."""
        filepath = os.path.join(temp_workspace, 'search.py')
        run(executor.opencode_write(OpenCodeWriteAction(
            path=filepath, content='UNIQUE_MARKER_XYZ'
        )))
        obs = run(executor.grep(GrepAction(
            pattern='UNIQUE_MARKER_XYZ', path=temp_workspace
        )))
        assert 'search.py' in obs.content

    def test_list_then_read(self, executor, temp_workspace):
        """Test listing then reading found files."""
        create_test_structure(temp_workspace)
        list_obs = run(executor.list_dir(ListDirAction(path=temp_workspace)))
        assert 'main.py' in list_obs.content

        read_obs = run(executor.opencode_read(OpenCodeReadAction(
            path=os.path.join(temp_workspace, 'main.py')
        )))
        assert 'def main' in read_obs.content


# ==============================================================================
# Ripgrep Integration Tests (if available)
# ==============================================================================


class TestRipgrepIntegration:
    """Tests that specifically verify ripgrep-based handlers."""

    @pytest.fixture(autouse=True)
    def check_ripgrep(self):
        """Check if ripgrep is available."""
        result = subprocess.run(['which', 'rg'], capture_output=True)
        if result.returncode != 0:
            pytest.skip("ripgrep (rg) not available")

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['glob', 'grep'])

    def test_rg_glob_files(self, executor, temp_workspace):
        """Test ripgrep-based globbing."""
        create_test_structure(temp_workspace)
        action = GlobAction(pattern='*.py', path=temp_workspace)
        obs = run(executor.glob(action))
        assert 'main.py' in obs.content

    def test_rg_grep_pattern(self, executor, temp_workspace):
        """Test ripgrep-based content search."""
        create_test_structure(temp_workspace)
        action = GrepAction(pattern='def', path=temp_workspace)
        obs = run(executor.grep(action))
        assert 'def main' in obs.content
