"""Comprehensive unit tests for OpenCode runtime handler implementations.

These tests directly call the actual handler methods in action_execution_server.py
using real file operations on temporary directories, without requiring Docker.
"""

import asyncio
import base64
import json
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
    return asyncio.run(coro)


# ==============================================================================
# OpenCode Read Handler Tests
# ==============================================================================


class TestOpenCodeReadHandler:
    """Tests for the opencode_read handler calling the actual handler method."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['opencode_read'])

    def test_read_file_line_number_format(self, executor, temp_workspace):
        """Read uses the current XML envelope and 1-based line numbers."""
        filepath = create_test_file(
            temp_workspace, 'test.txt', 'line 1\nline 2\nline 3'
        )
        action = OpenCodeReadAction(path=filepath)
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        assert obs.success is True
        assert obs.exit_code == 0
        assert obs.content == (
            f'<path>{filepath}</path>\n'
            '<type>file</type>\n'
            '<content>\n'
            '1: line 1\n'
            '2: line 2\n'
            '3: line 3\n\n'
            '(End of file - total 3 lines)\n'
            '</content>'
        )

    def test_read_file_with_offset(self, executor, temp_workspace):
        """Offset is a 1-based source line and continuation is also 1-based."""
        content = '\n'.join([f'line {i}' for i in range(1, 101)])
        filepath = create_test_file(temp_workspace, 'test.txt', content)
        action = OpenCodeReadAction(path=filepath, offset=50, limit=5)
        obs = run(executor.opencode_read(action))
        assert obs.content == (
            f'<path>{filepath}</path>\n'
            '<type>file</type>\n'
            '<content>\n'
            '50: line 50\n'
            '51: line 51\n'
            '52: line 52\n'
            '53: line 53\n'
            '54: line 54\n\n'
            '(Showing lines 50-54 of 100. Use offset=55 to continue.)\n'
            '</content>'
        )

    def test_read_file_with_limit(self, executor, temp_workspace):
        """Test reading file with line limit."""
        content = '\n'.join([f'line {i}' for i in range(1, 101)])
        filepath = create_test_file(temp_workspace, 'test.txt', content)
        action = OpenCodeReadAction(path=filepath, limit=3)
        obs = run(executor.opencode_read(action))
        assert obs.content == (
            f'<path>{filepath}</path>\n'
            '<type>file</type>\n'
            '<content>\n'
            '1: line 1\n'
            '2: line 2\n'
            '3: line 3\n\n'
            '(Showing lines 1-3 of 100. Use offset=4 to continue.)\n'
            '</content>'
        )

    def test_read_nonexistent_file(self, executor, temp_workspace):
        """Test reading a nonexistent file returns error."""
        action = OpenCodeReadAction(path=os.path.join(temp_workspace, 'nonexistent.py'))
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
        action = OpenCodeReadAction(path=os.path.join(temp_workspace, 'text.py'))
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'print("hello")' in obs.content

    def test_read_strips_utf8_bom_from_model_body(self, executor, temp_workspace):
        filepath = os.path.join(temp_workspace, 'bom.txt')
        with open(filepath, 'wb') as target:
            target.write(b'\xef\xbb\xbfhello\n')

        obs = run(executor.opencode_read(OpenCodeReadAction(path=filepath)))

        assert isinstance(obs, CmdOutputObservation)
        assert '\ufeff' not in obs.content
        assert '\n1: hello\n\n' in obs.content

    @pytest.mark.parametrize(
        ('signature', 'expected'),
        [
            (b'\x89PNG\r\n\x1a\nrest', 'Image read successfully'),
            (b'\xff\xd8\xffrest', 'Image read successfully'),
            (b'GIF89arest', 'Image read successfully'),
            (b'RIFF\x04\x00\x00\x00WEBPrest', 'Image read successfully'),
            (b'%PDF-1.7\nrest', 'PDF read successfully'),
        ],
    )
    def test_read_magic_sniffs_extensionless_media(
        self,
        executor,
        temp_workspace,
        signature,
        expected,
    ):
        filepath = os.path.join(temp_workspace, 'attachment')
        with open(filepath, 'wb') as target:
            target.write(signature)

        obs = run(executor.opencode_read(OpenCodeReadAction(path=filepath)))

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == expected
        assert obs.success is True
        assert obs.exit_code == 0

    @pytest.mark.parametrize(
        ('extension', 'expected'),
        [
            ('.png', 'Image read successfully'),
            ('.pdf', 'PDF read successfully'),
        ],
    )
    def test_read_media_extensions_have_success_metadata(
        self, executor, temp_workspace, extension, expected
    ):
        filepath = create_test_file(
            temp_workspace, f'attachment{extension}', 'placeholder'
        )

        obs = run(executor.opencode_read(OpenCodeReadAction(path=filepath)))

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == expected
        assert obs.success is True
        assert obs.exit_code == 0

    def test_read_directory_returns_immediate_entries(self, executor, temp_workspace):
        """Read accepts directories and returns only their immediate entries."""
        subdir = os.path.join(temp_workspace, 'subdir')
        os.makedirs(subdir)
        create_test_file(subdir, 'alpha.txt', '')
        os.makedirs(os.path.join(subdir, 'nested'))
        create_test_file(subdir, 'nested/not-shown.txt', '')
        action = OpenCodeReadAction(path=subdir)
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        assert obs.success is True
        assert obs.exit_code == 0
        assert obs.content == (
            f'<path>{subdir}</path>\n'
            '<type>directory</type>\n'
            '<entries>\n'
            'alpha.txt\n'
            'nested/\n\n'
            '(2 entries)\n'
            '</entries>'
        )

    def test_read_file_suggestions(self, executor, temp_workspace):
        """Test similar filenames are suggested when not found."""
        create_test_file(temp_workspace, 'mymodule.py', 'content')
        create_test_file(temp_workspace, 'mymodule_test.py', 'content')
        action = OpenCodeReadAction(path=os.path.join(temp_workspace, 'mymodule'))
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, ErrorObservation)
        assert 'did you mean' in obs.content.lower()

    def test_read_empty_file(self, executor, temp_workspace):
        """Test reading an empty file."""
        create_test_file(temp_workspace, 'empty.txt', '')
        action = OpenCodeReadAction(path=os.path.join(temp_workspace, 'empty.txt'))
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        filepath = os.path.join(temp_workspace, 'empty.txt')
        assert obs.content == (
            f'<path>{filepath}</path>\n'
            '<type>file</type>\n'
            '<content>\n\n\n'
            '(End of file - total 0 lines)\n'
            '</content>'
        )

    def test_read_file_long_line_truncation(self, executor, temp_workspace):
        """Test that long lines are truncated."""
        long_line = 'x' * 3000
        filepath = create_test_file(temp_workspace, 'long.txt', long_line)
        action = OpenCodeReadAction(path=filepath)
        obs = run(executor.opencode_read(action))
        assert obs.content == (
            f'<path>{filepath}</path>\n'
            '<type>file</type>\n'
            '<content>\n'
            f'1: {"x" * 2000}... (line truncated to 2000 chars)\n\n'
            '(End of file - total 1 lines)\n'
            '</content>'
        )

    def test_read_file_has_more_indicator(self, executor, temp_workspace):
        """Test that output indicates when more lines exist."""
        content = '\n'.join([f'line {i}' for i in range(1, 50)])
        filepath = create_test_file(temp_workspace, 'big.txt', content)
        action = OpenCodeReadAction(path=filepath, limit=5)
        obs = run(executor.opencode_read(action))
        assert obs.content == (
            f'<path>{filepath}</path>\n'
            '<type>file</type>\n'
            '<content>\n'
            '1: line 1\n'
            '2: line 2\n'
            '3: line 3\n'
            '4: line 4\n'
            '5: line 5\n\n'
            '(Showing lines 1-5 of 49. Use offset=6 to continue.)\n'
            '</content>'
        )

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
        action = OpenCodeReadAction(path=os.path.join(temp_workspace, 'uni.txt'))
        obs = run(executor.opencode_read(action))
        assert 'naïve café ✅' in obs.content

    def test_read_file_uses_path_type_and_content_tags(self, executor, temp_workspace):
        """The model-visible envelope uses path, type, and content tags."""
        filepath = create_test_file(temp_workspace, 'test.txt', 'content')
        action = OpenCodeReadAction(path=filepath)
        obs = run(executor.opencode_read(action))
        assert obs.content == (
            f'<path>{filepath}</path>\n'
            '<type>file</type>\n'
            '<content>\n'
            '1: content\n\n'
            '(End of file - total 1 lines)\n'
            '</content>'
        )

    def test_read_file_truncates_long_lines(self, executor, temp_workspace):
        """Long lines (3000+ chars) should be truncated.

        From opencode read.test.ts.
        """
        long_line = 'x' * 3000
        create_test_file(temp_workspace, 'long_line.txt', long_line)
        action = OpenCodeReadAction(path=os.path.join(temp_workspace, 'long_line.txt'))
        obs = run(executor.opencode_read(action))
        assert isinstance(obs, CmdOutputObservation)
        assert f'1: {"x" * 2000}... (line truncated to 2000 chars)' in obs.content
        assert long_line not in obs.content

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
        action = OpenCodeReadAction(path=os.path.join(temp_workspace, 'small.txt'))
        obs = run(executor.opencode_read(action))
        assert 'hello world' in obs.content
        # Should contain end-of-file indicator
        assert 'end of file' in obs.content.lower() or 'total' in obs.content.lower()

    def test_read_file_truncation_with_offset_and_limit(self, executor, temp_workspace):
        """Offset + limit should paginate correctly with truncation indicators.

        From opencode read.test.ts.
        """
        lines = '\n'.join(f'line {i}' for i in range(1, 101))
        filepath = create_test_file(temp_workspace, 'many.txt', lines)
        action = OpenCodeReadAction(
            path=filepath,
            offset=10,
            limit=5,
        )
        obs = run(executor.opencode_read(action))
        assert obs.content == (
            f'<path>{filepath}</path>\n'
            '<type>file</type>\n'
            '<content>\n'
            '10: line 10\n'
            '11: line 11\n'
            '12: line 12\n'
            '13: line 13\n'
            '14: line 14\n\n'
            '(Showing lines 10-14 of 100. Use offset=15 to continue.)\n'
            '</content>'
        )

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
        action = OpenCodeReadAction(path=os.path.join(temp_workspace, 'schema.fbs'))
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

    def test_write_preserves_existing_utf8_bom(self, executor, temp_workspace):
        filepath = os.path.join(temp_workspace, 'bom.txt')
        with open(filepath, 'wb') as target:
            target.write(b'\xef\xbb\xbfold')

        obs = run(
            executor.opencode_write(
                OpenCodeWriteAction(path=filepath, content='new content')
            )
        )

        assert isinstance(obs, FileWriteObservation)
        with open(filepath, 'rb') as target:
            assert target.read() == b'\xef\xbb\xbfnew content'

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
        content = (
            'def hello():\n    print("Hello")\n\ndef goodbye():\n    print("Bye")\n'
        )
        action = OpenCodeWriteAction(path=filepath, content=content)
        obs = run(executor.opencode_write(action))
        with open(filepath) as f:
            assert f.read() == content

    def test_write_returns_success_message(self, executor, temp_workspace):
        """Test that write returns success message."""
        filepath = os.path.join(temp_workspace, 'test.txt')
        action = OpenCodeWriteAction(path=filepath, content='hello')
        obs = run(executor.opencode_write(action))
        assert obs.content == 'Wrote file successfully.'

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
        assert obs.success is True
        assert obs.exit_code == 0

    def test_glob_finds_json_files(self, executor, temp_workspace):
        """Glob returns absolute paths without an extra envelope."""
        create_test_structure(temp_workspace)
        action = GlobAction(pattern='*.json', path=temp_workspace)
        obs = run(executor.glob(action))
        assert obs.content == os.path.join(temp_workspace, 'config.json')

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
        assert obs.content == 'No files found'

    def test_glob_preserves_rg_order_and_uses_exact_truncation_footer(
        self, executor, temp_workspace
    ):
        """Glob does not apply the obsolete modification-time sort."""
        relative_paths = ['z-last.py', 'a-first.py'] + [
            f'file-{index:03}.py' for index in range(98)
        ]
        rg_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='\n'.join(relative_paths) + '\n',
            stderr='',
        )
        action = GlobAction(pattern='*.py', path=temp_workspace)
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=rg_result,
        ):
            obs = run(executor.glob(action))

        absolute_paths = [os.path.join(temp_workspace, path) for path in relative_paths]
        assert obs.content == (
            '\n'.join(absolute_paths)
            + '\n\n(Results are truncated: showing first 100 results. '
            'Consider using a more specific path or pattern.)'
        )

    def test_glob_decodes_invalid_utf8_filename_output_lossily(
        self, executor, temp_workspace
    ):
        rg_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b'normal.py\ninvalid-\xff.py\nunsafe\x00.py\n',
            stderr=b'',
        )
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=rg_result,
        ):
            obs = run(executor.glob(GlobAction(pattern='*.py', path=temp_workspace)))

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content.splitlines() == [
            os.path.join(temp_workspace, 'normal.py'),
            os.path.join(temp_workspace, 'invalid-�.py'),
        ]
        assert obs.success is True
        assert obs.exit_code == 0

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

    def test_glob_basename_pattern_matches_recursively(self, executor, temp_workspace):
        """Ripgrep basename globs match files below the search directory."""
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

    def test_missing_rg_fallback_supports_braces_paths_and_git_exclusion(
        self, executor, temp_workspace
    ):
        expected = {
            create_test_file(temp_workspace, 'src/root.py', ''),
            create_test_file(temp_workspace, 'src/deep/code.pyi', ''),
            create_test_file(temp_workspace, 'src/.hidden.py', ''),
            create_test_file(temp_workspace, 'tests/test_code.py', ''),
        }
        create_test_file(temp_workspace, '.gitignore', 'vendor/\n')
        create_test_file(temp_workspace, '.ignore', 'ignored*.py\n')
        create_test_file(temp_workspace, '.rgignore', 'generated/\n')
        create_test_file(temp_workspace, 'src/deep/code.txt', '')
        create_test_file(temp_workspace, 'src/vendor/pkg/dependency.py', '')
        # The positive rg glob explicitly selects this file and therefore
        # overrides the matching .ignore rule.
        expected.add(create_test_file(temp_workspace, 'src/deep/ignored-output.py', ''))
        create_test_file(temp_workspace, 'src/generated/output.py', '')
        create_test_file(temp_workspace, '.git/hidden.py', '')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = run(
                executor.glob(
                    GlobAction(
                        pattern='{src,tests}/**/*.{py,pyi}',
                        path=temp_workspace,
                    )
                )
            )

        assert isinstance(obs, CmdOutputObservation)
        assert set(obs.content.splitlines()) == expected
        assert '.git' not in obs.content
        assert 'vendor' not in obs.content
        assert 'ignored-output.py' in obs.content
        assert 'generated' not in obs.content
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_negative_glob_honors_nested_ignore_rules(
        self, executor, temp_workspace
    ):
        expected = {
            create_test_file(temp_workspace, 'root.txt', ''),
            create_test_file(temp_workspace, 'nested/keep.txt', ''),
            create_test_file(temp_workspace, 'nested/deep/anchored.txt', ''),
        }
        create_test_file(
            temp_workspace,
            'nested/.gitignore',
            '/anchored.txt\nignored/\n*.txt\n!keep.txt\n!deep/anchored.txt\n',
        )
        create_test_file(
            temp_workspace,
            'nested/.ignore',
            'ignored-by-ignore.txt\n',
        )
        create_test_file(
            temp_workspace,
            'nested/.rgignore',
            'ignored-by-rgignore.txt\n',
        )
        create_test_file(temp_workspace, 'nested/anchored.txt', '')
        create_test_file(temp_workspace, 'nested/drop.txt', '')
        create_test_file(temp_workspace, 'nested/ignored/secret.txt', '')
        create_test_file(temp_workspace, 'nested/ignored-by-ignore.txt', '')
        create_test_file(temp_workspace, 'nested/ignored-by-rgignore.txt', '')

        # A positive rg glob overrides ignore files. Use a negative-only glob
        # to verify the fallback's ordinary nested-ignore behavior.
        action = GlobAction(pattern='!*.bin', path=temp_workspace)
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            fallback = run(executor.glob(action))

        assert isinstance(fallback, CmdOutputObservation)
        assert set(fallback.content.splitlines()) == expected
        assert fallback.success is True
        assert fallback.exit_code == 0

    @pytest.mark.parametrize(
        ('pattern', 'expected_relative_paths'),
        [
            ('*.py', ('.root.py', 'visible.py')),
            ('*', ('.root.py', 'visible.py', '.hidden/deep/a.py')),
            ('**/*', ('.root.py', 'visible.py', '.hidden/deep/a.py')),
        ],
    )
    def test_missing_rg_matches_real_rg_hidden_directory_glob_rules(
        self,
        executor,
        temp_workspace,
        pattern,
        expected_relative_paths,
    ):
        import shutil

        if shutil.which('rg') is None:
            pytest.skip('ripgrep is required for the parity half of this test')

        for relative_path in ('.root.py', 'visible.py', '.hidden/deep/a.py'):
            create_test_file(temp_workspace, relative_path, '')
        expected = {
            os.path.join(temp_workspace, relative_path)
            for relative_path in expected_relative_paths
        }

        action = GlobAction(pattern=pattern, path=temp_workspace)
        primary = run(executor.glob(action))
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            fallback = run(executor.glob(action))

        assert isinstance(primary, CmdOutputObservation)
        assert isinstance(fallback, CmdOutputObservation)
        assert set(primary.content.splitlines()) == expected
        assert set(fallback.content.splitlines()) == expected

    @pytest.mark.parametrize(
        ('pattern', 'expected_names'),
        [
            ('*.txt', {'keep.txt', 'ignored.txt'}),
            (
                '*',
                {
                    '.gitignore',
                    'keep.txt',
                    'ignored.txt',
                    'ignored-dir/nested.txt',
                },
            ),
        ],
    )
    def test_missing_rg_glob_matches_real_rg_positive_glob_ignore_override(
        self,
        executor,
        temp_workspace,
        pattern,
        expected_names,
    ):
        import shutil

        if shutil.which('rg') is None:
            pytest.skip('ripgrep is required for the parity half of this test')

        create_test_file(
            temp_workspace,
            '.gitignore',
            'ignored.txt\nignored-dir/\n',
        )
        for name in ('keep.txt', 'ignored.txt', 'ignored-dir/nested.txt'):
            create_test_file(temp_workspace, name, '')

        action = GlobAction(pattern=pattern, path=temp_workspace)
        primary = run(executor.glob(action))
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            fallback = run(executor.glob(action))

        expected = {os.path.join(temp_workspace, name) for name in expected_names}
        assert isinstance(primary, CmdOutputObservation)
        assert isinstance(fallback, CmdOutputObservation)
        assert set(primary.content.splitlines()) == expected
        assert set(fallback.content.splitlines()) == expected

    @pytest.mark.parametrize(
        ('field', 'expected'),
        [
            ('pattern', 'pattern must not contain NUL bytes'),
            ('path', 'path must not contain NUL bytes'),
        ],
    )
    def test_glob_rejects_nul_subprocess_arguments(
        self,
        executor,
        temp_workspace,
        field,
        expected,
    ):
        kwargs = {'pattern': '*.py', 'path': temp_workspace}
        kwargs[field] = f'{kwargs[field]}\x00suffix'

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run'
        ) as run_rg:
            obs = run(executor.glob(GlobAction(**kwargs)))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == expected
        run_rg.assert_not_called()

    def test_missing_rg_fallback_reports_enumeration_cap(
        self, executor, temp_workspace
    ):
        from openhands.runtime import action_execution_server

        create_test_file(temp_workspace, 'first.txt', '')
        create_test_file(temp_workspace, 'second.txt', '')

        with (
            patch.object(
                action_execution_server.subprocess,
                'run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ),
            patch.object(
                action_execution_server,
                '_NATIVE_SEARCH_MAX_ENTRIES',
                1,
            ),
        ):
            obs = run(executor.glob(GlobAction(pattern='*.py', path=temp_workspace)))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'glob fallback failed: search enumerated more than 1 filesystem entries'
        )


# ==============================================================================
# Grep Handler Tests
# ==============================================================================


class TestGrepHandler:
    """Tests for the grep handler calling the actual handler method."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['grep'])

    def test_grep_finds_pattern(self, executor, temp_workspace):
        """Grep groups exact line results below absolute file headings."""
        records = [
            {
                'type': 'match',
                'data': {
                    'path': {'text': 'first.py'},
                    'line_number': 2,
                    'lines': {'text': 'def first():'},
                },
            },
            {
                'type': 'match',
                'data': {
                    'path': {'text': 'first.py'},
                    'line_number': 7,
                    'lines': {'text': 'def second():'},
                },
            },
            {
                'type': 'match',
                'data': {
                    'path': {'text': 'nested/third.py'},
                    'line_number': 1,
                    'lines': {'text': 'def third():'},
                },
            },
        ]
        rg_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='\n'.join(json.dumps(record) for record in records),
            stderr='',
        )
        action = GrepAction(pattern='def', path=temp_workspace)
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=rg_result,
        ):
            obs = run(executor.grep(action))

        first = os.path.join(temp_workspace, 'first.py')
        third = os.path.join(temp_workspace, 'nested', 'third.py')
        assert isinstance(obs, CmdOutputObservation)
        assert obs.success is True
        assert obs.exit_code == 0
        assert obs.content == (
            'Found 3 matches\n'
            f'{first}:\n'
            '  Line 2: def first():\n'
            '  Line 7: def second():\n\n'
            f'{third}:\n'
            '  Line 1: def third():'
        )

    def test_grep_decodes_bytes_json_and_skips_unsafe_payloads(
        self, executor, temp_workspace
    ):
        records = [
            {
                'type': 'match',
                'data': {
                    'path': {
                        'bytes': base64.b64encode(b'invalid-\xff.py').decode('ascii')
                    },
                    'line_number': 7,
                    'lines': {
                        'bytes': base64.b64encode(b'needle \xff\n').decode('ascii')
                    },
                },
            },
            {
                'type': 'match',
                'data': {
                    'path': {'bytes': 'not-valid-base64!'},
                    'line_number': 8,
                    'lines': {'text': 'must be skipped'},
                },
            },
            {
                'type': 'match',
                'data': {
                    'path': {'text': 'unsafe\x00.py'},
                    'line_number': 9,
                    'lines': {'text': 'must be skipped'},
                },
            },
            [],
        ]
        rg_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b'\n'.join(json.dumps(record).encode('utf-8') for record in records),
            stderr=b'',
        )
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=rg_result,
        ):
            obs = run(executor.grep(GrepAction(pattern='needle', path=temp_workspace)))

        expected_path = os.path.join(temp_workspace, 'invalid-�.py')
        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == (
            f'Found 1 matches\n{expected_path}:\n  Line 7: needle �\n'
        )
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_grep_matches_real_rg_nested_ignore_rules(
        self, executor, temp_workspace
    ):
        import shutil

        if shutil.which('rg') is None:
            pytest.skip('ripgrep is required for the parity half of this test')

        expected = {
            create_test_file(temp_workspace, 'visible.txt', 'needle\n'),
            create_test_file(temp_workspace, 'nested/keep.py', 'needle\n'),
            create_test_file(
                temp_workspace,
                'nested/deep/anchored.txt',
                'needle\n',
            ),
        }
        create_test_file(
            temp_workspace,
            'nested/.gitignore',
            '/anchored.txt\nignored/\n*.py\n!keep.py\n',
        )
        create_test_file(
            temp_workspace,
            'nested/.ignore',
            'ignored-by-ignore.txt\n',
        )
        create_test_file(
            temp_workspace,
            'nested/.rgignore',
            'ignored-by-rgignore.txt\n',
        )
        create_test_file(temp_workspace, 'nested/anchored.txt', 'needle\n')
        create_test_file(temp_workspace, 'nested/drop.py', 'needle\n')
        create_test_file(
            temp_workspace,
            'nested/ignored/secret.txt',
            'needle\n',
        )
        create_test_file(
            temp_workspace,
            'nested/ignored-by-ignore.txt',
            'needle\n',
        )
        create_test_file(
            temp_workspace,
            'nested/ignored-by-rgignore.txt',
            'needle\n',
        )

        action = GrepAction(pattern='needle', path=temp_workspace)
        primary = run(executor.grep(action))
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            fallback = run(executor.grep(action))

        def result_paths(observation):
            return {
                line[:-1]
                for line in observation.content.splitlines()
                if line.startswith(temp_workspace) and line.endswith(':')
            }

        assert isinstance(primary, CmdOutputObservation)
        assert isinstance(fallback, CmdOutputObservation)
        assert result_paths(primary) == expected
        assert result_paths(fallback) == expected
        assert primary.success is True
        assert fallback.success is True
        assert primary.exit_code == fallback.exit_code == 0

    @pytest.mark.parametrize(
        ('include', 'expected_names'),
        [
            ('*.txt', {'keep.txt', 'ignored.txt'}),
            (
                '*',
                {'keep.txt', 'ignored.txt', 'ignored-dir/nested.txt'},
            ),
        ],
    )
    def test_missing_rg_grep_matches_real_rg_positive_glob_ignore_override(
        self,
        executor,
        temp_workspace,
        include,
        expected_names,
    ):
        import shutil

        if shutil.which('rg') is None:
            pytest.skip('ripgrep is required for the parity half of this test')

        create_test_file(
            temp_workspace,
            '.gitignore',
            'ignored.txt\nignored-dir/\n',
        )
        for name in ('keep.txt', 'ignored.txt', 'ignored-dir/nested.txt'):
            create_test_file(temp_workspace, name, 'needle\n')

        action = GrepAction(
            pattern='needle',
            include=include,
            path=temp_workspace,
        )
        primary = run(executor.grep(action))
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            fallback = run(executor.grep(action))

        def result_paths(observation):
            return {
                line[:-1]
                for line in observation.content.splitlines()
                if line.startswith(temp_workspace) and line.endswith(':')
            }

        expected = {os.path.join(temp_workspace, name) for name in expected_names}
        assert isinstance(primary, CmdOutputObservation)
        assert isinstance(fallback, CmdOutputObservation)
        assert result_paths(primary) == expected
        assert result_paths(fallback) == expected
        assert primary.exit_code == fallback.exit_code == 0

    @pytest.mark.parametrize(
        ('field', 'expected'),
        [
            ('pattern', 'pattern must not contain NUL bytes'),
            ('include', 'include must not contain NUL bytes'),
            ('path', 'path must not contain NUL bytes'),
        ],
    )
    def test_grep_rejects_nul_subprocess_arguments(
        self,
        executor,
        temp_workspace,
        field,
        expected,
    ):
        kwargs = {'pattern': 'needle', 'path': temp_workspace}
        if field == 'pattern':
            kwargs['pattern'] = 'needle\x00suffix'
        elif field == 'include':
            kwargs['include'] = '*.py\x00suffix'
        else:
            kwargs['path'] = f'{temp_workspace}\x00suffix'

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run'
        ) as run_rg:
            obs = run(executor.grep(GrepAction(**kwargs)))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == expected
        run_rg.assert_not_called()

    def test_grep_with_line_numbers(self, executor, temp_workspace):
        """Test grep output includes line numbers."""
        filepath = create_test_file(temp_workspace, 'test.py', 'line1\nTARGET\nline3')
        action = GrepAction(pattern='TARGET', path=temp_workspace)
        obs = run(executor.grep(action))
        assert obs.content == (f'Found 1 matches\n{filepath}:\n  Line 2: TARGET\n')

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
        assert obs.content == 'No files found'

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
        filepath = create_test_file(temp_workspace, 'case.txt', 'Hello\nhello\nHELLO')
        action = GrepAction(pattern='hello', path=temp_workspace)
        obs = run(executor.grep(action))
        assert obs.content == (f'Found 1 matches\n{filepath}:\n  Line 2: hello\n')

    def test_grep_nonexistent_path(self, executor, temp_workspace):
        """Test grep on nonexistent path returns error."""
        action = GrepAction(
            pattern='test', path=os.path.join(temp_workspace, 'nonexistent')
        )
        obs = run(executor.grep(action))
        assert isinstance(obs, ErrorObservation)
        assert 'not exist' in obs.content.lower()

    def test_grep_basename_include_matches_recursively(self, executor, temp_workspace):
        """Ripgrep basename includes apply throughout the search directory."""
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
        filepath = create_test_file(
            temp_workspace, 'multi.py', 'TODO first\nother\nTODO second'
        )
        action = GrepAction(pattern='TODO', path=temp_workspace)
        obs = run(executor.grep(action))
        assert obs.content == (
            'Found 2 matches\n'
            f'{filepath}:\n'
            '  Line 1: TODO first\n\n'
            '  Line 3: TODO second'
        )

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
        """No matches returns OpenCode's exact empty-result body.

        From opencode grep.test.ts.
        """
        create_test_file(temp_workspace, 'test.txt', 'hello world')
        action = GrepAction(pattern='xyznonexistentpatternxyz123', path=temp_workspace)
        obs = run(executor.grep(action))
        assert obs.content == 'No files found'

    def test_grep_invalid_regex_returns_error(self, executor, temp_workspace):
        """Invalid regex pattern (unmatched paren) returns informative ErrorObservation.

        Ripgrep's non-search-error exit code and stderr must not be converted to
        the successful ``No files found`` body.
        """
        create_test_file(
            temp_workspace, 'func.py', 'def write_records(data):\n    pass'
        )
        action = GrepAction(pattern='write_records(', path=temp_workspace)
        obs = run(executor.grep(action))
        assert isinstance(obs, ErrorObservation), (
            f'Expected ErrorObservation for invalid regex, got: {type(obs).__name__}: {obs.content}'
        )
        assert 'regex parse error' in obs.content.lower()

    def test_grep_valid_regex_alternation_works(self, executor, temp_workspace):
        """Valid regex with alternation (|) should still work."""
        create_test_file(temp_workspace, 'a.py', 'def foo(): pass')
        create_test_file(temp_workspace, 'b.py', 'def bar(): pass')
        action = GrepAction(pattern='foo|bar', path=temp_workspace)
        obs = run(executor.grep(action))
        assert isinstance(obs, CmdOutputObservation)
        assert 'foo' in obs.content
        assert 'bar' in obs.content

    def test_missing_rg_fallback_handles_globs_hidden_binary_and_unreadable(
        self, executor, temp_workspace
    ):
        from pathlib import Path

        expected = create_test_file(temp_workspace, 'src/good.py', 'needle\n')
        hidden = create_test_file(
            temp_workspace,
            '.hidden/good.pyi',
            'needle hidden\n',
        )
        create_test_file(temp_workspace, '.gitignore', 'node_modules/\n')
        create_test_file(temp_workspace, '.ignore', 'ignored*.py\n')
        create_test_file(temp_workspace, '.rgignore', 'generated/\n')
        create_test_file(temp_workspace, '.git/secret.py', 'needle git\n')
        create_test_file(
            temp_workspace,
            'nested/node_modules/pkg/dependency.py',
            'needle dependency\n',
        )
        ignored = create_test_file(
            temp_workspace,
            'nested/ignored-output.py',
            'needle ignored\n',
        )
        create_test_file(
            temp_workspace,
            'nested/generated/output.py',
            'needle generated\n',
        )
        create_test_file(temp_workspace, 'src/excluded.txt', 'needle text\n')
        unreadable = create_test_file(
            temp_workspace,
            'src/unreadable.py',
            'needle unreadable\n',
        )
        binary = os.path.join(temp_workspace, 'src', 'binary.py')
        with open(binary, 'wb') as target:
            target.write(b'needle before nul\nthen\x00binary\n')

        original_open = Path.open

        def selective_open(path, *args, **kwargs):
            if str(path) == unreadable:
                raise PermissionError(13, os.strerror(13), unreadable)
            return original_open(path, *args, **kwargs)

        with (
            patch(
                'openhands.runtime.action_execution_server.subprocess.run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ),
            patch(
                'openhands.runtime.action_execution_server.Path.open',
                autospec=True,
                side_effect=selective_open,
            ),
        ):
            obs = run(
                executor.grep(
                    GrepAction(
                        pattern='needle',
                        path=temp_workspace,
                        include='**/*.{py,pyi}',
                    )
                )
            )

        assert isinstance(obs, CmdOutputObservation)
        assert expected in obs.content
        assert hidden in obs.content
        assert 'binary.py' not in obs.content
        assert 'unreadable.py' not in obs.content
        assert '.git' not in obs.content
        assert 'node_modules' not in obs.content
        # The positive include explicitly selects this file, overriding the
        # matching .ignore rule just as rg does.
        assert ignored in obs.content
        assert 'generated' not in obs.content
        assert 'excluded.txt' not in obs.content
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_explicit_file_bypasses_root_ignore_rules(
        self, executor, temp_workspace
    ):
        create_test_file(temp_workspace, '.gitignore', 'ignored.py\n')
        expected = create_test_file(
            temp_workspace,
            'ignored.py',
            'needle\n',
        )

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = run(executor.grep(GrepAction(pattern='needle', path=expected)))

        assert isinstance(obs, CmdOutputObservation)
        assert expected in obs.content
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_fallback_catastrophic_regex_times_out(
        self, executor, temp_workspace
    ):
        import time

        from openhands.runtime import action_execution_server

        create_test_file(
            temp_workspace,
            'catastrophic.txt',
            ('a' * 100_000) + '!\n',
        )
        started_at = time.monotonic()

        with (
            patch.object(
                action_execution_server.subprocess,
                'run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ),
            patch.object(
                action_execution_server,
                '_NATIVE_SEARCH_TIMEOUT_SECONDS',
                0.005,
            ),
        ):
            obs = run(
                executor.grep(
                    GrepAction(
                        pattern=r'(a+)+$',
                        path=temp_workspace,
                    )
                )
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'grep fallback timed out after 30 seconds'
        assert time.monotonic() - started_at < 1

    def test_missing_rg_fallback_invalid_regex_is_controlled(
        self, executor, temp_workspace
    ):
        create_test_file(temp_workspace, 'source.py', 'content')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = run(
                executor.grep(GrepAction(pattern='unclosed(', path=temp_workspace))
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content.startswith(
            'grep fallback failed: invalid regular expression:'
        )

    def test_missing_rg_fallback_reports_enumeration_cap(
        self, executor, temp_workspace
    ):
        from openhands.runtime import action_execution_server

        create_test_file(temp_workspace, 'first.txt', 'other')
        create_test_file(temp_workspace, 'second.txt', 'other')

        with (
            patch.object(
                action_execution_server.subprocess,
                'run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ),
            patch.object(
                action_execution_server,
                '_NATIVE_SEARCH_MAX_ENTRIES',
                1,
            ),
        ):
            obs = run(executor.grep(GrepAction(pattern='needle', path=temp_workspace)))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'grep fallback failed: search enumerated more than 1 filesystem entries'
        )

    def test_missing_rg_fallback_rejects_oversized_newline_free_line(
        self, executor, temp_workspace
    ):
        from openhands.runtime import action_execution_server

        source = os.path.join(temp_workspace, 'huge.txt')
        with open(source, 'wb') as target:
            target.write(b'x' * 17)

        with (
            patch.object(
                action_execution_server.subprocess,
                'run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ),
            patch.object(
                action_execution_server,
                '_NATIVE_SEARCH_MAX_LINE_BYTES',
                16,
            ),
        ):
            obs = run(executor.grep(GrepAction(pattern='x', path=temp_workspace)))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'grep fallback failed: file contains a line exceeding '
            'the 16-byte fallback limit'
        )

    def test_missing_rg_fallback_stops_reading_at_match_cap(
        self, executor, temp_workspace
    ):
        from openhands.runtime import action_execution_server

        source = os.path.join(temp_workspace, 'many.txt')
        with open(source, 'wb') as target:
            target.write((b'needle\n' * 100) + (b'x' * 33))

        with (
            patch.object(
                action_execution_server.subprocess,
                'run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ),
            patch.object(
                action_execution_server,
                '_NATIVE_SEARCH_MAX_LINE_BYTES',
                32,
            ),
        ):
            obs = run(executor.grep(GrepAction(pattern='needle', path=temp_workspace)))

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content.startswith('Found 100 matches (more matches available)')
        assert 'Results truncated.' in obs.content
        assert obs.success is True
        assert obs.exit_code == 0


# ==============================================================================
# ListDir Handler Tests
# ==============================================================================


class TestListDirHandler:
    """Tests for the list_dir handler calling the actual handler method."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(temp_workspace, ['list_dir'])

    def test_list_dir_finds_files(self, executor, temp_workspace):
        """Legacy list_dir uses the read-directory body and immediate entries."""
        create_test_structure(temp_workspace)
        action = ListDirAction(path=temp_workspace)
        obs = run(executor.list_dir(action))
        assert isinstance(obs, CmdOutputObservation)
        assert obs.success is True
        assert obs.exit_code == 0
        assert obs.content == (
            f'<path>{temp_workspace}</path>\n'
            '<type>directory</type>\n'
            '<entries>\n'
            '.gitignore\n'
            'config.json\n'
            'main.py\n'
            'README.md\n'
            'src/\n'
            'tests/\n'
            'utils.py\n\n'
            '(7 entries)\n'
            '</entries>'
        )

    def test_list_dir_does_not_recurse(self, executor, temp_workspace):
        """Directory entries are immediate rather than a recursive tree."""
        create_test_file(temp_workspace, 'root.py', '')
        create_test_file(temp_workspace, 'src/core.py', '')
        action = ListDirAction(path=temp_workspace)
        obs = run(executor.list_dir(action))
        assert obs.content == (
            f'<path>{temp_workspace}</path>\n'
            '<type>directory</type>\n'
            '<entries>\n'
            'root.py\n'
            'src/\n\n'
            '(2 entries)\n'
            '</entries>'
        )

    def test_list_dir_empty_directory(self, executor, temp_workspace):
        """Test listing an empty directory."""
        empty_dir = os.path.join(temp_workspace, 'empty')
        os.makedirs(empty_dir)
        action = ListDirAction(path=empty_dir)
        obs = run(executor.list_dir(action))
        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == (
            f'<path>{empty_dir}</path>\n'
            '<type>directory</type>\n'
            '<entries>\n\n\n'
            '(0 entries)\n'
            '</entries>'
        )

    def test_list_dir_relative_path(self, executor, temp_workspace):
        """Test listing with a relative path."""
        os.makedirs(os.path.join(temp_workspace, 'mydir'))
        create_test_file(temp_workspace, 'mydir/a.txt', '')
        action = ListDirAction(path='mydir')
        obs = run(executor.list_dir(action))
        assert isinstance(obs, CmdOutputObservation)
        mydir = os.path.join(temp_workspace, 'mydir')
        assert obs.content == (
            f'<path>{mydir}</path>\n'
            '<type>directory</type>\n'
            '<entries>\n'
            'a.txt\n\n'
            '(1 entries)\n'
            '</entries>'
        )

    def test_list_dir_lists_ignored_names_but_not_their_contents(
        self, executor, temp_workspace
    ):
        """The legacy alias lists every immediate entry without ignore rules."""
        create_test_file(temp_workspace, 'main.py', '')
        create_test_file(temp_workspace, 'node_modules/pkg/index.js', '')
        create_test_file(temp_workspace, '__pycache__/module.pyc', '')
        action = ListDirAction(path=temp_workspace)
        obs = run(executor.list_dir(action))
        assert obs.content == (
            f'<path>{temp_workspace}</path>\n'
            '<type>directory</type>\n'
            '<entries>\n'
            '__pycache__/\n'
            'main.py\n'
            'node_modules/\n\n'
            '(3 entries)\n'
            '</entries>'
        )


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
        assert obs.content == '[]'
        assert obs.todos == []

    def test_todo_write_adds_items(self, executor):
        """Todo write echoes pretty, unescaped Unicode JSON."""
        todos = [
            {
                'content': 'Review café ✅',
                'status': 'pending',
                'priority': 'high',
            },
            {
                'content': 'Ship release',
                'status': 'in_progress',
                'priority': 'medium',
            },
        ]
        action = TodoWriteAction(todos=todos)
        obs = run(executor.todo_write(action))
        assert obs.success is True
        assert obs.todos == todos
        assert obs.content == (
            '[\n'
            '  {\n'
            '    "content": "Review café ✅",\n'
            '    "status": "pending",\n'
            '    "priority": "high"\n'
            '  },\n'
            '  {\n'
            '    "content": "Ship release",\n'
            '    "status": "in_progress",\n'
            '    "priority": "medium"\n'
            '  }\n'
            ']'
        )

    def test_todo_write_then_read(self, executor):
        """Test writing then reading todos."""
        todos = [
            {'content': 'Task', 'status': 'pending', 'priority': 'low'},
        ]
        expected = json.dumps(todos, indent=2, ensure_ascii=False)
        write_obs = run(executor.todo_write(TodoWriteAction(todos=todos)))
        assert write_obs.content == expected

        read_action = TodoReadAction()
        obs = run(executor.todo_read(read_action))
        assert obs.content == expected
        assert obs.todos == todos

    def test_todo_write_uses_generic_opencode_truncation(self, executor):
        todos = [{'id': index} for index in range(1_500)]
        expected = json.dumps(todos, indent=2, ensure_ascii=False)

        obs = run(executor.todo_write(TodoWriteAction(todos=todos)))

        assert ' lines truncated...\n\n' in obs.content
        path_prefix = 'Full output saved to: '
        path_start = obs.content.index(path_prefix) + len(path_prefix)
        path_end = obs.content.index('\n', path_start)
        output_path = obs.content[path_start:path_end]
        try:
            with open(output_path, encoding='utf-8') as saved_output:
                assert saved_output.read() == expected
        finally:
            os.unlink(output_path)

    def test_todo_write_replaces_existing_state(self, executor):
        """Each todo write replaces, rather than merges with, prior state."""
        write1 = TodoWriteAction(
            todos=[
                {'content': 'Old task', 'status': 'pending', 'priority': 'low'},
            ]
        )
        run(executor.todo_write(write1))

        replacement = [
            {
                'content': 'New task',
                'status': 'completed',
                'priority': 'high',
            },
        ]
        obs = run(executor.todo_write(TodoWriteAction(todos=replacement)))
        assert obs.success is True
        assert obs.todos == replacement
        assert obs.content == json.dumps(replacement, indent=2, ensure_ascii=False)

    def test_todo_write_empty_list_clears_existing(self, executor):
        """Replacing with an empty list clears all retained todo state."""
        run(
            executor.todo_write(
                TodoWriteAction(
                    todos=[
                        {'content': 'First', 'status': 'pending', 'priority': 'low'},
                    ]
                )
            )
        )
        obs = run(executor.todo_write(TodoWriteAction(todos=[])))
        assert obs.todos == []
        assert obs.content == '[]'


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
        action = QuestionAction(
            questions=[
                {'id': 'q1', 'text': 'What framework?', 'options': ['React', 'Vue']},
            ]
        )
        obs = run(executor.question(action))
        assert 'What framework?' in obs.content


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestOpenCodeIntegration:
    """Integration tests combining multiple handler operations."""

    @pytest.fixture
    def executor(self, temp_workspace):
        return _make_executor(
            temp_workspace,
            [
                'opencode_read',
                'opencode_write',
                'glob',
                'grep',
                'list_dir',
            ],
        )

    def test_write_then_read(self, executor, temp_workspace):
        """Test full write then read workflow."""
        filepath = os.path.join(temp_workspace, 'workflow.py')
        write_action = OpenCodeWriteAction(
            path=filepath, content='def hello():\n    print("Hello")\n'
        )
        run(executor.opencode_write(write_action))

        read_action = OpenCodeReadAction(path=filepath)
        obs = run(executor.opencode_read(read_action))
        assert obs.content == (
            f'<path>{filepath}</path>\n'
            '<type>file</type>\n'
            '<content>\n'
            '1: def hello():\n'
            '2:     print("Hello")\n\n'
            '(End of file - total 2 lines)\n'
            '</content>'
        )

    def test_write_then_glob(self, executor, temp_workspace):
        """Test write then glob to find the written file."""
        filepath = os.path.join(temp_workspace, 'written.py')
        run(
            executor.opencode_write(
                OpenCodeWriteAction(path=filepath, content='content')
            )
        )
        obs = run(executor.glob(GlobAction(pattern='*.py', path=temp_workspace)))
        assert 'written.py' in obs.content

    def test_write_then_grep(self, executor, temp_workspace):
        """Test write then grep to find content."""
        filepath = os.path.join(temp_workspace, 'search.py')
        run(
            executor.opencode_write(
                OpenCodeWriteAction(path=filepath, content='UNIQUE_MARKER_XYZ')
            )
        )
        obs = run(
            executor.grep(GrepAction(pattern='UNIQUE_MARKER_XYZ', path=temp_workspace))
        )
        assert 'search.py' in obs.content

    def test_list_then_read(self, executor, temp_workspace):
        """Test listing then reading found files."""
        create_test_structure(temp_workspace)
        list_obs = run(executor.list_dir(ListDirAction(path=temp_workspace)))
        assert 'main.py' in list_obs.content

        read_obs = run(
            executor.opencode_read(
                OpenCodeReadAction(path=os.path.join(temp_workspace, 'main.py'))
            )
        )
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
            pytest.skip('ripgrep (rg) not available')

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
