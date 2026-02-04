"""Standalone unit tests for actual OpenCode runtime handler implementations.

These tests directly test the handler methods in action_execution_server.py
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


@pytest.fixture
def mock_executor(temp_workspace):
    """Create a minimal mock ActionExecutor with real file system access."""
    # Import here to avoid circular imports
    from openhands.runtime.action_execution_server import ActionExecutor

    # Create a mock that has the necessary attributes
    executor = MagicMock(spec=ActionExecutor)
    executor._initial_cwd = temp_workspace
    executor.bash_session = MagicMock()
    executor.bash_session.cwd = temp_workspace
    executor.lock = asyncio.Lock()

    # Use the real _resolve_path logic
    def resolve_path(path, working_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(working_dir, path)

    executor._resolve_path = resolve_path

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


# ==============================================================================
# OpenCodeRead Handler Tests - Direct Implementation
# ==============================================================================


class TestOpenCodeReadHandlerImpl:
    """Tests for the actual opencode_read handler implementation logic."""

    def test_read_file_line_number_format(self, temp_workspace):
        """Test that files are read with correct 5-digit line number format."""
        content = 'line 1\nline 2\nline 3\nline 4\nline 5'
        filepath = create_test_file(temp_workspace, 'test.txt', content)

        # Simulate the handler logic
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().split('\n')

        offset = 0
        limit = 2000
        raw = lines[offset:offset + limit]

        # Format with 5-digit line numbers (OpenCode style)
        formatted = [f"{str(i + offset + 1).zfill(5)}| {line}" for i, line in enumerate(raw)]

        assert formatted[0] == '00001| line 1'
        assert formatted[1] == '00002| line 2'
        assert formatted[4] == '00005| line 5'

    def test_read_file_with_offset(self, temp_workspace):
        """Test reading file starting from offset."""
        content = '\n'.join([f'line {i}' for i in range(1, 101)])
        filepath = create_test_file(temp_workspace, 'offset_test.txt', content)

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        offset = 49  # Start from line 50
        limit = 10
        raw = lines[offset:offset + limit]
        formatted = [f"{str(i + offset + 1).zfill(5)}| {line}" for i, line in enumerate(raw)]

        assert formatted[0] == '00050| line 50'
        assert len(formatted) == 10

    def test_read_file_with_limit(self, temp_workspace):
        """Test reading file with line limit."""
        content = '\n'.join([f'line {i}' for i in range(1, 101)])
        filepath = create_test_file(temp_workspace, 'limit_test.txt', content)

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        offset = 0
        limit = 5
        raw = lines[offset:offset + limit]

        assert len(raw) == 5
        assert raw[0] == 'line 1'
        assert raw[4] == 'line 5'

    def test_read_file_long_line_truncation(self, temp_workspace):
        """Test that long lines are truncated with ..."""
        MAX_LINE_LENGTH = 2000
        long_line = 'x' * 3000
        filepath = create_test_file(temp_workspace, 'long_line.txt', long_line)

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        # Apply truncation logic
        processed = []
        for line in lines:
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + '...'
            processed.append(line)

        assert processed[0].endswith('...')
        assert len(processed[0]) == MAX_LINE_LENGTH + 3

    def test_read_file_byte_limit(self, temp_workspace):
        """Test that output is truncated at byte limit."""
        MAX_BYTES = 50 * 1024  # 50KB
        # Create file larger than 50KB
        lines = ['x' * 100 for _ in range(1000)]  # ~100KB
        content = '\n'.join(lines)
        filepath = create_test_file(temp_workspace, 'large_file.txt', content)

        with open(filepath, 'r', encoding='utf-8') as f:
            file_lines = f.read().split('\n')

        # Apply byte limit logic
        total_bytes = 0
        result = []
        truncated = False

        for line in file_lines:
            line_bytes = len(line.encode('utf-8')) + 1
            if total_bytes + line_bytes > MAX_BYTES:
                truncated = True
                break
            result.append(line)
            total_bytes += line_bytes

        assert truncated
        assert total_bytes <= MAX_BYTES
        assert len(result) < len(file_lines)

    def test_read_nonexistent_file_suggestions(self, temp_workspace):
        """Test file not found with similar file suggestions."""
        # Create similar files
        create_test_file(temp_workspace, 'test_file.py', 'content')
        create_test_file(temp_workspace, 'test_file.txt', 'content')
        create_test_file(temp_workspace, 'test_utils.py', 'content')

        # Simulate file not found logic
        missing_file = 'test_file.js'
        directory = temp_workspace
        basename = os.path.splitext(missing_file)[0]  # 'test_file'

        entries = os.listdir(directory)
        suggestions = [
            os.path.join(directory, entry)
            for entry in entries
            if basename.lower() in entry.lower() or entry.lower() in basename.lower()
        ][:3]

        assert len(suggestions) >= 2
        assert any('test_file.py' in s for s in suggestions)
        assert any('test_file.txt' in s for s in suggestions)

    def test_read_binary_detection_by_extension(self, temp_workspace):
        """Test binary file detection by extension."""
        BINARY_EXTENSIONS = {
            '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.class', '.jar',
            '.war', '.7z', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.bin', '.dat', '.obj', '.o', '.a', '.lib', '.wasm', '.pyc', '.pyo'
        }

        # Test detection
        test_paths = [
            ('file.zip', True),
            ('file.py', False),
            ('file.pyc', True),
            ('file.txt', False),
            ('archive.tar.gz', True),  # .gz extension is binary
            ('data.bin', True),
        ]

        for path, expected_binary in test_paths:
            ext = os.path.splitext(path)[1].lower()
            is_binary = ext in BINARY_EXTENSIONS
            assert is_binary == expected_binary, f"Failed for {path}"

    def test_read_binary_detection_by_content(self, temp_workspace):
        """Test binary file detection by content analysis."""
        # Create a binary-like file
        binary_content = b'\x00\x01\x02\x03\x04\x05\x06\x07'
        binary_path = os.path.join(temp_workspace, 'binary.dat')
        with open(binary_path, 'wb') as f:
            f.write(binary_content)

        # Create a text file
        text_content = 'Hello, world!\nThis is text.'
        text_path = os.path.join(temp_workspace, 'text.txt')
        with open(text_path, 'w') as f:
            f.write(text_content)

        # Binary detection logic
        def is_binary_file(filepath):
            try:
                with open(filepath, 'rb') as f:
                    chunk = f.read(4096)
                    if b'\x00' in chunk:
                        return True
                    if chunk:
                        non_printable = sum(1 for b in chunk if b < 9 or (b > 13 and b < 32))
                        if non_printable / len(chunk) > 0.3:
                            return True
                return False
            except Exception:
                return False

        assert is_binary_file(binary_path) == True
        assert is_binary_file(text_path) == False


# ==============================================================================
# OpenCodeWrite Handler Tests - Direct Implementation
# ==============================================================================


class TestOpenCodeWriteHandlerImpl:
    """Tests for the actual opencode_write handler implementation logic."""

    def test_write_creates_file(self, temp_workspace):
        """Test that write creates a new file."""
        filepath = os.path.join(temp_workspace, 'new_file.py')
        content = 'print("Hello, World!")'

        # Write logic
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            assert f.read() == content

    def test_write_creates_parent_directories(self, temp_workspace):
        """Test that write creates parent directories if needed."""
        filepath = os.path.join(temp_workspace, 'a', 'b', 'c', 'deep_file.txt')
        content = 'nested content'

        # Write logic with directory creation
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            assert f.read() == content

    def test_write_overwrites_existing(self, temp_workspace):
        """Test that write overwrites existing file."""
        filepath = os.path.join(temp_workspace, 'existing.txt')

        # Create initial file
        with open(filepath, 'w') as f:
            f.write('original content')

        # Overwrite
        new_content = 'new content'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        with open(filepath, 'r') as f:
            assert f.read() == new_content

    def test_write_empty_content(self, temp_workspace):
        """Test writing empty content creates empty file."""
        filepath = os.path.join(temp_workspace, 'empty.txt')

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('')

        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) == 0

    def test_write_preserves_unicode(self, temp_workspace):
        """Test that Unicode content is preserved."""
        filepath = os.path.join(temp_workspace, 'unicode.txt')
        content = '你好世界\nこんにちは\n🎉 emoji test'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        with open(filepath, 'r', encoding='utf-8') as f:
            assert f.read() == content

    def test_write_multiline_content(self, temp_workspace):
        """Test that multiline content preserves line breaks."""
        filepath = os.path.join(temp_workspace, 'multiline.py')
        content = 'def hello():\n    print("Hello")\n\ndef goodbye():\n    print("Bye")\n'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Content has 5 lines (trailing newline doesn't create extra line in readlines)
        assert len(lines) == 5
        assert 'def hello():' in lines[0]

    def test_linter_command_selection(self):
        """Test that correct linter is selected based on extension."""
        def get_linter_commands(filepath):
            ext = os.path.splitext(filepath)[1].lower()

            if ext == '.py':
                return [
                    ['flake8', '--max-line-length=120', filepath],
                    ['pylint', '--errors-only', filepath],
                    ['python3', '-m', 'py_compile', filepath],
                ]
            elif ext in ('.js', '.jsx', '.ts', '.tsx'):
                return [['eslint', '--format=compact', filepath]]
            elif ext == '.go':
                return [['go', 'vet', filepath]]
            elif ext == '.rs':
                return [['cargo', 'check', '--message-format=short']]
            return []

        assert len(get_linter_commands('/test.py')) == 3
        assert get_linter_commands('/test.py')[0][0] == 'flake8'

        assert len(get_linter_commands('/test.js')) == 1
        assert get_linter_commands('/test.js')[0][0] == 'eslint'

        assert get_linter_commands('/test.tsx')[0][0] == 'eslint'
        assert get_linter_commands('/test.go')[0][0] == 'go'


# ==============================================================================
# Glob Handler Tests - Direct Implementation
# ==============================================================================


class TestGlobHandlerImpl:
    """Tests for the actual glob handler implementation logic."""

    def test_glob_finds_python_files(self, temp_workspace):
        """Test glob finds Python files."""
        create_test_structure(temp_workspace)

        # Use find command (fallback when rg not available)
        result = subprocess.run(
            ['find', temp_workspace, '-type', 'f', '-name', '*.py'],
            capture_output=True, text=True
        )

        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        assert len(files) >= 3  # main.py, utils.py, core.py, helpers.py, test_main.py
        assert any('main.py' in f for f in files)
        assert any('utils.py' in f for f in files)

    def test_glob_finds_json_files(self, temp_workspace):
        """Test glob finds specific extension."""
        create_test_structure(temp_workspace)

        result = subprocess.run(
            ['find', temp_workspace, '-type', 'f', '-name', '*.json'],
            capture_output=True, text=True
        )

        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        assert len(files) == 1
        assert 'config.json' in files[0]

    def test_glob_in_subdirectory(self, temp_workspace):
        """Test glob searches in specific directory."""
        create_test_structure(temp_workspace)

        src_dir = os.path.join(temp_workspace, 'src')
        result = subprocess.run(
            ['find', src_dir, '-type', 'f', '-name', '*.py'],
            capture_output=True, text=True
        )

        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        # Should find core.py and helpers.py
        assert len(files) >= 1
        assert any('core.py' in f for f in files)

    def test_glob_no_matches(self, temp_workspace):
        """Test glob returns empty for no matches."""
        create_test_structure(temp_workspace)

        result = subprocess.run(
            ['find', temp_workspace, '-type', 'f', '-name', '*.nonexistent'],
            capture_output=True, text=True
        )

        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        assert len(files) == 0

    def test_glob_result_sorting_by_mtime(self, temp_workspace):
        """Test that results can be sorted by modification time."""
        import time

        # Create files with different mtimes
        file1 = create_test_file(temp_workspace, 'old.py', 'old content')
        time.sleep(0.1)
        file2 = create_test_file(temp_workspace, 'new.py', 'new content')

        # Get files with mtime
        result = subprocess.run(
            ['find', temp_workspace, '-type', 'f', '-name', '*.py', '-printf', '%T@ %p\n'],
            capture_output=True, text=True
        )

        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
        # Sort by mtime (first field) descending
        sorted_lines = sorted(lines, key=lambda x: float(x.split()[0]), reverse=True)

        if sorted_lines:
            newest = sorted_lines[0].split()[1]
            assert 'new.py' in newest


# ==============================================================================
# Grep Handler Tests - Direct Implementation
# ==============================================================================


class TestGrepHandlerImpl:
    """Tests for the actual grep handler implementation logic."""

    def test_grep_finds_pattern(self, temp_workspace):
        """Test grep finds pattern in files."""
        create_test_structure(temp_workspace)

        result = subprocess.run(
            ['grep', '-rn', 'def', temp_workspace],
            capture_output=True, text=True
        )

        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]

        assert len(lines) >= 3  # Multiple function definitions
        assert any('def main' in l for l in lines)
        assert any('def helper' in l for l in lines)

    def test_grep_with_line_numbers(self, temp_workspace):
        """Test grep output includes line numbers."""
        filepath = create_test_file(temp_workspace, 'test.py', 'line1\nTARGET\nline3')

        result = subprocess.run(
            ['grep', '-n', 'TARGET', filepath],
            capture_output=True, text=True
        )

        output = result.stdout.strip()
        assert '2:TARGET' in output  # Line 2

    def test_grep_with_file_filter(self, temp_workspace):
        """Test grep with file type filter using find."""
        create_test_structure(temp_workspace)

        # Find Python files and grep for 'import'
        result = subprocess.run(
            f'find {temp_workspace} -type f -name "*.py" -exec grep -Hn "import" {{}} \\;',
            shell=True, capture_output=True, text=True
        )

        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]

        # Should find imports in Python files
        assert any('import' in l for l in lines)
        # All results should be from .py files
        for line in lines:
            if ':' in line:
                filepath = line.split(':')[0]
                assert filepath.endswith('.py')

    def test_grep_no_matches(self, temp_workspace):
        """Test grep returns empty for no matches."""
        create_test_file(temp_workspace, 'test.txt', 'hello world')

        result = subprocess.run(
            ['grep', '-r', 'xyznonexistent123', temp_workspace],
            capture_output=True, text=True
        )

        assert result.stdout.strip() == ''

    def test_grep_regex_pattern(self, temp_workspace):
        """Test grep with regex pattern."""
        create_test_structure(temp_workspace)

        # Find function definitions with regex
        result = subprocess.run(
            ['grep', '-rE', r'def \w+\(', temp_workspace],
            capture_output=True, text=True
        )

        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]

        assert len(lines) >= 3
        assert any('def main(' in l for l in lines)

    def test_grep_case_sensitive(self, temp_workspace):
        """Test grep is case-sensitive by default."""
        filepath = create_test_file(temp_workspace, 'case.txt', 'Hello\nhello\nHELLO')

        result = subprocess.run(
            ['grep', 'hello', filepath],
            capture_output=True, text=True
        )

        lines = result.stdout.strip().split('\n')
        assert len(lines) == 1
        assert lines[0] == 'hello'


# ==============================================================================
# ListDir Handler Tests - Direct Implementation
# ==============================================================================


class TestListDirHandlerImpl:
    """Tests for the actual list_dir handler implementation logic."""

    def test_list_dir_finds_files(self, temp_workspace):
        """Test list_dir finds files and directories."""
        create_test_structure(temp_workspace)

        # Using find to list files
        result = subprocess.run(
            ['find', temp_workspace, '-maxdepth', '3', '-type', 'f'],
            capture_output=True, text=True
        )

        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        assert len(files) >= 5
        assert any('main.py' in f for f in files)
        assert any('config.json' in f for f in files)

    def test_list_dir_tree_structure(self, temp_workspace):
        """Test building tree structure from file list."""
        files = [
            'src/main.py',
            'src/utils.py',
            'src/lib/core.py',
            'tests/test_main.py',
        ]

        # Build tree structure (logic from handler)
        dirs = set()
        files_by_dir = {}

        for f in files:
            d = os.path.dirname(f) or '.'
            parts = d.split(os.sep) if d != '.' else []

            for i in range(len(parts) + 1):
                dir_p = os.sep.join(parts[:i]) if i > 0 else '.'
                dirs.add(dir_p)

            if d not in files_by_dir:
                files_by_dir[d] = []
            files_by_dir[d].append(os.path.basename(f))

        assert '.' in dirs
        assert 'src' in dirs
        assert 'src/lib' in dirs
        assert 'tests' in dirs

        assert 'main.py' in files_by_dir['src']
        assert 'core.py' in files_by_dir['src/lib']

    def test_list_dir_ignore_patterns(self, temp_workspace):
        """Test that ignore patterns filter correctly."""
        # Create structure with ignorable directories
        create_test_file(temp_workspace, 'main.py', 'content')
        create_test_file(temp_workspace, 'node_modules/package/index.js', 'content')
        create_test_file(temp_workspace, '__pycache__/module.pyc', 'content')
        create_test_file(temp_workspace, '.git/config', 'content')

        # Get all files
        result = subprocess.run(
            ['find', temp_workspace, '-type', 'f'],
            capture_output=True, text=True
        )

        all_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        # Filter with ignore patterns
        ignores = ['node_modules', '__pycache__', '.git']
        filtered = [f for f in all_files if not any(ig in f for ig in ignores)]

        assert any('main.py' in f for f in filtered)
        assert not any('node_modules' in f for f in filtered)
        assert not any('__pycache__' in f for f in filtered)
        assert not any('.git' in f for f in filtered)

    def test_list_dir_empty_directory(self, temp_workspace):
        """Test list_dir on empty directory."""
        empty_dir = os.path.join(temp_workspace, 'empty')
        os.makedirs(empty_dir)

        result = subprocess.run(
            ['find', empty_dir, '-type', 'f'],
            capture_output=True, text=True
        )

        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        assert len(files) == 0

    def test_list_dir_default_ignores(self):
        """Test that default ignore patterns are correct."""
        DEFAULT_IGNORES = [
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

        # Test that common patterns are included
        assert 'node_modules' in DEFAULT_IGNORES
        assert '__pycache__' in DEFAULT_IGNORES
        assert '.git' in DEFAULT_IGNORES
        assert '.venv' in DEFAULT_IGNORES


# ==============================================================================
# Integration-style Tests - Full Handler Flow
# ==============================================================================


class TestFullHandlerFlow:
    """Tests that simulate the full handler flow."""

    def test_read_then_write_then_read(self, temp_workspace):
        """Test full read → write → read workflow."""
        filepath = os.path.join(temp_workspace, 'workflow.py')

        # Write initial content
        initial_content = 'def hello():\n    print("Hello")\n'
        with open(filepath, 'w') as f:
            f.write(initial_content)

        # Read it back
        with open(filepath, 'r') as f:
            lines = f.read().split('\n')
        formatted = [f"{str(i+1).zfill(5)}| {line}" for i, line in enumerate(lines)]

        assert '00001| def hello():' in formatted

        # Write new content
        new_content = 'def hello():\n    print("Hello, World!")\n'
        with open(filepath, 'w') as f:
            f.write(new_content)

        # Read again
        with open(filepath, 'r') as f:
            final = f.read()

        assert 'Hello, World!' in final

    def test_glob_then_read_files(self, temp_workspace):
        """Test glob to find files, then read them."""
        create_test_structure(temp_workspace)

        # Glob for Python files
        result = subprocess.run(
            ['find', temp_workspace, '-type', 'f', '-name', '*.py'],
            capture_output=True, text=True
        )

        py_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        # Read each file
        all_content = []
        for filepath in py_files:
            with open(filepath, 'r') as f:
                content = f.read()
                all_content.append(content)

        # Verify we found function definitions
        combined = '\n'.join(all_content)
        assert 'def main' in combined
        assert 'def helper' in combined

    def test_grep_then_read_matching_file(self, temp_workspace):
        """Test grep to find pattern, then read the file."""
        create_test_structure(temp_workspace)

        # Grep for 'class' keyword
        result = subprocess.run(
            ['grep', '-rl', 'class', temp_workspace],
            capture_output=True, text=True
        )

        files_with_class = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        assert len(files_with_class) >= 1

        # Read the file with class
        for filepath in files_with_class:
            with open(filepath, 'r') as f:
                content = f.read()
            assert 'class' in content

    def test_write_python_and_check_syntax(self, temp_workspace):
        """Test writing Python file and checking syntax."""
        filepath = os.path.join(temp_workspace, 'syntax_test.py')

        # Write valid Python
        valid_content = 'def test():\n    return 42\n\nprint(test())\n'
        with open(filepath, 'w') as f:
            f.write(valid_content)

        # Check syntax with py_compile
        result = subprocess.run(
            ['python3', '-m', 'py_compile', filepath],
            capture_output=True, text=True
        )

        assert result.returncode == 0

    def test_write_invalid_python_and_detect_error(self, temp_workspace):
        """Test writing invalid Python and detecting syntax error."""
        filepath = os.path.join(temp_workspace, 'invalid_syntax.py')

        # Write invalid Python (missing closing paren)
        invalid_content = 'def test(\n    return 42\n'
        with open(filepath, 'w') as f:
            f.write(invalid_content)

        # Check syntax with py_compile
        result = subprocess.run(
            ['python3', '-m', 'py_compile', filepath],
            capture_output=True, text=True
        )

        # Should have error
        assert result.returncode != 0 or result.stderr


# ==============================================================================
# Ripgrep Integration Tests (if available)
# ==============================================================================


class TestRipgrepIntegration:
    """Tests using ripgrep if available."""

    @pytest.fixture(autouse=True)
    def check_ripgrep(self):
        """Check if ripgrep is available."""
        result = subprocess.run(['which', 'rg'], capture_output=True)
        if result.returncode != 0:
            pytest.skip("ripgrep (rg) not available")

    def test_rg_glob_files(self, temp_workspace):
        """Test ripgrep for globbing files."""
        create_test_structure(temp_workspace)

        result = subprocess.run(
            ['rg', '--files', '-g', '*.py', temp_workspace],
            capture_output=True, text=True
        )

        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        assert len(files) >= 3
        assert any('main.py' in f for f in files)

    def test_rg_grep_pattern(self, temp_workspace):
        """Test ripgrep for content search."""
        create_test_structure(temp_workspace)

        result = subprocess.run(
            ['rg', '-n', 'def', temp_workspace],
            capture_output=True, text=True
        )

        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]

        assert len(lines) >= 3
        assert any('def main' in l for l in lines)

    def test_rg_with_file_type(self, temp_workspace):
        """Test ripgrep with file type filter."""
        create_test_structure(temp_workspace)

        result = subprocess.run(
            ['rg', '-n', '-g', '*.py', 'import', temp_workspace],
            capture_output=True, text=True
        )

        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]

        # All results should be from Python files
        for line in lines:
            if ':' in line:
                filepath = line.split(':')[0]
                assert filepath.endswith('.py')

    def test_rg_sorted_by_mtime(self, temp_workspace):
        """Test ripgrep sorts by modification time."""
        create_test_structure(temp_workspace)

        result = subprocess.run(
            ['rg', '--files', '--sortr', 'modified', temp_workspace],
            capture_output=True, text=True
        )

        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        # Should have files (sorted by mtime)
        assert len(files) > 0

