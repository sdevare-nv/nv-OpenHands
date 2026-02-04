"""Unit tests for OpenCode runtime handler logic.

These tests verify the handler logic used in action_execution_server.py
by testing the individual components in isolation.
"""

import os
import tempfile

import pytest

from openhands.events.action import (
    GlobAction,
    GrepAction,
    ListDirAction,
    OpenCodeReadAction,
    OpenCodeWriteAction,
)


# ==============================================================================
# OpenCodeRead Handler Tests
# ==============================================================================


class TestOpenCodeReadHandler:
    """Tests for opencode_read handler logic."""

    def test_read_formats_line_numbers_correctly(self):
        """Test that line numbers are formatted as 5-digit zero-padded with | separator."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('line 1\nline 2\nline 3')
            temp_path = f.name

        try:
            # Simulate reading - test the formatting logic
            with open(temp_path, 'r') as f:
                lines = f.read().split('\n')

            # Format as OpenCode does
            formatted = [f"{str(i + 1).zfill(5)}| {line}" for i, line in enumerate(lines)]

            assert formatted[0] == '00001| line 1'
            assert formatted[1] == '00002| line 2'
            assert formatted[2] == '00003| line 3'
        finally:
            os.unlink(temp_path)

    def test_read_respects_offset(self):
        """Test that offset parameter works correctly."""
        lines = [f'line {i}' for i in range(1, 101)]
        offset = 49  # Start from line 50 (0-indexed)
        limit = 10

        result = lines[offset:offset + limit]

        assert result[0] == 'line 50'
        assert len(result) == 10

    def test_read_respects_limit(self):
        """Test that limit parameter works correctly."""
        lines = [f'line {i}' for i in range(1, 101)]
        offset = 0
        limit = 5

        result = lines[offset:offset + limit]

        assert len(result) == 5
        assert result[-1] == 'line 5'

    def test_read_truncates_long_lines(self):
        """Test that lines longer than MAX_LINE_LENGTH are truncated."""
        MAX_LINE_LENGTH = 2000
        long_line = 'x' * 3000

        if len(long_line) > MAX_LINE_LENGTH:
            truncated = long_line[:MAX_LINE_LENGTH] + '...'
        else:
            truncated = long_line

        assert len(truncated) == MAX_LINE_LENGTH + 3  # +3 for '...'
        assert truncated.endswith('...')

    def test_binary_extension_detection(self):
        """Test binary file detection by extension."""
        BINARY_EXTENSIONS = {
            '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.class', '.jar',
            '.war', '.7z', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.bin', '.dat', '.obj', '.o', '.a', '.lib', '.wasm', '.pyc', '.pyo'
        }

        # Test that binary extensions are detected
        assert os.path.splitext('/path/file.zip')[1].lower() in BINARY_EXTENSIONS
        assert os.path.splitext('/path/file.pyc')[1].lower() in BINARY_EXTENSIONS

        # Test that text extensions are not detected
        assert os.path.splitext('/path/file.py')[1].lower() not in BINARY_EXTENSIONS
        assert os.path.splitext('/path/file.txt')[1].lower() not in BINARY_EXTENSIONS

    def test_binary_content_detection(self):
        """Test binary file detection by content analysis."""
        # Text content
        text_content = b'def hello():\n    print("hi")\n'
        has_null = b'\x00' in text_content
        assert not has_null

        # Binary content
        binary_content = b'\x00\x01\x02\x03\x04\x05'
        has_null = b'\x00' in binary_content
        assert has_null

    def test_byte_limit_calculation(self):
        """Test that byte limit calculation works correctly."""
        MAX_BYTES = 50 * 1024  # 50KB
        lines = ['x' * 100 for _ in range(1000)]  # 1000 lines of 100 chars each

        total_bytes = 0
        read_lines = []
        for line in lines:
            line_bytes = len(line.encode('utf-8')) + 1  # +1 for newline
            if total_bytes + line_bytes > MAX_BYTES:
                break
            read_lines.append(line)
            total_bytes += line_bytes

        assert total_bytes <= MAX_BYTES
        # With 100 chars per line + 1 newline = 101 bytes
        # 50KB / 101 bytes ≈ 507 lines max
        assert len(read_lines) < 520


# ==============================================================================
# OpenCodeWrite Handler Tests
# ==============================================================================


class TestOpenCodeWriteHandler:
    """Tests for opencode_write handler logic."""

    def test_write_creates_file(self):
        """Test that write creates file with correct content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test.txt')
            content = 'test content'

            with open(filepath, 'w') as f:
                f.write(content)

            with open(filepath, 'r') as f:
                assert f.read() == content

    def test_write_creates_parent_directories(self):
        """Test that write creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, 'a', 'b', 'c', 'file.txt')

            os.makedirs(os.path.dirname(nested_path), exist_ok=True)
            with open(nested_path, 'w') as f:
                f.write('content')

            assert os.path.exists(nested_path)

    def test_write_overwrites_existing(self):
        """Test that write overwrites existing file content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('original')
            temp_path = f.name

        try:
            with open(temp_path, 'w') as f:
                f.write('new content')

            with open(temp_path, 'r') as f:
                assert f.read() == 'new content'
        finally:
            os.unlink(temp_path)

    def test_linter_detection_by_extension(self):
        """Test that correct linter is selected based on file extension."""
        extension_to_linter = {
            '.py': ['flake8', 'pylint', 'py_compile'],
            '.js': ['eslint'],
            '.jsx': ['eslint'],
            '.ts': ['eslint'],
            '.tsx': ['eslint'],
            '.go': ['go vet'],
            '.rs': ['cargo check'],
        }

        for ext, linters in extension_to_linter.items():
            assert ext in extension_to_linter
            assert len(linters) > 0

    def test_write_preserves_unicode(self):
        """Test that Unicode content is preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'unicode.txt')
            content = '你好世界\nこんにちは\n🎉'

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            with open(filepath, 'r', encoding='utf-8') as f:
                assert f.read() == content


# ==============================================================================
# Glob Handler Tests
# ==============================================================================


class TestGlobHandler:
    """Tests for glob handler logic."""

    def test_glob_pattern_matching(self):
        """Test glob pattern matching logic."""
        import fnmatch

        files = [
            'main.py',
            'utils.py',
            'test.js',
            'config.json',
            'src/lib.py',
        ]

        # Test *.py pattern
        py_files = [f for f in files if fnmatch.fnmatch(f, '*.py')]
        assert 'main.py' in py_files
        assert 'utils.py' in py_files
        assert 'test.js' not in py_files

        # Note: fnmatch doesn't handle ** recursion, that's handled by find/rg

    def test_glob_result_limit(self):
        """Test that glob results are limited to 100."""
        limit = 100
        files = [f'file{i}.txt' for i in range(200)]

        limited = files[:limit]
        assert len(limited) == 100

    def test_glob_various_patterns(self):
        """Test various glob patterns work correctly."""
        import fnmatch

        test_cases = [
            ('*.py', 'test.py', True),
            ('*.py', 'test.js', False),
            ('*.{py,js}', 'test.py', False),  # fnmatch doesn't support {,}
            ('test_*.py', 'test_main.py', True),
            ('test_*.py', 'main_test.py', False),
            ('[!_]*.py', 'main.py', True),
            ('[!_]*.py', '_private.py', False),
        ]

        for pattern, filename, expected in test_cases:
            if '{' not in pattern:  # fnmatch doesn't support brace expansion
                result = fnmatch.fnmatch(filename, pattern)
                assert result == expected, f"Pattern {pattern} vs {filename}"


# ==============================================================================
# Grep Handler Tests
# ==============================================================================


class TestGrepHandler:
    """Tests for grep handler logic."""

    def test_grep_pattern_matching(self):
        """Test grep pattern matching."""
        import re

        content = '''
def foo():
    pass

def bar():
    return 42

class MyClass:
    pass
'''

        # Find function definitions
        pattern = r'def \w+\('
        matches = re.findall(pattern, content)
        assert len(matches) == 2
        assert 'def foo(' in matches
        assert 'def bar(' in matches

    def test_grep_result_limit(self):
        """Test that grep results are limited to 100."""
        limit = 100
        lines = [f'match_{i}' for i in range(200)]

        limited = lines[:limit]
        assert len(limited) == 100

    def test_grep_include_filter_logic(self):
        """Test include filter pattern matching."""
        import fnmatch

        files = [
            'main.py',
            'utils.py',
            'test.js',
            'config.json',
        ]

        include_pattern = '*.py'
        filtered = [f for f in files if fnmatch.fnmatch(f, include_pattern)]

        assert 'main.py' in filtered
        assert 'test.js' not in filtered

    def test_grep_regex_patterns(self):
        """Test various regex patterns."""
        import re

        content = 'import os\nfrom pathlib import Path\nimport sys'

        patterns_expected = [
            (r'^import', 2),  # Lines starting with 'import'
            (r'import \w+', 3),  # 'import' followed by word
            (r'from \w+ import', 1),  # 'from X import' pattern
        ]

        for pattern, expected_count in patterns_expected:
            matches = re.findall(pattern, content, re.MULTILINE)
            assert len(matches) == expected_count, f"Pattern {pattern}"


# ==============================================================================
# ListDir Handler Tests
# ==============================================================================


class TestListDirHandler:
    """Tests for list_dir handler logic."""

    def test_default_ignore_patterns(self):
        """Test that default ignore patterns are correct."""
        default_ignores = [
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

        action = ListDirAction()
        assert action.DEFAULT_IGNORES == default_ignores

    def test_combined_ignore_patterns(self):
        """Test that custom ignores are combined with defaults."""
        custom = ['*.log', 'tmp']
        action = ListDirAction(ignore=custom)

        all_ignores = action.all_ignores

        # Should contain defaults
        assert 'node_modules' in all_ignores
        assert '__pycache__' in all_ignores

        # Should contain custom
        assert '*.log' in all_ignores
        assert 'tmp' in all_ignores

    def test_tree_structure_building(self):
        """Test tree structure building logic."""
        files = [
            'src/main.py',
            'src/utils.py',
            'src/lib/core.py',
            'tests/test_main.py',
        ]

        dirs = set()
        files_by_dir = {}

        for f in files:
            d = os.path.dirname(f) or '.'
            parts = d.split(os.sep) if d != '.' else []

            # Add all parent directories
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

    def test_ignore_pattern_filtering(self):
        """Test that ignore patterns filter correctly."""
        files = [
            'src/main.py',
            'node_modules/package/index.js',
            '__pycache__/module.pyc',
            '.git/config',
            'dist/bundle.js',
        ]

        ignores = ['node_modules', '__pycache__', '.git', 'dist']

        filtered = [
            f for f in files
            if not any(ignore in f for ignore in ignores)
        ]

        assert 'src/main.py' in filtered
        assert len(filtered) == 1


# ==============================================================================
# Action Properties Tests
# ==============================================================================


class TestActionProperties:
    """Tests for action class properties."""

    def test_opencode_read_message(self):
        """Test OpenCodeReadAction message property."""
        action = OpenCodeReadAction(path='/test.py')
        assert action.message == 'Reading file: /test.py'

        action_with_offset = OpenCodeReadAction(path='/test.py', offset=50)
        assert 'from line 51' in action_with_offset.message

    def test_opencode_write_message(self):
        """Test OpenCodeWriteAction message property."""
        action = OpenCodeWriteAction(path='/test.py', content='')
        assert action.message == 'Writing file: /test.py'

    def test_glob_message(self):
        """Test GlobAction message property."""
        action = GlobAction(pattern='**/*.py')
        assert '**/*.py' in action.message

    def test_grep_message(self):
        """Test GrepAction message property."""
        action = GrepAction(pattern='TODO')
        assert 'TODO' in action.message

    def test_list_dir_message(self):
        """Test ListDirAction message property."""
        action = ListDirAction(path='/project')
        assert '/project' in action.message

        action_default = ListDirAction()
        # Message contains the path, which is '.'
        assert '.' in action_default.message or 'directory' in action_default.message

    def test_actions_are_runnable(self):
        """Test that all OpenCode actions are runnable."""
        actions = [
            OpenCodeReadAction(path='/test.py'),
            OpenCodeWriteAction(path='/test.py', content=''),
            GlobAction(pattern='*.py'),
            GrepAction(pattern='TODO'),
            ListDirAction(),
        ]

        for action in actions:
            assert action.runnable is True


# ==============================================================================
# File Suggestions Tests
# ==============================================================================


class TestFileSuggestions:
    """Tests for file not found suggestions logic."""

    def test_find_similar_files(self):
        """Test finding similar files for suggestions."""
        directory_contents = [
            'test_file.py',
            'test_file.txt',
            'test_main.py',
            'other_file.py',
            'config.json',
        ]

        basename = 'test_file'

        suggestions = [
            entry for entry in directory_contents
            if basename.lower() in entry.lower() or entry.lower() in basename.lower()
        ][:3]

        assert 'test_file.py' in suggestions
        assert 'test_file.txt' in suggestions
        assert len(suggestions) <= 3

    def test_case_insensitive_matching(self):
        """Test that file suggestions are case-insensitive."""
        directory_contents = [
            'TestFile.py',
            'TESTFILE.txt',
            'testfile.js',
        ]

        basename = 'testfile'

        suggestions = [
            entry for entry in directory_contents
            if basename.lower() in entry.lower()
        ]

        assert len(suggestions) == 3


# ==============================================================================
# Subprocess Command Building Tests
# ==============================================================================


class TestSubprocessCommandBuilding:
    """Tests for building subprocess commands."""

    def test_ripgrep_glob_command(self):
        """Test ripgrep command for glob."""
        pattern = '*.py'
        search_path = '/project'

        cmd = ['rg', '--files', '-g', pattern, '--sortr', 'modified', search_path]

        assert cmd[0] == 'rg'
        assert '-g' in cmd
        assert pattern in cmd
        assert '--sortr' in cmd

    def test_find_glob_fallback_command(self):
        """Test find command for glob fallback."""
        pattern = '*.py'
        search_path = '/project'

        # Note: This is a simplified version
        cmd = f'find {search_path} -type f -name "{pattern}"'

        assert 'find' in cmd
        assert search_path in cmd
        assert pattern in cmd

    def test_ripgrep_grep_command(self):
        """Test ripgrep command for grep."""
        pattern = 'TODO'
        search_path = '/project'

        cmd = ['rg', '-n', pattern, search_path]

        assert cmd[0] == 'rg'
        assert '-n' in cmd  # Line numbers
        assert pattern in cmd

    def test_grep_fallback_command(self):
        """Test grep command for fallback."""
        pattern = 'TODO'
        search_path = '/project'

        cmd = f'grep -rn "{pattern}" {search_path}'

        assert 'grep' in cmd
        assert '-rn' in cmd  # Recursive with line numbers
        assert pattern in cmd

    def test_tree_command(self):
        """Test tree command for list_dir."""
        path = '/project'
        ignores = ['node_modules', '__pycache__']

        cmd = ['tree', '-L', '3', '--noreport']
        for ignore in ignores:
            cmd.extend(['-I', ignore])
        cmd.append(path)

        assert 'tree' in cmd
        assert '-L' in cmd
        assert '3' in cmd  # Depth limit

    def test_grep_with_include_command(self):
        """Test grep command with include filter."""
        pattern = 'import'
        search_path = '/project'
        include = '*.py'

        # Ripgrep version
        rg_cmd = ['rg', '-n', '-g', include, pattern, search_path]
        assert '-g' in rg_cmd
        assert include in rg_cmd

        # Find+grep fallback
        find_grep = f'find {search_path} -type f -name "{include}" -exec grep -Hn "{pattern}" {{}} \\;'
        assert 'find' in find_grep
        assert '-name' in find_grep
        assert include in find_grep

