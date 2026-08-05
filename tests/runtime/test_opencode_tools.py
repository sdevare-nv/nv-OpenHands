"""Integration tests for OpenCode-style tools runtime execution.

These tests verify the runtime handlers for:
- OpenCodeReadAction (opencode_read)
- OpenCodeWriteAction (opencode_write)
- GlobAction (glob)
- GrepAction (grep)
- ListDirAction (list_dir)

Tests run against the actual runtime (Docker/Local/CLI) to ensure
proper execution of file operations.
"""

import pytest
from conftest import _close_test_runtime, _load_runtime

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    CmdRunAction,
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
# Test Fixtures and Helpers
# ==============================================================================


def _run_action(runtime, action):
    """Execute an action and return the observation."""
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    return obs


def _create_test_file(runtime, path: str, content: str):
    """Helper to create a test file in the sandbox."""
    action = CmdRunAction(
        command=f'mkdir -p "$(dirname "{path}")" && cat > "{path}" << \'TESTEOF\'\n{content}\nTESTEOF'
    )
    action.set_hard_timeout(30)
    obs = runtime.run_action(action)
    assert isinstance(obs, CmdOutputObservation), f"Failed to create file: {obs}"
    return obs


def _create_test_directory_structure(runtime, base_path: str):
    """Create a test directory structure for glob/grep tests."""
    # Create directory structure
    files = {
        f'{base_path}/src/main.py': 'def main():\n    print("Hello")\n\nmain()',
        f'{base_path}/src/utils.py': '# Utility functions\ndef helper():\n    return 42',
        f'{base_path}/src/lib/core.py': 'class Core:\n    pass',
        f'{base_path}/tests/test_main.py': 'import pytest\n\ndef test_main():\n    assert True',
        f'{base_path}/tests/test_utils.py': 'def test_helper():\n    pass',
        f'{base_path}/docs/readme.md': '# Project README\n\nThis is a test project.',
        f'{base_path}/config.json': '{"name": "test", "version": "1.0.0"}',
        f'{base_path}/.gitignore': 'node_modules/\n__pycache__/\n*.pyc',
    }

    for path, content in files.items():
        _create_test_file(runtime, path, content)

    return files


def _expected_file_read(
    path: str,
    lines: list[tuple[int, str]],
    footer: str,
) -> str:
    """Build the exact model-visible OpenCode read body."""
    numbered = '\n'.join(f'{number}: {line}' for number, line in lines)
    return (
        f'<path>{path}</path>\n'
        '<type>file</type>\n'
        f'<content>\n{numbered}\n\n{footer}\n</content>'
    )


def _expected_directory_read(path: str, entries: list[str]) -> str:
    """Build the exact model-visible OpenCode directory body."""
    listing = '\n'.join(entries)
    return (
        f'<path>{path}</path>\n'
        '<type>directory</type>\n'
        f'<entries>\n{listing}\n\n({len(entries)} entries)\n</entries>'
    )


# ==============================================================================
# OpenCodeReadAction Tests
# ==============================================================================


class TestOpenCodeRead:
    """Tests for OpenCodeReadAction runtime handler."""

    def test_read_existing_file(self, temp_dir, runtime_cls, run_as_openhands):
        """Test reading an existing file returns correct content with line numbers."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            # Create test file
            sandbox_path = config.workspace_mount_path_in_sandbox
            test_content = 'line 1\nline 2\nline 3\nline 4\nline 5'
            _create_test_file(runtime, f'{sandbox_path}/test.txt', test_content)

            # Read file
            action = OpenCodeReadAction(path=f'{sandbox_path}/test.txt')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_file_read(
                f'{sandbox_path}/test.txt',
                [(number, f'line {number}') for number in range(1, 6)],
                '(End of file - total 5 lines)',
            )
        finally:
            _close_test_runtime(runtime)

    def test_read_with_offset(self, temp_dir, runtime_cls, run_as_openhands):
        """Test reading file with offset parameter."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            test_content = '\n'.join([f'line {i}' for i in range(1, 21)])
            _create_test_file(runtime, f'{sandbox_path}/offset_test.txt', test_content)

            # OpenCode offsets are one-based, so offset=10 starts at line 10.
            action = OpenCodeReadAction(
                path=f'{sandbox_path}/offset_test.txt', offset=10
            )
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_file_read(
                f'{sandbox_path}/offset_test.txt',
                [(number, f'line {number}') for number in range(10, 21)],
                '(End of file - total 20 lines)',
            )

            # JavaScript's `offset || 1` makes zero an alias for the first line.
            zero_action = OpenCodeReadAction(
                path=f'{sandbox_path}/offset_test.txt', offset=0, limit=1
            )
            zero_action.set_hard_timeout(30)
            zero_obs = _run_action(runtime, zero_action)
            assert isinstance(zero_obs, CmdOutputObservation)
            assert zero_obs.content == _expected_file_read(
                f'{sandbox_path}/offset_test.txt',
                [(1, 'line 1')],
                '(Showing lines 1-1 of 20. Use offset=2 to continue.)',
            )

            out_of_range = OpenCodeReadAction(
                path=f'{sandbox_path}/offset_test.txt', offset=21
            )
            out_of_range.set_hard_timeout(30)
            error = _run_action(runtime, out_of_range)
            assert isinstance(error, ErrorObservation)
            assert error.content == (
                'Offset 21 is out of range for this file (20 lines)'
            )
        finally:
            _close_test_runtime(runtime)

    def test_read_with_limit(self, temp_dir, runtime_cls, run_as_openhands):
        """Test reading file with limit parameter."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            test_content = '\n'.join([f'line {i}' for i in range(1, 101)])
            _create_test_file(runtime, f'{sandbox_path}/limit_test.txt', test_content)

            # Read only first 5 lines
            action = OpenCodeReadAction(path=f'{sandbox_path}/limit_test.txt', limit=5)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_file_read(
                f'{sandbox_path}/limit_test.txt',
                [(number, f'line {number}') for number in range(1, 6)],
                '(Showing lines 1-5 of 100. Use offset=6 to continue.)',
            )
        finally:
            _close_test_runtime(runtime)

    def test_read_nonexistent_file(self, temp_dir, runtime_cls, run_as_openhands):
        """Test reading non-existent file returns proper error with suggestions."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            # Create a similar file for suggestions
            _create_test_file(runtime, f'{sandbox_path}/existing_file.py', 'content')

            action = OpenCodeReadAction(path=f'{sandbox_path}/existing_file')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, ErrorObservation)
            assert obs.content == (
                f'File not found: {sandbox_path}/existing_file\n\n'
                'Did you mean one of these?\n'
                f'{sandbox_path}/existing_file.py'
            )
        finally:
            _close_test_runtime(runtime)

    def test_read_empty_file(self, temp_dir, runtime_cls, run_as_openhands):
        """Test reading an empty file."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            # Create empty file
            action = CmdRunAction(command=f'touch {sandbox_path}/empty.txt')
            action.set_hard_timeout(30)
            runtime.run_action(action)

            action = OpenCodeReadAction(path=f'{sandbox_path}/empty.txt')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_file_read(
                f'{sandbox_path}/empty.txt',
                [],
                '(End of file - total 0 lines)',
            )
        finally:
            _close_test_runtime(runtime)

    def test_read_long_lines_truncation(self, temp_dir, runtime_cls, run_as_openhands):
        """Test that very long lines are truncated."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            # OpenCode keeps exactly 2,000 characters, then adds this suffix.
            long_line = 'x' * 2001
            _create_test_file(runtime, f'{sandbox_path}/long_line.txt', long_line)

            action = OpenCodeReadAction(path=f'{sandbox_path}/long_line.txt')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            shown = 'x' * 2000 + '... (line truncated to 2000 chars)'
            assert obs.content == _expected_file_read(
                f'{sandbox_path}/long_line.txt',
                [(1, shown)],
                '(End of file - total 1 lines)',
            )
        finally:
            _close_test_runtime(runtime)

    def test_read_directory_uses_directory_body(
        self, temp_dir, runtime_cls, run_as_openhands
    ):
        """Test read's directory branch uses OpenCode's immediate-entry body."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            directory = f'{sandbox_path}/read_directory'
            _create_test_file(runtime, f'{directory}/z.txt', 'z')
            _create_test_file(runtime, f'{directory}/nested/a.txt', 'a')

            action = OpenCodeReadAction(path=directory)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_directory_read(
                directory, ['nested/', 'z.txt']
            )
        finally:
            _close_test_runtime(runtime)


# ==============================================================================
# OpenCodeWriteAction Tests
# ==============================================================================


class TestOpenCodeWrite:
    """Tests for OpenCodeWriteAction runtime handler."""

    def test_write_new_file(self, temp_dir, runtime_cls, run_as_openhands):
        """Test writing a new file."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            test_content = 'print("Hello, World!")'

            action = OpenCodeWriteAction(
                path=f'{sandbox_path}/new_file.py',
                content=test_content
            )
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, FileWriteObservation)
            assert obs.content == 'Wrote file successfully.'

            # Verify file exists with correct content
            verify = CmdRunAction(command=f'cat {sandbox_path}/new_file.py')
            verify.set_hard_timeout(30)
            verify_obs = runtime.run_action(verify)
            assert test_content in verify_obs.content
        finally:
            _close_test_runtime(runtime)

    def test_write_with_nested_directory(self, temp_dir, runtime_cls, run_as_openhands):
        """Test writing creates parent directories if needed."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            nested_path = f'{sandbox_path}/new/nested/dir/file.txt'

            action = OpenCodeWriteAction(
                path=nested_path,
                content='nested content'
            )
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, FileWriteObservation)
            assert obs.content == 'Wrote file successfully.'

            # Verify directory was created
            verify = CmdRunAction(command=f'test -f {nested_path} && echo "exists"')
            verify.set_hard_timeout(30)
            verify_obs = runtime.run_action(verify)
            assert 'exists' in verify_obs.content
        finally:
            _close_test_runtime(runtime)

    def test_write_overwrite_existing(self, temp_dir, runtime_cls, run_as_openhands):
        """Test writing overwrites existing file."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_file(runtime, f'{sandbox_path}/overwrite.txt', 'original content')

            # Overwrite with new content
            action = OpenCodeWriteAction(
                path=f'{sandbox_path}/overwrite.txt',
                content='new content'
            )
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, FileWriteObservation)
            assert obs.content == 'Wrote file successfully.'

            # Verify content was overwritten
            verify = CmdRunAction(command=f'cat {sandbox_path}/overwrite.txt')
            verify.set_hard_timeout(30)
            verify_obs = runtime.run_action(verify)
            assert 'new content' in verify_obs.content
            assert 'original content' not in verify_obs.content
        finally:
            _close_test_runtime(runtime)

    def test_write_empty_content(self, temp_dir, runtime_cls, run_as_openhands):
        """Test writing empty content creates empty file."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox

            action = OpenCodeWriteAction(
                path=f'{sandbox_path}/empty_write.txt',
                content=''
            )
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, FileWriteObservation)
            assert obs.content == 'Wrote file successfully.'

            # Verify file is empty
            verify = CmdRunAction(command=f'wc -c < {sandbox_path}/empty_write.txt')
            verify.set_hard_timeout(30)
            verify_obs = runtime.run_action(verify)
            assert '0' in verify_obs.content.strip()
        finally:
            _close_test_runtime(runtime)

    def test_write_multiline_content(self, temp_dir, runtime_cls, run_as_openhands):
        """Test writing multiline content preserves line breaks."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            multiline = 'line 1\nline 2\nline 3'

            action = OpenCodeWriteAction(
                path=f'{sandbox_path}/multiline.txt',
                content=multiline
            )
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, FileWriteObservation)
            assert obs.content == 'Wrote file successfully.'

            # Verify line count
            verify = CmdRunAction(command=f'wc -l < {sandbox_path}/multiline.txt')
            verify.set_hard_timeout(30)
            verify_obs = runtime.run_action(verify)
            # Should have 2 newlines (3 lines)
            assert int(verify_obs.content.strip()) >= 2
        finally:
            _close_test_runtime(runtime)

    def test_write_python_without_lsp_diagnostics(
        self, temp_dir, runtime_cls, run_as_openhands
    ):
        """Test writing Python returns only the base body without an LSP."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            # Intentionally bad Python (syntax error)
            bad_python = 'def foo(\n    print("missing paren"'

            action = OpenCodeWriteAction(
                path=f'{sandbox_path}/bad.py',
                content=bad_python
            )
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            # File should still be written
            assert isinstance(obs, FileWriteObservation)
            assert obs.content == 'Wrote file successfully.'
        finally:
            _close_test_runtime(runtime)


# ==============================================================================
# GlobAction Tests
# ==============================================================================


class TestGlob:
    """Tests for GlobAction runtime handler."""

    def test_glob_find_python_files(self, temp_dir, runtime_cls, run_as_openhands):
        """Test glob finds Python files."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            files = _create_test_directory_structure(runtime, sandbox_path)

            action = GlobAction(pattern='*.py', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert set(obs.content.splitlines()) == {
                path for path in files if path.endswith('.py')
            }
        finally:
            _close_test_runtime(runtime)

    def test_glob_recursive_pattern(self, temp_dir, runtime_cls, run_as_openhands):
        """Test glob with recursive pattern finds files in subdirectories."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            files = _create_test_directory_structure(runtime, sandbox_path)

            action = GlobAction(pattern='**/*.py', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert set(obs.content.splitlines()) == {
                path for path in files if path.endswith('.py')
            }
        finally:
            _close_test_runtime(runtime)

    def test_glob_no_matches(self, temp_dir, runtime_cls, run_as_openhands):
        """Test glob returns appropriate message when no files match."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            # Create a simple file
            _create_test_file(runtime, f'{sandbox_path}/test.txt', 'content')

            action = GlobAction(pattern='*.nonexistent', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == 'No files found'
        finally:
            _close_test_runtime(runtime)

    def test_glob_specific_extension(self, temp_dir, runtime_cls, run_as_openhands):
        """Test glob finds only specific file extensions."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            action = GlobAction(pattern='*.json', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == f'{sandbox_path}/config.json'
        finally:
            _close_test_runtime(runtime)

    def test_glob_in_specific_directory(self, temp_dir, runtime_cls, run_as_openhands):
        """Test glob searches only in specified directory."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            # Search only in tests directory
            action = GlobAction(pattern='*.py', path=f'{sandbox_path}/tests')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert set(obs.content.splitlines()) == {
                f'{sandbox_path}/tests/test_main.py',
                f'{sandbox_path}/tests/test_utils.py',
            }
        finally:
            _close_test_runtime(runtime)

    def test_glob_exactly_100_results_has_truncation_footer(
        self, temp_dir, runtime_cls, run_as_openhands
    ):
        """Test OpenCode's inclusive 100-result truncation boundary."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            directory = f'{sandbox_path}/glob_boundary'
            create = CmdRunAction(
                command=(
                    f'mkdir -p "{directory}" && '
                    f'touch "{directory}"/file-{{000..099}}.py'
                )
            )
            create.set_hard_timeout(30)
            create_obs = runtime.run_action(create)
            assert isinstance(create_obs, CmdOutputObservation)
            assert create_obs.exit_code == 0

            action = GlobAction(pattern='*.py', path=directory)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            listing, footer = obs.content.rsplit('\n\n', maxsplit=1)
            assert set(listing.splitlines()) == {
                f'{directory}/file-{index:03}.py' for index in range(100)
            }
            assert footer == (
                '(Results are truncated: showing first 100 results. '
                'Consider using a more specific path or pattern.)'
            )
        finally:
            _close_test_runtime(runtime)


# ==============================================================================
# GrepAction Tests
# ==============================================================================


class TestGrep:
    """Tests for GrepAction runtime handler."""

    def test_grep_simple_pattern(self, temp_dir, runtime_cls, run_as_openhands):
        """Test grep finds simple pattern."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            action = GrepAction(pattern='def', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content.startswith('Found 4 matches\n')
            assert f'{sandbox_path}/src/main.py:\n  Line 1: def main():\n' in obs.content
            assert f'{sandbox_path}/src/utils.py:\n  Line 2: def helper():\n' in obs.content
            assert (
                f'{sandbox_path}/tests/test_main.py:\n'
                '  Line 3: def test_main():\n'
            ) in obs.content
            assert (
                f'{sandbox_path}/tests/test_utils.py:\n'
                '  Line 1: def test_helper():\n'
            ) in obs.content
        finally:
            _close_test_runtime(runtime)

    def test_grep_with_include_filter(self, temp_dir, runtime_cls, run_as_openhands):
        """Test grep with file type filter."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            # Search only in Python files
            action = GrepAction(pattern='import', path=sandbox_path, include='*.py')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == (
                'Found 1 matches\n'
                f'{sandbox_path}/tests/test_main.py:\n'
                '  Line 1: import pytest\n'
            )
        finally:
            _close_test_runtime(runtime)

    def test_grep_no_matches(self, temp_dir, runtime_cls, run_as_openhands):
        """Test grep returns appropriate message when no matches."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_file(runtime, f'{sandbox_path}/test.txt', 'hello world')

            action = GrepAction(pattern='xyznonexistent123', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == 'No files found'
        finally:
            _close_test_runtime(runtime)

    def test_grep_regex_pattern(self, temp_dir, runtime_cls, run_as_openhands):
        """Test grep with regex pattern."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            # Search for function definitions
            action = GrepAction(pattern=r'def \w+\(', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content.startswith('Found 4 matches\n')
            assert obs.content.count('  Line ') == 4
        finally:
            _close_test_runtime(runtime)

    def test_grep_case_sensitive(self, temp_dir, runtime_cls, run_as_openhands):
        """Test grep is case-sensitive by default."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_file(
                runtime,
                f'{sandbox_path}/case_test.txt',
                'Hello\nhello\nHELLO'
            )

            # Search for lowercase 'hello'
            action = GrepAction(pattern='hello', path=f'{sandbox_path}/case_test.txt')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == (
                'Found 1 matches\n'
                f'{sandbox_path}/case_test.txt:\n'
                '  Line 2: hello\n'
            )
        finally:
            _close_test_runtime(runtime)

    def test_grep_multiline_context(self, temp_dir, runtime_cls, run_as_openhands):
        """Test grep shows OpenCode's grouped path and line format."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            content = 'line1\nTARGET_PATTERN\nline3'
            _create_test_file(runtime, f'{sandbox_path}/grep_test.txt', content)

            action = GrepAction(pattern='TARGET_PATTERN', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == (
                'Found 1 matches\n'
                f'{sandbox_path}/grep_test.txt:\n'
                '  Line 2: TARGET_PATTERN\n'
            )
        finally:
            _close_test_runtime(runtime)

    def test_grep_exactly_100_matches_has_truncation_markers(
        self, temp_dir, runtime_cls, run_as_openhands
    ):
        """Test OpenCode's inclusive 100-match truncation boundary."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            path = f'{sandbox_path}/grep_boundary.txt'
            content = '\n'.join(f'MATCH {index:03}' for index in range(1, 101))
            _create_test_file(runtime, path, content)

            action = GrepAction(pattern='MATCH', path=path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content.startswith(
                f'Found 100 matches (more matches available)\n{path}:\n'
            )
            assert obs.content.count('  Line ') == 100
            assert '  Line 1: MATCH 001\n' in obs.content
            assert '  Line 100: MATCH 100\n' in obs.content
            assert obs.content.endswith(
                '\n\n\n'
                '(Results truncated. Consider using a more specific path or pattern.)'
            )
        finally:
            _close_test_runtime(runtime)


# ==============================================================================
# ListDirAction Tests
# ==============================================================================


class TestListDir:
    """Tests for ListDirAction runtime handler."""

    def test_list_dir_basic(self, temp_dir, runtime_cls, run_as_openhands):
        """Test basic directory listing."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            action = ListDirAction(path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_directory_read(
                sandbox_path,
                ['.gitignore', 'config.json', 'docs/', 'src/', 'tests/'],
            )
        finally:
            _close_test_runtime(runtime)

    def test_list_dir_lists_all_immediate_entries(
        self, temp_dir, runtime_cls, run_as_openhands
    ):
        """Test the legacy alias follows OpenCode read-directory semantics."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox

            # Directory reads expose every immediate entry, including names that
            # search tools commonly ignore.
            entries = [
                f'{sandbox_path}/node_modules/package/index.js',
                f'{sandbox_path}/__pycache__/module.pyc',
                f'{sandbox_path}/.git/config',
                f'{sandbox_path}/src/main.py',
            ]
            for path in entries:
                _create_test_file(runtime, path, 'content')

            action = ListDirAction(path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_directory_read(
                sandbox_path,
                ['.git/', '__pycache__/', 'node_modules/', 'src/'],
            )
        finally:
            _close_test_runtime(runtime)

    def test_list_dir_legacy_ignore_does_not_change_body(
        self, temp_dir, runtime_cls, run_as_openhands
    ):
        """Test retained legacy arguments do not alter OpenCode's read body."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            # Ignore tests directory
            action = ListDirAction(path=sandbox_path, ignore=['tests'])
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_directory_read(
                sandbox_path,
                ['.gitignore', 'config.json', 'docs/', 'src/', 'tests/'],
            )
        finally:
            _close_test_runtime(runtime)

    def test_list_dir_empty_directory(self, temp_dir, runtime_cls, run_as_openhands):
        """Test listing empty directory."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox

            # Create empty directory
            empty_dir = f'{sandbox_path}/empty_dir'
            create_action = CmdRunAction(command=f'mkdir -p {empty_dir}')
            create_action.set_hard_timeout(30)
            runtime.run_action(create_action)

            action = ListDirAction(path=empty_dir)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_directory_read(empty_dir, [])
        finally:
            _close_test_runtime(runtime)

    def test_list_dir_nested_structure(self, temp_dir, runtime_cls, run_as_openhands):
        """Test listing reports immediate entries, not a recursive tree."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            action = ListDirAction(path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert obs.content == _expected_directory_read(
                sandbox_path,
                ['.gitignore', 'config.json', 'docs/', 'src/', 'tests/'],
            )
        finally:
            _close_test_runtime(runtime)

    def test_list_dir_nonexistent(self, temp_dir, runtime_cls, run_as_openhands):
        """Test listing non-existent directory."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox

            action = ListDirAction(path=f'{sandbox_path}/nonexistent_dir_12345')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, ErrorObservation)
            assert obs.content == f'File not found: {sandbox_path}/nonexistent_dir_12345'
        finally:
            _close_test_runtime(runtime)


# ==============================================================================
# Combined Workflow Tests
# ==============================================================================


class TestCombinedWorkflows:
    """Tests for combined tool workflows."""

    def test_write_then_read(self, temp_dir, runtime_cls, run_as_openhands):
        """Test write followed by read returns correct content."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            test_content = 'def hello():\n    print("Hello!")\n\nhello()'

            # Write file
            write_action = OpenCodeWriteAction(
                path=f'{sandbox_path}/workflow_test.py',
                content=test_content
            )
            write_action.set_hard_timeout(30)
            write_obs = _run_action(runtime, write_action)
            assert isinstance(write_obs, FileWriteObservation)
            assert write_obs.content == 'Wrote file successfully.'

            # Read file back
            read_action = OpenCodeReadAction(path=f'{sandbox_path}/workflow_test.py')
            read_action.set_hard_timeout(30)
            read_obs = _run_action(runtime, read_action)

            assert isinstance(read_obs, CmdOutputObservation)
            assert read_obs.content == _expected_file_read(
                f'{sandbox_path}/workflow_test.py',
                [
                    (1, 'def hello():'),
                    (2, '    print("Hello!")'),
                    (3, ''),
                    (4, 'hello()'),
                ],
                '(End of file - total 4 lines)',
            )
        finally:
            _close_test_runtime(runtime)

    def test_glob_then_read_multiple(self, temp_dir, runtime_cls, run_as_openhands):
        """Test glob to find files, then read them."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            # Find Python files
            glob_action = GlobAction(pattern='*.py', path=f'{sandbox_path}/src')
            glob_action.set_hard_timeout(30)
            glob_obs = _run_action(runtime, glob_action)

            assert isinstance(glob_obs, CmdOutputObservation)
            assert set(glob_obs.content.splitlines()) == {
                f'{sandbox_path}/src/main.py',
                f'{sandbox_path}/src/utils.py',
                f'{sandbox_path}/src/lib/core.py',
            }
        finally:
            _close_test_runtime(runtime)

    def test_grep_to_find_then_read(self, temp_dir, runtime_cls, run_as_openhands):
        """Test grep to find pattern, then read matching file."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            # Find files with 'class' keyword
            grep_action = GrepAction(pattern='class', path=sandbox_path)
            grep_action.set_hard_timeout(30)
            grep_obs = _run_action(runtime, grep_action)

            assert isinstance(grep_obs, CmdOutputObservation)
            assert grep_obs.content == (
                'Found 1 matches\n'
                f'{sandbox_path}/src/lib/core.py:\n'
                '  Line 1: class Core:\n'
            )
        finally:
            _close_test_runtime(runtime)
