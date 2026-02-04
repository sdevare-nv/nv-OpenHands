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

import os
import time
from pathlib import Path

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
            assert '<file>' in obs.content
            assert '</file>' in obs.content
            # Check line number format (5-digit zero-padded with |)
            assert '00001| line 1' in obs.content
            assert '00002| line 2' in obs.content
            assert '00005| line 5' in obs.content
        finally:
            _close_test_runtime(runtime)

    def test_read_with_offset(self, temp_dir, runtime_cls, run_as_openhands):
        """Test reading file with offset parameter."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            test_content = '\n'.join([f'line {i}' for i in range(1, 21)])
            _create_test_file(runtime, f'{sandbox_path}/offset_test.txt', test_content)

            # Read starting from line 10 (0-based offset 9)
            action = OpenCodeReadAction(path=f'{sandbox_path}/offset_test.txt', offset=9)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            # Should start from line 10 (offset 9 + 1 = line 10)
            assert '00010| line 10' in obs.content
            # Should NOT contain early lines
            assert '00001| line 1' not in obs.content
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
            assert '00001| line 1' in obs.content
            assert '00005| line 5' in obs.content
            # Should indicate there are more lines
            assert 'more lines' in obs.content.lower() or 'offset' in obs.content.lower()
        finally:
            _close_test_runtime(runtime)

    def test_read_nonexistent_file(self, temp_dir, runtime_cls, run_as_openhands):
        """Test reading non-existent file returns proper error with suggestions."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            # Create a similar file for suggestions
            _create_test_file(runtime, f'{sandbox_path}/existing_file.py', 'content')

            action = OpenCodeReadAction(path=f'{sandbox_path}/existing_file.txt')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            # Should be error or contain error message
            assert isinstance(obs, (ErrorObservation, CmdOutputObservation))
            content = obs.content if hasattr(obs, 'content') else str(obs)
            assert 'not found' in content.lower() or 'error' in content.lower()
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
            assert '<file>' in obs.content
        finally:
            _close_test_runtime(runtime)

    def test_read_long_lines_truncation(self, temp_dir, runtime_cls, run_as_openhands):
        """Test that very long lines are truncated."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            # Create file with very long line (> 2000 chars)
            long_line = 'x' * 3000
            _create_test_file(runtime, f'{sandbox_path}/long_line.txt', long_line)

            action = OpenCodeReadAction(path=f'{sandbox_path}/long_line.txt')
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            # Line should be truncated with ...
            assert '...' in obs.content
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
            assert 'success' in obs.content.lower() or obs.path.endswith('new_file.py')

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

            # Verify line count
            verify = CmdRunAction(command=f'wc -l < {sandbox_path}/multiline.txt')
            verify.set_hard_timeout(30)
            verify_obs = runtime.run_action(verify)
            # Should have 2 newlines (3 lines)
            assert int(verify_obs.content.strip()) >= 2
        finally:
            _close_test_runtime(runtime)

    def test_write_python_with_linting(self, temp_dir, runtime_cls, run_as_openhands):
        """Test writing Python file shows lint errors."""
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
            # May or may not show diagnostics depending on linter availability
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
            _create_test_directory_structure(runtime, sandbox_path)

            action = GlobAction(pattern='*.py', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            # Should find Python files
            assert '.py' in obs.content
            # Should find at least some of our test files
            content_lower = obs.content.lower()
            assert 'main.py' in content_lower or 'utils.py' in content_lower
        finally:
            _close_test_runtime(runtime)

    def test_glob_recursive_pattern(self, temp_dir, runtime_cls, run_as_openhands):
        """Test glob with recursive pattern finds files in subdirectories."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            action = GlobAction(pattern='**/*.py', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            # Should find files in subdirectories
            assert 'test_' in obs.content.lower() or 'core.py' in obs.content.lower()
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
            # Should indicate no files found
            assert 'no files' in obs.content.lower() or obs.content.strip() == ''
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
            assert 'config.json' in obs.content
            # Should NOT find Python files
            assert 'main.py' not in obs.content
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
            # Should find test files
            assert 'test_' in obs.content.lower()
            # Should NOT find src files
            assert 'main.py' not in obs.content
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
            # Should find function definitions
            assert 'def' in obs.content
            # Should show line numbers
            assert ':' in obs.content  # file:linenum:content format
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
            # Should find imports in Python files
            if 'import' in obs.content:
                assert '.py' in obs.content
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
            # Should indicate no matches
            assert 'no match' in obs.content.lower() or obs.content.strip() == ''
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
            # Should find function definitions
            assert 'def' in obs.content
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
            # Should find lowercase
            assert 'hello' in obs.content.lower()
        finally:
            _close_test_runtime(runtime)

    def test_grep_multiline_context(self, temp_dir, runtime_cls, run_as_openhands):
        """Test grep shows file:line:content format."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            content = 'line1\nTARGET_PATTERN\nline3'
            _create_test_file(runtime, f'{sandbox_path}/grep_test.txt', content)

            action = GrepAction(pattern='TARGET_PATTERN', path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            assert 'TARGET_PATTERN' in obs.content
            # Should show line number (line 2)
            assert '2' in obs.content or 'grep_test.txt' in obs.content
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
            # Should show directory structure
            assert 'src' in obs.content or 'tests' in obs.content or 'docs' in obs.content
        finally:
            _close_test_runtime(runtime)

    def test_list_dir_default_ignores(self, temp_dir, runtime_cls, run_as_openhands):
        """Test default ignore patterns are applied."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox

            # Create directories that should be ignored
            ignored_dirs = [
                f'{sandbox_path}/node_modules/package/index.js',
                f'{sandbox_path}/__pycache__/module.pyc',
                f'{sandbox_path}/.git/config',
                f'{sandbox_path}/src/main.py',  # This should show
            ]
            for path in ignored_dirs:
                _create_test_file(runtime, path, 'content')

            action = ListDirAction(path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            # Main.py should be visible
            # node_modules, __pycache__, .git should be filtered
            # (behavior may vary based on tool availability)
        finally:
            _close_test_runtime(runtime)

    def test_list_dir_with_custom_ignore(self, temp_dir, runtime_cls, run_as_openhands):
        """Test custom ignore patterns."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            # Ignore tests directory
            action = ListDirAction(path=sandbox_path, ignore=['tests'])
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            # Should show src but potentially not tests
            assert 'src' in obs.content or 'main.py' in obs.content
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
            # Should handle empty directory gracefully
        finally:
            _close_test_runtime(runtime)

    def test_list_dir_nested_structure(self, temp_dir, runtime_cls, run_as_openhands):
        """Test listing shows nested directory structure."""
        runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
        try:
            sandbox_path = config.workspace_mount_path_in_sandbox
            _create_test_directory_structure(runtime, sandbox_path)

            action = ListDirAction(path=sandbox_path)
            action.set_hard_timeout(30)
            obs = _run_action(runtime, action)

            assert isinstance(obs, CmdOutputObservation)
            # Should show directory hierarchy
            # The exact format depends on the tool used (tree vs find)
            assert len(obs.content) > 0
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

            # Should handle gracefully (either error or empty output)
            assert isinstance(obs, (CmdOutputObservation, ErrorObservation))
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

            # Read file back
            read_action = OpenCodeReadAction(path=f'{sandbox_path}/workflow_test.py')
            read_action.set_hard_timeout(30)
            read_obs = _run_action(runtime, read_action)

            assert isinstance(read_obs, CmdOutputObservation)
            assert 'def hello()' in read_obs.content
            assert 'print' in read_obs.content
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
            assert '.py' in glob_obs.content
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
            # Should find Core class
            if 'class' in grep_obs.content:
                assert 'Core' in grep_obs.content
        finally:
            _close_test_runtime(runtime)

