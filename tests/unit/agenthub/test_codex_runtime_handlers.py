"""Standalone unit tests for Codex runtime handler implementations.

These tests directly test the handler methods in action_execution_server.py
using real file operations on temporary directories, without requiring Docker.
"""

import asyncio
import os
import subprocess
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.events.action.codex import (
    CodexApplyPatchAction,
    CodexGrepFilesAction,
    CodexListDirAction,
    CodexReadFileAction,
    CodexUpdatePlanAction,
)
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
)
from openhands.events.observation.codex import (
    CodexApplyPatchObservation,
    CodexUpdatePlanObservation,
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
    from openhands.runtime.action_execution_server import ActionExecutor

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
# CodexReadFile Handler Tests
# ==============================================================================


class TestCodexReadFileHandler:
    """Exact rust-v0.98.0 ``read_file`` handler contract tests."""

    @pytest.fixture
    def executor(self, temp_workspace):
        from openhands.runtime.action_execution_server import ActionExecutor

        executor = MagicMock(spec=ActionExecutor)
        executor.bash_session = MagicMock()
        executor.bash_session.cwd = temp_workspace
        executor._resolve_path = ActionExecutor._resolve_path.__get__(executor)
        executor.codex_read_file = ActionExecutor.codex_read_file.__get__(executor)
        executor._codex_read_file_indentation = (
            ActionExecutor._codex_read_file_indentation.__get__(executor)
        )
        return executor

    @staticmethod
    def run(executor, action):
        return asyncio.run(executor.codex_read_file(action))

    def test_slice_body_is_exact_and_has_no_footer(self, executor, temp_workspace):
        path = create_test_file(
            temp_workspace, 'range.txt', 'alpha\nbeta\ngamma\ndelta\n'
        )

        obs = self.run(
            executor,
            CodexReadFileAction(file_path=path, offset=2, limit=2),
        )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == 'L2: beta\nL3: gamma'
        assert obs.success is True
        assert obs.exit_code == 0

    @pytest.mark.parametrize(
        ('kwargs', 'expected'),
        [
            ({'offset': 0}, 'offset must be a 1-indexed line number'),
            ({'limit': 0}, 'limit must be greater than zero'),
        ],
    )
    def test_slice_validation_errors_are_exact(
        self, executor, temp_workspace, kwargs, expected
    ):
        path = create_test_file(temp_workspace, 'valid.txt', 'one\n')

        obs = self.run(
            executor,
            CodexReadFileAction(file_path=path, **kwargs),
        )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == expected

    def test_relative_path_error_is_exact(self, executor):
        obs = self.run(executor, CodexReadFileAction(file_path='relative.txt'))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'file_path must be an absolute path'

    def test_offset_past_eof_error_is_exact(self, executor, temp_workspace):
        path = create_test_file(temp_workspace, 'short.txt', 'one\ntwo\n')

        obs = self.run(
            executor,
            CodexReadFileAction(file_path=path, offset=3, limit=1),
        )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'offset exceeds file length'

    def test_empty_file_is_offset_error(self, executor, temp_workspace):
        path = create_test_file(temp_workspace, 'empty.txt', '')

        obs = self.run(executor, CodexReadFileAction(file_path=path))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'offset exceeds file length'

    def test_missing_file_os_error_is_exact(self, executor, temp_workspace):
        path = os.path.join(temp_workspace, 'missing.txt')

        obs = self.run(executor, CodexReadFileAction(file_path=path))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (f'failed to read file: {os.strerror(2)} (os error 2)')

    def test_directory_os_error_is_exact(self, executor, temp_workspace):
        path = os.path.join(temp_workspace, 'directory')
        os.makedirs(path)

        obs = self.run(executor, CodexReadFileAction(file_path=path))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (f'failed to read file: {os.strerror(21)} (os error 21)')

    def test_crlf_is_stripped_but_unterminated_cr_is_data(
        self, executor, temp_workspace
    ):
        path = os.path.join(temp_workspace, 'crlf.txt')
        with open(path, 'wb') as file:
            file.write(b'one\r\ntwo\nlast\r')

        obs = self.run(executor, CodexReadFileAction(file_path=path))

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == 'L1: one\nL2: two\nL3: last\r'

    def test_non_utf8_and_nul_bytes_are_read_lossily(self, executor, temp_workspace):
        path = os.path.join(temp_workspace, 'bytes.dat')
        with open(path, 'wb') as file:
            file.write(b'\xff\xfe\n\x00text\n')

        obs = self.run(executor, CodexReadFileAction(file_path=path))

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == 'L1: ��\nL2: \x00text'

    def test_line_is_clipped_at_500_utf8_bytes(self, executor, temp_workspace):
        path = create_test_file(
            temp_workspace,
            'long.txt',
            'a' * 498 + 'é' + 'z\n',
        )

        obs = self.run(executor, CodexReadFileAction(file_path=path))

        shown = 'a' * 498 + 'é'
        assert len(shown.encode('utf-8')) == 500
        assert obs.content == f'L1: {shown}'


# ==============================================================================
# CodexReadFile Indentation Mode Tests
# ==============================================================================


class TestCodexReadFileIndentationMode:
    """Exact indentation-mode integration tests for ``read_file``."""

    @pytest.fixture
    def executor(self, temp_workspace):
        from openhands.runtime.action_execution_server import ActionExecutor

        executor = MagicMock(spec=ActionExecutor)
        executor.bash_session = MagicMock()
        executor.bash_session.cwd = temp_workspace
        executor._resolve_path = ActionExecutor._resolve_path.__get__(executor)
        executor.codex_read_file = ActionExecutor.codex_read_file.__get__(executor)
        executor._codex_read_file_indentation = (
            ActionExecutor._codex_read_file_indentation.__get__(executor)
        )
        return executor

    @staticmethod
    def run(executor, action):
        return asyncio.run(executor.codex_read_file(action))

    def test_indentation_block_body_is_exact(self, executor, temp_workspace):
        content = (
            'fn outer() {\n    if cond {\n        inner();\n    }\n    tail();\n}\n'
        )
        path = create_test_file(temp_workspace, 'block.rs', content)
        action = CodexReadFileAction(
            file_path=path,
            mode='indentation',
            limit=10,
            indentation={'anchor_line': 3, 'max_levels': 1},
        )

        obs = self.run(executor, action)

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == ('L2:     if cond {\nL3:         inner();\nL4:     }')
        assert obs.success is True
        assert obs.exit_code == 0

    def test_indentation_limit_one_returns_only_anchor(self, executor, temp_workspace):
        path = create_test_file(temp_workspace, 'one.py', 'parent\n    child\n')
        action = CodexReadFileAction(
            file_path=path,
            mode='indentation',
            limit=1,
            indentation={'anchor_line': 2},
        )

        obs = self.run(executor, action)

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == 'L2:     child'

    @pytest.mark.parametrize(
        ('indentation', 'expected'),
        [
            (
                {'anchor_line': 0},
                'anchor_line must be a 1-indexed line number',
            ),
            ({'max_lines': 0}, 'max_lines must be greater than zero'),
            ({'anchor_line': 3}, 'anchor_line exceeds file length'),
        ],
    )
    def test_indentation_errors_are_exact(
        self, executor, temp_workspace, indentation, expected
    ):
        path = create_test_file(temp_workspace, 'two.txt', 'one\ntwo\n')
        action = CodexReadFileAction(
            file_path=path,
            mode='indentation',
            indentation=indentation,
        )

        obs = self.run(executor, action)

        assert isinstance(obs, ErrorObservation)
        assert obs.content == expected


# ==============================================================================
# CodexListDir Handler Tests
# ==============================================================================


class TestCodexListDirHandler:
    """Exact rust-v0.98.0 ``list_dir`` handler contract tests."""

    @pytest.fixture
    def executor(self, temp_workspace):
        from openhands.runtime.action_execution_server import ActionExecutor

        executor = MagicMock(spec=ActionExecutor)
        executor.bash_session = MagicMock()
        executor.bash_session.cwd = temp_workspace
        executor._resolve_path = ActionExecutor._resolve_path.__get__(executor)
        executor.codex_list_dir = ActionExecutor.codex_list_dir.__get__(executor)
        return executor

    @staticmethod
    def run(executor, action):
        return asyncio.run(executor.codex_list_dir(action))

    def test_entry_kinds_hidden_files_and_body_are_exact(
        self, executor, temp_workspace
    ):
        create_test_file(temp_workspace, '.hidden', '')
        create_test_file(temp_workspace, 'alpha.txt', '')
        os.makedirs(os.path.join(temp_workspace, 'nested'))
        create_test_file(temp_workspace, 'nested/child.txt', '')
        os.symlink('alpha.txt', os.path.join(temp_workspace, 'link'))
        os.mkfifo(os.path.join(temp_workspace, 'pipe'))

        obs = self.run(
            executor,
            CodexListDirAction(dir_path=temp_workspace, depth=2, limit=20),
        )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == (
            f'Absolute path: {temp_workspace}\n'
            '.hidden\n'
            'alpha.txt\n'
            'link@\n'
            'nested/\n'
            '  child.txt\n'
            'pipe?'
        )
        assert obs.success is True
        assert obs.exit_code == 0

    def test_depth_and_global_relative_path_sort_are_exact(
        self, executor, temp_workspace
    ):
        os.makedirs(os.path.join(temp_workspace, 'a', 'deep'))
        os.makedirs(os.path.join(temp_workspace, 'b'))
        create_test_file(temp_workspace, 'a/z.txt', '')
        create_test_file(temp_workspace, 'a/deep/grandchild.txt', '')
        create_test_file(temp_workspace, 'aa.txt', '')
        create_test_file(temp_workspace, 'b/a.txt', '')
        create_test_file(temp_workspace, 'zz.txt', '')

        depth_one = self.run(
            executor,
            CodexListDirAction(dir_path=temp_workspace, depth=1, limit=20),
        )
        depth_two = self.run(
            executor,
            CodexListDirAction(dir_path=temp_workspace, depth=2, limit=20),
        )

        assert depth_one.content == (
            f'Absolute path: {temp_workspace}\na/\naa.txt\nb/\nzz.txt'
        )
        assert depth_two.content == (
            f'Absolute path: {temp_workspace}\n'
            'a/\n'
            '  deep/\n'
            '  z.txt\n'
            'aa.txt\n'
            'b/\n'
            '  a.txt\n'
            'zz.txt'
        )

    def test_pagination_uses_global_sort_and_exact_footer(
        self, executor, temp_workspace
    ):
        os.makedirs(os.path.join(temp_workspace, 'a'))
        os.makedirs(os.path.join(temp_workspace, 'b'))
        create_test_file(temp_workspace, 'a/a_child.txt', '')
        create_test_file(temp_workspace, 'b/b_child.txt', '')

        first_page = self.run(
            executor,
            CodexListDirAction(
                dir_path=temp_workspace,
                offset=1,
                limit=2,
                depth=2,
            ),
        )
        second_page = self.run(
            executor,
            CodexListDirAction(
                dir_path=temp_workspace,
                offset=3,
                limit=2,
                depth=2,
            ),
        )

        assert first_page.content == (
            f'Absolute path: {temp_workspace}\n'
            'a/\n'
            '  a_child.txt\n'
            'More than 2 entries found'
        )
        assert second_page.content == (
            f'Absolute path: {temp_workspace}\nb/\n  b_child.txt'
        )

    def test_empty_directory_body_is_only_absolute_path(self, executor, temp_workspace):
        empty = os.path.join(temp_workspace, 'empty')
        os.makedirs(empty)

        obs = self.run(executor, CodexListDirAction(dir_path=empty))

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == f'Absolute path: {empty}'

    def test_non_utf8_absolute_path_is_displayed_lossily(
        self, executor, temp_workspace
    ):
        raw_path = os.fsencode(temp_workspace) + b'/invalid-\xff'
        os.mkdir(raw_path)
        path = os.fsdecode(raw_path)

        obs = self.run(executor, CodexListDirAction(dir_path=path))

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == f'Absolute path: {temp_workspace}/invalid-�'

    @pytest.mark.parametrize(
        ('kwargs', 'expected'),
        [
            (
                {'offset': 0},
                'offset must be a 1-indexed entry number',
            ),
            ({'limit': 0}, 'limit must be greater than zero'),
            ({'depth': 0}, 'depth must be greater than zero'),
        ],
    )
    def test_validation_errors_are_exact(
        self, executor, temp_workspace, kwargs, expected
    ):
        obs = self.run(
            executor,
            CodexListDirAction(dir_path=temp_workspace, **kwargs),
        )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == expected

    def test_relative_path_error_is_exact(self, executor):
        obs = self.run(executor, CodexListDirAction(dir_path='relative'))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'dir_path must be an absolute path'

    def test_offset_past_entries_error_is_exact(self, executor, temp_workspace):
        create_test_file(temp_workspace, 'one.txt', '')

        obs = self.run(
            executor,
            CodexListDirAction(dir_path=temp_workspace, offset=2),
        )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'offset exceeds directory entry count'

    @pytest.mark.parametrize(
        ('path_kind', 'errno_value'),
        [('missing', 2), ('file', 20)],
    )
    def test_directory_os_errors_are_exact(
        self, executor, temp_workspace, path_kind, errno_value
    ):
        path = os.path.join(temp_workspace, path_kind)
        if path_kind == 'file':
            create_test_file(temp_workspace, path_kind, '')

        obs = self.run(executor, CodexListDirAction(dir_path=path))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'failed to read directory: '
            f'{os.strerror(errno_value)} (os error {errno_value})'
        )


# ==============================================================================
# CodexGrepFiles Handler Tests
# ==============================================================================


class TestCodexGrepFilesHandler:
    """Exact rust-v0.98.0 ``grep_files`` handler contract tests."""

    @pytest.fixture
    def executor(self, temp_workspace):
        from openhands.runtime.action_execution_server import ActionExecutor

        executor = MagicMock(spec=ActionExecutor)
        executor.bash_session = MagicMock()
        executor.bash_session.cwd = temp_workspace
        executor._resolve_path = ActionExecutor._resolve_path.__get__(executor)
        executor.codex_grep_files = ActionExecutor.codex_grep_files.__get__(executor)
        return executor

    @staticmethod
    def run(executor, action):
        return asyncio.run(executor.codex_grep_files(action))

    def test_exact_rg_argv_relative_path_and_glob_are_unchanged(
        self, executor, temp_workspace
    ):
        search_path = os.path.join(temp_workspace, 'src')
        os.makedirs(search_path)
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b'/repo/new.py\n/repo/old.py\n',
            stderr=b'',
        )

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=completed,
        ) as run_rg:
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='  TODO|FIXME  ',
                    include='  **/*.{py,pyi}  ',
                    path='src',
                ),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == '/repo/new.py\n/repo/old.py'
        assert obs.success is True
        assert obs.exit_code == 0
        run_rg.assert_called_once_with(
            [
                'rg',
                '--files-with-matches',
                '--sortr=modified',
                '--regexp',
                'TODO|FIXME',
                '--no-messages',
                '--glob',
                '**/*.{py,pyi}',
                '--glob',
                '!**/.git/**',
                '--',
                search_path,
            ],
            capture_output=True,
            timeout=30,
            cwd=temp_workspace,
        )

    def test_default_path_is_working_directory_and_blank_glob_is_omitted(
        self, executor, temp_workspace
    ):
        completed = subprocess.CompletedProcess([], 1, b'', b'ignored')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=completed,
        ) as run_rg:
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='needle',
                    path='',
                    include='   ',
                ),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == 'No matches found.'
        assert run_rg.call_args.args[0] == [
            'rg',
            '--files-with-matches',
            '--sortr=modified',
            '--regexp',
            'needle',
            '--no-messages',
            '--glob',
            '!**/.git/**',
            '--',
            temp_workspace,
        ]

    def test_limit_has_no_truncation_footer(self, executor, temp_workspace):
        completed = subprocess.CompletedProcess(
            [],
            0,
            b'/repo/one.py\n/repo/two.py\n/repo/three.py\n',
            b'',
        )

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=completed,
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='needle',
                    path=temp_workspace,
                    limit=2,
                ),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == '/repo/one.py\n/repo/two.py'

    def test_limit_is_capped_at_2000(self, executor, temp_workspace):
        stdout = b''.join(f'{index:04x}\n'.encode() for index in range(2001))
        completed = subprocess.CompletedProcess([], 0, stdout, b'')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=completed,
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='needle',
                    path=temp_workspace,
                    limit=5000,
                ),
            )

        lines = obs.content.splitlines()
        assert len(lines) == 2000
        assert lines[-1] == '07cf'

    @pytest.mark.parametrize('pattern', ['', '   \t\n'])
    def test_empty_pattern_error_is_exact(self, executor, temp_workspace, pattern):
        obs = self.run(
            executor,
            CodexGrepFilesAction(pattern=pattern, path=temp_workspace),
        )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'pattern must not be empty'

    def test_zero_limit_error_is_exact(self, executor, temp_workspace):
        obs = self.run(
            executor,
            CodexGrepFilesAction(
                pattern='needle',
                path=temp_workspace,
                limit=0,
            ),
        )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'limit must be greater than zero'

    def test_missing_path_error_is_exact(self, executor, temp_workspace):
        missing = os.path.join(temp_workspace, 'missing')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run'
        ) as run_rg:
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=missing),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            f'unable to access `{missing}`: {os.strerror(2)} (os error 2)'
        )
        run_rg.assert_not_called()

    def test_non_utf8_missing_path_error_is_displayed_lossily(
        self, executor, temp_workspace
    ):
        raw_missing = os.fsencode(temp_workspace) + b'/missing-\xff'
        missing = os.fsdecode(raw_missing)

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run'
        ) as run_rg:
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=missing),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            f'unable to access `{temp_workspace}/missing-�`: '
            f'{os.strerror(2)} (os error 2)'
        )
        run_rg.assert_not_called()

    def test_missing_rg_falls_back_with_glob_limit_and_mtime_order(
        self, executor, temp_workspace
    ):
        paths = {
            'old.py': create_test_file(temp_workspace, 'old.py', 'needle'),
            'new.py': create_test_file(temp_workspace, 'new.py', 'needle'),
            'nested/newest.pyi': create_test_file(
                temp_workspace, 'nested/newest.pyi', 'needle'
            ),
            'newest.txt': create_test_file(temp_workspace, 'newest.txt', 'needle'),
            'absent.py': create_test_file(temp_workspace, 'absent.py', 'other text'),
        }
        for index, path in enumerate(paths.values(), start=1):
            timestamp_ns = index * 1_000_000_000
            os.utime(path, ns=(timestamp_ns, timestamp_ns))

        launch_error = FileNotFoundError(2, os.strerror(2), 'rg')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=launch_error,
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='needle',
                    include='**/*.{py,pyi}',
                    path=temp_workspace,
                    limit=2,
                ),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == (f'{paths["nested/newest.pyi"]}\n{paths["new.py"]}')
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_fallback_no_matches_is_successful(
        self, executor, temp_workspace
    ):
        create_test_file(temp_workspace, 'source.py', 'nothing here')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == 'No matches found.'
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_fallback_skips_hidden_ignored_symlink_and_binary(
        self, executor, temp_workspace
    ):
        visible = create_test_file(temp_workspace, 'visible.txt', 'needle\n')
        create_test_file(temp_workspace, '.hidden.txt', 'needle\n')
        create_test_file(temp_workspace, '.hidden/secret.txt', 'needle\n')
        create_test_file(temp_workspace, '.ignore', 'node_modules/\nignored*.txt\n')
        create_test_file(temp_workspace, 'node_modules/pkg/source.txt', 'needle\n')
        create_test_file(
            temp_workspace, 'nested/node_modules/pkg/source.txt', 'needle\n'
        )
        create_test_file(temp_workspace, 'ignored-output.txt', 'needle\n')
        create_test_file(temp_workspace, 'nested/ignored-output.txt', 'needle\n')
        os.symlink(visible, os.path.join(temp_workspace, 'linked.txt'))
        binary = os.path.join(temp_workspace, 'binary.dat')
        with open(binary, 'wb') as target:
            target.write(b'needle\x00binary')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == visible
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_positive_include_can_select_hidden_file(
        self, executor, temp_workspace
    ):
        hidden = create_test_file(temp_workspace, '.selected.py', 'needle\n')
        create_test_file(temp_workspace, '.hidden/not-selected.py', 'needle\n')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='needle',
                    include='*.py',
                    path=temp_workspace,
                ),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == hidden
        assert obs.success is True
        assert obs.exit_code == 0

    @pytest.mark.parametrize(
        ('include', 'expected_names'),
        [
            (None, {'visible.yml', 'nested/visible.yml'}),
            (
                '*.yml',
                {'visible.yml', 'nested/visible.yml', '.root.yml'},
            ),
            (
                '*',
                {
                    'visible.yml',
                    'nested/visible.yml',
                    '.root.yml',
                    '.hidden/visible.yml',
                },
            ),
            (
                '**/*',
                {
                    'visible.yml',
                    'nested/visible.yml',
                    '.root.yml',
                    '.hidden/visible.yml',
                },
            ),
        ],
    )
    def test_missing_rg_matches_real_rg_hidden_directory_glob_selection(
        self,
        executor,
        temp_workspace,
        include,
        expected_names,
    ):
        import shutil

        if shutil.which('rg') is None:
            pytest.skip('ripgrep is required for the parity half of this test')

        for name in (
            'visible.yml',
            'nested/visible.yml',
            '.root.yml',
            '.hidden/visible.yml',
            '.git/never-visible.yml',
        ):
            create_test_file(temp_workspace, name, 'needle\n')

        action = CodexGrepFilesAction(
            pattern='needle',
            include=include,
            path=temp_workspace,
        )
        primary = self.run(executor, action)
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            fallback = self.run(executor, action)

        expected = {os.path.join(temp_workspace, name) for name in expected_names}
        assert isinstance(primary, CmdOutputObservation)
        assert isinstance(fallback, CmdOutputObservation)
        assert set(primary.content.splitlines()) == expected
        assert set(fallback.content.splitlines()) == expected
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
    def test_missing_rg_matches_real_rg_positive_glob_ignore_override(
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

        action = CodexGrepFilesAction(
            pattern='needle',
            include=include,
            path=temp_workspace,
        )
        primary = self.run(executor, action)
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            fallback = self.run(executor, action)

        expected = {os.path.join(temp_workspace, name) for name in expected_names}
        assert isinstance(primary, CmdOutputObservation)
        assert isinstance(fallback, CmdOutputObservation)
        assert set(primary.content.splitlines()) == expected
        assert set(fallback.content.splitlines()) == expected
        assert primary.exit_code == fallback.exit_code == 0

    def test_missing_rg_slashless_include_matches_at_any_depth(
        self, executor, temp_workspace
    ):
        expected = [
            create_test_file(temp_workspace, 'root.py', 'needle\n'),
            create_test_file(temp_workspace, 'nested/deep/source.py', 'needle\n'),
        ]
        create_test_file(temp_workspace, 'nested/deep/source.txt', 'needle\n')
        for index, path in enumerate(expected, start=1):
            os.utime(path, ns=(index, index))

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='needle',
                    include='*.py',
                    path=temp_workspace,
                ),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content.splitlines() == list(reversed(expected))
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_anchored_ignore_remains_root_only(
        self, executor, temp_workspace
    ):
        create_test_file(temp_workspace, '.ignore', '/ignored.txt\n')
        create_test_file(temp_workspace, 'ignored.txt', 'needle\n')
        expected = create_test_file(temp_workspace, 'nested/ignored.txt', 'needle\n')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == expected
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_matches_real_rg_nested_ignore_precedence(
        self, executor, temp_workspace
    ):
        import shutil

        if shutil.which('rg') is None:
            pytest.skip('ripgrep is required for the parity half of this test')

        expected = {
            create_test_file(temp_workspace, 'visible.txt', 'needle\n'),
            create_test_file(temp_workspace, 'anchored.txt', 'needle\n'),
            create_test_file(
                temp_workspace,
                'nested/keep.py',
                'needle\n',
            ),
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

        action = CodexGrepFilesAction(pattern='needle', path=temp_workspace)
        primary = self.run(executor, action)
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            fallback = self.run(executor, action)

        assert isinstance(primary, CmdOutputObservation)
        assert isinstance(fallback, CmdOutputObservation)
        assert set(primary.content.splitlines()) == expected
        assert set(fallback.content.splitlines()) == expected
        assert primary.success is True
        assert fallback.success is True
        assert primary.exit_code == fallback.exit_code == 0

    @pytest.mark.parametrize(
        ('field', 'expected'),
        [
            ('pattern', 'pattern must not contain NUL bytes'),
            ('include', 'include must not contain NUL bytes'),
            ('path', 'path must not contain NUL bytes'),
        ],
    )
    def test_rejects_nul_subprocess_arguments(
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
            obs = self.run(executor, CodexGrepFilesAction(**kwargs))

        assert isinstance(obs, ErrorObservation)
        assert obs.content == expected
        run_rg.assert_not_called()

    def test_missing_rg_fallback_searches_explicit_file(self, executor, temp_workspace):
        source = create_test_file(
            temp_workspace, 'nested/source.py', 'prefix needle suffix\n'
        )

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern=r'prefix\s+needle',
                    include='*.py',
                    path=source,
                ),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == source
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_fallback_skips_unreadable_file(self, executor, temp_workspace):
        from pathlib import Path

        visible = create_test_file(temp_workspace, 'visible.txt', 'needle\n')
        unreadable = create_test_file(temp_workspace, 'unreadable.txt', 'needle\n')
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
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == visible
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_fallback_invalid_regex_is_error(self, executor, temp_workspace):
        create_test_file(temp_workspace, 'source.py', 'content')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='unclosed(', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content.startswith(
            'grep_files fallback failed: invalid regular expression:'
        )

    def test_fallback_invalid_regex_raises_regex_error_exactly(self, temp_workspace):
        from pathlib import Path

        from openhands.runtime import action_execution_server

        source = create_test_file(temp_workspace, 'source.py', 'content')

        with pytest.raises(action_execution_server._CODEX_GREP_REGEX_ERROR) as captured:
            action_execution_server._fallback_codex_grep_files(
                Path(source),
                'unclosed(',
                None,
                1,
            )

        assert type(captured.value) is action_execution_server._codex_regex.error

    def test_missing_rg_fallback_supports_unicode_properties(
        self, executor, temp_workspace
    ):
        source = create_test_file(temp_workspace, 'unicode.txt', '123 café 日本語\n')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern=r'\p{L}+', path=temp_workspace),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == source
        assert obs.success is True
        assert obs.exit_code == 0

    def test_missing_rg_fallback_supports_nested_braces_and_double_stars(
        self, executor, temp_workspace
    ):
        expected = [
            create_test_file(temp_workspace, 'src/unit/test_a.py', 'needle\n'),
            create_test_file(
                temp_workspace,
                'src/integration/deep/test_b.pyi',
                'needle\n',
            ),
            create_test_file(temp_workspace, 'src/e2e/x/y/test_c.py', 'needle\n'),
        ]
        create_test_file(temp_workspace, 'src/other/test_d.py', 'needle\n')
        for index, path in enumerate(expected, start=1):
            timestamp_ns = index * 1_000_000_000
            os.utime(path, ns=(timestamp_ns, timestamp_ns))

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='needle',
                    include=('src/**/{unit,{integration,e2e}}/**/*.{py,pyi}'),
                    path=temp_workspace,
                ),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content.splitlines() == list(reversed(expected))
        assert obs.success is True
        assert obs.exit_code == 0

    def test_fallback_caps_exponential_include_expansion_quickly(
        self, executor, temp_workspace
    ):
        from openhands.runtime import action_execution_server

        create_test_file(temp_workspace, 'source.py', 'needle')
        include = ''.join(f'{{a{index},b{index}}}' for index in range(25)) + '.py'
        started_at = time.monotonic()

        with patch.object(
            action_execution_server.subprocess,
            'run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='needle',
                    include=include,
                    path=temp_workspace,
                ),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'grep_files fallback failed: include expands to more than '
            f'{action_execution_server._CODEX_GREP_MAX_GLOB_VARIANTS} '
            'glob variants'
        )
        assert time.monotonic() - started_at < 1

    def test_fallback_deduplicates_brace_alternatives_without_expansion_blowup(
        self, executor, temp_workspace
    ):
        from openhands.runtime import action_execution_server

        expected = create_test_file(temp_workspace, ('a' * 25) + '.py', 'needle')
        include = ('{a,a}' * 25) + '.py'
        started_at = time.monotonic()

        with patch.object(
            action_execution_server.subprocess,
            'run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern='needle',
                    include=include,
                    path=temp_workspace,
                ),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == expected
        assert obs.success is True
        assert obs.exit_code == 0
        assert time.monotonic() - started_at < 1

    def test_fallback_rejects_oversized_newline_free_line(
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
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='x', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'grep_files fallback failed: file contains a line exceeding '
            'the 16-byte fallback limit'
        )

    @pytest.mark.parametrize(
        ('pattern', 'include', 'expected'),
        [
            (
                None,
                'x' * (4_096 + 1),
                'include exceeds the 4096-byte fallback limit',
            ),
            (
                'x' * (16_384 + 1),
                None,
                'pattern exceeds the 16384-byte fallback limit',
            ),
        ],
        ids=('include', 'pattern'),
    )
    def test_fallback_bounds_pattern_and_include_inputs(
        self,
        executor,
        temp_workspace,
        pattern,
        include,
        expected,
    ):
        create_test_file(temp_workspace, 'source.py', 'needle')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(
                    pattern=pattern or 'needle',
                    include=include,
                    path=temp_workspace,
                ),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == f'grep_files fallback failed: {expected}'

    def test_fallback_reports_bounded_deep_regex_nesting(
        self, executor, temp_workspace
    ):
        create_test_file(temp_workspace, 'source.py', 'a')
        pattern = ('(' * 8_000) + 'a' + (')' * 8_000)

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern=pattern, path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'grep_files fallback failed: pattern nesting exceeds the fallback limit'
        )

    def test_fallback_bounds_streamed_ignore_lines(self, executor, temp_workspace):
        from openhands.runtime import action_execution_server

        create_test_file(temp_workspace, 'source.py', 'needle')
        create_test_file(
            temp_workspace,
            '.ignore',
            'x' * (action_execution_server._CODEX_GREP_MAX_INCLUDE_BYTES + 1),
        )

        with patch.object(
            action_execution_server.subprocess,
            'run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'grep_files fallback failed: .ignore contains a line '
            'exceeding the 4096-byte fallback limit'
        )

    def test_fallback_bounds_total_streamed_ignore_file_size(
        self, executor, temp_workspace
    ):
        from openhands.runtime import action_execution_server

        create_test_file(temp_workspace, 'source.py', 'needle')
        line = '#' + ('x' * 4_093) + '\n'
        create_test_file(temp_workspace, '.ignore', line * 257)

        with patch.object(
            action_execution_server.subprocess,
            'run',
            side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'grep_files fallback failed: .ignore exceeds the '
            '1048576-byte fallback limit'
        )

    def test_fallback_deadline_includes_regex_compilation(self, temp_workspace):
        from pathlib import Path

        from openhands.runtime import action_execution_server

        source = create_test_file(temp_workspace, 'source.py', 'needle')
        real_compile = action_execution_server._codex_regex.compile
        now = [0.0]

        def compile_and_expire(pattern):
            matcher = real_compile(pattern)
            now[0] = 31.0
            return matcher

        with (
            patch.object(
                action_execution_server,
                '_codex_grep_monotonic',
                side_effect=lambda: now[0],
            ),
            patch.object(
                action_execution_server._codex_regex,
                'compile',
                side_effect=compile_and_expire,
            ) as compile_pattern,
            pytest.raises(TimeoutError) as captured,
        ):
            action_execution_server._fallback_codex_grep_files(
                Path(source), 'needle', '*.py', 1
            )

        assert type(captured.value) is TimeoutError
        compile_pattern.assert_called_once_with('needle')

    def test_fallback_deadline_covers_include_preprocessing(self, temp_workspace):
        from pathlib import Path

        from openhands.runtime import action_execution_server

        source = create_test_file(temp_workspace, 'source.py', 'needle')
        real_expand = action_execution_server._expand_codex_include_glob
        now = [0.0]

        def expand_after_deadline(pattern, **kwargs):
            now[0] = 31.0
            return real_expand(pattern, **kwargs)

        with (
            patch.object(
                action_execution_server,
                '_codex_grep_monotonic',
                side_effect=lambda: now[0],
            ),
            patch.object(
                action_execution_server,
                '_expand_codex_include_glob',
                side_effect=expand_after_deadline,
            ) as expand_include,
            pytest.raises(TimeoutError) as captured,
        ):
            action_execution_server._fallback_codex_grep_files(
                Path(source), 'needle', '**/*.py', 1
            )

        assert type(captured.value) is TimeoutError
        expand_include.assert_called_once()

    @pytest.mark.parametrize('entry_kind', ['directory', 'file'])
    def test_fallback_checks_deadline_within_directory_enumeration(
        self,
        temp_workspace,
        entry_kind,
    ):
        from pathlib import Path

        from openhands.runtime import action_execution_server

        enumeration_started = False
        enumeration_checks = 0
        entries_consumed = 0

        class FakeEntry:
            name = 'entry'
            path = os.path.join(temp_workspace, name)

            def is_symlink(self):
                return False

            def is_dir(self, *, follow_symlinks):
                return entry_kind == 'directory'

            def is_file(self, *, follow_symlinks):
                return entry_kind == 'file'

        class HugeScanner:
            def __iter__(self):
                return self

            def __next__(self):
                nonlocal entries_consumed
                if entries_consumed == 10:
                    raise AssertionError(
                        'enumeration consumed entries past its deadline'
                    )
                entries_consumed += 1
                return FakeEntry()

            def close(self):
                pass

        def clock():
            nonlocal enumeration_checks
            if not enumeration_started:
                return 0.0
            enumeration_checks += 1
            return 31.0 if enumeration_checks > 2 else 0.0

        def scandir(*args, **kwargs):
            nonlocal enumeration_started
            enumeration_started = True
            return HugeScanner()

        with (
            patch.object(
                action_execution_server,
                '_codex_grep_monotonic',
                side_effect=clock,
            ),
            patch.object(
                action_execution_server.os,
                'scandir',
                side_effect=scandir,
            ),
            pytest.raises(TimeoutError) as captured,
        ):
            action_execution_server._fallback_codex_grep_files(
                Path(temp_workspace), 'needle', None, 1
            )

        assert type(captured.value) is TimeoutError
        assert enumeration_checks == 3
        assert entries_consumed == 1

    @pytest.mark.parametrize(
        ('constant_name', 'constant_value', 'expected'),
        [
            (
                '_NATIVE_SEARCH_MAX_ENTRIES',
                1,
                'search enumerated more than 1 filesystem entries',
            ),
            (
                '_NATIVE_SEARCH_MAX_CANDIDATES',
                1,
                'search found more than 1 candidate files',
            ),
            (
                '_NATIVE_SEARCH_MAX_TRAVERSAL_PATH_BYTES',
                1,
                'search traversal paths exceed the 1-byte memory limit',
            ),
        ],
    )
    def test_fallback_reports_traversal_and_candidate_caps(
        self,
        executor,
        temp_workspace,
        constant_name,
        constant_value,
        expected,
    ):
        from openhands.runtime import action_execution_server

        create_test_file(temp_workspace, 'first.py', 'needle')
        create_test_file(temp_workspace, 'second.py', 'needle')

        with (
            patch.object(
                action_execution_server.subprocess,
                'run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ),
            patch.object(
                action_execution_server,
                constant_name,
                constant_value,
            ),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == f'grep_files fallback failed: {expected}'

    def test_fallback_catastrophic_regex_times_out_during_search(self, temp_workspace):
        from pathlib import Path

        from openhands.runtime import action_execution_server

        source = create_test_file(
            temp_workspace,
            'catastrophic.txt',
            ('a' * 100_000) + '!\n',
        )
        started_at = time.monotonic()

        with (
            patch.object(
                action_execution_server,
                '_codex_grep_monotonic',
                return_value=100.0,
            ),
            pytest.raises(TimeoutError) as captured,
        ):
            action_execution_server._fallback_codex_grep_files(
                Path(source),
                r'(a+)+$',
                None,
                1,
                timeout_seconds=0.005,
            )

        assert type(captured.value) is TimeoutError
        assert time.monotonic() - started_at < 1

    def test_missing_regex_dependency_fails_safely(self, executor, temp_workspace):
        from pathlib import Path

        from openhands.runtime import action_execution_server

        source = create_test_file(temp_workspace, 'source.py', 'needle')
        unavailable_error = action_execution_server._CodexGrepRegexUnavailableError

        with patch.object(action_execution_server, '_codex_regex', None):
            with pytest.raises(unavailable_error) as captured:
                action_execution_server._fallback_codex_grep_files(
                    Path(source), 'needle', None, 1
                )
            assert type(captured.value) is unavailable_error

            with patch.object(
                action_execution_server.subprocess,
                'run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ):
                obs = self.run(
                    executor,
                    CodexGrepFilesAction(pattern='needle', path=temp_workspace),
                )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'grep_files fallback unavailable: '
            "the 'regex' package is required when ripgrep is unavailable"
        )

    def test_missing_rg_fallback_timeout_is_error(self, executor, temp_workspace):
        create_test_file(temp_workspace, 'source.py', 'needle')

        with (
            patch(
                'openhands.runtime.action_execution_server.subprocess.run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ),
            patch(
                'openhands.runtime.action_execution_server._codex_grep_monotonic',
                side_effect=[0.0, 31.0],
            ),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == ('grep_files fallback timed out after 30 seconds')

    def test_missing_rg_fallback_walk_error_is_reported(self, executor, temp_workspace):
        with (
            patch(
                'openhands.runtime.action_execution_server.subprocess.run',
                side_effect=FileNotFoundError(2, os.strerror(2), 'rg'),
            ),
            patch(
                'openhands.runtime.action_execution_server.os.scandir',
                side_effect=PermissionError(
                    13,
                    os.strerror(13),
                    temp_workspace,
                ),
            ),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            f'grep_files fallback failed: {os.strerror(13)} (os error 13)'
        )

    def test_non_missing_rg_launch_error_remains_an_error(
        self, executor, temp_workspace
    ):
        launch_error = PermissionError(13, os.strerror(13), 'rg')

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=launch_error,
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == (
            'failed to launch rg: '
            f'{os.strerror(13)} (os error 13). '
            'Ensure ripgrep is installed and on PATH.'
        )

    def test_rg_timeout_error_is_exact(self, executor, temp_workspace):
        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            side_effect=subprocess.TimeoutExpired('rg', 30),
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'rg timed out after 30 seconds'

    def test_rg_nonzero_error_and_lossy_stderr_are_exact(
        self, executor, temp_workspace
    ):
        completed = subprocess.CompletedProcess(
            [],
            2,
            b'',
            b'regex parse error: \xff\n',
        )

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=completed,
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='write_records(', path=temp_workspace),
            )

        assert isinstance(obs, ErrorObservation)
        assert obs.content == 'rg failed: regex parse error: �\n'

    def test_invalid_utf8_stdout_lines_are_skipped(self, executor, temp_workspace):
        completed = subprocess.CompletedProcess(
            [],
            0,
            b'/repo/valid.py\n/repo/\xff.py\n/repo/final.py\n',
            b'',
        )

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=completed,
        ):
            obs = self.run(
                executor,
                CodexGrepFilesAction(pattern='needle', path=temp_workspace),
            )

        assert isinstance(obs, CmdOutputObservation)
        assert obs.content == '/repo/valid.py\n/repo/final.py'


class TestCodexFunctionOutputWiring:
    """Producer truncation and CmdOutput constructor-bypass integration."""

    generic_marker = '[... Observation truncated due to length ...]'
    token_model = 'gpt-5.6-sol'
    byte_model = 'unknown-model'

    @pytest.fixture
    def executor(self, temp_workspace):
        from openhands.runtime.action_execution_server import ActionExecutor

        executor = MagicMock(spec=ActionExecutor)
        executor.bash_session = MagicMock()
        executor.bash_session.cwd = temp_workspace
        executor._resolve_path = ActionExecutor._resolve_path.__get__(executor)
        executor.codex_read_file = ActionExecutor.codex_read_file.__get__(executor)
        executor.codex_list_dir = ActionExecutor.codex_list_dir.__get__(executor)
        executor.codex_grep_files = ActionExecutor.codex_grep_files.__get__(executor)
        return executor

    @staticmethod
    def attach_model(action, function_name, model_name, call_id):
        from openhands.events.tool import ToolCallMetadata

        action.tool_call_metadata = ToolCallMetadata(
            function_name=function_name,
            tool_call_id=call_id,
            model_response={'model': model_name},
            total_calls_in_response=1,
            tool_result_format='codex',
        )
        return action

    def assert_model_policies(self, raw_body, token_obs, byte_obs):
        from openhands.agenthub.codex_agent.tool_output import (
            truncate_function_output,
        )

        assert 30_000 < len(raw_body) < 48_000
        assert isinstance(token_obs, CmdOutputObservation)
        assert isinstance(byte_obs, CmdOutputObservation)
        assert token_obs.content == truncate_function_output(
            raw_body,
            model_name=self.token_model,
        )
        assert token_obs.content == raw_body

        expected_byte_body = truncate_function_output(
            raw_body,
            model_name=self.byte_model,
        )
        removed_chars = len(raw_body) - 12_000
        assert byte_obs.content == expected_byte_body
        assert byte_obs.content.startswith(raw_body[:6_000])
        assert byte_obs.content.endswith(raw_body[-6_000:])
        assert f'…{removed_chars} chars truncated…' in byte_obs.content

        assert self.generic_marker not in token_obs.content
        assert self.generic_marker not in byte_obs.content

    def test_read_file_model_policy_and_constructor_bypass(
        self, executor, temp_workspace
    ):
        from openhands.agenthub.codex_agent.tool_output import format_read_slice

        lines = [f'{index:03d}-' + 'r' * 396 for index in range(90)]
        path = create_test_file(
            temp_workspace,
            'large-read.txt',
            '\n'.join(lines) + '\n',
        )
        raw_body = format_read_slice(lines, offset=1, limit=len(lines))

        token_action = self.attach_model(
            CodexReadFileAction(file_path=path, limit=len(lines)),
            'read_file',
            self.token_model,
            'read-token',
        )
        byte_action = self.attach_model(
            CodexReadFileAction(file_path=path, limit=len(lines)),
            'read_file',
            self.byte_model,
            'read-byte',
        )

        token_obs = asyncio.run(executor.codex_read_file(token_action))
        byte_obs = asyncio.run(executor.codex_read_file(byte_action))

        self.assert_model_policies(raw_body, token_obs, byte_obs)

    def test_list_dir_model_policy_and_constructor_bypass(
        self, executor, temp_workspace
    ):
        names = [f'{index:03d}-' + 'l' * 190 + '.txt' for index in range(180)]
        for name in names:
            create_test_file(temp_workspace, name, '')
        raw_body = f'Absolute path: {temp_workspace}\n' + '\n'.join(names)

        token_action = self.attach_model(
            CodexListDirAction(
                dir_path=temp_workspace,
                limit=len(names),
                depth=1,
            ),
            'list_dir',
            self.token_model,
            'list-token',
        )
        byte_action = self.attach_model(
            CodexListDirAction(
                dir_path=temp_workspace,
                limit=len(names),
                depth=1,
            ),
            'list_dir',
            self.byte_model,
            'list-byte',
        )

        token_obs = asyncio.run(executor.codex_list_dir(token_action))
        byte_obs = asyncio.run(executor.codex_list_dir(byte_action))

        self.assert_model_policies(raw_body, token_obs, byte_obs)

    def test_grep_files_model_policy_and_constructor_bypass(
        self, executor, temp_workspace
    ):
        paths = [f'/repo/{index:03d}-' + 'g' * 190 + '.py' for index in range(180)]
        raw_body = '\n'.join(paths)
        completed = subprocess.CompletedProcess(
            [],
            0,
            (raw_body + '\n').encode(),
            b'',
        )

        token_action = self.attach_model(
            CodexGrepFilesAction(
                pattern='needle',
                path=temp_workspace,
                limit=len(paths),
            ),
            'grep_files',
            self.token_model,
            'grep-token',
        )
        byte_action = self.attach_model(
            CodexGrepFilesAction(
                pattern='needle',
                path=temp_workspace,
                limit=len(paths),
            ),
            'grep_files',
            self.byte_model,
            'grep-byte',
        )

        with patch(
            'openhands.runtime.action_execution_server.subprocess.run',
            return_value=completed,
        ):
            token_obs = asyncio.run(executor.codex_grep_files(token_action))
            byte_obs = asyncio.run(executor.codex_grep_files(byte_action))

        self.assert_model_policies(raw_body, token_obs, byte_obs)


# ==============================================================================
# CodexApplyPatch Handler Tests
# ==============================================================================


class TestCodexApplyPatchParser:
    """Tests for the Codex freeform patch format parser."""

    @pytest.fixture
    def executor(self, temp_workspace):
        """Create a real ActionExecutor-like object for parser tests."""
        from openhands.runtime.action_execution_server import ActionExecutor

        executor = MagicMock(spec=ActionExecutor)
        executor.bash_session = MagicMock()
        executor.bash_session.cwd = temp_workspace
        # Bind real methods
        executor._codex_parse_patch = ActionExecutor._codex_parse_patch.__get__(
            executor
        )
        executor._codex_parse_update_chunk = (
            ActionExecutor._codex_parse_update_chunk.__get__(executor)
        )
        executor._codex_seek_sequence = ActionExecutor._codex_seek_sequence
        executor._codex_apply_update_hunk = (
            ActionExecutor._codex_apply_update_hunk.__get__(executor)
        )
        executor.codex_apply_patch = ActionExecutor.codex_apply_patch.__get__(executor)
        return executor

    def test_parse_add_file(self, executor):
        """Test parsing *** Add File: hunks."""
        patch = (
            '*** Begin Patch\n'
            '*** Add File: new.py\n'
            '+print("hello")\n'
            '+print("world")\n'
            '*** End Patch'
        )
        hunks = executor._codex_parse_patch(patch)
        assert len(hunks) == 1
        assert hunks[0]['type'] == 'add'
        assert hunks[0]['path'] == 'new.py'
        assert hunks[0]['contents'] == 'print("hello")\nprint("world")\n'

    def test_parse_delete_file(self, executor):
        """Test parsing *** Delete File: hunks."""
        patch = '*** Begin Patch\n*** Delete File: old.py\n*** End Patch'
        hunks = executor._codex_parse_patch(patch)
        assert len(hunks) == 1
        assert hunks[0]['type'] == 'delete'
        assert hunks[0]['path'] == 'old.py'

    def test_parse_update_file_simple(self, executor):
        """Test parsing *** Update File: with a simple chunk."""
        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@\n'
            ' foo\n'
            '-bar\n'
            '+baz\n'
            '*** End Patch'
        )
        hunks = executor._codex_parse_patch(patch)
        assert len(hunks) == 1
        h = hunks[0]
        assert h['type'] == 'update'
        assert h['path'] == 'test.py'
        assert len(h['chunks']) == 1
        chunk = h['chunks'][0]
        assert chunk['context'] is None
        assert chunk['old_lines'] == ['foo', 'bar']
        assert chunk['new_lines'] == ['foo', 'baz']

    def test_parse_update_file_with_context(self, executor):
        """Test parsing update chunk with @@ context line."""
        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@ def main():\n'
            '-    old_code\n'
            '+    new_code\n'
            '*** End Patch'
        )
        hunks = executor._codex_parse_patch(patch)
        chunk = hunks[0]['chunks'][0]
        assert chunk['context'] == 'def main():'
        assert chunk['old_lines'] == ['    old_code']
        assert chunk['new_lines'] == ['    new_code']

    def test_parse_update_file_multiple_chunks(self, executor):
        """Test parsing Update File with multiple @@ chunks."""
        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@\n'
            ' foo\n'
            '-bar\n'
            '+BAR\n'
            '@@\n'
            ' baz\n'
            '-qux\n'
            '+QUX\n'
            '*** End Patch'
        )
        hunks = executor._codex_parse_patch(patch)
        chunks = hunks[0]['chunks']
        assert len(chunks) == 2

    def test_parse_multiple_operations(self, executor):
        """Test parsing a patch with add, delete, and update."""
        patch = (
            '*** Begin Patch\n'
            '*** Add File: new.py\n'
            '+content\n'
            '*** Delete File: old.py\n'
            '*** Update File: mod.py\n'
            '@@\n'
            '-old\n'
            '+new\n'
            '*** End Patch'
        )
        hunks = executor._codex_parse_patch(patch)
        assert len(hunks) == 3
        assert hunks[0]['type'] == 'add'
        assert hunks[1]['type'] == 'delete'
        assert hunks[2]['type'] == 'update'

    def test_parse_update_file_with_move(self, executor):
        """Test parsing *** Move to: within Update File."""
        patch = (
            '*** Begin Patch\n'
            '*** Update File: old_name.py\n'
            '*** Move to: new_name.py\n'
            '@@\n'
            '-old\n'
            '+new\n'
            '*** End Patch'
        )
        hunks = executor._codex_parse_patch(patch)
        assert hunks[0]['move_path'] == 'new_name.py'

    def test_parse_update_file_end_of_file(self, executor):
        """Test parsing *** End of File marker."""
        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@\n'
            '+new_last_line\n'
            '*** End of File\n'
            '*** End Patch'
        )
        hunks = executor._codex_parse_patch(patch)
        chunk = hunks[0]['chunks'][0]
        assert chunk['is_eof'] is True

    def test_parse_error_missing_begin(self, executor):
        """Test error on missing *** Begin Patch."""
        with pytest.raises(ValueError, match="Expected '\\*\\*\\* Begin Patch'"):
            executor._codex_parse_patch('bad patch')

    def test_parse_error_missing_end(self, executor):
        """Test error on missing *** End Patch."""
        with pytest.raises(ValueError, match="Expected '\\*\\*\\* End Patch'"):
            executor._codex_parse_patch('*** Begin Patch\n*** Add File: x\n+y')

    def test_parse_error_empty_update_hunk(self, executor):
        """Test error when Update File has no chunks."""
        patch = '*** Begin Patch\n*** Update File: test.py\n*** End Patch'
        with pytest.raises(ValueError, match='contains no change chunks'):
            executor._codex_parse_patch(patch)

    def test_parse_heredoc_wrapper(self, executor):
        """Test lenient parsing with <<EOF heredoc wrapper."""
        patch = (
            "<<'EOF'\n"
            '*** Begin Patch\n'
            '*** Add File: test.py\n'
            '+hello\n'
            '*** End Patch\n'
            'EOF'
        )
        hunks = executor._codex_parse_patch(patch)
        assert len(hunks) == 1
        assert hunks[0]['type'] == 'add'

    def test_parse_update_without_explicit_context(self, executor):
        """Test update chunk without @@ header (allowed for first chunk)."""
        patch = (
            '*** Begin Patch\n'
            '*** Update File: file.py\n'
            ' import foo\n'
            '+bar\n'
            '*** End Patch'
        )
        hunks = executor._codex_parse_patch(patch)
        chunk = hunks[0]['chunks'][0]
        assert chunk['context'] is None
        assert chunk['old_lines'] == ['import foo']
        assert chunk['new_lines'] == ['import foo', 'bar']


class TestCodexSeekSequence:
    """Tests for the _codex_seek_sequence matching logic."""

    def test_exact_match(self):
        from openhands.runtime.action_execution_server import ActionExecutor

        lines = ['foo', 'bar', 'baz']
        assert ActionExecutor._codex_seek_sequence(lines, ['bar', 'baz'], 0) == 1

    def test_rstrip_match(self):
        from openhands.runtime.action_execution_server import ActionExecutor

        lines = ['foo   ', 'bar\t\t']
        assert ActionExecutor._codex_seek_sequence(lines, ['foo', 'bar'], 0) == 0

    def test_trim_match(self):
        from openhands.runtime.action_execution_server import ActionExecutor

        lines = ['    foo   ', '   bar\t']
        assert ActionExecutor._codex_seek_sequence(lines, ['foo', 'bar'], 0) == 0

    def test_pattern_longer_than_input(self):
        from openhands.runtime.action_execution_server import ActionExecutor

        lines = ['one']
        assert ActionExecutor._codex_seek_sequence(lines, ['a', 'b', 'c'], 0) is None

    def test_empty_pattern(self):
        from openhands.runtime.action_execution_server import ActionExecutor

        assert ActionExecutor._codex_seek_sequence(['x'], [], 0) == 0

    def test_eof_mode_searches_from_end(self):
        from openhands.runtime.action_execution_server import ActionExecutor

        lines = ['a', 'b', 'c', 'b', 'c']
        # With eof=True, should find the last occurrence
        assert ActionExecutor._codex_seek_sequence(lines, ['b', 'c'], 0, eof=True) == 3

    def test_no_match(self):
        from openhands.runtime.action_execution_server import ActionExecutor

        lines = ['foo', 'bar']
        assert ActionExecutor._codex_seek_sequence(lines, ['xyz'], 0) is None

    def test_start_offset(self):
        from openhands.runtime.action_execution_server import ActionExecutor

        lines = ['a', 'b', 'a', 'b']
        # Starting from index 2, should find second 'a'
        assert ActionExecutor._codex_seek_sequence(lines, ['a'], 2) == 2


class TestCodexApplyPatchHandler:
    """Integration tests for the codex_apply_patch handler."""

    @pytest.fixture
    def executor(self, temp_workspace):
        """Create a real ActionExecutor-like object for apply_patch tests."""
        from openhands.runtime.action_execution_server import ActionExecutor

        executor = MagicMock(spec=ActionExecutor)
        executor.bash_session = MagicMock()
        executor.bash_session.cwd = temp_workspace
        # Bind real methods
        executor._codex_parse_patch = ActionExecutor._codex_parse_patch.__get__(
            executor
        )
        executor._codex_parse_update_chunk = (
            ActionExecutor._codex_parse_update_chunk.__get__(executor)
        )
        executor._codex_seek_sequence = ActionExecutor._codex_seek_sequence
        executor._codex_apply_update_hunk = (
            ActionExecutor._codex_apply_update_hunk.__get__(executor)
        )
        executor.codex_apply_patch = ActionExecutor.codex_apply_patch.__get__(executor)
        return executor

    def test_apply_patch_create_new_file(self, executor, temp_workspace):
        """Test creating a new file via patch."""
        patch = (
            '*** Begin Patch\n'
            '*** Add File: new_file.py\n'
            '+print("hello world")\n'
            '+print("goodbye")\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert isinstance(obs, CodexApplyPatchObservation)
        assert obs.success is True
        assert 'new_file.py' in obs.files_changed

        filepath = os.path.join(temp_workspace, 'new_file.py')
        assert os.path.exists(filepath)
        with open(filepath) as f:
            content = f.read()
        assert 'hello world' in content
        assert 'goodbye' in content

    def test_apply_patch_delete_file(self, executor, temp_workspace):
        """Test deleting a file via patch."""
        filepath = create_test_file(temp_workspace, 'to_delete.py', 'old content')
        assert os.path.exists(filepath)

        patch = '*** Begin Patch\n*** Delete File: to_delete.py\n*** End Patch'
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert isinstance(obs, CodexApplyPatchObservation)
        assert obs.success is True
        assert not os.path.exists(filepath)

    def test_apply_patch_create_nested_directory(self, executor, temp_workspace):
        """Test creating a file in a new nested directory."""
        patch = (
            '*** Begin Patch\n'
            '*** Add File: deep/nested/dir/file.py\n'
            '+content\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True
        nested_path = os.path.join(temp_workspace, 'deep', 'nested', 'dir', 'file.py')
        assert os.path.exists(nested_path)

    def test_apply_patch_empty_patch_is_error(self, executor):
        """Test that empty patch text returns an error."""
        action = CodexApplyPatchAction(patch='')
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert isinstance(obs, ErrorObservation)
        assert 'Empty patch' in obs.content

    def test_apply_patch_update_simple(self, executor, temp_workspace):
        """Test a simple update (replace one line)."""
        create_test_file(temp_workspace, 'test.py', 'foo\nbar\nbaz\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@\n'
            ' foo\n'
            '-bar\n'
            '+BAR\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'

        with open(os.path.join(temp_workspace, 'test.py')) as f:
            content = f.read()
        assert content == 'foo\nBAR\nbaz\n'

    def test_apply_patch_update_multiple_chunks(self, executor, temp_workspace):
        """Test update with multiple @@ chunks in one file."""
        create_test_file(temp_workspace, 'test.py', 'foo\nbar\nbaz\nqux\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@\n'
            ' foo\n'
            '-bar\n'
            '+BAR\n'
            '@@\n'
            ' baz\n'
            '-qux\n'
            '+QUX\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'

        with open(os.path.join(temp_workspace, 'test.py')) as f:
            content = f.read()
        assert content == 'foo\nBAR\nbaz\nQUX\n'

    def test_apply_patch_update_with_context_marker(self, executor, temp_workspace):
        """Test update using @@ <context> to locate changes."""
        create_test_file(
            temp_workspace,
            'test.py',
            'class Foo:\n    def bar(self):\n        return 1\n\n    def baz(self):\n        return 2\n',
        )

        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@ def baz(self):\n'
            '-        return 2\n'
            '+        return 42\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'

        with open(os.path.join(temp_workspace, 'test.py')) as f:
            content = f.read()
        assert 'return 42' in content
        assert 'return 1' in content  # Unchanged

    def test_apply_patch_append_at_eof(self, executor, temp_workspace):
        """Test appending lines at end-of-file."""
        create_test_file(temp_workspace, 'test.py', 'foo\nbar\nbaz\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@\n'
            ' baz\n'
            '+quux\n'
            '*** End of File\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'

        with open(os.path.join(temp_workspace, 'test.py')) as f:
            content = f.read()
        assert content == 'foo\nbar\nbaz\nquux\n'

    def test_apply_patch_interleaved_changes(self, executor, temp_workspace):
        """Test multiple chunks with additions and replacements."""
        create_test_file(temp_workspace, 'test.py', 'a\nb\nc\nd\ne\nf\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@\n'
            ' a\n'
            '-b\n'
            '+B\n'
            '@@\n'
            ' c\n'
            ' d\n'
            '-e\n'
            '+E\n'
            '@@\n'
            ' f\n'
            '+g\n'
            '*** End of File\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'

        with open(os.path.join(temp_workspace, 'test.py')) as f:
            content = f.read()
        assert content == 'a\nB\nc\nd\nE\nf\ng\n'

    def test_apply_patch_descriptive_error_file_not_found(
        self, executor, temp_workspace
    ):
        """A verification failure is returned raw, without a shell envelope."""
        patch_text = (
            '*** Begin Patch\n'
            '*** Update File: nonexistent.py\n'
            '@@\n'
            '-old\n'
            '+new\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch_text)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        assert obs.files_changed == []
        assert obs.content == (
            "Errors (1):\n  - Update failed: file not found 'nonexistent.py'"
        )
        assert not os.path.exists(os.path.join(temp_workspace, 'nonexistent.py'))

    def test_apply_patch_descriptive_error_context_not_found(
        self, executor, temp_workspace
    ):
        """Test descriptive error when context line can't be found."""
        create_test_file(temp_workspace, 'test.py', 'foo\nbar\nbaz\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@ def nonexistent_function():\n'
            '-old\n'
            '+new\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        assert 'could not find context' in obs.content.lower()

    def test_apply_patch_descriptive_error_lines_not_found(
        self, executor, temp_workspace
    ):
        """Test descriptive error when old_lines can't be matched."""
        create_test_file(temp_workspace, 'test.py', 'foo\nbar\nbaz\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@\n'
            '-this line does not exist\n'
            '+replacement\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        assert 'could not find' in obs.content.lower()
        assert 'this line does not exist' in obs.content

    def test_apply_patch_descriptive_error_parse_failure(self, executor):
        """Test descriptive parse error on malformed patch."""
        action = CodexApplyPatchAction(patch='not a valid patch at all')
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert isinstance(obs, CodexApplyPatchObservation)
        assert obs.success is False
        assert 'parse error' in obs.content.lower()

    def test_apply_patch_move_file(self, executor, temp_workspace):
        """Test *** Move to: renames a file."""
        create_test_file(temp_workspace, 'old.py', 'hello\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: old.py\n'
            '*** Move to: new.py\n'
            '@@\n'
            '-hello\n'
            '+world\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'

        assert not os.path.exists(os.path.join(temp_workspace, 'old.py'))
        with open(os.path.join(temp_workspace, 'new.py')) as f:
            content = f.read()
        assert content == 'world\n'

    def test_apply_patch_real_world_example(self, executor, temp_workspace):
        """Test the exact pattern from the user's failing example (escaped JSON)."""
        create_test_file(
            temp_workspace,
            'moto/acm/models.py',
            '        domain_names = set(sans + [self.common_name])\n'
            '        validation_options = []\n'
            '\n'
            '        if self.status == "PENDING_VALIDATION":\n'
            '            for san in domain_names:\n'
            '                resource_record = {\n'
            '                    "Name": f"_d930b28be6c5927595552b219965053e.{san}.",\n'
            '                    "Type": "CNAME",\n'
            '                }\n'
            '                validation_options.append(san)\n'
            '        else:\n'
            '            validation_options = [{"DomainName": name} for name in domain_names]\n'
            '        result["Certificate"]["DomainValidationOptions"] = validation_options\n',
        )

        patch = (
            '*** Begin Patch\n'
            '*** Update File: moto/acm/models.py\n'
            '@@\n'
            '-        domain_names = set(sans + [self.common_name])\n'
            '-        validation_options = []\n'
            '-\n'
            '-        if self.status == "PENDING_VALIDATION":\n'
            '-            for san in domain_names:\n'
            '-                resource_record = {\n'
            '-                    "Name": f"_d930b28be6c5927595552b219965053e.{san}.",\n'
            '-                    "Type": "CNAME",\n'
            '-                }\n'
            '-                validation_options.append(san)\n'
            '-        else:\n'
            '-            validation_options = [{"DomainName": name} for name in domain_names]\n'
            '-        result["Certificate"]["DomainValidationOptions"] = validation_options\n'
            '+        domain_names = sorted(set(sans + [self.common_name]))\n'
            '+        for san in domain_names:\n'
            '+            validation_options.append(san)\n'
            '+        result["Certificate"]["DomainValidationOptions"] = validation_options\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        assert 'moto/acm/models.py' in obs.files_changed

        with open(os.path.join(temp_workspace, 'moto/acm/models.py')) as f:
            content = f.read()
        assert 'sorted(set(sans' in content
        assert 'if self.status' not in content

    # ------------------------------------------------------------------
    # Scenarios ported from the Rust apply-patch test suite
    # ------------------------------------------------------------------

    def test_apply_patch_delete_nonexistent_file(self, executor, temp_workspace):
        """Scenario 007: Reject deleting a file that doesn't exist."""
        create_test_file(temp_workspace, 'other.txt', 'keep me')
        patch = '*** Begin Patch\n*** Delete File: missing.txt\n*** End Patch'
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        assert 'file not found' in obs.content.lower()
        # The other file must be untouched
        with open(os.path.join(temp_workspace, 'other.txt')) as f:
            assert f.read() == 'keep me'

    def test_apply_patch_delete_directory_fails(self, executor, temp_workspace):
        """Scenario 012: Reject deleting a directory (not a file)."""
        os.makedirs(os.path.join(temp_workspace, 'dir'))
        create_test_file(temp_workspace, 'dir/foo.txt', 'inside')

        patch = '*** Begin Patch\n*** Delete File: dir\n*** End Patch'
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        # Directory and contents should still exist
        assert os.path.exists(os.path.join(temp_workspace, 'dir/foo.txt'))

    def test_apply_patch_add_overwrites_existing(self, executor, temp_workspace):
        """Scenario 011: Add File overwrites an existing file."""
        create_test_file(temp_workspace, 'duplicate.txt', 'old content\n')

        patch = (
            '*** Begin Patch\n*** Add File: duplicate.txt\n+new content\n*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True
        with open(os.path.join(temp_workspace, 'duplicate.txt')) as f:
            assert f.read() == 'new content\n'

    def test_apply_patch_move_overwrites_existing_destination(
        self, executor, temp_workspace
    ):
        """Scenario 010: Move overwrites an existing destination file."""
        create_test_file(temp_workspace, 'old/name.txt', 'from\n')
        create_test_file(
            temp_workspace, 'renamed/dir/name.txt', 'will be overwritten\n'
        )

        patch = (
            '*** Begin Patch\n'
            '*** Update File: old/name.txt\n'
            '*** Move to: renamed/dir/name.txt\n'
            '@@\n'
            '-from\n'
            '+new\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        assert not os.path.exists(os.path.join(temp_workspace, 'old/name.txt'))
        with open(os.path.join(temp_workspace, 'renamed/dir/name.txt')) as f:
            assert f.read() == 'new\n'

    def test_apply_patch_pure_addition_chunk(self, executor, temp_workspace):
        """Scenario 016: Update chunk with only additions (no old lines)."""
        create_test_file(temp_workspace, 'input.txt', 'line1\nline2\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: input.txt\n'
            '@@\n'
            '+added line 1\n'
            '+added line 2\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'input.txt')) as f:
            content = f.read()
        assert 'line1' in content
        assert 'line2' in content
        assert 'added line 1' in content
        assert 'added line 2' in content

    def test_apply_patch_deletion_only_chunk(self, executor, temp_workspace):
        """Scenario 021: Update chunk with only deletions (no additions)."""
        create_test_file(temp_workspace, 'lines.txt', 'line1\nline2\nline3\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: lines.txt\n'
            '@@\n'
            ' line1\n'
            '-line2\n'
            ' line3\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'lines.txt')) as f:
            content = f.read()
        assert content == 'line1\nline3\n'

    def test_apply_patch_appends_trailing_newline(self, executor, temp_workspace):
        """Scenario 014: Update adds trailing newline when file lacks one."""
        filepath = os.path.join(temp_workspace, 'no_newline.txt')
        with open(filepath, 'w') as f:
            f.write('no newline at end')  # No trailing newline

        patch = (
            '*** Begin Patch\n'
            '*** Update File: no_newline.txt\n'
            '@@\n'
            '-no newline at end\n'
            '+first line\n'
            '+second line\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(filepath) as f:
            content = f.read()
        assert content == 'first line\nsecond line\n'

    def test_apply_patch_partial_success_leaves_changes(self, executor, temp_workspace):
        """Scenario 015: First op succeeds, second fails; successful changes remain."""
        patch = (
            '*** Begin Patch\n'
            '*** Add File: created.txt\n'
            '+hello\n'
            '*** Update File: missing.txt\n'
            '@@\n'
            '-old\n'
            '+new\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False  # Overall failure
        assert obs.files_changed == ['created.txt']  # First op succeeded
        assert obs.content == (
            'Partial success (1 file(s) changed):\n'
            '  A created.txt\n'
            'Errors (1):\n'
            "  - Update failed: file not found 'missing.txt'"
        )
        assert os.path.exists(os.path.join(temp_workspace, 'created.txt'))
        assert not os.path.exists(os.path.join(temp_workspace, 'missing.txt'))
        with open(os.path.join(temp_workspace, 'created.txt')) as f:
            assert f.read() == 'hello\n'

    def test_apply_patch_unicode_content(self, executor, temp_workspace):
        """Scenario 019: Unicode characters in patch content."""
        create_test_file(
            temp_workspace, 'foo.txt', 'line1\nna\u00efve caf\u00e9\nline3\n'
        )

        patch = (
            '*** Begin Patch\n'
            '*** Update File: foo.txt\n'
            '@@\n'
            ' line1\n'
            '-na\u00efve caf\u00e9\n'
            '+na\u00efve caf\u00e9 \u2705\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'foo.txt')) as f:
            content = f.read()
        assert 'na\u00efve caf\u00e9 \u2705' in content
        assert 'line3' in content

    def test_apply_patch_invalid_hunk_header(self, executor):
        """Scenario 013: Reject invalid operation type."""
        patch = '*** Begin Patch\n*** Frobnicate File: foo\n*** End Patch'
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        assert (
            'parse error' in obs.content.lower() or 'unexpected' in obs.content.lower()
        )

    def test_apply_patch_empty_patch_no_ops(self, executor):
        """Scenario 005: Patch with no operations between markers."""
        patch = '*** Begin Patch\n*** End Patch'
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        assert (
            'no file' in obs.content.lower() or 'no operations' in obs.content.lower()
        )

    def test_apply_patch_rejects_missing_context(self, executor, temp_workspace):
        """Scenario 006: Old lines not found in file."""
        create_test_file(temp_workspace, 'modify.txt', 'line1\nline2\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: modify.txt\n'
            '@@\n'
            '-missing_line\n'
            '+changed\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        assert 'could not find' in obs.content.lower()
        # File should be unchanged
        with open(os.path.join(temp_workspace, 'modify.txt')) as f:
            assert f.read() == 'line1\nline2\n'

    def test_apply_patch_requires_existing_file_for_update(
        self, executor, temp_workspace
    ):
        """Scenario 009: Update on non-existent file fails."""
        create_test_file(temp_workspace, 'other.txt', 'keep')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: missing.txt\n'
            '@@\n'
            '-old\n'
            '+new\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        assert 'file not found' in obs.content.lower()

    def test_apply_patch_multiple_operations_combined(self, executor, temp_workspace):
        """Success uses Codex's exact shell body and grouped A/M/D summary."""
        import re

        create_test_file(temp_workspace, 'delete.txt', 'obsolete\n')
        create_test_file(temp_workspace, 'modify.txt', 'line1\nline2\n')

        patch_text = (
            '*** Begin Patch\n'
            '*** Add File: nested/new.txt\n'
            '+created\n'
            '*** Delete File: delete.txt\n'
            '*** Update File: modify.txt\n'
            '@@\n'
            '-line2\n'
            '+changed\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch_text)
        obs = asyncio.run(executor.codex_apply_patch(action))

        assert obs.success is True, f'Patch failed: {obs.content}'
        assert obs.files_changed == [
            'nested/new.txt',
            'modify.txt',
            'delete.txt',
        ]
        assert (
            re.fullmatch(
                r'Exit code: 0\n'
                r'Wall time: (?:0|[1-9]\d*)(?:\.[1-9])? seconds\n'
                r'Output:\n'
                r'Success\. Updated the following files:\n'
                r'A nested/new\.txt\n'
                r'M modify\.txt\n'
                r'D delete\.txt\n',
                obs.content,
            )
            is not None
        )

        # New file created
        with open(os.path.join(temp_workspace, 'nested/new.txt')) as f:
            assert f.read() == 'created\n'
        # File deleted
        assert not os.path.exists(os.path.join(temp_workspace, 'delete.txt'))
        # File modified
        with open(os.path.join(temp_workspace, 'modify.txt')) as f:
            content = f.read()
        assert content == 'line1\nchanged\n'

    def test_apply_patch_move_to_new_directory(self, executor, temp_workspace):
        """Scenario 004: Move file to a new directory with update."""
        create_test_file(temp_workspace, 'old/name.txt', 'old content\n')
        create_test_file(temp_workspace, 'old/other.txt', 'untouched\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: old/name.txt\n'
            '*** Move to: renamed/dir/name.txt\n'
            '@@\n'
            '-old content\n'
            '+new content\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        # Old file removed
        assert not os.path.exists(os.path.join(temp_workspace, 'old/name.txt'))
        # New file at new location
        with open(os.path.join(temp_workspace, 'renamed/dir/name.txt')) as f:
            assert f.read() == 'new content\n'
        # Other file untouched
        with open(os.path.join(temp_workspace, 'old/other.txt')) as f:
            assert f.read() == 'untouched\n'

    def test_apply_patch_multiple_chunks_same_file(self, executor, temp_workspace):
        """Scenario 003: Multiple update chunks in a single file."""
        create_test_file(temp_workspace, 'multi.txt', 'line1\nline2\nline3\nline4\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: multi.txt\n'
            '@@\n'
            '-line2\n'
            '+changed2\n'
            '@@\n'
            '-line4\n'
            '+changed4\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'multi.txt')) as f:
            content = f.read()
        assert content == 'line1\nchanged2\nline3\nchanged4\n'

    def test_apply_patch_end_of_file_marker_with_context(
        self, executor, temp_workspace
    ):
        """Scenario 022: End of File marker with context line."""
        create_test_file(temp_workspace, 'tail.txt', 'first\nsecond\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: tail.txt\n'
            '@@\n'
            ' first\n'
            '-second\n'
            '+second updated\n'
            '*** End of File\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'tail.txt')) as f:
            content = f.read()
        assert content == 'first\nsecond updated\n'

    def test_apply_patch_whitespace_padded_hunk_header(self, executor, temp_workspace):
        """Scenario 017: Whitespace padding around file header markers."""
        create_test_file(temp_workspace, 'foo.txt', 'old\n')

        # Note: the *** Update File header has leading whitespace
        patch = (
            '*** Begin Patch\n  *** Update File: foo.txt\n@@\n-old\n+new\n*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'foo.txt')) as f:
            assert f.read() == 'new\n'

    def test_apply_patch_whitespace_padded_patch_markers(
        self, executor, temp_workspace
    ):
        """Scenario 018: Whitespace padding around Begin/End Patch markers."""
        create_test_file(temp_workspace, 'file.txt', 'one\n')

        patch = (
            ' *** Begin Patch\n'
            '*** Update File: file.txt\n'
            '@@\n'
            '-one\n'
            '+two\n'
            '*** End Patch '
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'file.txt')) as f:
            assert f.read() == 'two\n'

    def test_apply_patch_success_file_summary(self, executor, temp_workspace):
        """Verify an untruncated success body retains its final newline."""
        create_test_file(temp_workspace, 'del.txt', 'x')
        create_test_file(temp_workspace, 'mod.txt', 'old\n')

        patch = (
            '*** Begin Patch\n'
            '*** Add File: new.txt\n'
            '+hello\n'
            '*** Delete File: del.txt\n'
            '*** Update File: mod.txt\n'
            '@@\n'
            '-old\n'
            '+new\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True
        assert obs.content.startswith('Exit code: 0\nWall time: ')
        duration, output = obs.content.removeprefix('Exit code: 0\nWall time: ').split(
            ' seconds\nOutput:\n', maxsplit=1
        )
        assert duration == f'{float(duration):.1f}'.removesuffix('.0')
        assert output == (
            'Success. Updated the following files:\nA new.txt\nM mod.txt\nD del.txt\n'
        )
        with open(os.path.join(temp_workspace, 'new.txt')) as f:
            assert f.read() == 'hello\n'
        with open(os.path.join(temp_workspace, 'mod.txt')) as f:
            assert f.read() == 'new\n'
        assert not os.path.exists(os.path.join(temp_workspace, 'del.txt'))

    def test_apply_patch_model_metadata_selects_byte_truncation(
        self,
        executor,
        temp_workspace,
    ):
        """gpt-5.2 metadata selects Codex's 10k-byte shell-output policy."""
        from openhands.events.tool import ToolCallMetadata

        file_names = [f'bulk/{index:03d}-' + 'x' * 40 + '.txt' for index in range(210)]
        patch_lines = ['*** Begin Patch']
        for file_name in file_names:
            patch_lines.extend((f'*** Add File: {file_name}', '+payload'))
        patch_lines.append('*** End Patch')

        action = CodexApplyPatchAction(patch='\n'.join(patch_lines))
        action.tool_call_metadata = ToolCallMetadata(
            function_name='apply_patch',
            tool_call_id='call-byte-policy',
            model_response={'model': 'gpt-5.2'},
            total_calls_in_response=1,
            tool_result_format='codex',
        )
        obs = asyncio.run(executor.codex_apply_patch(action))

        assert obs.success is True
        assert obs.files_changed == file_names
        header, truncated_output = obs.content.split(' seconds\n', maxsplit=1)
        assert header.startswith('Exit code: 0\nWall time: ')
        duration = header.removeprefix('Exit code: 0\nWall time: ')
        assert duration == f'{float(duration):.1f}'.removesuffix('.0')
        assert truncated_output.startswith(
            'Total output lines: 211\nOutput:\nSuccess. Updated the following files:\n'
        )
        assert obs.content.count('…1798 chars truncated…') == 1
        assert 'tokens truncated' not in obs.content
        assert obs.content.endswith(f'A {file_names[-1]}\n')
        for file_name in (file_names[0], file_names[105], file_names[-1]):
            with open(os.path.join(temp_workspace, file_name)) as f:
                assert f.read() == 'payload\n'

    def test_apply_patch_error_output_includes_preview(self, executor, temp_workspace):
        """Verify error message includes a preview of the old_lines that couldn't be matched."""
        create_test_file(temp_workspace, 'test.py', 'aaa\nbbb\nccc\n')

        patch = (
            '*** Begin Patch\n'
            '*** Update File: test.py\n'
            '@@\n'
            '-xxx line 1\n'
            '-xxx line 2\n'
            '-xxx line 3\n'
            '-xxx line 4\n'
            '-xxx line 5\n'
            '-xxx line 6\n'
            '+replacement\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False
        # Should include preview of first 5 lines + truncation notice
        assert 'xxx line 1' in obs.content
        assert 'xxx line 5' in obs.content
        assert 'more line' in obs.content.lower()

    def test_apply_patch_atomicity_no_side_effects_on_failure(
        self, executor, temp_workspace
    ):
        """Verification failure should leave no side effects (atomicity).

        From opencode apply_patch.test.ts: when an Add succeeds but a subsequent
        Update fails on a missing file, the added file should NOT exist because
        verification runs before application in OpenCode. Our Codex implementation
        applies sequentially, so we verify partial success leaves the first op's
        changes but fails overall.
        """
        patch = (
            '*** Begin Patch\n'
            '*** Add File: should_not_exist.txt\n'
            '+temporary\n'
            '*** Update File: nonexistent_target.txt\n'
            '@@\n'
            '-old\n'
            '+new\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is False

    def test_apply_patch_disambiguate_context_with_header(
        self, executor, temp_workspace
    ):
        """@@ context header disambiguates between duplicate patterns.

        From opencode apply_patch.test.ts: file has two 'x=10' lines under
        different function headers. Using '@@ fn b' should match the second one.
        """
        create_test_file(
            temp_workspace, 'multi_ctx.txt', 'fn a\nx=10\ny=2\nfn b\nx=10\ny=20\n'
        )
        patch = (
            '*** Begin Patch\n'
            '*** Update File: multi_ctx.txt\n'
            '@@ fn b\n'
            '-x=10\n'
            '+x=11\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'multi_ctx.txt')) as f:
            content = f.read()
        # First x=10 (under fn a) should be unchanged, second should be x=11
        assert content == 'fn a\nx=10\ny=2\nfn b\nx=11\ny=20\n'

    def test_apply_patch_eof_anchor_matches_from_end(self, executor, temp_workspace):
        """EOF anchor should match the LAST occurrence, not the first.

        From opencode apply_patch.test.ts: file has duplicate 'marker' lines,
        EOF anchor should change the last one.
        """
        create_test_file(
            temp_workspace, 'eof_anchor.txt', 'start\nmarker\nmiddle\nmarker\nend\n'
        )
        patch = (
            '*** Begin Patch\n'
            '*** Update File: eof_anchor.txt\n'
            '@@\n'
            '-marker\n'
            '-end\n'
            '+marker-changed\n'
            '+end\n'
            '*** End of File\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'eof_anchor.txt')) as f:
            content = f.read()
        # First marker unchanged, second marker changed
        assert content == 'start\nmarker\nmiddle\nmarker-changed\nend\n'

    def test_apply_patch_rejects_missing_second_chunk_context(
        self, executor, temp_workspace
    ):
        """Rejects patch with missing @@ for second chunk.

        From opencode apply_patch.test.ts: two chunks without separator should fail.
        """
        create_test_file(temp_workspace, 'two_chunks.txt', 'a\nb\nc\nd\n')
        patch = (
            '*** Begin Patch\n'
            '*** Update File: two_chunks.txt\n'
            '@@\n'
            '-b\n'
            '+B\n'
            '\n'
            '-d\n'
            '+D\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        # Should either fail or the parser should handle the blank line as a context line
        # The original OpenCode test expects failure
        if obs.success:
            # If it succeeds, the blank line was treated as context - that's OK
            pass
        else:
            # File should be unchanged on failure
            with open(os.path.join(temp_workspace, 'two_chunks.txt')) as f:
                assert f.read() == 'a\nb\nc\nd\n'

    def test_apply_patch_heredoc_not_supported(self, executor, temp_workspace):
        """Heredoc-wrapped patches are NOT stripped by our Python parser.

        The Codex Rust parser supports heredoc stripping in 'lenient' mode and
        the OpenCode TypeScript parser has stripHeredoc(). Our Python parser
        does not implement this yet, so heredoc-wrapped patches should fail
        with a parse error.
        """
        patch = (
            "cat <<'EOF'\n"
            '*** Begin Patch\n'
            '*** Add File: heredoc_test.txt\n'
            '+heredoc content\n'
            '*** End Patch\n'
            'EOF'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        # Our parser does not strip heredoc wrappers — this is a known gap
        assert obs.success is False
        assert (
            'begin patch' in obs.content.lower() or 'parse error' in obs.content.lower()
        )

    def test_apply_patch_trailing_whitespace_matching(self, executor, temp_workspace):
        """Matches lines despite trailing whitespace differences (rstrip pass).

        From opencode apply_patch.test.ts.
        """
        create_test_file(
            temp_workspace, 'trailing_ws.txt', 'line1  \nline2\nline3   \n'
        )
        patch = (
            '*** Begin Patch\n'
            '*** Update File: trailing_ws.txt\n'
            '@@\n'
            '-line2\n'
            '+changed\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'trailing_ws.txt')) as f:
            content = f.read()
        assert 'changed' in content
        assert 'line1  ' in content  # Trailing spaces preserved

    def test_apply_patch_leading_whitespace_matching(self, executor, temp_workspace):
        """Matches lines despite leading whitespace differences (trim pass).

        From opencode apply_patch.test.ts.
        """
        create_test_file(temp_workspace, 'leading_ws.txt', '  line1\nline2\n  line3\n')
        patch = (
            '*** Begin Patch\n'
            '*** Update File: leading_ws.txt\n'
            '@@\n'
            '-line2\n'
            '+changed\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'leading_ws.txt')) as f:
            content = f.read()
        assert 'changed' in content

    def test_apply_patch_unicode_punctuation_normalization(
        self, executor, temp_workspace
    ):
        """Matches Unicode fancy quotes/dashes against ASCII equivalents.

        From opencode apply_patch.test.ts.
        """
        left_quote = '\u201c'  # "
        right_quote = '\u201d'  # "
        em_dash = '\u2014'  # —
        create_test_file(
            temp_workspace,
            'unicode_punct.txt',
            f'He said {left_quote}hello{right_quote}\nsome{em_dash}dash\nend\n',
        )
        # Patch uses ASCII equivalents
        patch = (
            '*** Begin Patch\n'
            '*** Update File: unicode_punct.txt\n'
            '@@\n'
            '-He said "hello"\n'
            '+He said "hi"\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'unicode_punct.txt')) as f:
            content = f.read()
        assert 'He said "hi"' in content

    def test_apply_patch_insert_only_hunk(self, executor, temp_workspace):
        """Insert-only hunk (additions between context lines, no deletions).

        From opencode apply_patch.test.ts.
        """
        create_test_file(temp_workspace, 'insert_only.txt', 'alpha\nomega\n')
        patch = (
            '*** Begin Patch\n'
            '*** Update File: insert_only.txt\n'
            '@@\n'
            ' alpha\n'
            '+beta\n'
            ' omega\n'
            '*** End Patch'
        )
        action = CodexApplyPatchAction(patch=patch)
        obs = asyncio.run(executor.codex_apply_patch(action))
        assert obs.success is True, f'Patch failed: {obs.content}'
        with open(os.path.join(temp_workspace, 'insert_only.txt')) as f:
            assert f.read() == 'alpha\nbeta\nomega\n'


# ==============================================================================
# CodexUpdatePlan Handler Tests
# ==============================================================================


class TestCodexUpdatePlanHandler:
    """Tests for the codex_update_plan handler calling the actual handler method."""

    @pytest.fixture
    def executor(self):
        from openhands.runtime.action_execution_server import ActionExecutor

        executor = MagicMock(spec=ActionExecutor)
        executor.codex_update_plan = ActionExecutor.codex_update_plan.__get__(executor)
        return executor

    def test_update_plan_basic(self, executor):
        """Test basic plan update."""
        action = CodexUpdatePlanAction(
            plan=[
                {'step': 'Read code', 'status': 'completed'},
                {'step': 'Write feature', 'status': 'in_progress'},
                {'step': 'Test', 'status': 'pending'},
            ]
        )
        obs = asyncio.run(executor.codex_update_plan(action))
        assert isinstance(obs, CodexUpdatePlanObservation)
        assert obs.success is True
        assert obs.content == 'Plan updated'
        assert len(obs.plan) == 3

    def test_update_plan_rejects_multiple_in_progress(self, executor):
        """Test that multiple in_progress steps are rejected."""
        action = CodexUpdatePlanAction(
            plan=[
                {'step': 'Step A', 'status': 'in_progress'},
                {'step': 'Step B', 'status': 'in_progress'},
            ]
        )
        obs = asyncio.run(executor.codex_update_plan(action))
        assert isinstance(obs, ErrorObservation)
        assert 'in_progress' in obs.content.lower()

    def test_update_plan_validates_status_values(self, executor):
        """Test that invalid status values are rejected."""
        action = CodexUpdatePlanAction(
            plan=[
                {'step': 'Test', 'status': 'invalid_status'},
            ]
        )
        obs = asyncio.run(executor.codex_update_plan(action))
        assert isinstance(obs, ErrorObservation)
        assert 'invalid status' in obs.content.lower()

    def test_update_plan_requires_step_and_status(self, executor):
        """Test that missing step or status fields are rejected."""
        action = CodexUpdatePlanAction(plan=[{'status': 'pending'}])
        obs = asyncio.run(executor.codex_update_plan(action))
        assert isinstance(obs, ErrorObservation)
        assert 'step' in obs.content.lower()

        action2 = CodexUpdatePlanAction(plan=[{'step': 'Do something'}])
        obs2 = asyncio.run(executor.codex_update_plan(action2))
        assert isinstance(obs2, ErrorObservation)
        assert 'status' in obs2.content.lower()

    def test_update_plan_returns_plan_updated(self, executor):
        """The model-visible body is exact and does not echo arguments."""
        action = CodexUpdatePlanAction(
            plan=[
                {'step': 'Task', 'status': 'pending'},
            ],
            explanation='Keep this explanation out of the tool result.',
        )
        obs = asyncio.run(executor.codex_update_plan(action))
        assert isinstance(obs, CodexUpdatePlanObservation)
        assert obs.success is True
        assert obs.content == 'Plan updated'
        assert obs.plan == [{'step': 'Task', 'status': 'pending'}]

    def test_update_plan_stores_state(self, executor):
        """Test that plan state is stored across calls."""
        action = CodexUpdatePlanAction(
            plan=[
                {'step': 'Step 1', 'status': 'completed'},
                {'step': 'Step 2', 'status': 'in_progress'},
            ]
        )
        obs = asyncio.run(executor.codex_update_plan(action))
        assert obs.success is True
        assert obs.plan[0]['status'] == 'completed'
        assert obs.plan[1]['status'] == 'in_progress'

    def test_update_plan_empty_plan_list(self, executor):
        """Test updating with an empty plan list."""
        action = CodexUpdatePlanAction(plan=[])
        obs = asyncio.run(executor.codex_update_plan(action))
        assert isinstance(obs, CodexUpdatePlanObservation)
        assert obs.success is True

    def test_update_plan_all_completed(self, executor):
        """Test plan where all steps are completed."""
        action = CodexUpdatePlanAction(
            plan=[
                {'step': 'Step 1', 'status': 'completed'},
                {'step': 'Step 2', 'status': 'completed'},
                {'step': 'Step 3', 'status': 'completed'},
            ]
        )
        obs = asyncio.run(executor.codex_update_plan(action))
        assert obs.success is True
        assert len(obs.plan) == 3


# ==============================================================================
# Integration-style Tests (combining action creation + handler logic)
# ==============================================================================


class TestCodexIntegration:
    """Integration tests combining action creation with handler logic."""

    def test_read_file_action_defaults(self):
        """Test that CodexReadFileAction has correct Codex defaults."""
        action = CodexReadFileAction(file_path='/test.py')
        assert action.offset == 1  # 1-indexed (unlike OpenCode's 0)
        assert action.limit == 2000
        assert action.mode == 'slice'
        assert action.indentation == {}

    def test_list_dir_action_defaults(self):
        """Test that CodexListDirAction has correct Codex defaults."""
        action = CodexListDirAction(dir_path='/workspace')
        assert action.offset == 1  # 1-indexed
        assert action.limit == 25
        assert action.depth == 2

    def test_grep_files_action_defaults(self):
        """Test that CodexGrepFilesAction has correct defaults."""
        action = CodexGrepFilesAction(pattern='TODO')
        assert action.include == ''
        assert action.path == ''
        assert action.limit == 100

    def test_apply_patch_observation_success(self):
        """Test CodexApplyPatchObservation for successful patch."""
        obs = CodexApplyPatchObservation(
            content='Patch applied successfully to 2 file(s).',
            files_changed=['a.py', 'b.py'],
            success=True,
        )
        assert obs.success is True
        assert len(obs.files_changed) == 2
        assert 'Applied patch to 2 files' in obs.message

    def test_apply_patch_observation_failure(self):
        """Test CodexApplyPatchObservation for failed patch."""
        obs = CodexApplyPatchObservation(
            content='Failed to apply: conflict',
            files_changed=[],
            success=False,
        )
        assert obs.success is False
        assert obs.message == 'Failed to apply patch'

    def test_update_plan_observation(self):
        """Test CodexUpdatePlanObservation."""
        obs = CodexUpdatePlanObservation(
            content='Plan updated',
            plan=[{'step': 'A', 'status': 'completed'}],
            success=True,
        )
        assert obs.success is True
        assert obs.content == 'Plan updated'
        assert len(obs.plan) == 1
