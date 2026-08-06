"""This is the main file for the runtime client.

It is responsible for executing actions received from OpenHands backend and producing observations.

NOTE: this will be executed inside the docker sandbox.
"""

import argparse
import asyncio
import base64
import fnmatch
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Generator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, NamedTuple
from zipfile import ZipFile

try:
    import regex as _codex_regex
except ImportError:
    _codex_regex = None  # type: ignore[assignment]

import puremagic
from binaryornot.check import is_binary
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader

# Use OpenCodeEditor with fuzzy matching instead of default OHEditor
try:
    from openhands.agenthub.opencode_agent.opencode_editor import (
        OpenCodeEditor as OHEditor,
    )
except ImportError:
    # Fallback to standard OHEditor if OpenCodeEditor not available (e.g., in sandbox)
    from openhands_aci.editor.editor import OHEditor
from openhands_aci.editor.exceptions import ToolError
from openhands_aci.editor.results import ToolResult
from openhands_aci.utils.diff import get_diff
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException
from uvicorn import run

from openhands.agenthub.opencode_agent.tool_output import (
    GrepMatch,
    format_glob,
    format_grep,
    format_read_directory,
    format_read_file,
    format_shell_output,
    format_truncated_tool_output,
    split_file_lines,
    tail_shell_output,
    truncate_tool_output_head,
)
from openhands.agenthub.codex_agent.tool_output import (
    format_apply_patch_success as format_codex_apply_patch_success,
    format_os_error as format_codex_os_error,
    format_read_indentation as format_codex_read_indentation,
    format_read_slice as format_codex_read_slice,
    format_shell_output as format_codex_shell_output,
    split_file_lines as split_codex_file_lines,
    take_utf8_prefix as take_codex_utf8_prefix,
    truncate_function_output as truncate_codex_function_output,
)
from openhands.core.config.mcp_config import MCPStdioServerConfig
from openhands.core.exceptions import BrowserUnavailableException
from openhands.core.logger import get_uvicorn_json_log_config
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    ApplyPatchAction,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    GlobAction,
    GrepAction,
    IPythonRunCellAction,
    ListDirAction,
    OpenCodeReadAction,
    OpenCodeWriteAction,
    QuestionAction,
    TodoReadAction,
    TodoWriteAction,
)
from openhands.events.action.codex import (
    CodexApplyPatchAction,
    CodexGrepFilesAction,
    CodexListDirAction,
    CodexReadFileAction,
    CodexUpdatePlanAction,
)
from openhands.events.action.terminus_2 import Terminus2CmdRunAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileDownloadObservation,
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
    IPythonRunCellObservation,
    Observation,
    TodoReadObservation,
    TodoWriteObservation,
)
from openhands.events.observation.opencode import (
    ApplyPatchObservation,
    QuestionObservation,
)
from openhands.events.observation.codex import (
    CodexApplyPatchObservation,
    CodexUpdatePlanObservation,
)
from openhands.events.observation.terminus_2 import Terminus2CmdOutputObservation
from openhands.events.serialization import event_from_dict, event_to_dict
from openhands.runtime.browser import browse
from openhands.runtime.browser.browser_env import BrowserEnv
from openhands.runtime.file_viewer_server import start_file_viewer_server

# Import our custom MCP Proxy Manager
from openhands.runtime.mcp.proxy import MCPProxyManager
from openhands.runtime.plugins import ALL_PLUGINS, JupyterPlugin, Plugin, VSCodePlugin
from openhands.runtime.utils import find_available_tcp_port
from openhands.runtime.utils.bash import BashSession
from openhands.runtime.utils.files import insert_lines, read_lines
from openhands.runtime.utils.memory_monitor import MemoryMonitor
from openhands.runtime.utils.runtime_init import init_user_and_working_directory
from openhands.runtime.utils.system_stats import (
    get_system_stats,
    update_last_execution_time,
)
from openhands.utils.async_utils import call_sync_from_async, wait_all

if sys.platform == 'win32':
    from openhands.runtime.utils.windows_bash import WindowsPowershellSession


class ActionRequest(BaseModel):
    action: dict


ROOT_GID = 0

SESSION_API_KEY = os.environ.get('SESSION_API_KEY')
api_key_header = APIKeyHeader(name='X-Session-API-Key', auto_error=False)


def _tool_model_name(action: Action) -> str | None:
    metadata = action.tool_call_metadata
    if metadata is None:
        return None
    return getattr(metadata.model_response, 'model', None)


def _nul_argument_error(
    value: str | os.PathLike[str] | None,
    name: str,
) -> str | None:
    """Return a controlled error for an argument subprocess cannot accept."""
    if value is None:
        return None
    if '\x00' in os.fspath(value):
        return f'{name} must not contain NUL bytes'
    return None


def _decode_process_output(value: str | bytes) -> str:
    """Decode subprocess output without failing on filesystem byte names."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


def _decode_rg_json_value(payload: Any) -> str | None:
    """Decode ripgrep's JSON text-or-base64 value without propagating errors."""
    if not isinstance(payload, dict):
        return None
    text_value = payload.get('text')
    if isinstance(text_value, str):
        return text_value
    bytes_value = payload.get('bytes')
    if not isinstance(bytes_value, str):
        return None
    try:
        raw_value = base64.b64decode(bytes_value, validate=True)
    except ValueError:
        return None
    return raw_value.decode('utf-8', errors='replace')


_CODEX_GREP_MAX_PATTERN_BYTES = 16_384
_CODEX_GREP_MAX_INCLUDE_BYTES = 4_096
_CODEX_GREP_MAX_GLOB_VARIANTS = 256
_CODEX_GREP_MAX_GLOB_WORK = 4_096
_CODEX_GREP_MAX_GLOB_DEPTH = 64
_CODEX_GREP_MAX_IGNORE_FILE_BYTES = 1_048_576
_CODEX_GREP_MAX_IGNORE_PATTERNS = 10_000
_CODEX_GREP_MAX_IGNORE_CONTEXT_BYTES = 4_194_304
_NATIVE_SEARCH_TIMEOUT_SECONDS = 30
_NATIVE_SEARCH_MAX_ENTRIES = 100_000
_NATIVE_SEARCH_MAX_OPEN_DIRECTORIES = 256
_NATIVE_SEARCH_MAX_TRAVERSAL_PATH_BYTES = 1_048_576
_NATIVE_SEARCH_MAX_CANDIDATES = 50_000
_NATIVE_SEARCH_MAX_CANDIDATE_PATH_BYTES = 16_777_216
_NATIVE_SEARCH_MAX_LINE_BYTES = 1_048_576
_OPENCODE_SEARCH_RESULT_LIMIT = 100


class _CodexGrepPreprocessingError(ValueError):
    """Raised when fallback grep input cannot be processed safely."""


class _NativeSearchLimitError(RuntimeError):
    """Raised when a native fallback exceeds a bounded traversal limit."""


class _CodexIgnoreRule(NamedTuple):
    base_relative: str
    pattern: str
    storage_bytes: int


class _CodexIgnoreContext(NamedTuple):
    layers: tuple[tuple[_CodexIgnoreRule, ...], ...] = ()
    pattern_count: int = 0
    storage_bytes: int = 0


def _path_storage_size(path: str | os.PathLike[str]) -> int:
    """Return the bytes needed to retain a filesystem path."""
    return len(os.fsencode(path))


def _iter_bounded_search_files(
    search_path: Path,
    *,
    ensure_deadline: Callable[[], None],
    should_descend: Callable[[str, _CodexIgnoreContext], bool],
    load_ignore_context: Callable[[Path, str, _CodexIgnoreContext], _CodexIgnoreContext]
    | None = None,
) -> Generator[tuple[Path, str, _CodexIgnoreContext], None, None]:
    """Yield regular files with lazy, deadline-aware ``scandir`` traversal."""
    # Keeping one iterator per depth avoids eagerly retaining every sibling
    # directory. Both the open-directory stack and its path storage are capped.
    stack: list[tuple[Any, str, int, _CodexIgnoreContext]] = []
    stack_path_bytes = 0
    entry_count = 0

    def push_directory(
        directory: Path,
        relative_path: str,
        parent_context: _CodexIgnoreContext,
    ) -> None:
        nonlocal stack_path_bytes
        ensure_deadline()
        path_bytes = _path_storage_size(directory)
        if len(stack) >= _NATIVE_SEARCH_MAX_OPEN_DIRECTORIES:
            raise _NativeSearchLimitError(
                'search exceeds the '
                f'{_NATIVE_SEARCH_MAX_OPEN_DIRECTORIES}-directory traversal limit'
            )
        if stack_path_bytes + path_bytes > _NATIVE_SEARCH_MAX_TRAVERSAL_PATH_BYTES:
            raise _NativeSearchLimitError(
                'search traversal paths exceed the '
                f'{_NATIVE_SEARCH_MAX_TRAVERSAL_PATH_BYTES}-byte memory limit'
            )
        context = (
            load_ignore_context(directory, relative_path, parent_context)
            if load_ignore_context is not None
            else parent_context
        )
        ensure_deadline()
        scanner = os.scandir(directory)
        try:
            ensure_deadline()
        except BaseException:
            scanner.close()
            raise
        stack.append((scanner, relative_path, path_bytes, context))
        stack_path_bytes += path_bytes

    try:
        push_directory(search_path, '', _CodexIgnoreContext())
        while stack:
            ensure_deadline()
            scanner, parent_relative, path_bytes, context = stack[-1]
            try:
                entry = next(scanner)
            except StopIteration:
                scanner.close()
                stack.pop()
                stack_path_bytes -= path_bytes
                continue
            ensure_deadline()
            entry_count += 1
            if entry_count > _NATIVE_SEARCH_MAX_ENTRIES:
                raise _NativeSearchLimitError(
                    'search enumerated more than '
                    f'{_NATIVE_SEARCH_MAX_ENTRIES} filesystem entries'
                )

            relative_path = (
                f'{parent_relative}/{entry.name}' if parent_relative else entry.name
            )
            try:
                is_symlink = entry.is_symlink()
                ensure_deadline()
                if is_symlink:
                    continue
                is_directory = entry.is_dir(follow_symlinks=False)
                ensure_deadline()
                if is_directory:
                    ensure_deadline()
                    if should_descend(relative_path, context):
                        push_directory(Path(entry.path), relative_path, context)
                    continue
                is_file = entry.is_file(follow_symlinks=False)
                ensure_deadline()
                if not is_file:
                    continue
            except OSError:
                # Match rg --no-messages for an individual entry that races
                # with traversal or cannot be inspected.
                continue
            yield Path(entry.path), relative_path.replace(os.sep, '/'), context
    finally:
        for scanner, _, _, _ in reversed(stack):
            scanner.close()


def _iter_bounded_binary_lines(
    source: Any,
    *,
    ensure_deadline: Callable[[], None],
) -> Generator[tuple[int, bytes], None, None]:
    """Yield file lines without allocating an unbounded newline-free line."""
    line_number = 0
    while True:
        ensure_deadline()
        raw_text = source.readline(_NATIVE_SEARCH_MAX_LINE_BYTES + 1)
        ensure_deadline()
        if not raw_text:
            return
        if len(raw_text) > _NATIVE_SEARCH_MAX_LINE_BYTES:
            raise _NativeSearchLimitError(
                'file contains a line exceeding the '
                f'{_NATIVE_SEARCH_MAX_LINE_BYTES}-byte fallback limit'
            )
        line_number += 1
        yield line_number, raw_text


def _ensure_codex_grep_before_deadline(deadline: float | None) -> None:
    if deadline is not None and _codex_grep_monotonic() >= deadline:
        raise TimeoutError


def _validate_codex_grep_input(
    value: str,
    *,
    name: str,
    max_bytes: int,
    deadline: float,
) -> None:
    """Bound fallback inputs before regex or glob preprocessing."""
    _ensure_codex_grep_before_deadline(deadline)
    if len(value) > max_bytes:
        raise _CodexGrepPreprocessingError(
            f'{name} exceeds the {max_bytes}-byte fallback limit'
        )
    try:
        encoded_size = len(value.encode('utf-8'))
    except UnicodeEncodeError as error:
        raise _CodexGrepPreprocessingError(f'{name} is not valid UTF-8') from error
    if encoded_size > max_bytes:
        raise _CodexGrepPreprocessingError(
            f'{name} exceeds the {max_bytes}-byte fallback limit'
        )
    _ensure_codex_grep_before_deadline(deadline)


def _expand_codex_include_glob(
    pattern: str,
    *,
    deadline: float | None = None,
    max_variants: int = _CODEX_GREP_MAX_GLOB_VARIANTS,
) -> list[str]:
    """Expand rg brace and zero-directory ``**`` forms within strict caps."""
    variants: set[str] = set()
    expanded_brace_states: set[str] = set()
    work = 0

    def consume_work() -> None:
        nonlocal work
        work += 1
        if work > _CODEX_GREP_MAX_GLOB_WORK:
            raise _CodexGrepPreprocessingError(
                'include expansion exceeds the '
                f'{_CODEX_GREP_MAX_GLOB_WORK}-step fallback work limit'
            )

    def add_variant(variant: str) -> None:
        variants.add(variant)
        if len(variants) > max_variants:
            raise _CodexGrepPreprocessingError(
                f'include expands to more than {max_variants} glob variants'
            )

    def expand_braces(current: str, depth: int, logical_variants: int) -> None:
        _ensure_codex_grep_before_deadline(deadline)
        consume_work()
        if depth > _CODEX_GREP_MAX_GLOB_DEPTH:
            raise _CodexGrepPreprocessingError(
                'include glob nesting exceeds the fallback limit'
            )
        if current in expanded_brace_states:
            return
        expanded_brace_states.add(current)
        brace = re.search(r'\{([^{}]*,[^{}]*)\}', current)
        if brace is None:
            expand_double_stars(current)
            return
        # Syntactically repeated alternatives do not create distinct glob
        # variants and must not inflate the logical expansion bound.
        options = list(dict.fromkeys(brace.group(1).split(',')))
        if logical_variants > max_variants // len(options):
            raise _CodexGrepPreprocessingError(
                f'include expands to more than {max_variants} glob variants'
            )
        next_logical_variants = logical_variants * len(options)
        for option in options:
            consume_work()
            expand_braces(
                current[: brace.start()] + option + current[brace.end() :],
                depth + 1,
                next_logical_variants,
            )

    def expand_double_stars(current: str) -> None:
        pending = [current]
        seen = {current}
        while pending:
            _ensure_codex_grep_before_deadline(deadline)
            consume_work()
            candidate = pending.pop()
            add_variant(candidate)
            for marker in (match.start() for match in re.finditer(r'\*\*/', candidate)):
                consume_work()
                without_marker = candidate[:marker] + candidate[marker + 3 :]
                if without_marker not in seen:
                    seen.add(without_marker)
                    if len(seen) > max_variants:
                        raise _CodexGrepPreprocessingError(
                            f'include expands to more than {max_variants} glob variants'
                        )
                    pending.append(without_marker)

    expand_braces(pattern, 0, 1)
    return sorted(variants)


def _prepare_codex_include(
    include: str | None,
    *,
    deadline: float | None = None,
) -> tuple[bool, tuple[re.Pattern[str], ...]] | None:
    """Preprocess one include exactly once for a fallback search."""
    if include is None:
        return None

    negated = include.startswith('!')
    pattern = include[1:] if negated else include
    variants = tuple(
        re.compile(
            _translate_codex_glob(
                variant if '/' in variant else f'**/{variant}',
                deadline=deadline,
            )
        )
        for variant in _expand_codex_include_glob(pattern, deadline=deadline)
    )
    _ensure_codex_grep_before_deadline(deadline)
    return negated, variants


def _translate_codex_glob(
    pattern: str,
    *,
    deadline: float | None = None,
) -> str:
    """Translate an rg-style path glob into an anchored safe regex."""
    if pattern.startswith('/'):
        pattern = pattern[1:]
    pieces = [r'\A']
    index = 0
    while index < len(pattern):
        _ensure_codex_grep_before_deadline(deadline)
        if pattern.startswith('**/', index):
            pieces.append(r'(?:[^/]+/)*')
            index += 3
            continue
        if pattern.startswith('**', index):
            pieces.append(r'.*')
            index += 2
            continue

        character = pattern[index]
        if character == '*':
            pieces.append(r'[^/]*')
        elif character == '?':
            pieces.append(r'[^/]')
        elif character == '[':
            closing = index + 1
            if closing < len(pattern) and pattern[closing] in ('!', ']'):
                closing += 1
            while closing < len(pattern) and pattern[closing] != ']':
                closing += 1
            if closing == len(pattern):
                pieces.append(r'\[')
            else:
                shell_class = pattern[index : closing + 1]
                pieces.append(fnmatch.translate(shell_class)[4:-3])
                index = closing
        else:
            pieces.append(re.escape(character))
        index += 1
    pieces.append(r'\Z')
    _ensure_codex_grep_before_deadline(deadline)
    return ''.join(pieces)


def _matches_prepared_codex_include(
    relative_path: str,
    prepared: tuple[bool, tuple[re.Pattern[str], ...]] | None,
) -> bool:
    """Match a slash-normalized path against a preprocessed include."""
    if prepared is None:
        return True
    negated, variants = prepared
    matched = any(variant.match(relative_path) for variant in variants)
    return not matched if negated else matched


def _matches_codex_include(
    relative_path: str,
    include: str | None,
    *,
    deadline: float | None = None,
) -> bool:
    """Match an rg-style include glob against a slash-normalized path."""
    return _matches_prepared_codex_include(
        relative_path,
        _prepare_codex_include(include, deadline=deadline),
    )


def _positive_codex_include_matches(
    relative_path: str,
    include: str | None,
    prepared: tuple[bool, tuple[re.Pattern[str], ...]] | None,
) -> bool:
    """Return whether a positive rg glob explicitly selects one path."""
    return (
        include is not None
        and not include.startswith('!')
        and _matches_prepared_codex_include(relative_path, prepared)
    )


def _load_codex_ignore_patterns(
    directory: Path,
    *,
    base_relative: str = '',
    inherited: _CodexIgnoreContext | None = None,
    deadline: float | None = None,
) -> _CodexIgnoreContext:
    """Add one directory's bounded ignore rules to its inherited context."""
    context = inherited or _CodexIgnoreContext()
    local_rules: list[_CodexIgnoreRule] = []
    local_storage_bytes = 0
    for ignore_name in ('.gitignore', '.ignore', '.rgignore'):
        _ensure_codex_grep_before_deadline(deadline)
        ignore_path = directory / ignore_name
        ignore_display = (
            f'{base_relative}/{ignore_name}' if base_relative else ignore_name
        )
        try:
            if not ignore_path.is_file():
                continue
            source = ignore_path.open('rb')
        except OSError:
            continue
        consumed_bytes = 0
        with source:
            while True:
                _ensure_codex_grep_before_deadline(deadline)
                line = source.readline(_CODEX_GREP_MAX_INCLUDE_BYTES + 1)
                if not line:
                    break
                consumed_bytes += len(line)
                if consumed_bytes > _CODEX_GREP_MAX_IGNORE_FILE_BYTES:
                    raise _CodexGrepPreprocessingError(
                        f'{ignore_display} exceeds the '
                        f'{_CODEX_GREP_MAX_IGNORE_FILE_BYTES}-byte '
                        'fallback limit'
                    )
                if len(line) > _CODEX_GREP_MAX_INCLUDE_BYTES and not line.endswith(
                    (b'\n', b'\r')
                ):
                    raise _CodexGrepPreprocessingError(
                        f'{ignore_display} contains a line exceeding the '
                        f'{_CODEX_GREP_MAX_INCLUDE_BYTES}-byte '
                        'fallback limit'
                    )
                decoded = line.decode('utf-8', errors='replace').rstrip('\r\n')
                if not decoded or decoded.lstrip().startswith('#'):
                    continue
                storage_bytes = len(os.fsencode(base_relative)) + len(
                    decoded.encode('utf-8')
                )
                local_rules.append(
                    _CodexIgnoreRule(base_relative, decoded, storage_bytes)
                )
                local_storage_bytes += storage_bytes
                if (
                    context.pattern_count + len(local_rules)
                    > _CODEX_GREP_MAX_IGNORE_PATTERNS
                ):
                    raise _CodexGrepPreprocessingError(
                        'ignore files contain more than '
                        f'{_CODEX_GREP_MAX_IGNORE_PATTERNS} patterns'
                    )
                if (
                    context.storage_bytes + local_storage_bytes
                    > _CODEX_GREP_MAX_IGNORE_CONTEXT_BYTES
                ):
                    raise _CodexGrepPreprocessingError(
                        'active ignore rules exceed the '
                        f'{_CODEX_GREP_MAX_IGNORE_CONTEXT_BYTES}-byte memory limit'
                    )
    if not local_rules:
        return context
    return _CodexIgnoreContext(
        context.layers + (tuple(local_rules),),
        context.pattern_count + len(local_rules),
        context.storage_bytes + local_storage_bytes,
    )


def _matches_codex_ignore(
    relative_path: str,
    *,
    is_directory: bool,
    patterns: _CodexIgnoreContext,
    deadline: float | None = None,
) -> bool:
    """Apply common gitignore forms without requiring another dependency."""
    ignored = False
    for layer in patterns.layers:
        for rule in layer:
            _ensure_codex_grep_before_deadline(deadline)
            if rule.base_relative:
                prefix = f'{rule.base_relative}/'
                if not relative_path.startswith(prefix):
                    continue
                local_path = relative_path[len(prefix) :]
            else:
                local_path = relative_path

            raw_pattern = rule.pattern
            negated = raw_pattern.startswith('!')
            pattern = raw_pattern[1:] if negated else raw_pattern
            directory_only = pattern.endswith('/')
            if directory_only and not is_directory:
                continue
            pattern = pattern.rstrip('/')
            if not pattern:
                continue

            anchored = pattern.startswith('/')
            pattern = pattern.lstrip('/')
            if not anchored and '/' not in pattern:
                # A slashless pattern applies to a component at any depth
                # below the directory containing its ignore file.
                matched = fnmatch.fnmatchcase(local_path.rsplit('/', 1)[-1], pattern)
            else:
                include_pattern = f'/{pattern}' if anchored else pattern
                matched = _matches_codex_include(
                    local_path,
                    include_pattern,
                    deadline=deadline,
                )
            if matched:
                ignored = not negated
    return ignored


_codex_grep_monotonic = time.monotonic


class _CodexGrepRegexUnavailableError(RuntimeError):
    """Raised when the safe regex engine is unavailable for fallback grep."""


_CODEX_GREP_REGEX_ERROR = _codex_regex.error if _codex_regex is not None else re.error


def _fallback_codex_grep_files(
    search_path: Path,
    pattern: str,
    include: str | None,
    limit: int,
    *,
    timeout_seconds: float = 30,
) -> list[str]:
    """Find matching files without relying on an external search binary."""
    deadline = _codex_grep_monotonic() + timeout_seconds
    _validate_codex_grep_input(
        pattern,
        name='pattern',
        max_bytes=_CODEX_GREP_MAX_PATTERN_BYTES,
        deadline=deadline,
    )
    if include is not None:
        _validate_codex_grep_input(
            include,
            name='include',
            max_bytes=_CODEX_GREP_MAX_INCLUDE_BYTES,
            deadline=deadline,
        )
    if _codex_regex is None:
        raise _CodexGrepRegexUnavailableError(
            "the 'regex' package is required when ripgrep is unavailable"
        )
    try:
        matcher = _codex_regex.compile(pattern)
    except RecursionError as error:
        raise _CodexGrepPreprocessingError(
            'pattern nesting exceeds the fallback limit'
        ) from error
    _ensure_codex_grep_before_deadline(deadline)
    prepared_include = _prepare_codex_include(
        include,
        deadline=deadline,
    )
    _ensure_codex_grep_before_deadline(deadline)
    candidates: list[tuple[int, str, Path]] = []
    candidate_path_bytes = 0
    explicit_file = search_path.is_file()

    def add_candidate(candidate: Path, relative_path: str) -> None:
        nonlocal candidate_path_bytes
        _ensure_codex_grep_before_deadline(deadline)
        if not _matches_prepared_codex_include(relative_path, prepared_include):
            return
        try:
            display_path = str(candidate)
            display_path.encode('utf-8')
            modified_ns = candidate.stat().st_mtime_ns
        except (OSError, UnicodeEncodeError):
            return
        if len(candidates) >= _NATIVE_SEARCH_MAX_CANDIDATES:
            raise _NativeSearchLimitError(
                'search found more than '
                f'{_NATIVE_SEARCH_MAX_CANDIDATES} candidate files'
            )
        path_bytes = _path_storage_size(display_path)
        if candidate_path_bytes + path_bytes > _NATIVE_SEARCH_MAX_CANDIDATE_PATH_BYTES:
            raise _NativeSearchLimitError(
                'candidate paths exceed the '
                f'{_NATIVE_SEARCH_MAX_CANDIDATE_PATH_BYTES}-byte memory limit'
            )
        candidates.append((modified_ns, display_path, candidate))
        candidate_path_bytes += path_bytes

    if explicit_file:
        add_candidate(search_path, search_path.name)
    else:

        def should_descend(
            relative_path: str,
            ignore_context: _CodexIgnoreContext,
        ) -> bool:
            directory = relative_path.rsplit('/', 1)[-1]
            if directory == '.git':
                return False
            explicitly_selected = _positive_codex_include_matches(
                relative_path,
                include,
                prepared_include,
            )
            # A positive rg glob can whitelist a hidden directory when the
            # directory path itself matches (for example ``*`` or ``**/*``).
            # Without that explicit selection, rg keeps hidden directories
            # out of the traversal. Directory symlinks are rejected earlier.
            if directory.startswith('.') and not explicitly_selected:
                return False
            ignored = _matches_codex_ignore(
                relative_path,
                is_directory=True,
                patterns=ignore_context,
                deadline=deadline,
            )
            return not ignored or explicitly_selected

        def load_ignore_context(
            directory: Path,
            base_relative: str,
            inherited: _CodexIgnoreContext,
        ) -> _CodexIgnoreContext:
            return _load_codex_ignore_patterns(
                directory,
                base_relative=base_relative,
                inherited=inherited,
                deadline=deadline,
            )

        for candidate, relative_path, ignore_context in _iter_bounded_search_files(
            search_path,
            ensure_deadline=lambda: _ensure_codex_grep_before_deadline(deadline),
            should_descend=should_descend,
            load_ignore_context=load_ignore_context,
        ):
            filename = relative_path.rsplit('/', 1)[-1]
            explicitly_selected = _positive_codex_include_matches(
                relative_path,
                include,
                prepared_include,
            )
            # A positive rg glob may explicitly select a hidden file.
            if filename.startswith('.') and not explicitly_selected:
                continue
            ignored = _matches_codex_ignore(
                relative_path,
                is_directory=False,
                patterns=ignore_context,
                deadline=deadline,
            )
            if ignored and not explicitly_selected:
                continue
            add_candidate(candidate, relative_path)

    # Newest first, with lexical order as the deterministic mtime tie-breaker.
    candidates.sort(key=lambda item: (-item[0], item[1]))

    matches: list[str] = []
    for _, display_path, candidate in candidates:
        if _codex_grep_monotonic() >= deadline:
            raise TimeoutError
        try:
            with candidate.open('rb') as source:
                matched = False
                binary = False
                for _, line in _iter_bounded_binary_lines(
                    source,
                    ensure_deadline=lambda: _ensure_codex_grep_before_deadline(
                        deadline
                    ),
                ):
                    if not explicit_file and b'\x00' in line:
                        binary = True
                        break
                    remaining_seconds = deadline - _codex_grep_monotonic()
                    if remaining_seconds <= 0:
                        raise TimeoutError
                    if matcher.search(
                        line.decode('utf-8', errors='replace'),
                        timeout=remaining_seconds,
                    ):
                        matched = True
                if matched and not binary:
                    matches.append(display_path)
        except TimeoutError:
            raise
        except OSError:
            # Match rg --no-messages: an unreadable individual file does not
            # add noise to otherwise useful search results.
            continue
        if len(matches) == limit:
            break
    return matches


def _fallback_opencode_glob(
    search_path: Path,
    pattern: str,
    *,
    timeout_seconds: float = 30,
) -> list[str]:
    """Find OpenCode glob results with a bounded native traversal."""
    deadline = _codex_grep_monotonic() + timeout_seconds
    _validate_codex_grep_input(
        pattern,
        name='pattern',
        max_bytes=_CODEX_GREP_MAX_INCLUDE_BYTES,
        deadline=deadline,
    )
    prepared_pattern = _prepare_codex_include(pattern, deadline=deadline)
    files: list[str] = []
    result_path_bytes = 0

    def should_descend(
        relative_path: str,
        ignore_context: _CodexIgnoreContext,
    ) -> bool:
        directory_name = relative_path.rsplit('/', 1)[-1]
        if directory_name == '.git':
            return False
        explicitly_selected = _positive_codex_include_matches(
            relative_path,
            pattern,
            prepared_pattern,
        )
        # A positive rg glob can whitelist a hidden directory only when the
        # directory path itself matches (for example ``*`` or ``**/*``).
        if directory_name.startswith('.') and not explicitly_selected:
            return False
        ignored = _matches_codex_ignore(
            relative_path,
            is_directory=True,
            patterns=ignore_context,
            deadline=deadline,
        )
        return not ignored or explicitly_selected

    def load_ignore_context(
        directory: Path,
        base_relative: str,
        inherited: _CodexIgnoreContext,
    ) -> _CodexIgnoreContext:
        return _load_codex_ignore_patterns(
            directory,
            base_relative=base_relative,
            inherited=inherited,
            deadline=deadline,
        )

    traversal = _iter_bounded_search_files(
        search_path,
        ensure_deadline=lambda: _ensure_codex_grep_before_deadline(deadline),
        should_descend=should_descend,
        load_ignore_context=load_ignore_context,
    )
    try:
        for candidate, relative_path, ignore_context in traversal:
            matches_pattern = _matches_prepared_codex_include(
                relative_path,
                prepared_pattern,
            )
            if not matches_pattern:
                continue
            explicitly_selected = _positive_codex_include_matches(
                relative_path,
                pattern,
                prepared_pattern,
            )
            if (
                relative_path.rsplit('/', 1)[-1].startswith('.')
                and not explicitly_selected
            ):
                continue
            ignored = _matches_codex_ignore(
                relative_path,
                is_directory=False,
                patterns=ignore_context,
                deadline=deadline,
            )
            if ignored and not explicitly_selected:
                continue
            display_path = os.fsencode(candidate).decode('utf-8', errors='replace')
            path_bytes = _path_storage_size(display_path)
            if result_path_bytes + path_bytes > _NATIVE_SEARCH_MAX_CANDIDATE_PATH_BYTES:
                raise _NativeSearchLimitError(
                    'result paths exceed the '
                    f'{_NATIVE_SEARCH_MAX_CANDIDATE_PATH_BYTES}-byte memory limit'
                )
            files.append(display_path)
            result_path_bytes += path_bytes
            if len(files) == _OPENCODE_SEARCH_RESULT_LIMIT:
                break
    finally:
        traversal.close()
    return files


def _fallback_opencode_grep(
    search_path: Path,
    pattern: str,
    include: str | None,
    *,
    timeout_seconds: float = 30,
) -> list[GrepMatch]:
    """Find OpenCode grep matches with bounded traversal and regex timeouts."""
    deadline = _codex_grep_monotonic() + timeout_seconds
    include = include or None
    _validate_codex_grep_input(
        pattern,
        name='pattern',
        max_bytes=_CODEX_GREP_MAX_PATTERN_BYTES,
        deadline=deadline,
    )
    if include is not None:
        _validate_codex_grep_input(
            include,
            name='include',
            max_bytes=_CODEX_GREP_MAX_INCLUDE_BYTES,
            deadline=deadline,
        )
    if _codex_regex is None:
        raise _CodexGrepRegexUnavailableError(
            "the 'regex' package is required when ripgrep is unavailable"
        )
    try:
        matcher = _codex_regex.compile(pattern)
    except RecursionError as error:
        raise _CodexGrepPreprocessingError(
            'pattern nesting exceeds the fallback limit'
        ) from error
    _ensure_codex_grep_before_deadline(deadline)
    prepared_include = _prepare_codex_include(include, deadline=deadline)
    matches: list[GrepMatch] = []
    result_path_bytes = 0
    search_is_directory = search_path.is_dir()

    def should_descend(
        relative_path: str,
        ignore_context: _CodexIgnoreContext,
    ) -> bool:
        if relative_path.rsplit('/', 1)[-1] == '.git':
            return False
        explicitly_selected = _positive_codex_include_matches(
            relative_path,
            include,
            prepared_include,
        )
        ignored = _matches_codex_ignore(
            relative_path,
            is_directory=True,
            patterns=ignore_context,
            deadline=deadline,
        )
        return not ignored or explicitly_selected

    def load_ignore_context(
        directory: Path,
        base_relative: str,
        inherited: _CodexIgnoreContext,
    ) -> _CodexIgnoreContext:
        return _load_codex_ignore_patterns(
            directory,
            base_relative=base_relative,
            inherited=inherited,
            deadline=deadline,
        )

    if search_is_directory:
        traversal = _iter_bounded_search_files(
            search_path,
            ensure_deadline=lambda: _ensure_codex_grep_before_deadline(deadline),
            should_descend=should_descend,
            load_ignore_context=load_ignore_context,
        )
    else:
        traversal = None

    try:
        candidates = (
            traversal
            if traversal is not None
            else iter(((search_path, search_path.name, _CodexIgnoreContext()),))
        )
        for candidate, relative_path, ignore_context in candidates:
            _ensure_codex_grep_before_deadline(deadline)
            if not _matches_prepared_codex_include(
                relative_path,
                prepared_include,
            ):
                continue
            explicitly_selected = _positive_codex_include_matches(
                relative_path,
                include,
                prepared_include,
            )
            if search_is_directory:
                ignored = _matches_codex_ignore(
                    relative_path,
                    is_directory=False,
                    patterns=ignore_context,
                    deadline=deadline,
                )
                if ignored and not explicitly_selected:
                    continue
            display_path = os.fsencode(candidate).decode('utf-8', errors='replace')
            file_matches: list[GrepMatch] = []
            binary = False
            try:
                with candidate.open('rb') as source:
                    for line_number, raw_text in _iter_bounded_binary_lines(
                        source,
                        ensure_deadline=lambda: _ensure_codex_grep_before_deadline(
                            deadline
                        ),
                    ):
                        if b'\x00' in raw_text:
                            binary = True
                            break
                        remaining_seconds = deadline - _codex_grep_monotonic()
                        if remaining_seconds <= 0:
                            raise TimeoutError
                        text = raw_text.decode('utf-8', errors='replace')
                        if matcher.search(text, timeout=remaining_seconds):
                            file_matches.append(
                                GrepMatch(display_path, line_number, text)
                            )
                            if (
                                len(matches) + len(file_matches)
                                == _OPENCODE_SEARCH_RESULT_LIMIT
                            ):
                                break
            except TimeoutError:
                raise
            except OSError:
                continue
            if binary:
                continue
            if file_matches:
                path_bytes = _path_storage_size(display_path)
                if (
                    result_path_bytes + path_bytes
                    > _NATIVE_SEARCH_MAX_CANDIDATE_PATH_BYTES
                ):
                    raise _NativeSearchLimitError(
                        'result paths exceed the '
                        f'{_NATIVE_SEARCH_MAX_CANDIDATE_PATH_BYTES}-byte memory limit'
                    )
                matches.extend(file_matches)
                result_path_bytes += path_bytes
            if len(matches) == _OPENCODE_SEARCH_RESULT_LIMIT:
                break
    finally:
        if traversal is not None:
            traversal.close()
    return matches


def verify_api_key(api_key: str = Depends(api_key_header)):
    if SESSION_API_KEY and api_key != SESSION_API_KEY:
        raise HTTPException(status_code=403, detail='Invalid API Key')
    return api_key


def _execute_file_editor(
    editor: OHEditor,
    command: str,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | str | None = None,
    enable_linting: bool = False,
) -> tuple[str, tuple[str | None, str | None]]:
    """Execute file editor command and handle exceptions.

    Args:
        editor: The OHEditor instance
        command: Editor command to execute
        path: File path
        file_text: Optional file text content
        view_range: Optional view range tuple (start, end)
        old_str: Optional string to replace
        new_str: Optional replacement string
        insert_line: Optional line number for insertion (can be int or str)
        enable_linting: Whether to enable linting

    Returns:
        tuple: A tuple containing the output string and a tuple of old and new file content
    """
    result: ToolResult | None = None

    # Convert insert_line from string to int if needed
    if insert_line is not None and isinstance(insert_line, str):
        try:
            insert_line = int(insert_line)
        except ValueError:
            return (
                f"ERROR:\nInvalid insert_line value: '{insert_line}'. Expected an integer.",
                (None, None),
            )

    try:
        result = editor(
            command=command,
            path=path,
            file_text=file_text,
            view_range=view_range,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
            enable_linting=enable_linting,
        )
    except ToolError as e:
        result = ToolResult(error=e.message)
    except TypeError as e:
        # Handle unexpected arguments or type errors
        return f'ERROR:\n{str(e)}', (None, None)

    if result.error:
        return f'ERROR:\n{result.error}', (None, None)

    if not result.output:
        logger.warning(f'No output from file_editor for {path}')
        return '', (None, None)

    return result.output, (result.old_content, result.new_content)


class ActionExecutor:
    """ActionExecutor is running inside docker sandbox.

    It is responsible for executing actions received from OpenHands backend and producing observations.
    """

    def __init__(
        self,
        plugins_to_load: list[Plugin],
        work_dir: str,
        username: str,
        user_id: int,
        enable_browser: bool,
        browsergym_eval_env: str | None,
    ) -> None:
        self.plugins_to_load = plugins_to_load
        self._initial_cwd = work_dir
        self.username = username
        self.user_id = user_id
        _updated_user_id = init_user_and_working_directory(
            username=username, user_id=self.user_id, initial_cwd=work_dir
        )
        if _updated_user_id is not None:
            self.user_id = _updated_user_id

        self.bash_session: BashSession | 'WindowsPowershellSession' | None = None  # type: ignore[name-defined]
        self.lock = asyncio.Lock()
        self.plugins: dict[str, Plugin] = {}
        self.file_editor = OHEditor(workspace_root=self._initial_cwd)
        self.enable_browser = enable_browser
        self.browser: BrowserEnv | None = None
        self.browser_init_task: asyncio.Task | None = None
        self.browsergym_eval_env = browsergym_eval_env

        if (not self.enable_browser) and self.browsergym_eval_env:
            raise BrowserUnavailableException(
                'Browser environment is not enabled in config, but browsergym_eval_env is set'
            )

        self.start_time = time.time()
        self.last_execution_time = self.start_time
        self._initialized = False
        self.downloaded_files: list[str] = []
        self.downloads_directory = '/workspace/.downloads'
        self._todos: list[dict] = []  # In-memory todo list storage

        self.max_memory_gb: int | None = None
        if _override_max_memory_gb := os.environ.get('RUNTIME_MAX_MEMORY_GB', None):
            self.max_memory_gb = int(_override_max_memory_gb)
            logger.info(
                f'Setting max memory to {self.max_memory_gb}GB (according to the RUNTIME_MAX_MEMORY_GB environment variable)'
            )
        else:
            logger.info('No max memory limit set, using all available system memory')

        self.memory_monitor = MemoryMonitor(
            enable=os.environ.get('RUNTIME_MEMORY_MONITOR', 'False').lower()
            in ['true', '1', 'yes']
        )
        self.memory_monitor.start_monitoring()

    @property
    def initial_cwd(self):
        return self._initial_cwd

    async def _init_browser_async(self):
        """Initialize the browser asynchronously."""
        if not self.enable_browser:
            logger.info('Browser environment is not enabled in config')
            return

        if sys.platform == 'win32':
            logger.warning('Browser environment not supported on windows')
            return

        logger.debug('Initializing browser asynchronously')
        try:
            self.browser = BrowserEnv(self.browsergym_eval_env)
            logger.debug('Browser initialized asynchronously')
        except Exception as e:
            logger.exception(f'Failed to initialize browser: {e}')
            self.browser = None

    async def _ensure_browser_ready(self):
        """Ensure the browser is ready for use."""
        if self.browser is None:
            if self.browser_init_task is None:
                # Start browser initialization if it hasn't been started
                self.browser_init_task = asyncio.create_task(self._init_browser_async())
            elif self.browser_init_task.done():
                # If the task is done but browser is still None, restart initialization
                self.browser_init_task = asyncio.create_task(self._init_browser_async())

            # Wait for browser to be initialized
            if self.browser_init_task:
                logger.debug('Waiting for browser to be ready...')
                await self.browser_init_task

            # Check if browser was successfully initialized
            if self.browser is None:
                raise BrowserUnavailableException('Browser initialization failed')

        # If we get here, the browser is ready
        logger.debug('Browser is ready')

    def _create_bash_session(self, cwd: str | None = None):
        if sys.platform == 'win32':
            return WindowsPowershellSession(  # type: ignore[name-defined]
                work_dir=cwd or self._initial_cwd,
                username=self.username,
                no_change_timeout_seconds=int(
                    os.environ.get('NO_CHANGE_TIMEOUT_SECONDS', 10)
                ),
                max_memory_mb=self.max_memory_gb * 1024 if self.max_memory_gb else None,
            )
        else:
            bash_session = BashSession(
                work_dir=cwd or self._initial_cwd,
                username=self.username,
                no_change_timeout_seconds=int(
                    os.environ.get('NO_CHANGE_TIMEOUT_SECONDS', 10)
                ),
                max_memory_mb=self.max_memory_gb * 1024 if self.max_memory_gb else None,
            )
            bash_session.initialize()
            return bash_session

    async def ainit(self):
        # bash needs to be initialized first
        logger.debug('Initializing bash session')
        self.bash_session = self._create_bash_session()
        logger.debug('Bash session initialized')

        # Start browser initialization in the background
        self.browser_init_task = asyncio.create_task(self._init_browser_async())
        logger.debug('Browser initialization started in background')

        await wait_all(
            (self._init_plugin(plugin) for plugin in self.plugins_to_load),
            timeout=int(os.environ.get('INIT_PLUGIN_TIMEOUT', '120')),
        )
        logger.debug('All plugins initialized')

        # This is a temporary workaround
        # TODO: refactor AgentSkills to be part of JupyterPlugin
        # AFTER ServerRuntime is deprecated
        logger.debug('Initializing AgentSkills')
        if 'agent_skills' in self.plugins and 'jupyter' in self.plugins:
            obs = await self.run_ipython(
                IPythonRunCellAction(
                    code='from openhands.runtime.plugins.agent_skills.agentskills import *\n'
                )
            )
            logger.debug(f'AgentSkills initialized: {obs}')

        logger.debug('Initializing bash commands')
        await self._init_bash_commands()

        logger.debug('Runtime client initialized.')
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def _init_plugin(self, plugin: Plugin):
        assert self.bash_session is not None
        # VSCode plugin needs runtime_id for path-based routing when using Gateway API
        if isinstance(plugin, VSCodePlugin):
            runtime_id = os.environ.get('RUNTIME_ID')
            await plugin.initialize(self.username, runtime_id=runtime_id)
        else:
            await plugin.initialize(self.username)
        self.plugins[plugin.name] = plugin
        logger.debug(f'Initializing plugin: {plugin.name}')

        if isinstance(plugin, JupyterPlugin):
            # Escape backslashes in Windows path
            cwd = self.bash_session.cwd.replace('\\', '/')
            await self.run_ipython(
                IPythonRunCellAction(code=f'import os; os.chdir(r"{cwd}")')
            )

    async def _init_bash_commands(self):
        # You can add any bash commands you want to run on startup here
        # It is empty because: Git configuration is now handled by the runtime client after connection
        INIT_COMMANDS = []
        is_windows = sys.platform == 'win32'

        # Determine no-pager command
        if is_windows:
            no_pager_cmd = 'function git { git.exe --no-pager $args }'
        else:
            no_pager_cmd = 'alias git="git --no-pager"'

        INIT_COMMANDS.append(no_pager_cmd)

        # Hack: for some reason when you set the openhands user to anything but root, tmux changes out
        # of the mount directory on the first invocation.
        if self.user_id != 0:
            INIT_COMMANDS.append(f'cd {self._initial_cwd}')

        logger.info(f'Initializing by running {len(INIT_COMMANDS)} bash commands...')
        for command in INIT_COMMANDS:
            action = CmdRunAction(command=command)
            action.set_hard_timeout(300)
            logger.debug(f'Executing init command: {command}')
            obs = await self.run(action)
            assert isinstance(obs, CmdOutputObservation)
            logger.debug(
                f'Init command outputs (exit code: {obs.exit_code}): {obs.content}'
            )
            assert obs.exit_code == 0
        logger.debug('Bash init commands completed')

    async def run_action(self, action) -> Observation:
        async with self.lock:
            action_type = action.action
            observation = await getattr(self, action_type)(action)
            return observation

    async def run(
        self, action: CmdRunAction
    ) -> CmdOutputObservation | ErrorObservation:
        try:
            bash_session = self.bash_session
            static_session = None
            if action.is_static:
                bash_session = self._create_bash_session(action.cwd)
                static_session = bash_session
            assert bash_session is not None
            started_at = time.monotonic()
            primary_error: BaseException | None = None
            try:
                obs = await call_sync_from_async(bash_session.execute, action)
                duration_seconds = time.monotonic() - started_at
            except BaseException as error:
                primary_error = error
                raise
            finally:
                if static_session is not None:
                    try:
                        await call_sync_from_async(static_session.close)
                    except BaseException:
                        if primary_error is None:
                            raise
                        logger.exception(
                            'Error closing static shell session after command interruption'
                        )
            result_format = (
                getattr(action.tool_call_metadata, 'tool_result_format', None)
                if action.tool_call_metadata is not None
                else None
            )
            is_opencode = result_format == 'opencode'
            if is_opencode and isinstance(obs, CmdOutputObservation):
                raw_output = obs.content
                tail, truncated = tail_shell_output(raw_output)
                output_path = None
                if truncated:
                    with tempfile.NamedTemporaryFile(
                        mode='w',
                        encoding='utf-8',
                        prefix='opencode-tool-',
                        delete=False,
                    ) as saved_output:
                        saved_output.write(raw_output)
                        output_path = saved_output.name

                suffix = obs.metadata.suffix
                timed_out = '[The command timed out after ' in suffix
                aborted = 'CTRL+C was sent' in suffix
                metadata_messages = []
                if (
                    '[The command has no new output after ' in suffix
                    or ' is NOT executed. The previous command is still running'
                    in suffix
                ):
                    metadata_messages.append(
                        suffix.strip().removeprefix('[').removesuffix(']')
                    )
                timeout_ms = (
                    round(float(action.timeout) * 1000)
                    if timed_out and action.timeout is not None
                    else None
                )
                obs.content = format_shell_output(
                    tail,
                    output_path=output_path,
                    timeout_ms=timeout_ms,
                    aborted=aborted,
                    metadata_messages=metadata_messages,
                )
            elif result_format == 'codex' and isinstance(obs, CmdOutputObservation):
                # Prefix/suffix strings are OpenHands operational annotations,
                # not bytes captured from the command. Codex's shell body wraps
                # only the captured output (plus its own timeout prefix).
                raw_output = obs.content
                suffix = obs.metadata.suffix
                timed_out = '[The command timed out after ' in suffix

                timed_out_ms = None
                exit_code = obs.exit_code
                if timed_out:
                    # Rust's Duration::as_millis reports the measured elapsed
                    # duration and floors sub-millisecond precision.
                    timed_out_ms = int(max(duration_seconds, 0.0) * 1000)
                    exit_code = 124
                    if not action.is_static and hasattr(
                        bash_session, 'recover_after_timeout'
                    ):
                        # A hard timeout is terminal for Codex's shell call.
                        # Reclaim the persistent pane before returning so the
                        # next call cannot inherit output or a live process.
                        await call_sync_from_async(bash_session.recover_after_timeout)

                obs.content = format_codex_shell_output(
                    raw_output,
                    exit_code=exit_code,
                    duration_seconds=duration_seconds,
                    model_name=_tool_model_name(action),
                    timed_out_ms=timed_out_ms,
                )
            return obs
        except Exception as e:
            logger.exception(f'Error running command: {e}')
            return ErrorObservation(str(e))

    async def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        assert self.bash_session is not None
        if 'jupyter' in self.plugins:
            _jupyter_plugin: JupyterPlugin = self.plugins['jupyter']  # type: ignore
            # This is used to make AgentSkills in Jupyter aware of the
            # current working directory in Bash
            jupyter_cwd = getattr(self, '_jupyter_cwd', None)
            if self.bash_session.cwd != jupyter_cwd:
                logger.debug(
                    f'{self.bash_session.cwd} != {jupyter_cwd} -> reset Jupyter PWD'
                )
                # escape windows paths
                cwd = self.bash_session.cwd.replace('\\', '/')
                reset_jupyter_cwd_code = f'import os; os.chdir("{cwd}")'
                _aux_action = IPythonRunCellAction(code=reset_jupyter_cwd_code)
                _reset_obs: IPythonRunCellObservation = await _jupyter_plugin.run(
                    _aux_action
                )
                logger.debug(
                    f'Changed working directory in IPython to: {self.bash_session.cwd}. Output: {_reset_obs}'
                )
                self._jupyter_cwd = self.bash_session.cwd

            obs: IPythonRunCellObservation = await _jupyter_plugin.run(action)
            obs.content = obs.content.rstrip()

            if action.include_extra:
                obs.content += (
                    f'\n[Jupyter current working directory: {self.bash_session.cwd}]'
                )
                obs.content += f'\n[Jupyter Python interpreter: {_jupyter_plugin.python_interpreter_path}]'
            return obs
        else:
            raise RuntimeError(
                'JupyterRequirement not found. Unable to run IPython action.'
            )

    def _resolve_path(self, path: str, working_dir: str) -> str:
        filepath = Path(path)
        if not filepath.is_absolute():
            return str(Path(working_dir) / filepath)
        return str(filepath)

    async def read(self, action: FileReadAction) -> Observation:
        assert self.bash_session is not None

        # Cannot read binary files
        if is_binary(action.path):
            return ErrorObservation('ERROR_BINARY_FILE')

        if action.impl_source == FileReadSource.OH_ACI:
            result_str, _ = _execute_file_editor(
                self.file_editor,
                command='view',
                path=action.path,
                view_range=action.view_range,
            )

            return FileReadObservation(
                content=result_str,
                path=action.path,
                impl_source=FileReadSource.OH_ACI,
            )

        # NOTE: the client code is running inside the sandbox,
        # so there's no need to check permission
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)
        try:
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                with open(filepath, 'rb') as file:
                    image_data = file.read()
                    encoded_image = base64.b64encode(image_data).decode('utf-8')
                    mime_type, _ = mimetypes.guess_type(filepath)
                    if mime_type is None:
                        mime_type = 'image/png'  # default to PNG if mime type cannot be determined
                    encoded_image = f'data:{mime_type};base64,{encoded_image}'

                return FileReadObservation(path=filepath, content=encoded_image)
            elif filepath.lower().endswith('.pdf'):
                with open(filepath, 'rb') as file:
                    pdf_data = file.read()
                    encoded_pdf = base64.b64encode(pdf_data).decode('utf-8')
                    encoded_pdf = f'data:application/pdf;base64,{encoded_pdf}'
                return FileReadObservation(path=filepath, content=encoded_pdf)
            elif filepath.lower().endswith(('.mp4', '.webm', '.ogg')):
                with open(filepath, 'rb') as file:
                    video_data = file.read()
                    encoded_video = base64.b64encode(video_data).decode('utf-8')
                    mime_type, _ = mimetypes.guess_type(filepath)
                    if mime_type is None:
                        mime_type = 'video/mp4'  # default to MP4 if MIME type cannot be determined
                    encoded_video = f'data:{mime_type};base64,{encoded_video}'

                return FileReadObservation(path=filepath, content=encoded_video)

            with open(filepath, 'r', encoding='utf-8') as file:
                lines = read_lines(file.readlines(), action.start, action.end)
        except FileNotFoundError:
            return ErrorObservation(
                f'File not found: {filepath}. Your current working directory is {working_dir}.'
            )
        except UnicodeDecodeError:
            return ErrorObservation(f'File could not be decoded as utf-8: {filepath}.')
        except IsADirectoryError:
            return ErrorObservation(
                f'Path is a directory: {filepath}. You can only read files'
            )

        code_view = ''.join(lines)
        return FileReadObservation(path=filepath, content=code_view)

    async def write(self, action: FileWriteAction) -> Observation:
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)

        insert = action.content.split('\n')
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        file_exists = os.path.exists(filepath)
        if file_exists:
            file_stat = os.stat(filepath)
        else:
            file_stat = None

        mode = 'w' if not file_exists else 'r+'
        try:
            with open(filepath, mode, encoding='utf-8') as file:
                if mode != 'w':
                    all_lines = file.readlines()
                    new_file = insert_lines(insert, all_lines, action.start, action.end)
                else:
                    new_file = [i + '\n' for i in insert]

                file.seek(0)
                file.writelines(new_file)
                file.truncate()

        except FileNotFoundError:
            return ErrorObservation(f'File not found: {filepath}')
        except IsADirectoryError:
            return ErrorObservation(
                f'Path is a directory: {filepath}. You can only write to files'
            )
        except UnicodeDecodeError:
            return ErrorObservation(f'File could not be decoded as utf-8: {filepath}')

        # Attempt to handle file permissions
        try:
            if file_exists:
                assert file_stat is not None
                # restore the original file permissions if the file already exists
                os.chmod(filepath, file_stat.st_mode)
                os.chown(filepath, file_stat.st_uid, file_stat.st_gid)
            else:
                # set the new file permissions if the file is new
                os.chmod(filepath, 0o664)
                os.chown(filepath, self.user_id, self.user_id)
        except PermissionError as e:
            return ErrorObservation(
                f'File {filepath} written, but failed to change ownership and permissions: {e}'
            )
        return FileWriteObservation(content='', path=filepath)

    async def edit(self, action: FileEditAction) -> Observation:
        assert action.impl_source == FileEditSource.OH_ACI
        is_opencode = (
            action.tool_call_metadata is not None
            and getattr(action.tool_call_metadata, 'tool_result_format', None)
            == 'opencode'
        )
        if is_opencode:
            from openhands.agenthub.opencode_agent.opencode_impl import (
                replace_with_fuzzy_matching,
            )

            assert self.bash_session is not None
            filepath = os.path.abspath(
                self._resolve_path(action.path, self.bash_session.cwd)
            )
            old_string = action.old_str or ''
            new_string = action.new_str or ''
            if not action.path:
                return ErrorObservation('filePath is required')
            if old_string == new_string:
                return ErrorObservation(
                    'No changes to apply: oldString and newString are identical.'
                )
            if old_string == '' and os.path.exists(filepath):
                return ErrorObservation(
                    'oldString cannot be empty when editing an existing file. '
                    'Provide the exact text to replace, or use write for an '
                    'intentional full-file replacement.'
                )
            if old_string and not os.path.exists(filepath):
                return ErrorObservation(f'File {filepath} not found')
            if os.path.isdir(filepath):
                return ErrorObservation(f'Path is a directory, not a file: {filepath}')

            if old_string == '':
                old_content = ''
                new_content = new_string
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
            else:
                try:
                    with open(
                        filepath,
                        'r',
                        encoding='utf-8',
                        errors='replace',
                        newline='',
                    ) as source:
                        old_content = source.read()
                except OSError as exc:
                    return ErrorObservation(str(exc))
                ending = '\r\n' if '\r\n' in old_content else '\n'
                normalized_old = old_string.replace('\r\n', '\n').replace('\n', ending)
                normalized_new = new_string.replace('\r\n', '\n').replace('\n', ending)
                try:
                    new_content = replace_with_fuzzy_matching(
                        old_content,
                        normalized_old,
                        normalized_new,
                        replace_all=action.replace_all,
                    )
                except ValueError as exc:
                    return ErrorObservation(str(exc))

            try:
                with open(filepath, 'w', encoding='utf-8', newline='') as target:
                    target.write(new_content)
            except OSError as exc:
                return ErrorObservation(str(exc))
            return FileEditObservation(
                content='Edit applied successfully.',
                path=filepath,
                old_content=old_content,
                new_content=new_content,
                impl_source=FileEditSource.OH_ACI,
                diff=get_diff(
                    old_contents=old_content,
                    new_contents=new_content,
                    filepath=filepath,
                ),
            )

        result_str, (old_content, new_content) = _execute_file_editor(
            self.file_editor,
            command=action.command,
            path=action.path,
            file_text=action.file_text,
            old_str=action.old_str,
            new_str=action.new_str,
            insert_line=action.insert_line,
            enable_linting=False,
        )

        return FileEditObservation(
            content=result_str,
            path=action.path,
            old_content=action.old_str,
            new_content=action.new_str,
            impl_source=FileEditSource.OH_ACI,
            diff=get_diff(
                old_contents=old_content or '',
                new_contents=new_content or '',
                filepath=action.path,
            ),
        )

    # =========================================================================
    # OpenCode-style action handlers
    # =========================================================================

    async def opencode_read(self, action: OpenCodeReadAction) -> Observation:
        """Execute a read with the current OpenCode model-visible body."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        filepath = os.path.abspath(self._resolve_path(action.path, working_dir))

        BINARY_EXTENSIONS = {
            '.zip',
            '.tar',
            '.gz',
            '.exe',
            '.dll',
            '.so',
            '.class',
            '.jar',
            '.war',
            '.7z',
            '.doc',
            '.docx',
            '.xls',
            '.xlsx',
            '.ppt',
            '.pptx',
            '.odt',
            '.ods',
            '.odp',
            '.bin',
            '.dat',
            '.obj',
            '.o',
            '.a',
            '.lib',
            '.wasm',
            '.pyc',
            '.pyo',
        }

        if not os.path.exists(filepath):
            directory = os.path.dirname(filepath) or '.'
            basename = os.path.basename(filepath)
            if os.path.isdir(directory):
                try:
                    suggestions = [
                        os.path.join(directory, entry)
                        for entry in os.listdir(directory)
                        if basename.lower() in entry.lower()
                        or entry.lower() in basename.lower()
                    ][:3]
                    if suggestions:
                        return ErrorObservation(
                            f'File not found: {filepath}\n\n'
                            'Did you mean one of these?\n' + '\n'.join(suggestions)
                        )
                except OSError:
                    pass
            return ErrorObservation(f'File not found: {filepath}')

        if os.path.isdir(filepath):
            try:
                entries = []
                with os.scandir(filepath) as iterator:
                    for entry in iterator:
                        suffix = '/' if entry.is_dir(follow_symlinks=True) else ''
                        entries.append(entry.name + suffix)
                entries.sort(key=lambda item: (item.casefold(), item))
                output = format_read_directory(
                    filepath,
                    entries,
                    offset=action.offset,
                    limit=action.limit,
                )
            except (OSError, ValueError) as exc:
                return ErrorObservation(str(exc))
            return CmdOutputObservation(
                content=output,
                command=f'opencode_read {filepath}',
                command_id=-1,
                exit_code=0,
                max_content_size=None,
            )

        ext = os.path.splitext(filepath)[1].lower()
        if ext in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
            return CmdOutputObservation(
                content='Image read successfully',
                command=f'opencode_read {filepath}',
                command_id=-1,
                exit_code=0,
                max_content_size=None,
            )
        if ext == '.pdf':
            return CmdOutputObservation(
                content='PDF read successfully',
                command=f'opencode_read {filepath}',
                command_id=-1,
                exit_code=0,
                max_content_size=None,
            )
        if ext in BINARY_EXTENSIONS:
            return ErrorObservation(f'Cannot read binary file: {filepath}')

        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(4096)
                if chunk.startswith(b'%PDF-'):
                    return CmdOutputObservation(
                        content='PDF read successfully',
                        command=f'opencode_read {filepath}',
                        command_id=-1,
                        exit_code=0,
                        max_content_size=None,
                    )
                is_image = (
                    chunk.startswith(b'\x89PNG\r\n\x1a\n')
                    or chunk.startswith(b'\xff\xd8\xff')
                    or chunk.startswith((b'GIF87a', b'GIF89a'))
                    or (
                        len(chunk) >= 12
                        and chunk.startswith(b'RIFF')
                        and chunk[8:12] == b'WEBP'
                    )
                )
                if is_image:
                    return CmdOutputObservation(
                        content='Image read successfully',
                        command=f'opencode_read {filepath}',
                        command_id=-1,
                        exit_code=0,
                        max_content_size=None,
                    )
                if b'\x00' in chunk:
                    return ErrorObservation(f'Cannot read binary file: {filepath}')
                if chunk:
                    non_printable = sum(
                        1 for b in chunk if b < 9 or (b > 13 and b < 32)
                    )
                    if non_printable / len(chunk) > 0.3:
                        return ErrorObservation(f'Cannot read binary file: {filepath}')
        except OSError as exc:
            return ErrorObservation(str(exc))

        try:
            with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = split_file_lines(f.read())
            output = format_read_file(
                filepath,
                lines,
                offset=action.offset,
                limit=action.limit,
            )
        except (OSError, ValueError) as exc:
            return ErrorObservation(str(exc))

        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f'opencode_read {filepath}',
            exit_code=0,
            max_content_size=None,
        )

    async def opencode_write(self, action: OpenCodeWriteAction) -> Observation:
        """Write a file and return OpenCode's text body."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)

        # Create directory if needed
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                return ErrorObservation(f'Failed to create directory: {e}')

        # Preserve an existing UTF-8 BOM, matching OpenCode's Bom.join helper.
        preserve_bom = False
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'rb') as existing:
                    preserve_bom = existing.read(3) == b'\xef\xbb\xbf'
            except OSError:
                pass
        output_content = action.content
        encoding = 'utf-8'
        if preserve_bom:
            encoding = 'utf-8-sig'
            output_content = output_content.removeprefix('\ufeff')

        # Write file
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(output_content)
        except Exception as e:
            return ErrorObservation(f'Failed to write file: {e}')

        # This runtime does not expose OpenCode's LSP service. Raw output from
        # unrelated command-line linters is not an LSP diagnostic and must not be
        # placed in the OpenCode diagnostics block.
        return FileWriteObservation(content='Wrote file successfully.', path=filepath)

    async def glob(self, action: GlobAction) -> Observation:
        """Search for files and render OpenCode's glob body."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        for value, name in (
            (action.pattern, 'pattern'),
            (action.path, 'path'),
        ):
            error = _nul_argument_error(value, name)
            if error is not None:
                return ErrorObservation(error)
        search_path = os.path.abspath(self._resolve_path(action.path, working_dir))

        if not os.path.exists(search_path):
            return ErrorObservation(f'Path does not exist: {search_path}')
        if os.path.isfile(search_path):
            return ErrorObservation(f'glob path must be a directory: {search_path}')

        files: list[str] = []
        try:
            result = subprocess.run(
                [
                    'rg',
                    '--no-config',
                    '--files',
                    f'--glob={action.pattern}',
                    '--glob=!**/.git/**',
                    '.',
                ],
                capture_output=True,
                timeout=30,
                cwd=search_path,
            )
            if result.returncode == 0:
                files = []
                for raw_item in result.stdout.splitlines():
                    item = _decode_process_output(raw_item)
                    if not item or '\x00' in item:
                        continue
                    files.append(os.path.abspath(os.path.join(search_path, item)))
            elif result.returncode == 1:
                files = []
            else:
                return ErrorObservation(_decode_process_output(result.stderr).strip())
        except FileNotFoundError:
            try:
                files = _fallback_opencode_glob(
                    Path(search_path),
                    action.pattern,
                    timeout_seconds=_NATIVE_SEARCH_TIMEOUT_SECONDS,
                )
            except _CodexGrepPreprocessingError as error:
                return ErrorObservation(f'glob fallback failed: {error}')
            except _NativeSearchLimitError as error:
                return ErrorObservation(f'glob fallback failed: {error}')
            except TimeoutError:
                return ErrorObservation('glob fallback timed out after 30 seconds')
            except OSError as error:
                return ErrorObservation(
                    f'glob fallback failed: {format_codex_os_error(error)}'
                )
        except subprocess.TimeoutExpired:
            return ErrorObservation('glob search timed out after 30 seconds')

        return CmdOutputObservation(
            content=format_glob(files),
            command_id=-1,
            command=f'glob {action.pattern} {action.path}',
            exit_code=0,
            max_content_size=None,
        )

    async def grep(self, action: GrepAction) -> Observation:
        """Search file contents and render OpenCode's grouped grep body."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        for value, name in (
            (action.pattern, 'pattern'),
            (action.include, 'include'),
            (action.path, 'path'),
        ):
            error = _nul_argument_error(value, name)
            if error is not None:
                return ErrorObservation(error)
        search_path = os.path.abspath(self._resolve_path(action.path, working_dir))

        if not action.pattern:
            return ErrorObservation('pattern is required')
        if not os.path.exists(search_path):
            return ErrorObservation(f'Path does not exist: {search_path}')

        search_is_dir = os.path.isdir(search_path)
        cwd = search_path if search_is_dir else os.path.dirname(search_path)
        target = '.' if search_is_dir else os.path.basename(search_path)
        matches: list[GrepMatch] = []
        try:
            cmd = ['rg', '--no-config', '--json', '--hidden', '--no-messages']
            if action.include:
                cmd.append(f'--glob={action.include}')
            cmd.extend(['--glob=!**/.git/**', '--', action.pattern, target])
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
                cwd=cwd,
            )
            if result.returncode not in (0, 1):
                return ErrorObservation(_decode_process_output(result.stderr).strip())
            for raw_line in result.stdout.splitlines():
                try:
                    line = (
                        raw_line.decode('utf-8')
                        if isinstance(raw_line, bytes)
                        else raw_line
                    )
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get('type') != 'match':
                    continue
                data = record.get('data')
                if not isinstance(data, dict):
                    continue
                relative_path = _decode_rg_json_value(data.get('path'))
                text = _decode_rg_json_value(data.get('lines'))
                if (
                    relative_path is None
                    or text is None
                    or '\x00' in relative_path
                    or '\x00' in text
                    or os.path.isabs(relative_path)
                    or '..' in Path(relative_path).parts
                ):
                    continue
                try:
                    line_number = int(data['line_number'])
                except (KeyError, TypeError, ValueError):
                    continue
                display_path = os.path.abspath(os.path.join(cwd, relative_path))
                matches.append(
                    GrepMatch(
                        path=display_path,
                        line=line_number,
                        text=text,
                    )
                )
                if len(matches) == 100:
                    break
        except FileNotFoundError:
            try:
                matches = _fallback_opencode_grep(
                    Path(search_path),
                    action.pattern,
                    action.include,
                    timeout_seconds=_NATIVE_SEARCH_TIMEOUT_SECONDS,
                )
            except _CODEX_GREP_REGEX_ERROR as error:
                return ErrorObservation(
                    f'grep fallback failed: invalid regular expression: {error}'
                )
            except _CodexGrepPreprocessingError as error:
                return ErrorObservation(f'grep fallback failed: {error}')
            except _NativeSearchLimitError as error:
                return ErrorObservation(f'grep fallback failed: {error}')
            except _CodexGrepRegexUnavailableError as error:
                return ErrorObservation(f'grep fallback unavailable: {error}')
            except TimeoutError:
                return ErrorObservation('grep fallback timed out after 30 seconds')
            except OSError as error:
                return ErrorObservation(
                    f'grep fallback failed: {format_codex_os_error(error)}'
                )
        except subprocess.TimeoutExpired:
            return ErrorObservation('grep search timed out after 30 seconds')

        return CmdOutputObservation(
            content=format_grep(matches),
            command_id=-1,
            command=f'grep {action.pattern} {action.path}',
            exit_code=0,
            max_content_size=None,
        )

    async def list_dir(self, action: ListDirAction) -> Observation:
        """Legacy local tool using OpenCode's read-directory body."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        list_path = os.path.abspath(self._resolve_path(action.path, working_dir))
        if not os.path.exists(list_path):
            return ErrorObservation(f'File not found: {list_path}')
        if not os.path.isdir(list_path):
            return ErrorObservation(f'Path is not a directory: {list_path}')
        try:
            entries = []
            with os.scandir(list_path) as iterator:
                for entry in iterator:
                    suffix = '/' if entry.is_dir(follow_symlinks=True) else ''
                    entries.append(entry.name + suffix)
            entries.sort(key=lambda item: (item.casefold(), item))
        except OSError as exc:
            return ErrorObservation(str(exc))

        return CmdOutputObservation(
            content=format_read_directory(list_path, entries),
            command_id=-1,
            command=f'list_dir {action.path}',
            exit_code=0,
            max_content_size=None,
        )

    async def question(self, action: QuestionAction) -> Observation:
        """Handle a question action. Returns an observation with the questions.

        Note: In a full implementation, this would interact with the user.
        In sandbox/evaluation mode, we return the questions as-is since
        the controller handles user interaction.
        """
        return QuestionObservation(
            content=json.dumps(action.questions, indent=2),
            questions=action.questions,
        )

    async def apply_patch(self, action: ApplyPatchAction) -> Observation:
        """Apply a unified diff patch to files."""
        assert self.bash_session is not None
        try:
            # Write the patch to a temporary file and apply with git apply
            import tempfile

            patch_text = action.patchText
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.patch', delete=False
            ) as f:
                f.write(patch_text)
                patch_file = f.name

            try:
                result = subprocess.run(
                    ['git', 'apply', '--verbose', patch_file],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.bash_session.cwd,
                )
                if result.returncode == 0:
                    output = result.stdout.strip() or 'Patch applied successfully.'
                    # Try to extract changed files from verbose output
                    files_changed = [
                        line.split(':')[0].strip()
                        for line in result.stderr.strip().split('\n')
                        if line.strip()
                    ]
                    return ApplyPatchObservation(
                        content=output,
                        files_changed=files_changed,
                        success=True,
                    )
                else:
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    return ApplyPatchObservation(
                        content=f'Failed to apply patch: {error_msg}',
                        success=False,
                    )
            finally:
                os.unlink(patch_file)
        except Exception as e:
            logger.exception(f'Error applying patch: {e}')
            return ErrorObservation(f'Failed to apply patch: {str(e)}')

    async def todo_read(self, action: TodoReadAction) -> Observation:
        """Read the retained local todo list using todowrite's JSON body."""
        output = ActionExecutor._format_opencode_generic_output(
            json.dumps(self._todos, indent=2, ensure_ascii=False)
        )
        return TodoReadObservation(
            content=output,
            todos=list(self._todos),
        )

    async def todo_write(self, action: TodoWriteAction) -> Observation:
        """Replace the todo list and echo it like OpenCode's todowrite."""
        try:
            incoming_todos = action.todos
            if not isinstance(incoming_todos, list):
                return ErrorObservation('todos must be a list of todo objects')
            if not all(isinstance(todo, dict) for todo in incoming_todos):
                return ErrorObservation('todos must be a list of todo objects')
            self._todos = [dict(todo) for todo in incoming_todos]

            output = ActionExecutor._format_opencode_generic_output(
                json.dumps(self._todos, indent=2, ensure_ascii=False)
            )
            return TodoWriteObservation(
                content=output,
                todos=list(self._todos),
                success=True,
            )
        except Exception as e:
            logger.exception(f'Error updating todos: {e}')
            return ErrorObservation(f'Failed to update todos: {str(e)}')

    @staticmethod
    def _format_opencode_generic_output(output: str) -> str:
        preview = truncate_tool_output_head(output)
        if not preview.truncated:
            return output
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            prefix='opencode-tool-',
            delete=False,
        ) as saved_output:
            saved_output.write(output)
            output_path = saved_output.name
        return format_truncated_tool_output(preview, output_path=output_path)

    # =========================================================================
    # Codex-style action handlers
    # =========================================================================

    async def codex_read_file(self, action: CodexReadFileAction) -> Observation:
        """Execute the legacy Codex ``read_file`` result contract."""
        if action.offset <= 0:
            return ErrorObservation('offset must be a 1-indexed line number')
        if action.limit <= 0:
            return ErrorObservation('limit must be greater than zero')

        path = Path(action.file_path)
        if not path.is_absolute():
            return ErrorObservation('file_path must be an absolute path')

        try:
            lines = split_codex_file_lines(path.read_bytes())
        except OSError as error:
            return ErrorObservation(
                f'failed to read file: {format_codex_os_error(error)}'
            )

        try:
            if action.mode == 'indentation':
                indentation = action.indentation or {}
                output = format_codex_read_indentation(
                    lines,
                    offset=action.offset,
                    limit=action.limit,
                    anchor_line=indentation.get('anchor_line'),
                    max_levels=indentation.get('max_levels', 0),
                    include_siblings=indentation.get('include_siblings', False),
                    include_header=indentation.get('include_header', True),
                    max_lines=indentation.get('max_lines'),
                )
            else:
                output = format_codex_read_slice(
                    lines,
                    offset=action.offset,
                    limit=action.limit,
                )
        except ValueError as error:
            return ErrorObservation(str(error))

        output = truncate_codex_function_output(
            output,
            model_name=_tool_model_name(action),
        )
        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f'codex_read_file {path}',
            exit_code=0,
            max_content_size=None,
        )

    def _codex_read_file_indentation(
        self,
        lines: list[str],
        total_lines: int,
        filepath: str,
        action: CodexReadFileAction,
    ) -> Observation:
        """Compatibility wrapper retained for callers of the legacy helper."""
        del total_lines
        indentation = action.indentation or {}
        try:
            output = format_codex_read_indentation(
                lines,
                offset=action.offset,
                limit=action.limit,
                anchor_line=indentation.get('anchor_line'),
                max_levels=indentation.get('max_levels', 0),
                include_siblings=indentation.get('include_siblings', False),
                include_header=indentation.get('include_header', True),
                max_lines=indentation.get('max_lines'),
            )
        except ValueError as error:
            return ErrorObservation(str(error))
        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f'codex_read_file {filepath} (indentation mode)',
            exit_code=0,
            max_content_size=None,
        )

    async def codex_list_dir(self, action: CodexListDirAction) -> Observation:
        """Execute the legacy Codex ``list_dir`` result contract."""
        if action.offset <= 0:
            return ErrorObservation('offset must be a 1-indexed entry number')
        if action.limit <= 0:
            return ErrorObservation('limit must be greater than zero')
        if action.depth <= 0:
            return ErrorObservation('depth must be greater than zero')

        path = Path(action.dir_path)
        if not path.is_absolute():
            return ErrorObservation('dir_path must be an absolute path')

        # (truncated relative sort key, component display, depth, kind)
        entries: list[tuple[str, str, int, str]] = []
        queue: list[tuple[Path, Path, int]] = [(path, Path(), action.depth)]
        queue_index = 0

        while queue_index < len(queue):
            current_dir, prefix, remaining_depth = queue[queue_index]
            queue_index += 1
            try:
                with os.scandir(current_dir) as iterator:
                    current_entries = list(iterator)
            except OSError as error:
                return ErrorObservation(
                    f'failed to read directory: {format_codex_os_error(error)}'
                )

            collected: list[tuple[Path, Path, str, str, int, str]] = []
            for entry in current_entries:
                try:
                    if entry.is_symlink():
                        kind = 'symlink'
                    elif entry.is_dir(follow_symlinks=False):
                        kind = 'directory'
                    elif entry.is_file(follow_symlinks=False):
                        kind = 'file'
                    else:
                        kind = 'other'
                except OSError as error:
                    return ErrorObservation(
                        f'failed to inspect entry: {format_codex_os_error(error)}'
                    )

                relative_path = prefix / entry.name
                display_name = take_codex_utf8_prefix(
                    os.fsencode(entry.name).decode('utf-8', errors='replace'),
                    500,
                )
                normalized_path = (
                    os.fsencode(relative_path)
                    .decode('utf-8', errors='replace')
                    .replace('\\', '/')
                )
                sort_key = take_codex_utf8_prefix(normalized_path, 500)
                display_depth = len(prefix.parts)
                collected.append(
                    (
                        Path(entry.path),
                        relative_path,
                        sort_key,
                        display_name,
                        display_depth,
                        kind,
                    )
                )

            collected.sort(key=lambda item: item[2])
            for (
                entry_path,
                relative_path,
                sort_key,
                display_name,
                display_depth,
                kind,
            ) in collected:
                if kind == 'directory' and remaining_depth > 1:
                    queue.append((entry_path, relative_path, remaining_depth - 1))
                entries.append((sort_key, display_name, display_depth, kind))

        entries.sort(key=lambda item: item[0])
        absolute_display = os.fsencode(path).decode('utf-8', errors='replace')
        output_lines = [f'Absolute path: {absolute_display}']
        if entries:
            start_index = action.offset - 1
            if start_index >= len(entries):
                return ErrorObservation('offset exceeds directory entry count')
            capped_limit = min(action.limit, len(entries) - start_index)
            end_index = start_index + capped_limit
            suffixes = {
                'directory': '/',
                'symlink': '@',
                'other': '?',
                'file': '',
            }
            for _, display_name, display_depth, kind in entries[start_index:end_index]:
                output_lines.append(
                    f'{"  " * display_depth}{display_name}{suffixes[kind]}'
                )
            if end_index < len(entries):
                output_lines.append(f'More than {capped_limit} entries found')

        output = truncate_codex_function_output(
            '\n'.join(output_lines),
            model_name=_tool_model_name(action),
        )
        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f'codex_list_dir {path}',
            exit_code=0,
            max_content_size=None,
        )

    async def codex_grep_files(self, action: CodexGrepFilesAction) -> Observation:
        """Execute the legacy Codex ``grep_files`` result contract."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        pattern = action.pattern.strip()
        if not pattern:
            return ErrorObservation('pattern must not be empty')
        error = _nul_argument_error(pattern, 'pattern')
        if error is not None:
            return ErrorObservation(error)
        if action.limit <= 0:
            return ErrorObservation('limit must be greater than zero')

        limit = min(action.limit, 2000)
        search_path = Path(
            self._resolve_path(action.path, working_dir) if action.path else working_dir
        )
        include = (action.include or '').strip() or None
        if include is not None:
            error = _nul_argument_error(include, 'include')
            if error is not None:
                return ErrorObservation(error)
        error = _nul_argument_error(search_path, 'path')
        if error is not None:
            return ErrorObservation(error)
        try:
            search_path.stat()
        except OSError as error:
            search_path_display = os.fsencode(search_path).decode(
                'utf-8', errors='replace'
            )
            return ErrorObservation(
                f'unable to access `{search_path_display}`: '
                f'{format_codex_os_error(error)}'
            )

        command = [
            'rg',
            '--files-with-matches',
            '--sortr=modified',
            '--regexp',
            pattern,
            '--no-messages',
        ]
        if include:
            command.extend(('--glob', include))
        command.extend(('--glob', '!**/.git/**'))
        command.extend(('--', str(search_path)))

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=30,
                cwd=working_dir,
            )
        except subprocess.TimeoutExpired:
            return ErrorObservation('rg timed out after 30 seconds')
        except FileNotFoundError:
            try:
                files = _fallback_codex_grep_files(
                    search_path,
                    pattern,
                    include,
                    limit,
                )
            except _CODEX_GREP_REGEX_ERROR as error:
                return ErrorObservation(
                    f'grep_files fallback failed: invalid regular expression: {error}'
                )
            except _CodexGrepPreprocessingError as error:
                return ErrorObservation(f'grep_files fallback failed: {error}')
            except _NativeSearchLimitError as error:
                return ErrorObservation(f'grep_files fallback failed: {error}')
            except _CodexGrepRegexUnavailableError as error:
                return ErrorObservation(f'grep_files fallback unavailable: {error}')
            except TimeoutError:
                return ErrorObservation(
                    'grep_files fallback timed out after 30 seconds'
                )
            except OSError as error:
                return ErrorObservation(
                    f'grep_files fallback failed: {format_codex_os_error(error)}'
                )
        except OSError as error:
            return ErrorObservation(
                f'failed to launch rg: {format_codex_os_error(error)}. '
                'Ensure ripgrep is installed and on PATH.'
            )
        else:
            if result.returncode == 0:
                files = []
                for raw_line in result.stdout.split(b'\n'):
                    if not raw_line:
                        continue
                    try:
                        file_path = raw_line.decode('utf-8')
                    except UnicodeDecodeError:
                        continue
                    if file_path:
                        files.append(file_path)
                        if len(files) == limit:
                            break
            elif result.returncode == 1:
                files = []
            else:
                stderr = result.stderr.decode('utf-8', errors='replace')
                return ErrorObservation(f'rg failed: {stderr}')

        output = '\n'.join(files) if files else 'No matches found.'
        output = truncate_codex_function_output(
            output,
            model_name=_tool_model_name(action),
        )
        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f'codex_grep_files {pattern}',
            exit_code=0,
            max_content_size=None,
        )

    async def codex_apply_patch(self, action: CodexApplyPatchAction) -> Observation:
        """Apply a Codex freeform-format patch to files.

        The Codex patch format uses:
        *** Begin Patch / *** End Patch delimiters
        *** Add File: <path>    - create new files
        *** Delete File: <path> - delete files
        *** Update File: <path> - modify existing files
        *** Move to: <path>     - rename/move files (after Update File)
        @@ <context>            - context anchors within Update File chunks
        +/- lines for additions/removals
        space-prefixed context lines (both old and new)
        *** End of File         - mark end-of-file position
        """
        assert self.bash_session is not None
        started_at = time.monotonic()
        patch_text = action.patch

        if not patch_text.strip():
            return ErrorObservation('Empty patch provided.')

        try:
            hunks = self._codex_parse_patch(patch_text)
        except ValueError as e:
            return CodexApplyPatchObservation(
                content=f'Patch parse error: {e}',
                files_changed=[],
                success=False,
            )

        if not hunks:
            return CodexApplyPatchObservation(
                content='Patch parsed but contained no file operations.',
                files_changed=[],
                success=False,
            )

        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []
        errors: list[str] = []

        for hunk in hunks:
            hunk_type = hunk['type']
            path = hunk['path']
            full_path = os.path.join(self.bash_session.cwd, path)

            try:
                if hunk_type == 'add':
                    parent = os.path.dirname(full_path)
                    if parent and not os.path.exists(parent):
                        os.makedirs(parent, exist_ok=True)
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(hunk['contents'])
                    added.append(path)

                elif hunk_type == 'delete':
                    if not os.path.exists(full_path):
                        errors.append(f"Delete failed: file not found '{path}'")
                        continue
                    os.unlink(full_path)
                    deleted.append(path)

                elif hunk_type == 'update':
                    if not os.path.exists(full_path):
                        errors.append(f"Update failed: file not found '{path}'")
                        continue
                    if not os.path.isfile(full_path):
                        errors.append(f"Update failed: '{path}' is not a regular file")
                        continue

                    err = self._codex_apply_update_hunk(full_path, hunk['chunks'])
                    if err:
                        errors.append(f"Update failed for '{path}': {err}")
                        continue

                    move_path = hunk.get('move_path')
                    if move_path:
                        dest = os.path.join(self.bash_session.cwd, move_path)
                        parent = os.path.dirname(dest)
                        if parent and not os.path.exists(parent):
                            os.makedirs(parent, exist_ok=True)
                        os.rename(full_path, dest)
                        modified.append(move_path)
                    else:
                        modified.append(path)

            except Exception as e:
                errors.append(f"Error processing '{path}': {e}")

        files_changed = added + modified + deleted

        if errors:
            summary_parts = []
            if files_changed:
                summary_parts.append(
                    f'Partial success ({len(files_changed)} file(s) changed):'
                )
                for p in added:
                    summary_parts.append(f'  A {p}')
                for p in modified:
                    summary_parts.append(f'  M {p}')
                for p in deleted:
                    summary_parts.append(f'  D {p}')
            summary_parts.append(f'Errors ({len(errors)}):')
            for err in errors:
                summary_parts.append(f'  - {err}')
            return CodexApplyPatchObservation(
                content='\n'.join(summary_parts),
                files_changed=files_changed,
                success=False,
            )

        if not files_changed:
            return CodexApplyPatchObservation(
                content='Patch parsed successfully but no files were changed.',
                files_changed=[],
                success=False,
            )

        summary = format_codex_apply_patch_success(
            added=added,
            modified=modified,
            deleted=deleted,
        )
        output = format_codex_shell_output(
            summary,
            exit_code=0,
            duration_seconds=time.monotonic() - started_at,
            model_name=_tool_model_name(action),
        )
        return CodexApplyPatchObservation(
            content=output,
            files_changed=files_changed,
            success=True,
        )

    def _codex_parse_patch(self, patch_text: str) -> list[dict]:
        """Parse Codex freeform patch format into a list of hunk dicts.

        Returns a list of dicts, each with:
          {'type': 'add', 'path': str, 'contents': str}
          {'type': 'delete', 'path': str}
          {'type': 'update', 'path': str, 'move_path': str|None,
           'chunks': [{'context': str|None, 'old_lines': [str],
                        'new_lines': [str], 'is_eof': bool}]}

        Raises ValueError with a descriptive message on parse failure.
        """
        lines = patch_text.strip().splitlines()
        if not lines:
            raise ValueError('Patch text is empty')

        # Strip heredoc wrapper if present (lenient mode, like gpt-4.1)
        if lines[0].strip() in ('<<EOF', "<<'EOF'", '<<"EOF"'):
            if len(lines) >= 4 and lines[-1].strip().endswith('EOF'):
                lines = lines[1:-1]

        # Validate *** Begin Patch / *** End Patch boundaries
        if lines[0].strip() != '*** Begin Patch':
            raise ValueError(
                f"Expected '*** Begin Patch' on line 1, got: '{lines[0].strip()}'"
            )
        if lines[-1].strip() != '*** End Patch':
            raise ValueError(
                f"Expected '*** End Patch' on the last line, got: '{lines[-1].strip()}'"
            )

        # Work with content between markers
        content_lines = lines[1:-1]
        hunks: list[dict] = []
        i = 0

        while i < len(content_lines):
            line = content_lines[i].strip()

            # Skip blank lines between hunks
            if not line:
                i += 1
                continue

            if line.startswith('*** Add File: '):
                path = line[len('*** Add File: ') :]
                if not path:
                    raise ValueError(f"Empty path in '*** Add File:' on line {i + 2}")
                contents = ''
                i += 1
                while i < len(content_lines):
                    if content_lines[i].startswith('+'):
                        contents += content_lines[i][1:] + '\n'
                        i += 1
                    else:
                        break
                hunks.append(
                    {
                        'type': 'add',
                        'path': path,
                        'contents': contents,
                    }
                )

            elif line.startswith('*** Delete File: '):
                path = line[len('*** Delete File: ') :]
                if not path:
                    raise ValueError(
                        f"Empty path in '*** Delete File:' on line {i + 2}"
                    )
                hunks.append({'type': 'delete', 'path': path})
                i += 1

            elif line.startswith('*** Update File: '):
                path = line[len('*** Update File: ') :]
                if not path:
                    raise ValueError(
                        f"Empty path in '*** Update File:' on line {i + 2}"
                    )
                i += 1

                # Optional: *** Move to: <path>
                move_path = None
                if i < len(content_lines) and content_lines[i].strip().startswith(
                    '*** Move to: '
                ):
                    move_path = content_lines[i].strip()[len('*** Move to: ') :]
                    i += 1

                # Parse chunks within this Update File hunk
                chunks: list[dict] = []
                while i < len(content_lines):
                    raw = content_lines[i]
                    stripped = raw.strip()

                    # Skip blank lines between chunks
                    if not stripped:
                        i += 1
                        continue

                    # Stop at next file-level marker
                    if stripped.startswith('***'):
                        break

                    # Parse one chunk
                    chunk, lines_consumed = self._codex_parse_update_chunk(
                        content_lines, i, len(chunks) == 0
                    )
                    chunks.append(chunk)
                    i += lines_consumed

                if not chunks:
                    raise ValueError(
                        f"Update File hunk for '{path}' contains no change chunks"
                    )

                hunks.append(
                    {
                        'type': 'update',
                        'path': path,
                        'move_path': move_path,
                        'chunks': chunks,
                    }
                )

            else:
                raise ValueError(
                    f"Unexpected line {i + 2}: '{line}'. "
                    f"Expected '*** Add File:', '*** Delete File:', "
                    f"or '*** Update File:'"
                )

        return hunks

    def _codex_parse_update_chunk(
        self, lines: list[str], start: int, allow_missing_context: bool
    ) -> tuple[dict, int]:
        """Parse a single update chunk within an Update File hunk.

        Returns (chunk_dict, lines_consumed).
        chunk_dict has: context, old_lines, new_lines, is_eof
        """
        line = lines[start]

        # Check for @@ context marker
        context = None
        idx = start
        if line.strip() == '@@':
            context = None
            idx += 1
        elif line.startswith('@@ '):
            context = line[3:]
            idx += 1
        else:
            if not allow_missing_context:
                raise ValueError(
                    f"Expected '@@ ...' context marker on line {start + 2}, "
                    f"got: '{line.strip()}'"
                )

        old_lines: list[str] = []
        new_lines: list[str] = []
        is_eof = False
        parsed = 0

        while idx < len(lines):
            raw = lines[idx]

            # *** End of File marker
            if raw.strip() == '*** End of File':
                if parsed == 0:
                    raise ValueError(f'Empty update chunk at line {idx + 2}')
                is_eof = True
                idx += 1
                parsed += 1
                break

            # Next file-level hunk or next @@ chunk
            if raw.strip().startswith('***'):
                break
            if raw.startswith('@@') and parsed > 0:
                break

            first_char = raw[0] if raw else ''

            if first_char == ' ':
                # Context line: goes into both old and new
                old_lines.append(raw[1:])
                new_lines.append(raw[1:])
            elif first_char == '+':
                new_lines.append(raw[1:])
            elif first_char == '-':
                old_lines.append(raw[1:])
            elif raw == '':
                # Empty line interpreted as empty context
                old_lines.append('')
                new_lines.append('')
            else:
                if parsed == 0:
                    raise ValueError(
                        f"Unexpected line {idx + 2} in update chunk: '{raw}'. "
                        f"Lines must start with ' ' (context), '+' (add), "
                        f"or '-' (remove)"
                    )
                # Assume start of next chunk
                break

            idx += 1
            parsed += 1

        lines_consumed = idx - start
        return {
            'context': context,
            'old_lines': old_lines,
            'new_lines': new_lines,
            'is_eof': is_eof,
        }, lines_consumed

    @staticmethod
    def _codex_seek_sequence(
        lines: list[str],
        pattern: list[str],
        start: int,
        eof: bool = False,
    ) -> int | None:
        """Find a sequence of pattern lines within lines, starting at or after start.

        Matches with decreasing strictness: exact, rstrip, trim, unicode-normalized.
        When eof=True, searches from end of file first.
        Returns starting index or None.
        """
        if not pattern:
            return start
        if len(pattern) > len(lines):
            return None

        search_start = (
            len(lines) - len(pattern) if eof and len(lines) >= len(pattern) else start
        )
        end = len(lines) - len(pattern)

        # Exact match
        for i in range(search_start, end + 1):
            if lines[i : i + len(pattern)] == pattern:
                return i

        # rstrip match
        for i in range(search_start, end + 1):
            if all(lines[i + j].rstrip() == p.rstrip() for j, p in enumerate(pattern)):
                return i

        # trim match (strip both sides)
        for i in range(search_start, end + 1):
            if all(lines[i + j].strip() == p.strip() for j, p in enumerate(pattern)):
                return i

        # Unicode-normalized match
        def _normalize(s: str) -> str:
            result = []
            for c in s.strip():
                if c in '\u2010\u2011\u2012\u2013\u2014\u2015\u2212':
                    result.append('-')
                elif c in '\u2018\u2019\u201a\u201b':
                    result.append("'")
                elif c in '\u201c\u201d\u201e\u201f':
                    result.append('"')
                elif (
                    c
                    in '\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000'
                ):
                    result.append(' ')
                else:
                    result.append(c)
            return ''.join(result)

        for i in range(search_start, end + 1):
            if all(
                _normalize(lines[i + j]) == _normalize(p) for j, p in enumerate(pattern)
            ):
                return i

        return None

    def _codex_apply_update_hunk(
        self, full_path: str, chunks: list[dict]
    ) -> str | None:
        """Apply update chunks to a file. Returns error string or None on success."""
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                original_contents = f.read()
        except Exception as e:
            return f'Failed to read file: {e}'

        original_lines = original_contents.split('\n')
        # Drop trailing empty element from final newline (matches Rust behavior)
        if original_lines and original_lines[-1] == '':
            original_lines.pop()

        line_index = 0

        # Compute replacements: list of (start_idx, old_len, new_lines)
        replacements: list[tuple[int, int, list[str]]] = []

        for chunk_idx, chunk in enumerate(chunks):
            context = chunk.get('context')
            old_lines = chunk['old_lines']
            new_lines = chunk['new_lines']
            is_eof = chunk.get('is_eof', False)

            # If chunk has a context line, seek to it
            if context is not None:
                ctx_idx = self._codex_seek_sequence(
                    original_lines, [context], line_index, False
                )
                if ctx_idx is None:
                    return (
                        f'Chunk {chunk_idx + 1}: could not find context '
                        f"line '{context}' in file "
                        f'(searched from line {line_index + 1})'
                    )
                line_index = ctx_idx + 1

            # Pure addition (no old lines)
            if not old_lines:
                insertion_idx = (
                    len(original_lines) - 1
                    if original_lines and original_lines[-1] == ''
                    else len(original_lines)
                )
                replacements.append((insertion_idx, 0, new_lines))
                continue

            # Seek old_lines in the file
            pattern = old_lines
            found = self._codex_seek_sequence(
                original_lines, pattern, line_index, is_eof
            )

            new_slice = new_lines

            # Retry without trailing empty line (handles EOF edge cases)
            if found is None and pattern and pattern[-1] == '':
                pattern = pattern[:-1]
                if new_slice and new_slice[-1] == '':
                    new_slice = new_slice[:-1]
                found = self._codex_seek_sequence(
                    original_lines, pattern, line_index, is_eof
                )

            if found is None:
                # Build a descriptive error message
                preview = old_lines[:5]
                if len(old_lines) > 5:
                    preview.append(f'... ({len(old_lines) - 5} more lines)')
                preview_str = '\n'.join(f'  {l}' for l in preview)
                return (
                    f'Chunk {chunk_idx + 1}: could not find the expected '
                    f'lines in file (searched from line {line_index + 1}).\n'
                    f'Looking for:\n{preview_str}'
                )

            replacements.append((found, len(pattern), list(new_slice)))
            line_index = found + len(pattern)

        # Sort replacements by position
        replacements.sort(key=lambda r: r[0])

        # Apply replacements in reverse order so indices stay valid
        for start_idx, old_len, new_segment in reversed(replacements):
            del original_lines[start_idx : start_idx + old_len]
            for offset, new_line in enumerate(new_segment):
                original_lines.insert(start_idx + offset, new_line)

        # Ensure trailing newline
        if not original_lines or original_lines[-1] != '':
            original_lines.append('')

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(original_lines))
        except Exception as e:
            return f'Failed to write file: {e}'

        return None

    async def codex_update_plan(self, action: CodexUpdatePlanAction) -> Observation:
        """Update the task plan."""
        try:
            plan_items = action.plan
            if not isinstance(plan_items, list):
                return ErrorObservation('plan must be a list of plan items')

            # Validate plan items
            in_progress_count = 0
            for item in plan_items:
                if not isinstance(item, dict):
                    return ErrorObservation(
                        'Each plan item must be a dict with step and status'
                    )
                if 'step' not in item or 'status' not in item:
                    return ErrorObservation(
                        'Each plan item must have step and status fields'
                    )
                if item['status'] not in ('pending', 'in_progress', 'completed'):
                    return ErrorObservation(
                        f"Invalid status '{item['status']}'. Must be: pending, in_progress, completed"
                    )
                if item['status'] == 'in_progress':
                    in_progress_count += 1

            if in_progress_count > 1:
                return ErrorObservation('At most one step can be in_progress at a time')

            # Store the plan (reuse _todos storage for plan items)
            if not hasattr(self, '_plan'):
                self._plan: list[dict] = []
            self._plan = list(plan_items)

            return CodexUpdatePlanObservation(
                content='Plan updated',
                plan=list(self._plan),
                success=True,
            )
        except Exception as e:
            logger.exception(f'Error updating plan: {e}')
            return ErrorObservation(f'Failed to update plan: {str(e)}')

    def _format_terminal_screen(
        self, obs: CmdOutputObservation, command: str, pre_cwd: str | None = None
    ) -> str:
        """Format a CmdOutputObservation to look like a tmux capture-pane screen.

        The pre-command prompt uses pre_cwd (the directory before execution),
        and the post-command prompt uses the actual post-execution working_dir
        from metadata. This matches real terminal behavior where e.g.
        ``cd /app/src`` shows the old cwd before the command and the new cwd after.

        Produces output like:
            root@hostname:/app# cd /app/src
            root@hostname:/app/src#
        """
        meta = obs.metadata
        username = 'root'
        hostname = 'sandbox'
        post_cwd = meta.working_dir or '/'
        suffix = '#' if username == 'root' else '$'

        before_cwd = pre_cwd if pre_cwd else post_cwd
        pre_prompt = f'{username}@{hostname}:{before_cwd}{suffix} '
        post_prompt = f'{username}@{hostname}:{post_cwd}{suffix} '

        lines = [f'{pre_prompt}{command}']
        if obs.content.strip():
            lines.append(obs.content)
        lines.append(post_prompt)
        return '\n'.join(lines)

    async def terminus_2_cmd_run(
        self, action: Terminus2CmdRunAction
    ) -> Terminus2CmdOutputObservation | ErrorObservation:
        """Execute Terminus-2 keystroke action via BashSession.

        Converts keystrokes to a command, executes via the bash session,
        and returns terminal output formatted like the original Terminus-2
        tmux capture with appropriate prefix:
        - "Current Terminal Screen:" for initial captures (empty keystrokes)
          and timed-out commands
        - "New Terminal Output:" for normal command output
        """
        try:
            bash_session = self.bash_session
            assert bash_session is not None

            keystrokes = action.keystrokes
            duration = min(action.duration, 60)
            pre_cwd = bash_session.cwd

            if keystrokes == '' or keystrokes.strip() == '':
                cmd_action = CmdRunAction(command='pwd')
                cmd_action.set_hard_timeout(duration + 5, blocking=False)
                obs = await call_sync_from_async(bash_session.execute, cmd_action)
                screen = self._format_terminal_screen(obs, 'pwd', pre_cwd)
                terminal_state = f'Current Terminal Screen:\n{screen}'
                return Terminus2CmdOutputObservation(
                    content=terminal_state,
                    terminal_state=terminal_state,
                    timed_out=False,
                    command_keystrokes=keystrokes,
                )

            if keystrokes.strip() in ('C-c', 'C-d', 'C-z'):
                special_key = keystrokes.strip()
                cmd_action = CmdRunAction(command=special_key, is_input=True)
                cmd_action.set_hard_timeout(duration + 5, blocking=False)
                obs = await call_sync_from_async(bash_session.execute, cmd_action)
                screen = self._format_terminal_screen(
                    obs, f'^{"C" if special_key == "C-c" else "D"}', pre_cwd
                )
                terminal_state = f'New Terminal Output:\n{screen}'
                return Terminus2CmdOutputObservation(
                    content=terminal_state,
                    terminal_state=terminal_state,
                    timed_out=False,
                    command_keystrokes=keystrokes,
                )

            command = keystrokes.rstrip('\n')
            cmd_action = CmdRunAction(command=command)
            cmd_action.set_hard_timeout(duration + 10, blocking=False)
            obs = await call_sync_from_async(bash_session.execute, cmd_action)

            timed_out = False
            if hasattr(obs, 'metadata') and obs.metadata:
                timed_out = getattr(obs.metadata, 'exit_code', 0) == -1

            screen = self._format_terminal_screen(obs, command, pre_cwd)
            if timed_out:
                terminal_state = f'Current Terminal Screen:\n{screen}'
            else:
                terminal_state = f'New Terminal Output:\n{screen}'
            return Terminus2CmdOutputObservation(
                content=terminal_state,
                terminal_state=terminal_state,
                timed_out=timed_out,
                command_keystrokes=keystrokes,
            )
        except Exception as e:
            logger.exception(f'Error executing Terminus-2 keystrokes: {e}')
            return ErrorObservation(str(e))

    async def browse(self, action: BrowseURLAction) -> Observation:
        if self.browser is None:
            return ErrorObservation(
                'Browser functionality is not supported or disabled.'
            )
        await self._ensure_browser_ready()
        return await browse(action, self.browser, self.initial_cwd)

    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        if self.browser is None:
            return ErrorObservation(
                'Browser functionality is not supported or disabled.'
            )
        await self._ensure_browser_ready()
        browser_observation = await browse(action, self.browser, self.initial_cwd)
        if not browser_observation.error:
            return browser_observation
        else:
            curr_files = os.listdir(self.downloads_directory)
            new_download = False
            for file in curr_files:
                if file not in self.downloaded_files:
                    new_download = True
                    self.downloaded_files.append(file)
                    break  # FIXME: assuming only one file will be downloaded for simplicity

            if not new_download:
                return browser_observation
            else:
                # A new file is downloaded in self.downloads_directory, shift file to /workspace
                src_path = os.path.join(
                    self.downloads_directory, self.downloaded_files[-1]
                )
                # Guess extension of file using puremagic and add it to tgt_path file name
                file_ext = ''
                try:
                    guesses = puremagic.magic_file(src_path)
                    if len(guesses) > 0:
                        ext = guesses[0].extension.strip()
                        if len(ext) > 0:
                            file_ext = ext
                except Exception as _:
                    pass

                tgt_path = os.path.join(
                    '/workspace', f'file_{len(self.downloaded_files)}{file_ext}'
                )
                shutil.copy(src_path, tgt_path)
                file_download_obs = FileDownloadObservation(
                    content=f'Execution of the previous action {action.browser_actions} resulted in a file download. The downloaded file is saved at location: {tgt_path}',
                    file_path=tgt_path,
                )
                return file_download_obs

    def close(self):
        self.memory_monitor.stop_monitoring()
        if self.bash_session is not None:
            self.bash_session.close()
        if self.browser is not None:
            self.browser.close()


if __name__ == '__main__':
    logger.warning('Starting Action Execution Server')
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int, help='Port to listen on')
    parser.add_argument('--working-dir', type=str, help='Working directory')
    parser.add_argument('--plugins', type=str, help='Plugins to initialize', nargs='+')
    parser.add_argument(
        '--username', type=str, help='User to run as', default='openhands'
    )
    parser.add_argument('--user-id', type=int, help='User ID to run as', default=1000)
    parser.add_argument(
        '--enable-browser',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable the browser environment',
    )
    parser.add_argument(
        '--browsergym-eval-env',
        type=str,
        help='BrowserGym environment used for browser evaluation',
        default=None,
    )

    # example: python client.py 8000 --working-dir /workspace --plugins JupyterRequirement
    args = parser.parse_args()

    # Start the file viewer server in a separate thread
    logger.info('Starting file viewer server')
    # _file_viewer_port = find_available_tcp_port(
    #     min_port=args.port + 1, max_port=min(args.port + 1024, 65535)
    # )
    # server_url, _ = start_file_viewer_server(port=_file_viewer_port)
    # logger.info(f'File viewer server started at {server_url}')

    plugins_to_load: list[Plugin] = []
    if args.plugins:
        for plugin in args.plugins:
            if plugin not in ALL_PLUGINS:
                raise ValueError(f'Plugin {plugin} not found')
            plugins_to_load.append(ALL_PLUGINS[plugin]())  # type: ignore

    client: ActionExecutor | None = None
    mcp_proxy_manager: MCPProxyManager | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global client, mcp_proxy_manager
        logger.info('Initializing ActionExecutor...')
        client = ActionExecutor(
            plugins_to_load,
            work_dir=args.working_dir,
            username=args.username,
            user_id=args.user_id,
            enable_browser=args.enable_browser,
            browsergym_eval_env=args.browsergym_eval_env,
        )
        await client.ainit()
        logger.info('ActionExecutor initialized.')

        # Check if we're on Windows
        is_windows = sys.platform == 'win32'

        # Initialize and mount MCP Proxy Manager (skip on Windows or if disabled)
        if True:
            mcp_proxy_manager = None
        else:
            logger.info('Initializing MCP Proxy Manager...')
            # Create a MCP Proxy Manager
            mcp_proxy_manager = MCPProxyManager(
                auth_enabled=bool(SESSION_API_KEY),
                api_key=SESSION_API_KEY,
                logger_level=logger.getEffectiveLevel(),
            )
            mcp_proxy_manager.initialize()
            # Mount the proxy to the app
            allowed_origins = ['*']
            try:
                await mcp_proxy_manager.mount_to_app(app, allowed_origins)
            except Exception as e:
                logger.error(f'Error mounting MCP Proxy: {e}', exc_info=True)
                raise RuntimeError(f'Cannot mount MCP Proxy: {e}')

        yield

        # Clean up & release the resources
        logger.info('Shutting down MCP Proxy Manager...')
        if mcp_proxy_manager:
            del mcp_proxy_manager
            mcp_proxy_manager = None
        else:
            logger.info('MCP Proxy Manager instance not found for shutdown.')

        logger.info('Closing ActionExecutor...')
        if client:
            try:
                client.close()
                logger.info('ActionExecutor closed successfully.')
            except Exception as e:
                logger.error(f'Error closing ActionExecutor: {e}', exc_info=True)
        else:
            logger.info('ActionExecutor instance not found for closing.')
        logger.info('Shutdown complete.')

    app = FastAPI(lifespan=lifespan)

    # TODO below 3 exception handlers were recommended by Sonnet.
    # Are these something we should keep?
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception('Unhandled exception occurred:')
        return JSONResponse(
            status_code=500,
            content={'detail': 'An unexpected error occurred. Please try again later.'},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.exception(f'HTTP exception occurred: {exc.detail}')
        return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        logger.exception(f'Validation error occurred: {exc}')
        return JSONResponse(
            status_code=422,
            content={
                'detail': 'Invalid request parameters',
                'errors': str(exc.errors()),
            },
        )

    @app.middleware('http')
    async def authenticate_requests(request: Request, call_next):
        if request.url.path != '/alive' and request.url.path != '/server_info':
            try:
                verify_api_key(request.headers.get('X-Session-API-Key'))
            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code, content={'detail': e.detail}
                )
        response = await call_next(request)
        return response

    @app.get('/server_info')
    async def get_server_info():
        assert client is not None
        current_time = time.time()
        uptime = current_time - client.start_time
        idle_time = current_time - client.last_execution_time

        response = {
            'uptime': uptime,
            'idle_time': idle_time,
            'resources': get_system_stats(),
        }
        logger.info('Server info endpoint response: %s', response)
        return response

    @app.post('/execute_action')
    async def execute_action(action_request: ActionRequest):
        assert client is not None
        try:
            action = event_from_dict(action_request.action)
            if not isinstance(action, Action):
                raise HTTPException(status_code=400, detail='Invalid action type')
            client.last_execution_time = time.time()
            observation = await client.run_action(action)
            return event_to_dict(observation)
        except Exception as e:
            logger.exception(f'Error while running /execute_action: {str(e)}')
            raise HTTPException(
                status_code=500,
                detail=f'Internal server error: {str(e)}',
            )
        finally:
            update_last_execution_time()

    @app.post('/update_mcp_server')
    async def update_mcp_server(request: Request):
        # Check if we're on Windows
        is_windows = sys.platform == 'win32'

        # Access the global mcp_proxy_manager variable
        global mcp_proxy_manager

        if is_windows:
            # On Windows, just return a success response without doing anything
            logger.info(
                'MCP server update request received on Windows - skipping as MCP is disabled'
            )
            return JSONResponse(
                status_code=200,
                content={
                    'detail': 'MCP server update skipped (MCP is disabled on Windows)',
                    'router_error_log': '',
                },
            )

        # Non-Windows implementation
        if mcp_proxy_manager is None:
            raise HTTPException(
                status_code=500, detail='MCP Proxy Manager is not initialized'
            )

        # Get the request body
        mcp_tools_to_sync = await request.json()
        if not isinstance(mcp_tools_to_sync, list):
            raise HTTPException(
                status_code=400, detail='Request must be a list of MCP tools to sync'
            )
        logger.info(
            f'Updating MCP server with tools: {json.dumps(mcp_tools_to_sync, indent=2)}'
        )
        mcp_tools_to_sync = [MCPStdioServerConfig(**tool) for tool in mcp_tools_to_sync]
        try:
            await mcp_proxy_manager.update_and_remount(app, mcp_tools_to_sync, ['*'])
            logger.info('MCP Proxy Manager updated and remounted successfully')
            router_error_log = ''
        except Exception as e:
            logger.error(f'Error updating MCP Proxy Manager: {e}', exc_info=True)
            router_error_log = str(e)

        return JSONResponse(
            status_code=200,
            content={
                'detail': 'MCP server updated successfully',
                'router_error_log': router_error_log,
            },
        )

    @app.post('/upload_file')
    async def upload_file(
        file: UploadFile,
        destination: str = '/',
        recursive: bool = False,
    ):
        assert client is not None

        try:
            # Ensure the destination directory exists
            if not os.path.isabs(destination):
                raise HTTPException(
                    status_code=400, detail='Destination must be an absolute path'
                )

            full_dest_path = destination
            if not os.path.exists(full_dest_path):
                os.makedirs(full_dest_path, exist_ok=True)

            if recursive or file.filename.endswith('.zip'):
                # For recursive uploads, we expect a zip file
                if not file.filename.endswith('.zip'):
                    raise HTTPException(
                        status_code=400, detail='Recursive uploads must be zip files'
                    )

                zip_path = os.path.join(full_dest_path, file.filename)
                with open(zip_path, 'wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # Extract the zip file
                shutil.unpack_archive(zip_path, full_dest_path)
                os.remove(zip_path)  # Remove the zip file after extraction

                logger.debug(
                    f'Uploaded file {file.filename} and extracted to {destination}'
                )
            else:
                # For single file uploads
                file_path = os.path.join(full_dest_path, file.filename)
                with open(file_path, 'wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)
                logger.debug(f'Uploaded file {file.filename} to {destination}')

            return JSONResponse(
                content={
                    'filename': file.filename,
                    'destination': destination,
                    'recursive': recursive,
                },
                status_code=200,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/download_files')
    def download_file(path: str):
        logger.debug('Downloading files')
        try:
            if not os.path.isabs(path):
                raise HTTPException(
                    status_code=400, detail='Path must be an absolute path'
                )

            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail='File not found')

            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                with ZipFile(temp_zip, 'w') as zipf:
                    for root, _, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(
                                file_path, arcname=os.path.relpath(file_path, path)
                            )
                return FileResponse(
                    path=temp_zip.name,
                    media_type='application/zip',
                    filename=f'{os.path.basename(path)}.zip',
                    background=BackgroundTask(lambda: os.unlink(temp_zip.name)),
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/alive')
    async def alive():
        if client is None or not client.initialized:
            return {'status': 'not initialized'}
        return {'status': 'ok'}

    # ================================
    # VSCode-specific operations
    # ================================

    @app.get('/vscode/connection_token')
    async def get_vscode_connection_token():
        assert client is not None
        if 'vscode' in client.plugins:
            plugin: VSCodePlugin = client.plugins['vscode']  # type: ignore
            return {'token': plugin.vscode_connection_token}
        else:
            return {'token': None}

    # ================================
    # File-specific operations for UI
    # ================================

    @app.post('/list_files')
    async def list_files(request: Request):
        """List files in the specified path.

        This function retrieves a list of files from the agent's runtime file store,
        excluding certain system and hidden files/directories.

        To list files:
        ```sh
        curl -X POST -d '{"path": "/"}' http://localhost:3000/list_files
        ```

        Args:
            request (Request): The incoming request object.
            path (str, optional): The path to list files from. Defaults to '/'.

        Returns:
            list: A list of file names in the specified path.

        Raises:
            HTTPException: If there's an error listing the files.
        """
        assert client is not None

        # get request as dict
        request_dict = await request.json()
        path = request_dict.get('path', None)

        # Get the full path of the requested directory
        if path is None:
            full_path = client.initial_cwd
        elif os.path.isabs(path):
            full_path = path
        else:
            full_path = os.path.join(client.initial_cwd, path)

        if not os.path.exists(full_path):
            # if user just removed a folder, prevent server error 500 in UI
            return JSONResponse(content=[])

        try:
            # Check if the directory exists
            if not os.path.exists(full_path) or not os.path.isdir(full_path):
                return JSONResponse(content=[])

            entries = os.listdir(full_path)

            # Separate directories and files
            directories = []
            files = []
            for entry in entries:
                # Remove leading slash and any parent directory components
                entry_relative = entry.lstrip('/').split('/')[-1]

                # Construct the full path by joining the base path with the relative entry path
                full_entry_path = os.path.join(full_path, entry_relative)
                if os.path.exists(full_entry_path):
                    is_dir = os.path.isdir(full_entry_path)
                    if is_dir:
                        # add trailing slash to directories
                        # required by FE to differentiate directories and files
                        entry = entry.rstrip('/') + '/'
                        directories.append(entry)
                    else:
                        files.append(entry)

            # Sort directories and files separately
            directories.sort(key=lambda s: s.lower())
            files.sort(key=lambda s: s.lower())

            # Combine sorted directories and files
            sorted_entries = directories + files
            return JSONResponse(content=sorted_entries)

        except Exception as e:
            logger.exception(f'Error listing files: {e}')
            return JSONResponse(content=[])

    logger.debug(f'Starting action execution API on port {args.port}')
    # When LOG_JSON=1, provide a JSON log config to Uvicorn so error/access logs are structured
    log_config = None
    if os.getenv('LOG_JSON', '0') in ('1', 'true', 'True'):
        log_config = get_uvicorn_json_log_config()

    run(app, host='0.0.0.0', port=args.port, log_config=log_config, use_colors=False)
