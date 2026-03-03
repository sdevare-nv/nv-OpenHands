"""This is the main file for the runtime client.
It is responsible for executing actions received from OpenHands backend and producing observations.

NOTE: this will be executed inside the docker sandbox.
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from zipfile import ZipFile
import glob as glob_module
import subprocess

import puremagic
from binaryornot.check import is_binary
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
# Use OpenCodeEditor with fuzzy matching instead of default OHEditor
try:
    from openhands.agenthub.opencode_agent.opencode_editor import OpenCodeEditor as OHEditor
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
            if action.is_static:
                bash_session = self._create_bash_session(action.cwd)
            assert bash_session is not None
            obs = await call_sync_from_async(bash_session.execute, action)
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
        """Execute OpenCode-style file read with 5-digit line numbers."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)

        # Constants matching OpenCode behavior
        MAX_BYTES = 50 * 1024  # 50KB
        MAX_LINE_LENGTH = 2000
        BINARY_EXTENSIONS = {
            '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.class', '.jar',
            '.war', '.7z', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.bin', '.dat', '.obj', '.o', '.a', '.lib', '.wasm', '.pyc', '.pyo'
        }

        # Check if file exists
        if not os.path.exists(filepath):
            # Try to find suggestions
            directory = os.path.dirname(filepath) or '.'
            basename = os.path.basename(filepath)

            if os.path.isdir(directory):
                try:
                    entries = os.listdir(directory)
                    suggestions = [
                        os.path.join(directory, entry)
                        for entry in entries
                        if basename.lower() in entry.lower() or entry.lower() in basename.lower()
                    ][:3]

                    if suggestions:
                        return ErrorObservation(
                            f"File not found: {filepath}\n\nDid you mean one of these?\n"
                            + "\n".join(suggestions)
                        )
                except OSError:
                    pass

            return ErrorObservation(f"File not found: {filepath}")

        # Check if directory
        if os.path.isdir(filepath):
            return ErrorObservation(f"Path is a directory: {filepath}. You can only read files")

        # Check binary by extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext in BINARY_EXTENSIONS:
            return ErrorObservation(f"Cannot read binary file: {filepath}")

        # Check binary by content
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(4096)
                if b'\x00' in chunk:
                    return ErrorObservation(f"Cannot read binary file: {filepath}")
                if chunk:
                    non_printable = sum(1 for b in chunk if b < 9 or (b > 13 and b < 32))
                    if non_printable / len(chunk) > 0.3:
                        return ErrorObservation(f"Cannot read binary file: {filepath}")
        except Exception:
            pass

        # Read file
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.read().split('\n')
        except Exception as e:
            return ErrorObservation(f"Error reading file: {e}")

        # Process lines with offset and limit
        offset = action.offset
        limit = action.limit
        raw = []
        total_bytes = 0
        truncated_by_bytes = False

        for i in range(offset, min(len(lines), offset + limit)):
            line = lines[i]
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "..."

            line_bytes = len(line.encode('utf-8')) + (1 if raw else 0)
            if total_bytes + line_bytes > MAX_BYTES:
                truncated_by_bytes = True
                break

            raw.append(line)
            total_bytes += line_bytes

        # Format with 5-digit line numbers and | separator (OpenCode style)
        content_lines = [
            f"{str(i + offset + 1).zfill(5)}| {line}"
            for i, line in enumerate(raw)
        ]

        total_lines = len(lines)
        last_read_line = offset + len(raw)
        has_more_lines = total_lines > last_read_line
        truncated = has_more_lines or truncated_by_bytes

        output = "<file>\n"
        output += "\n".join(content_lines)

        if truncated_by_bytes:
            output += f"\n\n(Output truncated at {MAX_BYTES} bytes. Use 'offset' parameter to read beyond line {last_read_line})"
        elif has_more_lines:
            output += f"\n\n(File has more lines. Use 'offset' parameter to read beyond line {last_read_line})"
        else:
            output += f"\n\n(End of file - total {total_lines} lines)"

        output += "\n</file>"

        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f"opencode_read {filepath}",
        )

    async def opencode_write(self, action: OpenCodeWriteAction) -> Observation:
        """Execute OpenCode-style file write with LSP diagnostics."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)

        # Create directory if needed
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                return ErrorObservation(f"Failed to create directory: {e}")

        # Write file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(action.content)
        except Exception as e:
            return ErrorObservation(f"Failed to write file: {e}")

        output = "Wrote file successfully."

        # Run linter based on file extension
        ext = os.path.splitext(filepath)[1].lower()
        errors = []

        try:
            import subprocess

            if ext == '.py':
                # Try flake8, pylint, py_compile in order
                for linter_cmd in [
                    ['flake8', '--max-line-length=120', filepath],
                    ['pylint', '--errors-only', filepath],
                    ['python3', '-m', 'py_compile', filepath],
                ]:
                    try:
                        result = subprocess.run(
                            linter_cmd, capture_output=True, text=True, timeout=10
                        )
                        lint_output = result.stdout.strip() or result.stderr.strip()
                        if lint_output:
                            errors.extend(lint_output.split('\n')[:20])
                            break
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        continue

            elif ext in ('.js', '.jsx', '.ts', '.tsx'):
                try:
                    result = subprocess.run(
                        ['eslint', '--format=compact', filepath],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.stdout.strip():
                        errors.extend(result.stdout.strip().split('\n')[:20])
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass

            elif ext == '.go':
                try:
                    result = subprocess.run(
                        ['go', 'vet', filepath],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.stderr.strip():
                        errors.extend(result.stderr.strip().split('\n')[:20])
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass

            elif ext == '.rs':
                try:
                    result = subprocess.run(
                        ['cargo', 'check', '--message-format=short'],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.stderr.strip():
                        error_lines = [
                            l for l in result.stderr.strip().split('\n')
                            if 'error' in l.lower()
                        ][:20]
                        errors.extend(error_lines)
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass

        except Exception:
            pass

        if errors:
            output += f'\n\nLSP errors detected in this file, please fix:\n'
            output += f'<diagnostics file="{filepath}">\n'
            output += '\n'.join(errors)
            output += '\n</diagnostics>'

        return FileWriteObservation(content=output, path=filepath)

    async def glob(self, action: GlobAction) -> Observation:
        """Execute glob file search using ripgrep or Python glob."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        search_path = self._resolve_path(action.path, working_dir)

        # Validate path exists
        if not os.path.exists(search_path):
            return ErrorObservation(
                f"Path does not exist: {search_path}"
            )

        # Auto-prepend **/ to patterns without a path separator so that
        # simple patterns like "*.py" search recursively instead of only
        # matching at the root of the search path.
        pattern = action.pattern
        if '/' not in pattern:
            pattern = '**/' + pattern

        files = []
        truncated = False
        limit = 100
        rg_available = False

        # Try ripgrep first (fast, respects .gitignore)
        # Note: avoid --sortr flag as it requires ripgrep >= 13.0.0
        try:
            result = subprocess.run(
                ['rg', '--files', '-g', pattern, search_path],
                capture_output=True, text=True, timeout=30, cwd=working_dir
            )
            rg_available = True

            if result.returncode == 0 and result.stdout.strip():
                files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            elif result.returncode == 1:
                # Exit code 1 = no matches found (not an error)
                files = []
            elif result.returncode not in (0, 1):
                logger.warning(f"rg --files failed: {result.stderr.strip()}")
                rg_available = False
        except FileNotFoundError:
            rg_available = False
        except subprocess.TimeoutExpired:
            return ErrorObservation("glob search timed out after 30 seconds")

        # Fallback to Python's glob module (handles ** patterns natively)
        if not rg_available:
            try:
                full_pattern = os.path.join(search_path, pattern)
                files = [
                    f for f in glob_module.glob(full_pattern, recursive=True)
                    if os.path.isfile(f)
                ]
            except Exception:
                pass

        # Sort by modification time (newest first)
        try:
            files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        except (OSError, ValueError):
            pass

        # Apply limit
        if len(files) > limit:
            truncated = True
            files = files[:limit]

        # Build output
        if not files:
            output = "No files found"
        else:
            output = '\n'.join(files)
            if truncated:
                output += '\n\n(Results are truncated. Consider using a more specific path or pattern.)'

        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f"glob {action.pattern} {action.path}",
        )

    async def grep(self, action: GrepAction) -> Observation:
        """Execute grep content search using ripgrep or grep.

        Results are sorted by file modification time (newest first) and limited
        to 100 matches.  Uses ripgrep when available (respects .gitignore) with
        a fallback to grep -E for extended regex support (e.g. | alternation).
        """
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        search_path = self._resolve_path(action.path, working_dir)

        import shlex
        import subprocess

        # Validate path exists
        if not os.path.exists(search_path):
            return ErrorObservation(
                f"Path does not exist: {search_path}"
            )

        raw_lines: list[str] = []
        limit = 100
        rg_available = False

        # Ensure include pattern matches recursively (e.g., "*.py" -> "**/*.py")
        include = action.include
        if include and not include.startswith('**/'):
            include = '**/' + include

        # Try ripgrep first (respects .gitignore)
        try:
            cmd = ['rg', '-n', '--regexp', action.pattern, search_path]
            if include:
                cmd = ['rg', '-n', '--regexp', action.pattern, '-g', include, search_path]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=working_dir
            )
            rg_available = True

            if result.returncode == 0 and result.stdout.strip():
                raw_lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
            elif result.returncode == 1:
                # Exit code 1 = no matches found (not an error)
                raw_lines = []
            elif result.returncode == 2:
                # Exit code 2 = regex syntax error (e.g. unmatched parenthesis)
                stderr = result.stderr.strip()
                return ErrorObservation(
                    f"Invalid regex pattern: {action.pattern!r}. "
                    f"The pattern is treated as a regex. Characters like (, ), [, ], "
                    f"{{, }}, ., *, +, ? have special meaning and must be escaped "
                    f"with a backslash (e.g. 'write_records\\(' instead of 'write_records('). "
                    f"Detail: {stderr}"
                )
            elif result.returncode not in (0, 1):
                stderr = result.stderr.strip()
                logger.warning(f"rg failed: {stderr}")
                rg_available = False
        except FileNotFoundError:
            rg_available = False
        except subprocess.TimeoutExpired:
            return ErrorObservation("grep search timed out after 30 seconds")

        # Fallback to grep if rg is not available
        if not rg_available:
            try:
                if include:
                    # Strip **/ prefix for grep --include (doesn't understand **/*)
                    grep_include = include
                    if grep_include.startswith('**/'):
                        grep_include = grep_include[3:]
                    include_flag = f'--include={shlex.quote(grep_include)} '
                else:
                    include_flag = ''

                cmd_str = (
                    f'grep -rHn {include_flag}'
                    f'-E {shlex.quote(action.pattern)} {shlex.quote(search_path)}'
                )
                result = subprocess.run(
                    cmd_str,
                    shell=True, capture_output=True, text=True, timeout=30, cwd=working_dir
                )

                if result.returncode == 2:
                    # Exit code 2 = regex syntax error
                    return ErrorObservation(
                        f"Invalid regex pattern: {action.pattern!r}. "
                        f"The pattern is treated as a regex. Characters like (, ), [, ], "
                        f"{{, }}, ., *, +, ? have special meaning and must be escaped "
                        f"with a backslash (e.g. 'write_records\\(' instead of 'write_records('). "
                        f"Alternatively, remov the sepecial characters from the pattern."
                    )

                if result.stdout.strip():
                    raw_lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
            except subprocess.TimeoutExpired:
                return ErrorObservation("grep search timed out after 30 seconds")
            except Exception as e:
                logger.warning(f"grep fallback failed: {e}")

        if not raw_lines:
            output = "No matches found"
        else:
            # Sort results by file modification time (newest first).
            # Each line has the format  filepath:linenum:content
            def _mtime_key(line: str) -> float:
                filepath = line.split(':')[0]
                try:
                    return os.path.getmtime(filepath)
                except OSError:
                    return 0.0

            raw_lines.sort(key=_mtime_key, reverse=True)

            # Apply limit
            if len(raw_lines) > limit:
                output = '\n'.join(raw_lines[:limit])
                output += f'\n\n(Results truncated, showing {limit} of {len(raw_lines)}+ matches)'
            else:
                output = '\n'.join(raw_lines)

        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f"grep {action.pattern} {action.path}",
        )

    async def list_dir(self, action: ListDirAction) -> Observation:
        """Execute directory listing with tree structure."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        list_path = self._resolve_path(action.path, working_dir)

        import subprocess

        # Combine default and custom ignore patterns
        all_ignores = action.all_ignores

        files = []
        limit = 100

        # Try ripgrep first (respects .gitignore)
        try:
            cmd = ['rg', '--files']
            for pattern in all_ignores:
                cmd.extend(['-g', f'!{pattern}/**'])
            if list_path != '.':
                cmd.append(list_path)

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=working_dir
            )
            if result.returncode == 0 and result.stdout.strip():
                files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()][:limit]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Build tree structure if we have files
        if files:
            dirs = set()
            files_by_dir = {}

            for f in files:
                d = os.path.dirname(f) or '.'
                parts = d.split(os.sep) if d != '.' else []

                # Add all parent directories
                for i in range(len(parts) + 1):
                    dir_p = os.sep.join(parts[:i]) if i > 0 else '.'
                    dirs.add(dir_p)

                # Add file to its directory
                if d not in files_by_dir:
                    files_by_dir[d] = []
                files_by_dir[d].append(os.path.basename(f))

            def render_dir(dir_path: str, depth: int) -> str:
                output = ''
                if depth > 0:
                    output += '  ' * depth + os.path.basename(dir_path) + '/\n'

                # Get child directories
                children = sorted([
                    d for d in dirs
                    if os.path.dirname(d) == dir_path and d != dir_path
                ])

                # Render subdirectories first
                for child in children:
                    output += render_dir(child, depth + 1)

                # Render files
                for f in sorted(files_by_dir.get(dir_path, [])):
                    output += '  ' * (depth + 1) + f + '\n'

                return output

            abs_path = os.path.abspath(list_path)
            output = f"{abs_path}/\n" + render_dir('.', 0)
        else:
            # Fallback to tree or find
            try:
                # Try tree command
                ignore_args = []
                for p in all_ignores:
                    ignore_args.extend(['-I', p])

                result = subprocess.run(
                    ['tree', '-L', '3', '--noreport'] + ignore_args + [list_path],
                    capture_output=True, text=True, timeout=10, cwd=working_dir
                )
                output = result.stdout.strip()
            except FileNotFoundError:
                # Fallback to find
                try:
                    result = subprocess.run(
                        ['find', list_path, '-maxdepth', '3', '-type', 'f'],
                        capture_output=True, text=True, timeout=10, cwd=working_dir
                    )
                    lines = result.stdout.strip().split('\n')
                    # Filter out ignored patterns
                    filtered = [
                        l for l in lines
                        if l and not any(p in l for p in all_ignores)
                    ][:limit]
                    output = '\n'.join(filtered) if filtered else 'No files found'
                except Exception:
                    output = 'No files found'
            except subprocess.TimeoutExpired:
                output = 'Directory listing timed out'

        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f"list_dir {action.path}",
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
        """Read the current todo list."""
        return TodoReadObservation(
            content=json.dumps(self._todos, indent=2) if self._todos else '[]',
            todos=list(self._todos),
        )

    async def todo_write(self, action: TodoWriteAction) -> Observation:
        """Update the todo list with new or modified items."""
        try:
            incoming_todos = action.todos
            if not isinstance(incoming_todos, list):
                return ErrorObservation('todos must be a list of todo objects')

            # Build index of existing todos by id
            existing_by_id = {t['id']: t for t in self._todos if 'id' in t}

            # Merge incoming todos: update existing by id, add new ones
            for todo in incoming_todos:
                if not isinstance(todo, dict):
                    continue
                todo_id = todo.get('id')
                if todo_id and todo_id in existing_by_id:
                    # Update existing todo
                    existing_by_id[todo_id].update(todo)
                else:
                    # Add new todo
                    self._todos.append(todo)
                    if todo_id:
                        existing_by_id[todo_id] = todo

            return TodoWriteObservation(
                content=json.dumps(self._todos, indent=2),
                todos=list(self._todos),
                success=True,
            )
        except Exception as e:
            logger.exception(f'Error updating todos: {e}')
            return ErrorObservation(f'Failed to update todos: {str(e)}')

    # =========================================================================
    # Codex-style action handlers
    # =========================================================================

    async def codex_read_file(self, action: CodexReadFileAction) -> Observation:
        """Execute Codex-style file read with L{number}: format and 1-indexed lines."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.file_path, working_dir)

        # Check if file exists
        if not os.path.exists(filepath):
            # Try to find suggestions
            directory = os.path.dirname(filepath) or '.'
            basename = os.path.basename(filepath)

            if os.path.isdir(directory):
                try:
                    entries = os.listdir(directory)
                    suggestions = [
                        os.path.join(directory, entry)
                        for entry in entries
                        if basename.lower() in entry.lower() or entry.lower() in basename.lower()
                    ][:3]

                    if suggestions:
                        return ErrorObservation(
                            f"File not found: {filepath}\n\nDid you mean one of these?\n"
                            + "\n".join(suggestions)
                        )
                except OSError:
                    pass

            return ErrorObservation(f"File not found: {filepath}")

        # Check if directory
        if os.path.isdir(filepath):
            return ErrorObservation(f"Path is a directory: {filepath}. You can only read files")

        # Check binary by content
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(4096)
                if b'\x00' in chunk:
                    return ErrorObservation(f"Cannot read binary file: {filepath}")
                if chunk:
                    non_printable = sum(1 for b in chunk if b < 9 or (b > 13 and b < 32))
                    if non_printable / len(chunk) > 0.3:
                        return ErrorObservation(f"Cannot read binary file: {filepath}")
        except Exception:
            pass

        # Read file
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.read().split('\n')
        except Exception as e:
            return ErrorObservation(f"Error reading file: {e}")

        total_lines = len(lines)

        # Handle indentation mode
        if action.mode == 'indentation' and action.indentation:
            return self._codex_read_file_indentation(
                lines, total_lines, filepath, action
            )

        # Slice mode (default) - 1-indexed offset
        offset = max(action.offset, 1)  # Ensure >= 1
        limit = action.limit
        start_idx = offset - 1  # Convert to 0-indexed

        raw = []
        for i in range(start_idx, min(total_lines, start_idx + limit)):
            raw.append(lines[i])

        # Format with L{number}: (Codex style, 1-indexed)
        content_lines = [
            f"L{i + offset}: {line}"
            for i, line in enumerate(raw)
        ]

        last_read_line = offset + len(raw) - 1
        has_more = total_lines > (start_idx + len(raw))

        output = '\n'.join(content_lines)
        if has_more:
            output += f'\n\n(File has {total_lines} lines total. Use offset to read more.)'
        else:
            output += f'\n\n(End of file. Total lines: {total_lines})'

        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f"codex_read_file {filepath}",
        )

    def _codex_read_file_indentation(
        self, lines: list[str], total_lines: int, filepath: str,
        action: CodexReadFileAction,
    ) -> Observation:
        """Handle indentation-aware block reading mode."""
        indent_args = action.indentation
        anchor = indent_args.get('anchor_line', action.offset)
        anchor_idx = max(anchor - 1, 0)  # Convert to 0-indexed
        max_levels = indent_args.get('max_levels', 0)
        include_siblings = indent_args.get('include_siblings', False)
        include_header = indent_args.get('include_header', True)
        max_lines = indent_args.get('max_lines', action.limit)

        if anchor_idx >= total_lines:
            return ErrorObservation(
                f"Anchor line {anchor} is beyond end of file ({total_lines} lines)"
            )

        # Get the indentation level of the anchor line
        anchor_line = lines[anchor_idx]
        anchor_indent = len(anchor_line) - len(anchor_line.lstrip())

        # Find the block boundaries
        # Walk upward to find parent blocks based on max_levels
        start_idx = anchor_idx
        current_indent = anchor_indent
        levels_found = 0

        for i in range(anchor_idx - 1, -1, -1):
            line = lines[i]
            stripped = line.lstrip()
            if not stripped:  # Skip empty lines
                continue
            line_indent = len(line) - len(stripped)
            if line_indent < current_indent:
                levels_found += 1
                current_indent = line_indent
                start_idx = i
                if max_levels > 0 and levels_found >= max_levels:
                    break

        # Include header (doc comments/attributes above the block)
        if include_header and start_idx > 0:
            for i in range(start_idx - 1, -1, -1):
                line = lines[i].strip()
                if line.startswith('#') or line.startswith('//') or line.startswith('/*') or \
                   line.startswith('*') or line.startswith('"""') or line.startswith("'''") or \
                   line.startswith('@') or not line:
                    start_idx = i
                else:
                    break

        # Walk downward to find end of block
        end_idx = anchor_idx
        for i in range(anchor_idx + 1, total_lines):
            line = lines[i]
            stripped = line.lstrip()
            if not stripped:  # Include empty lines within block
                end_idx = i
                continue
            line_indent = len(line) - len(stripped)
            if line_indent <= anchor_indent and stripped:
                if include_siblings and line_indent == anchor_indent:
                    end_idx = i
                    continue
                break
            end_idx = i

        # Apply max_lines cap
        if max_lines and (end_idx - start_idx + 1) > max_lines:
            end_idx = start_idx + max_lines - 1

        # Collect lines
        raw = lines[start_idx:end_idx + 1]

        # Format with L{number}: (1-indexed)
        content_lines = [
            f"L{start_idx + 1 + i}: {line}"
            for i, line in enumerate(raw)
        ]

        output = '\n'.join(content_lines)
        output += f'\n\n(Showing lines {start_idx + 1}-{end_idx + 1} of {total_lines} total)'

        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f"codex_read_file {filepath} (indentation mode)",
        )

    async def codex_list_dir(self, action: CodexListDirAction) -> Observation:
        """Execute Codex-style directory listing with numbered entries and type labels."""
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        dir_path = self._resolve_path(action.dir_path, working_dir)

        if not os.path.exists(dir_path):
            return ErrorObservation(f"Directory not found: {dir_path}")

        if not os.path.isdir(dir_path):
            return ErrorObservation(f"Path is not a directory: {dir_path}")

        # Collect entries recursively up to depth
        entries: list[tuple[str, str]] = []  # (relative_path, type_label)

        def _collect_entries(current_path: str, rel_prefix: str, current_depth: int) -> None:
            if current_depth > action.depth:
                return
            try:
                items = sorted(os.listdir(current_path))
            except PermissionError:
                return

            for item in items:
                # Skip hidden files and common ignore patterns
                if item.startswith('.'):
                    continue

                full_path = os.path.join(current_path, item)
                rel_path = os.path.join(rel_prefix, item) if rel_prefix else item

                if os.path.isdir(full_path):
                    entries.append((rel_path, 'dir'))
                    if current_depth < action.depth:
                        _collect_entries(full_path, rel_path, current_depth + 1)
                else:
                    entries.append((rel_path, 'file'))

        _collect_entries(dir_path, '', 1)

        # Apply offset and limit (1-indexed offset)
        offset = max(action.offset, 1)
        start_idx = offset - 1
        end_idx = start_idx + action.limit

        paginated = entries[start_idx:end_idx]

        if not paginated:
            output = "No entries found."
        else:
            # Format as numbered entries with type labels
            output_lines = []
            for i, (rel_path, type_label) in enumerate(paginated):
                entry_num = start_idx + i + 1
                output_lines.append(f"{entry_num}. [{type_label}] {rel_path}")
            output = '\n'.join(output_lines)

            if end_idx < len(entries):
                output += f'\n\n(Showing {len(paginated)} of {len(entries)} entries. Use offset to see more.)'

        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f"codex_list_dir {dir_path}",
        )

    async def codex_grep_files(self, action: CodexGrepFilesAction) -> Observation:
        """Execute Codex-style grep: find files matching pattern, return paths sorted by mtime.

        Matches the original Codex implementation: uses ripgrep with --sortr=modified,
        --files-with-matches, --regexp, and --no-messages flags. Falls back to grep
        if ripgrep is not available.
        """
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        search_path = self._resolve_path(action.path, working_dir) if action.path else working_dir

        import shlex
        import subprocess

        pattern = action.pattern.strip()
        if not pattern:
            return ErrorObservation("pattern must not be empty")

        limit = min(action.limit, 2000) if action.limit > 0 else 100

        # Normalize include glob: ensure it matches recursively
        include = (action.include or '').strip() or None
        if include and not include.startswith('**/'):
            include = '**/' + include

        # Verify path exists
        if not os.path.exists(search_path):
            return ErrorObservation(f"unable to access `{search_path}`: path does not exist")

        files: list[str] = []
        rg_available = False

        # Try ripgrep first (matches Codex's Rust implementation exactly)
        try:
            cmd = ['rg', '--files-with-matches', '--sortr=modified', '--regexp', pattern, '--no-messages']
            if include:
                cmd.extend(['--glob', include])
            cmd.extend(['--', search_path])

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=working_dir
            )
            rg_available = True

            if result.returncode == 0 and result.stdout.strip():
                # rg found matches and already sorted by mtime
                files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            elif result.returncode == 1:
                # Exit code 1 = no matches (not an error)
                files = []
            elif result.returncode == 2:
                # Exit code 2 = regex syntax error (e.g. unmatched parenthesis)
                stderr = result.stderr.strip()
                return ErrorObservation(
                    f"Invalid regex pattern: {pattern!r}. "
                    f"The pattern is treated as a regex. Characters like (, ), [, ], "
                    f"{{, }}, ., *, +, ? have special meaning and must be escaped "
                    f"with a backslash (e.g. 'write_records\\(' instead of 'write_records('). "
                    f"Detail: {stderr}"
                )
            elif result.returncode not in (0, 1):
                # rg failed with a non-regex error
                stderr = result.stderr.strip()
                logger.warning(f"rg failed: {stderr}")
                # Fall through to grep fallback
                rg_available = False

        except FileNotFoundError:
            # rg not installed
            rg_available = False
        except subprocess.TimeoutExpired:
            return ErrorObservation("grep_files timed out after 30 seconds")

        # Fallback to grep if rg is not available
        if not rg_available:
            try:
                if include:
                    # Convert glob pattern to find-compatible: "**/*.py" -> "*.py"
                    find_pattern = include
                    if find_pattern.startswith('**/'):
                        find_pattern = find_pattern[3:]
                    include_flag = f'--include={shlex.quote(find_pattern)} '
                else:
                    include_flag = ''

                cmd_str = (
                    f'grep -rl {include_flag}'
                    f'-E {shlex.quote(pattern)} {shlex.quote(search_path)}'
                )
                result = subprocess.run(
                    cmd_str, shell=True, capture_output=True, text=True, timeout=30, cwd=working_dir
                )

                if result.returncode == 2:
                    # Exit code 2 = regex syntax error
                    return ErrorObservation(
                        f"Invalid regex pattern: {pattern!r}. "
                        f"The pattern is treated as a regex. Characters like (, ), [, ], "
                        f"{{, }}, ., *, +, ? have special meaning and must be escaped "
                        f"with a backslash (e.g. 'write_records\\(' instead of 'write_records('). "
                        f"Alternatively, remove the special characters from the pattern."
                    )

                if result.stdout.strip():
                    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

                # Sort by modification time (newest first) since grep doesn't sort
                try:
                    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
                except (OSError, ValueError):
                    pass

            except subprocess.TimeoutExpired:
                return ErrorObservation("grep_files timed out after 30 seconds")
            except Exception as e:
                return ErrorObservation(f"grep_files failed: {str(e)}")

        if not files:
            return CmdOutputObservation(
                content="No matches found.",
                command_id=-1,
                command=f"codex_grep_files {pattern}",
            )

        # Apply limit (rg results are already sorted by mtime)
        truncated = len(files) > limit
        if truncated:
            files = files[:limit]

        output = '\n'.join(files)
        if truncated:
            output += f'\n\n(Results truncated at {limit} files.)'

        return CmdOutputObservation(
            content=output,
            command_id=-1,
            command=f"codex_grep_files {pattern}",
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
                        errors.append(
                            f"Delete failed: file not found '{path}'"
                        )
                        continue
                    os.unlink(full_path)
                    deleted.append(path)

                elif hunk_type == 'update':
                    if not os.path.exists(full_path):
                        errors.append(
                            f"Update failed: file not found '{path}'"
                        )
                        continue
                    if not os.path.isfile(full_path):
                        errors.append(
                            f"Update failed: '{path}' is not a regular file"
                        )
                        continue

                    err = self._codex_apply_update_hunk(
                        full_path, hunk['chunks']
                    )
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
            summary_parts.append(
                f'Errors ({len(errors)}):'
            )
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

        summary = ['Patch applied successfully. Changed files:']
        for p in added:
            summary.append(f'  A {p}')
        for p in modified:
            summary.append(f'  M {p}')
        for p in deleted:
            summary.append(f'  D {p}')
        return CodexApplyPatchObservation(
            content='\n'.join(summary),
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
        if lines[0].strip() in ("<<EOF", "<<'EOF'", '<<"EOF"'):
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
                path = line[len('*** Add File: '):]
                if not path:
                    raise ValueError(
                        f"Empty path in '*** Add File:' on line {i + 2}"
                    )
                contents = ''
                i += 1
                while i < len(content_lines):
                    if content_lines[i].startswith('+'):
                        contents += content_lines[i][1:] + '\n'
                        i += 1
                    else:
                        break
                hunks.append({
                    'type': 'add',
                    'path': path,
                    'contents': contents,
                })

            elif line.startswith('*** Delete File: '):
                path = line[len('*** Delete File: '):]
                if not path:
                    raise ValueError(
                        f"Empty path in '*** Delete File:' on line {i + 2}"
                    )
                hunks.append({'type': 'delete', 'path': path})
                i += 1

            elif line.startswith('*** Update File: '):
                path = line[len('*** Update File: '):]
                if not path:
                    raise ValueError(
                        f"Empty path in '*** Update File:' on line {i + 2}"
                    )
                i += 1

                # Optional: *** Move to: <path>
                move_path = None
                if i < len(content_lines) and content_lines[i].strip().startswith('*** Move to: '):
                    move_path = content_lines[i].strip()[len('*** Move to: '):]
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

                hunks.append({
                    'type': 'update',
                    'path': path,
                    'move_path': move_path,
                    'chunks': chunks,
                })

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
                    raise ValueError(
                        f"Empty update chunk at line {idx + 2}"
                    )
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
            len(lines) - len(pattern) if eof and len(lines) >= len(pattern)
            else start
        )
        end = len(lines) - len(pattern)

        # Exact match
        for i in range(search_start, end + 1):
            if lines[i:i + len(pattern)] == pattern:
                return i

        # rstrip match
        for i in range(search_start, end + 1):
            if all(
                lines[i + j].rstrip() == p.rstrip()
                for j, p in enumerate(pattern)
            ):
                return i

        # trim match (strip both sides)
        for i in range(search_start, end + 1):
            if all(
                lines[i + j].strip() == p.strip()
                for j, p in enumerate(pattern)
            ):
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
                elif c in '\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000':
                    result.append(' ')
                else:
                    result.append(c)
            return ''.join(result)

        for i in range(search_start, end + 1):
            if all(
                _normalize(lines[i + j]) == _normalize(p)
                for j, p in enumerate(pattern)
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
                        f"Chunk {chunk_idx + 1}: could not find context "
                        f"line '{context}' in file "
                        f"(searched from line {line_index + 1})"
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
                    f"Chunk {chunk_idx + 1}: could not find the expected "
                    f"lines in file (searched from line {line_index + 1}).\n"
                    f"Looking for:\n{preview_str}"
                )

            replacements.append((found, len(pattern), list(new_slice)))
            line_index = found + len(pattern)

        # Sort replacements by position
        replacements.sort(key=lambda r: r[0])

        # Apply replacements in reverse order so indices stay valid
        for start_idx, old_len, new_segment in reversed(replacements):
            del original_lines[start_idx:start_idx + old_len]
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
                    return ErrorObservation('Each plan item must be a dict with step and status')
                if 'step' not in item or 'status' not in item:
                    return ErrorObservation('Each plan item must have step and status fields')
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
        username = meta.username or 'root'
        hostname = meta.hostname or 'sandbox'
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

            if keystrokes.strip() in ('C-c', 'C-d'):
                special_key = keystrokes.strip()
                cmd_action = CmdRunAction(command=special_key)
                cmd_action.set_hard_timeout(duration + 5, blocking=False)
                obs = await call_sync_from_async(bash_session.execute, cmd_action)
                screen = self._format_terminal_screen(obs, f'^{"C" if special_key == "C-c" else "D"}', pre_cwd)
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
