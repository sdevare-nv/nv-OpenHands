import json
import os
import re
import shlex
import signal
import time
import uuid
from enum import Enum
from typing import Any, Set

import bashlex
import libtmux
import psutil
import threading
import logging
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction
from openhands.events.observation import ErrorObservation
from openhands.events.observation.commands import (
    CMD_OUTPUT_METADATA_PS1_REGEX,
    CMD_OUTPUT_PS1_END,
    MAX_CMD_OUTPUT_SIZE,
    CmdOutputMetadata,
    CmdOutputObservation,
)
from openhands.runtime.utils.bash_constants import TIMEOUT_MESSAGE_TEMPLATE
from openhands.runtime.utils.command_blacklist import check_command_blacklist
from openhands.utils.shutdown_listener import should_continue

RUNTIME_USERNAME = os.getenv("RUNTIME_USERNAME")
SU_TO_USER = os.getenv("SU_TO_USER", "true").lower() in (
    "1",
    "true",
    "t",
    "yes",
    "y",
    "on",
)


class TmuxMemoryMonitor(threading.Thread):
    def __init__(self, tmux_server, limit_mb, interval=2.0):
        super().__init__(daemon=True)
        self.server = tmux_server
        self.limit_mb = limit_mb
        self.limit_bytes = limit_mb * 1024 * 1024
        self.interval = interval
        self.running = True
        print(f"[TmuxMemoryMonitor] initialized with limit: {self.limit_mb} MB", flush=True)

    def _get_server_pid(self):
        try:
            pid_str = self.server.cmd("display-message", "-p", "#{pid}").stdout[0]
            return int(pid_str)
        except (IndexError, ValueError, Exception):
            return None

    def get_tree_memory(self, parent_pid):
        total_mem = 0
        try:
            parent = psutil.Process(parent_pid)
            procs = [parent] + parent.children(recursive=True)
            for p in procs:
                try:
                    total_mem += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except psutil.NoSuchProcess:
            return 0
        return total_mem

    def kill_inner_processes(self):
        try:
            # Loop through all sessions -> windows -> panes
            for session in self.server.sessions:
                for window in session.windows:
                    for pane in window.panes:
                        try:
                            pane_pid_str = pane.cmd(
                                "display-message", "-p", "#{pane_pid}"
                            ).stdout[0]
                            pane_pid = int(pane_pid_str)

                            parent = psutil.Process(pane_pid)
                            shell_proc = BashSession._find_shell_proc(parent)
                            targets = shell_proc.children(recursive=True)
                            if targets:
                                print(
                                    f"[TmuxMemoryMonitor] Killing {len(targets)} command processes in pane {pane.id} "
                                    f"(shell PID: {shell_proc.pid}, keeping shell alive)",
                                    flush=True,
                                )
                                for child in targets:
                                    try:
                                        child.kill()
                                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                                        pass

                        except (psutil.NoSuchProcess, IndexError, ValueError):
                            continue
        except Exception as e:
            print(f"[TmuxMemoryMonitor] Error killing panes: {e}", flush=True)

    def run(self):
        print(f"[TmuxMemoryMonitor] started. Limit: {self.limit_mb} MB", flush=True)
        time.sleep(1)

        server_pid = self._get_server_pid()
        if not server_pid:
            print("[TmuxMemoryMonitor] Could not determine Tmux Server PID. Monitor aborting.", flush=True)
            return

        while self.running:
            used_bytes = self.get_tree_memory(server_pid)
            used_mb = used_bytes / (1024 * 1024)

            if used_bytes > self.limit_bytes:
                print(
                    f"[TmuxMemoryMonitor] MEMORY LIMIT EXCEEDED: {int(used_mb)}MB > {self.limit_mb}MB",
                    flush=True,
                )
                print(
                    "[TmuxMemoryMonitor] KILLING PROCESSES INSIDE TMUX (Server stays alive)...",
                    flush=True,
                )

                self.kill_inner_processes()
                time.sleep(10)

            time.sleep(self.interval)

    def stop(self):
        self.running = False


def split_bash_commands(commands: str) -> list[str]:
    if not commands.strip():
        return [""]
    try:
        parsed = bashlex.parse(commands)
    except (
        bashlex.errors.ParsingError,
        NotImplementedError,
        TypeError,
        AttributeError,
    ):
        # Added AttributeError to catch 'str' object has no attribute 'kind' error (issue #8369)
        logger.debug(
            f"Failed to parse bash commands\n"
            f"[input]: {commands}\n"
            f"The original command will be returned as is.",
            exc_info=True,
        )
        # If parsing fails, return the original commands
        return [commands]

    result: list[str] = []
    last_end = 0

    for node in parsed:
        start, end = node.pos

        # Include any text between the last command and this one
        if start > last_end:
            between = commands[last_end:start]
            logger.debug(f"BASH PARSING between: {between}")
            if result:
                result[-1] += between.rstrip()
            elif between.strip():
                # THIS SHOULD NOT HAPPEN
                result.append(between.rstrip())

        # Extract the command, preserving original formatting
        command = commands[start:end].rstrip()
        logger.debug(f"BASH PARSING command: {command}")
        result.append(command)

        last_end = end

    # Add any remaining text after the last command to the last command
    remaining = commands[last_end:].rstrip()
    logger.debug(f"BASH PARSING remaining: {remaining}")
    if last_end < len(commands) and result:
        result[-1] += remaining
        logger.debug(f"BASH PARSING result[-1] += remaining: {result[-1]}")
    elif last_end < len(commands):
        if remaining:
            result.append(remaining)
            logger.debug(f"BASH PARSING result.append(remaining): {result[-1]}")
    return result


def escape_bash_special_chars(command: str) -> str:
    r"""Escapes characters that have different interpretations in bash vs python.
    Specifically handles escape sequences like \;, \|, \&, etc.
    """
    if command.strip() == "":
        return ""

    try:
        parts = []
        last_pos = 0

        def visit_node(node: Any) -> None:
            nonlocal last_pos
            if (
                node.kind == "redirect"
                and hasattr(node, "heredoc")
                and node.heredoc is not None
            ):
                # We're entering a heredoc - preserve everything as-is until we see EOF
                # Store the heredoc end marker (usually 'EOF' but could be different)
                between = command[last_pos : node.pos[0]]
                parts.append(between)
                # Add the heredoc start marker
                parts.append(command[node.pos[0] : node.heredoc.pos[0]])
                # Add the heredoc content as-is
                parts.append(command[node.heredoc.pos[0] : node.heredoc.pos[1]])
                last_pos = node.pos[1]
                return

            if node.kind == "word":
                # Get the raw text between the last position and current word
                between = command[last_pos : node.pos[0]]
                word_text = command[node.pos[0] : node.pos[1]]

                # Add the between text, escaping special characters
                between = re.sub(r"\\([;&|><])", r"\\\\\1", between)
                parts.append(between)

                # Check if word_text is a quoted string or command substitution
                if (
                    (word_text.startswith('"') and word_text.endswith('"'))
                    or (word_text.startswith("'") and word_text.endswith("'"))
                    or (word_text.startswith("$(") and word_text.endswith(")"))
                    or (word_text.startswith("`") and word_text.endswith("`"))
                ):
                    # Preserve quoted strings, command substitutions, and heredoc content as-is
                    parts.append(word_text)
                else:
                    # Escape special chars in unquoted text
                    word_text = re.sub(r"\\([;&|><])", r"\\\\\1", word_text)
                    parts.append(word_text)

                last_pos = node.pos[1]
                return

            # Visit child nodes
            if hasattr(node, "parts"):
                for part in node.parts:
                    visit_node(part)

        # Process all nodes in the AST
        nodes = list(bashlex.parse(command))
        for node in nodes:
            between = command[last_pos : node.pos[0]]
            between = re.sub(r"\\([;&|><])", r"\\\\\1", between)
            parts.append(between)
            last_pos = node.pos[0]
            visit_node(node)

        # Handle any remaining text after the last word
        remaining = command[last_pos:]
        parts.append(remaining)
        return "".join(parts)
    except (bashlex.errors.ParsingError, NotImplementedError, TypeError):
        logger.debug(
            f"Failed to parse bash commands for special characters escape\n"
            f"[input]: {command}\n"
            f"The original command will be returned as is.",
            exc_info=True,
        )
        return command


class BashCommandStatus(Enum):
    CONTINUE = "continue"
    COMPLETED = "completed"
    NO_CHANGE_TIMEOUT = "no_change_timeout"
    HARD_TIMEOUT = "hard_timeout"


def _remove_command_prefix(
    command_output: str,
    command: str,
    *,
    preserve_output_whitespace: bool = False,
) -> str:
    if not preserve_output_whitespace:
        return command_output.lstrip().removeprefix(command.lstrip()).lstrip()

    # A terminal echoes the submitted command before stdout. Remove only that
    # exact echo and its line ending; leading whitespace in actual stdout is
    # part of OpenCode's raw shell body.
    candidate = command_output.lstrip('\r\n')
    if not candidate.startswith(command):
        return command_output
    candidate = candidate[len(command) :]
    if candidate.startswith('\r\n'):
        return candidate[2:]
    if candidate.startswith('\n'):
        return candidate[1:]
    return candidate


class BashSession:
    POLL_INTERVAL = 0.05
    HISTORY_LIMIT = 10_000
    PS1 = CmdOutputMetadata.to_ps1_prompt()

    def __init__(
        self,
        work_dir: str,
        username: str | None = None,
        no_change_timeout_seconds: int = 30,
        max_memory_mb: int | None = None,
    ):
        self.NO_CHANGE_TIMEOUT_SECONDS = no_change_timeout_seconds
        self.work_dir = work_dir
        self.username = username
        self._initialized = False
        self.max_memory_mb = max_memory_mb
        self.memory_monitor: TmuxMemoryMonitor | None = None
        self.session: libtmux.Session | None = None
        self._closed = True
        self._closing = False
        self._recovery_failed = False

    def initialize(self) -> None:
        self.server = libtmux.Server()
        _shell_command = "/bin/bash"
        if SU_TO_USER and self.username in list(
            filter(None, [RUNTIME_USERNAME, "root", "openhands"])
        ):
            # This starts a non-login (new) shell for the given user
            _shell_command = f"su {self.username} -"

        # FIXME: we will introduce memory limit using sysbox-runc in coming PR
        # # otherwise, we are running as the CURRENT USER (e.g., when running LocalRuntime)
        # if self.max_memory_mb is not None:
        #     window_command = (
        #         f'prlimit --as={self.max_memory_mb * 1024 * 1024} {_shell_command}'
        #     )
        # else:
        window_command = _shell_command

        logger.debug(
            f"Initializing bash session in {self.work_dir} with command: {window_command}"
        )
        session_name = f"openhands-{self.username}-{uuid.uuid4()}"
        self.session = self.server.new_session(
            session_name=session_name,
            start_directory=self.work_dir,  # This parameter is supported by libtmux
            kill_session=True,
            x=1000,
            y=1000,
        )
        self._closed = False
        try:
            tmux_memory_limit = int(os.getenv("TMUX_MEMORY_LIMIT", "32768"))
            self.memory_monitor = TmuxMemoryMonitor(
                self.server, limit_mb=tmux_memory_limit
            )
            self.memory_monitor.start()

            # Set history limit to a large number to avoid losing history
            # https://unix.stackexchange.com/questions/43414/unlimited-history-in-tmux
            self.session.set_option(
                "history-limit", str(self.HISTORY_LIMIT), global_=True
            )
            self.session.history_limit = self.HISTORY_LIMIT
            # We need to create a new pane because the initial pane's history limit is (default) 2000
            _initial_window = self.session.active_window
            self.window = self.session.new_window(
                window_name="bash",
                window_shell=window_command,
                start_directory=self.work_dir,  # This parameter is supported by libtmux
            )
            self.pane = self.window.active_pane
            logger.debug(
                f"pane: {self.pane}; history_limit: {self.session.history_limit}"
            )
            _initial_window.kill()

            # Disable interactive history expansion before accepting agent
            # commands. Otherwise valid code inside double quotes (for example
            # JavaScript ``!!value`` or ``!exclude``) is rewritten by Bash.
            # Also configure Bash to use a simple PS1 and disable PS2.
            self.pane.send_keys(
                'set +H; '
                f'export PROMPT_COMMAND=\'export PS1="{self.PS1}"\'; '
                'export PS2=""'
            )
            if self._wait_for_prompt(10.0) is None:
                raise RuntimeError(
                    "Bash session did not expose a valid configured prompt during "
                    "initialization"
                )
            self._clear_screen()
            if self._wait_for_prompt(10.0) is None:
                raise RuntimeError(
                    "Bash session prompt was not ready after initialization cleanup"
                )

            # Store the last command for interactive input handling
            self.prev_status: BashCommandStatus | None = None
            self.prev_output: str = ""
            logger.debug(f"Bash session initialized with work dir: {self.work_dir}")

            # Maintain the current working directory
            self._cwd = os.path.abspath(self.work_dir)
            self._initialized = True
            self._recovery_failed = False
        except BaseException:
            # Initialization failures must not leave a tmux session or monitor
            # behind. A cleanup failure is secondary and must not replace the
            # exception that made initialization fail.
            try:
                self.close()
            except BaseException:
                try:
                    logger.exception(
                        "Failed to clean up a partially initialized bash session"
                    )
                except BaseException:
                    # Logging can be unavailable during interpreter shutdown.
                    pass
            raise

    def __del__(self) -> None:
        """Ensure the session is closed when the object is destroyed."""
        try:
            self.close()
        except BaseException:
            try:
                logger.exception("Failed to close bash session during finalization")
            except BaseException:
                # Destructors must never surface failures, including while the
                # interpreter is tearing down module globals.
                pass

    def _get_pane_content(self, preserve_trailing: bool = False) -> str:
        """Capture the current pane content and update the buffer."""
        capture_args = ['capture-pane', '-J']
        if preserve_trailing:
            # tmux strips trailing spaces unless capture-pane is given -N.
            capture_args.append('-N')
        capture_args.extend(('-pS', '-'))
        content = "\n".join(
            map(
                # avoid double newlines
                lambda line: (
                    line.rstrip('\r\n') if preserve_trailing else line.rstrip()
                ),
                self.pane.cmd(*capture_args).stdout,
            )
        )
        return content

    def close(self) -> None:
        """Clean up the session and leave it permanently unusable.

        Teardown is idempotent after success and retryable after failure. The
        shell rejects execution immediately, but retains references only to
        resources whose cleanup failed so a later call can retry them. If both
        cleanup operations fail, the first failure is raised after the tmux
        kill has still been tried.
        """
        if getattr(self, "_closed", True):
            return
        if getattr(self, "_closing", False):
            return

        memory_monitor = getattr(self, "memory_monitor", None)
        session = getattr(self, "session", None)

        # Reject commands before invoking third-party cleanup hooks. A hook can
        # fail or call back into close(), so guard against re-entrant teardown.
        self._initialized = False
        self._closing = True

        cleanup_error: BaseException | None = None
        try:
            if memory_monitor is not None:
                try:
                    memory_monitor.stop()
                except BaseException as error:
                    cleanup_error = error
                else:
                    self.memory_monitor = None

            if session is not None:
                try:
                    session.kill()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
                    else:
                        try:
                            logger.exception(
                                "Failed to kill tmux session after monitor cleanup failed"
                            )
                        except BaseException:
                            pass
                else:
                    self.session = None
        finally:
            self._closing = False
            self._closed = (
                getattr(self, "memory_monitor", None) is None
                and getattr(self, "session", None) is None
            )

        if cleanup_error is not None:
            raise cleanup_error

    @property
    def cwd(self) -> str:
        return self._cwd

    @staticmethod
    def _find_shell_proc(pane_proc: psutil.Process) -> psutil.Process:
        """Walk down through single-child shell wrappers (e.g. su → bash)
        to find the actual shell process.

        Returns the deepest shell process in the chain. If the pane_proc
        itself is the shell, returns it unchanged.
        """
        shell_proc = pane_proc
        while True:
            kids = shell_proc.children(recursive=False)
            if not kids:
                break
            first = kids[0]
            try:
                name = first.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if len(kids) == 1 and name in ('bash', 'sh', 'zsh', 'fish', 'dash'):
                shell_proc = first
            else:
                break
        return shell_proc

    def _is_special_key(self, command: str) -> bool:
        """Check if the command is a special key."""
        # Special keys are of the form C-<key>
        _command = command.strip()
        return _command.startswith("C-") and len(_command) == 3

    def _send_keys_checked(self, keys: str, enter: bool = True) -> None:
        """Send keys to the tmux pane and log any errors.

        Unlike pane.send_keys(), this method checks the return value of
        the underlying tmux command and logs errors instead of silently
        swallowing them.
        """
        result = self.pane.cmd("send-keys", keys)
        if result.stderr:
            logger.error(
                f"tmux send-keys {keys!r} FAILED: {result.stderr} "
                f"(returncode={result.returncode}, pane_id={self.pane.pane_id})"
            )
        if enter:
            result = self.pane.cmd("send-keys", "Enter")
            if result.stderr:
                logger.error(
                    f"tmux send-keys Enter FAILED: {result.stderr} "
                    f"(returncode={result.returncode})"
                )

    def _kill_pane_processes(self) -> bool:
        """Kill foreground command processes in the tmux pane via SIGKILL.

        The pane process tree is: pane_init (su) → shell (bash) → commands.
        We must kill only the commands, NOT the shell, or the pane dies.
        Walks down to the deepest shell and kills its children.
        """
        try:
            pane_pid_str = self.pane.cmd(
                "display-message", "-p", "#{pane_pid}"
            ).stdout[0]
            pane_pid = int(pane_pid_str)
            print(f"[BASH_SIGNAL] Pane PID: {pane_pid}", flush=True)

            proc = psutil.Process(pane_pid)
            shell_proc = self._find_shell_proc(proc)

            targets = shell_proc.children(recursive=True)
            if not targets:
                print(f"[BASH_SIGNAL] Shell PID {shell_proc.pid} has no command children", flush=True)
                return False

            print(f"[BASH_SIGNAL] Shell: {shell_proc.pid} ({shell_proc.name()}) → killing {len(targets)} processes: {[(t.pid, t.name()) for t in targets]}", flush=True)
            for target in targets:
                try:
                    target.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            _, alive = psutil.wait_procs(targets, timeout=3)
            if alive:
                logger.error(
                    "Failed to reap all command processes after SIGKILL: %s",
                    [target.pid for target in alive],
                )
                return False
            print(f"[BASH_SIGNAL] Kill complete", flush=True)
            return True
        except (psutil.NoSuchProcess, IndexError, ValueError) as e:
            print(f"[BASH_SIGNAL] Kill failed: {type(e).__name__}: {e}", flush=True)
            return False
        except Exception as e:
            print(f"[BASH_SIGNAL] Kill failed unexpectedly: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def _clear_screen(self) -> None:
        """Clear the tmux pane screen and history."""
        self.pane.send_keys("C-l", enter=False)
        time.sleep(0.1)
        self.pane.cmd("clear-history")

    def _wait_for_prompt(self, timeout_seconds: float) -> str | None:
        """Wait until the pane has returned to an OpenHands shell prompt."""
        deadline = time.monotonic() + max(timeout_seconds, 0)
        while True:
            pane_content = self._get_pane_content()
            if self._has_terminal_prompt(pane_content):
                return pane_content
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.POLL_INTERVAL)

    @staticmethod
    def _has_terminal_prompt(
        pane_content: str,
        ps1_matches: list[re.Match[str]] | None = None,
    ) -> bool:
        """Return whether the pane ends at a valid configured shell prompt."""
        stripped_content = pane_content.rstrip()
        if not stripped_content.endswith(CMD_OUTPUT_PS1_END.rstrip()):
            return False
        if ps1_matches is None:
            # Initialization captures the echoed PROMPT_COMMAND assignment,
            # whose escaped JSON resembles a prompt but is not one. Parse it
            # silently while waiting instead of logging a warning per poll.
            matches = []
            for match in CMD_OUTPUT_METADATA_PS1_REGEX.finditer(pane_content):
                try:
                    json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
                matches.append(match)
        else:
            matches = ps1_matches
        return bool(matches and matches[-1].end() == len(stripped_content))

    def _mark_timeout_recovered(self, pane_content: str) -> None:
        """Reset bookkeeping after an interrupted command reaches the prompt."""
        ps1_matches = CmdOutputMetadata.matches_ps1_metadata(pane_content)
        if ps1_matches:
            metadata = CmdOutputMetadata.from_ps1_match(ps1_matches[-1])
            if metadata.working_dir:
                self._cwd = metadata.working_dir
        self.prev_status = BashCommandStatus.COMPLETED
        self.prev_output = ""
        self._ready_for_next_command()

    def _fail_timeout_recovery(self) -> bool:
        """Permanently reject a shell whose timed-out command was not reclaimed."""
        self._recovery_failed = True
        try:
            self.close()
        except Exception:
            logger.exception("Failed to close an unrecoverable bash session")
        return False

    def recover_after_timeout(self) -> bool:
        """Reclaim a terminal left busy by a timed-out command.

        A soft timeout intentionally leaves the process available for an
        ``is_input`` follow-up. When Codex instead submits a new command, or a
        hard timeout expires, this method interrupts and then kills only the
        command process tree. If the pane still cannot return to a prompt, the
        tmux session is closed and permanently rejected so later output cannot
        be attributed to the wrong command or run in a fresh, bare shell.

        Returns ``True`` when the existing shell was recovered and ``False``
        when it had to be closed.
        """
        if self.prev_status not in {
            BashCommandStatus.NO_CHANGE_TIMEOUT,
            BashCommandStatus.HARD_TIMEOUT,
        }:
            return True

        try:
            pane_content = self._wait_for_prompt(0)
            if pane_content is None:
                self._send_keys_checked("C-c", enter=False)
                pane_content = self._wait_for_prompt(1.0)

            if pane_content is None:
                self._kill_pane_processes()
                pane_content = self._wait_for_prompt(3.0)

            if pane_content is not None:
                self._mark_timeout_recovered(pane_content)
                return True
        except Exception:
            # Log the recovery error before attempting cleanup so a cleanup
            # exception cannot replace the original diagnostic.
            logger.exception("Exception while recovering timed-out bash command")
            return self._fail_timeout_recovery()

        logger.error("Timed-out command did not release the pane; closing it")
        return self._fail_timeout_recovery()

    @staticmethod
    def _prepare_codex_command(
        action: CmdRunAction,
        command: str,
        *,
        enable_pipefail: bool = False,
    ) -> str:
        """Apply Codex ``workdir`` and login-shell options to a command."""
        if action.is_input:
            return command

        script = command
        if enable_pipefail:
            script = f"set -o pipefail; {script}"
        if action.cwd is not None:
            # Put the caller's script on its own line so valid trailing shell
            # syntax (for example ``&``, ``;``, or a heredoc terminator) is not
            # followed by an injected semicolon or brace.
            script = f"builtin cd -- {shlex.quote(action.cwd)} || exit\n{script}"

        # A workdir alone should inherit the already initialized environment
        # and must not mutate the persistent shell's cwd.
        if action.login is None:
            return f"(\n{script}\n)" if action.cwd is not None else script

        shell_flag = "-lc" if action.login else "-c"
        return f"/bin/bash {shell_flag} {shlex.quote(script)}"

    def _get_command_output(
        self,
        command: str,
        raw_command_output: str,
        metadata: CmdOutputMetadata,
        continue_prefix: str = "",
        preserve_trailing: bool = False,
    ) -> str:
        """Get the command output with the previous command output removed.

        Args:
            command: The command that was executed.
            raw_command_output: The raw output from the command.
            metadata: The metadata object to store prefix/suffix in.
            continue_prefix: The prefix to add to the command output if it's a continuation of the previous command.
            preserve_trailing: Return trailing whitespace unchanged for tool
                protocols whose output body is byte-for-byte significant.
        """
        # remove the previous command output from the new output if any
        if self.prev_output:
            command_output = raw_command_output.removeprefix(self.prev_output)
            metadata.prefix = continue_prefix
        else:
            command_output = raw_command_output
        self.prev_output = raw_command_output  # update current command output anyway
        command_output = _remove_command_prefix(
            command_output,
            command,
            preserve_output_whitespace=preserve_trailing,
        )
        return command_output if preserve_trailing else command_output.rstrip()

    def _handle_completed_command(
        self,
        command: str,
        pane_content: str,
        ps1_matches: list[re.Match],
        hidden: bool,
        opencode_result: bool = False,
        observation_command: str | None = None,
    ) -> CmdOutputObservation:
        is_special_key = self._is_special_key(command)
        assert len(ps1_matches) >= 1, (
            f"Expected at least one PS1 metadata block, but got {len(ps1_matches)}.\n"
            f"---FULL OUTPUT---\n{pane_content!r}\n---END OF OUTPUT---"
        )
        metadata = CmdOutputMetadata.from_ps1_match(ps1_matches[-1])

        # Special case where the previous command output is truncated due to history limit
        # We should get the content BEFORE the last PS1 prompt
        get_content_before_last_match = bool(len(ps1_matches) == 1)

        # Update the current working directory if it has changed
        if metadata.working_dir != self._cwd and metadata.working_dir:
            logger.debug(
                f"directory_changed: {self._cwd}; {metadata.working_dir}; {command}"
            )
            self._cwd = metadata.working_dir

        logger.debug(f"COMMAND OUTPUT: {pane_content}")
        # Extract the command output between the two PS1 prompts
        raw_command_output = self._combine_outputs_between_matches(
            pane_content,
            ps1_matches,
            get_content_before_last_match=get_content_before_last_match,
            preserve_output_whitespace=opencode_result,
            include_after_last_match=False,
        )

        if get_content_before_last_match:
            # Count the number of lines in the truncated output
            num_lines = len(raw_command_output.splitlines())
            metadata.prefix = f"[Previous command outputs are truncated. Showing the last {num_lines} lines of the output below.]\n"

        metadata.suffix = (
            f"\n[The command completed with exit code {metadata.exit_code}.]"
            if not is_special_key
            else f"\n[The command completed with exit code {metadata.exit_code}. CTRL+{command[-1].upper()} was sent.]"
        )
        command_output = self._get_command_output(
            command,
            raw_command_output,
            metadata,
            preserve_trailing=opencode_result,
        )
        self.prev_status = BashCommandStatus.COMPLETED
        self.prev_output = ""  # Reset previous command output
        self._ready_for_next_command()
        return CmdOutputObservation(
            content=command_output,
            command=observation_command or command,
            metadata=metadata,
            hidden=hidden,
            max_content_size=None if opencode_result else MAX_CMD_OUTPUT_SIZE,
        )

    def _handle_nochange_timeout_command(
        self,
        command: str,
        pane_content: str,
        ps1_matches: list[re.Match],
        opencode_result: bool = False,
        observation_command: str | None = None,
    ) -> CmdOutputObservation:
        self.prev_status = BashCommandStatus.NO_CHANGE_TIMEOUT
        if len(ps1_matches) != 1:
            logger.warning(
                "Expected exactly one PS1 metadata block BEFORE the execution of a command, "
                f"but got {len(ps1_matches)} PS1 metadata blocks:\n---\n{pane_content!r}\n---"
            )
        raw_command_output = self._combine_outputs_between_matches(
            pane_content,
            ps1_matches,
            preserve_output_whitespace=opencode_result,
        )
        metadata = CmdOutputMetadata()  # No metadata available
        metadata.suffix = (
            f"\n[The command has no new output after {self.NO_CHANGE_TIMEOUT_SECONDS} seconds. "
            f"{TIMEOUT_MESSAGE_TEMPLATE}]"
        )
        command_output = self._get_command_output(
            command,
            raw_command_output,
            metadata,
            continue_prefix="[Below is the output of the previous command.]\n",
            preserve_trailing=opencode_result,
        )
        return CmdOutputObservation(
            content=command_output,
            command=observation_command or command,
            metadata=metadata,
            max_content_size=None if opencode_result else MAX_CMD_OUTPUT_SIZE,
        )

    def _handle_hard_timeout_command(
        self,
        command: str,
        pane_content: str,
        ps1_matches: list[re.Match],
        timeout: float,
        opencode_result: bool = False,
        observation_command: str | None = None,
    ) -> CmdOutputObservation:
        self.prev_status = BashCommandStatus.HARD_TIMEOUT
        if len(ps1_matches) != 1:
            logger.warning(
                "Expected exactly one PS1 metadata block BEFORE the execution of a command, "
                f"but got {len(ps1_matches)} PS1 metadata blocks:\n---\n{pane_content!r}\n---"
            )
        raw_command_output = self._combine_outputs_between_matches(
            pane_content,
            ps1_matches,
            preserve_output_whitespace=opencode_result,
        )
        metadata = CmdOutputMetadata()  # No metadata available
        metadata.suffix = (
            f"\n[The command timed out after {timeout} seconds. "
            f"{TIMEOUT_MESSAGE_TEMPLATE}]"
        )
        command_output = self._get_command_output(
            command,
            raw_command_output,
            metadata,
            continue_prefix="[Below is the output of the previous command.]\n",
            preserve_trailing=opencode_result,
        )

        return CmdOutputObservation(
            command=observation_command or command,
            content=command_output,
            metadata=metadata,
            max_content_size=None if opencode_result else MAX_CMD_OUTPUT_SIZE,
        )

    def _ready_for_next_command(self) -> None:
        """Reset the content buffer for a new command."""
        # Clear the current content
        self._clear_screen()

    def _combine_outputs_between_matches(
        self,
        pane_content: str,
        ps1_matches: list[re.Match],
        get_content_before_last_match: bool = False,
        preserve_output_whitespace: bool = False,
        include_after_last_match: bool = True,
    ) -> str:
        """Combine all outputs between PS1 matches.

        Args:
            pane_content: The full pane content containing PS1 prompts and command outputs
            ps1_matches: List of regex matches for PS1 prompts
            get_content_before_last_match: when there's only one PS1 match, whether to get
                the content before the last PS1 prompt (True) or after the last PS1 prompt (False)
            preserve_output_whitespace: remove only the prompt protocol's
                separator newlines without trimming command output
            include_after_last_match: include content typed or emitted after
                the last prompt; completed commands leave this disabled

        Returns:
            Combined string of all outputs between matches
        """
        if len(ps1_matches) == 1:
            if get_content_before_last_match:
                # The command output is the content before the last PS1 prompt
                end = ps1_matches[0].start()
                if (
                    preserve_output_whitespace
                    and end > 0
                    and pane_content[end - 1] == "\n"
                ):
                    end -= 1
                return pane_content[:end]
            else:
                # The command output is the content after the last PS1 prompt
                return pane_content[ps1_matches[0].end() + 1 :]
        elif len(ps1_matches) == 0:
            return pane_content

        if preserve_output_whitespace:
            output_segments: list[str] = []
            for index in range(len(ps1_matches) - 1):
                start = ps1_matches[index].end()
                if pane_content[start : start + 1] == "\n":
                    start += 1
                end = ps1_matches[index + 1].start()
                if end > start and pane_content[end - 1] == "\n":
                    end -= 1
                output_segments.append(pane_content[start:end])
            if include_after_last_match:
                start = ps1_matches[-1].end()
                if pane_content[start : start + 1] == "\n":
                    start += 1
                output_segments.append(pane_content[start:])
            combined_output = "".join(output_segments)
            logger.debug(f"COMBINED OUTPUT: {combined_output}")
            return combined_output

        combined_output = ""
        for i in range(len(ps1_matches) - 1):
            # Extract content between current and next PS1 prompt
            output_segment = pane_content[
                ps1_matches[i].end() + 1 : ps1_matches[i + 1].start()
            ]
            combined_output += output_segment + "\n"
        # Add the content after the last PS1 prompt
        combined_output += pane_content[ps1_matches[-1].end() + 1 :]
        logger.debug(f"COMBINED OUTPUT: {combined_output}")
        return combined_output

    def execute(self, action: CmdRunAction) -> CmdOutputObservation | ErrorObservation:
        """Execute a command in the bash session."""
        if self._recovery_failed:
            return ErrorObservation(
                content=(
                    "ERROR: The shell session could not be recovered after a "
                    "timed-out command and was closed. No command was executed."
                ),
                error_id="SHELL_RECOVERY_FAILED",
            )
        if not self._initialized or getattr(self, "_closed", False):
            raise RuntimeError("Bash session is not initialized")

        # Strip the command of any leading/trailing whitespace
        logger.debug(f"RECEIVED ACTION: {action}")
        requested_command = action.command.strip()
        command = requested_command
        is_input: bool = action.is_input
        result_format = (
            getattr(action.tool_call_metadata, 'tool_result_format', None)
            if action.tool_call_metadata is not None
            else None
        )
        codex_result = result_format == 'codex'
        opencode_result = (
            action.tool_call_metadata is not None
            and result_format in {'opencode', 'codex'}
        )

        print(f"COMMAND: {command}")
        print(f"IS INPUT: {is_input}")
        print(f"================================================")

        # If the previous command is not completed, we need to check if the command is empty
        if self.prev_status not in {
            BashCommandStatus.CONTINUE,
            BashCommandStatus.NO_CHANGE_TIMEOUT,
            BashCommandStatus.HARD_TIMEOUT,
        }:
            if command == "":
                return CmdOutputObservation(
                    content="ERROR: No previous running command to retrieve logs from.",
                    command="",
                    metadata=CmdOutputMetadata(),
                )
            if is_input:
                return CmdOutputObservation(
                    content="ERROR: No previous running command to interact with.",
                    command="",
                    metadata=CmdOutputMetadata(),
                )

        # A no-change timeout deliberately leaves the process available for
        # interactive input. If Codex submits a normal command instead, reclaim
        # the terminal and execute that command in this same tool call rather
        # than returning an endless series of "previous command" failures.
        if (
            codex_result
            and self.prev_status
            in {
                BashCommandStatus.NO_CHANGE_TIMEOUT,
                BashCommandStatus.HARD_TIMEOUT,
            }
            and command
            and not is_input
        ):
            if not self.recover_after_timeout():
                return ErrorObservation(
                    content=(
                        "ERROR: The previous timed-out command could not be "
                        "terminated cleanly. The shell session was closed and "
                        "the new command was not executed."
                    ),
                    error_id="SHELL_RECOVERY_FAILED",
                )

        # Check if the command is a single command or multiple commands
        splited_commands = split_bash_commands(command)
        if len(splited_commands) > 1:
            return ErrorObservation(
                content=(
                    f"ERROR: Cannot execute multiple commands at once.\n"
                    f"Please run each command separately OR chain them into a single command via && or ;\n"
                    f"Provided commands:\n{'\n'.join(f'({i + 1}) {cmd}' for i, cmd in enumerate(splited_commands))}"
                )
            )

        # Check if the command is blacklisted (only for non-input commands).
        # Trusted harness commands (setup/reset/eval scaffolding in
        # run_infer.py) set `bypass_blacklist=True` so the anti-cheat rules,
        # which are meant to constrain the *agent's* rollout, don't block the
        # harness's own git plumbing (e.g. `git merge-base --is-ancestor`).
        # The agent can never set this flag, so the blacklist stays enforced
        # for every agent-issued command.
        if not is_input and command and not getattr(action, 'bypass_blacklist', False):
            blacklist_result = check_command_blacklist(command)
            if blacklist_result.is_blocked:
                logger.warning(f"Command blocked by blacklist: {command!r}")
                return ErrorObservation(
                    content=blacklist_result.feedback,
                    error_id="COMMAND_BLACKLISTED",
                )
            if "kill" in command or "rm" in command:
                print(
                    f"[POSSIBLE BLACKLIST]Command {command} bypassed blacklist check",
                    flush=True,
                )

        # Get initial state before sending command
        initial_pane_output = self._get_pane_content(
            preserve_trailing=opencode_result
        )
        initial_ps1_matches = CmdOutputMetadata.matches_ps1_metadata(
            initial_pane_output
        )
        initial_ps1_count = len(initial_ps1_matches)
        completed_before_empty_poll = (
            is_input
            and command == ""
            and self.prev_status
            in {
                BashCommandStatus.NO_CHANGE_TIMEOUT,
                BashCommandStatus.HARD_TIMEOUT,
            }
            and initial_ps1_count > 0
            and initial_pane_output.rstrip().endswith(
                CMD_OUTPUT_PS1_END.rstrip()
            )
        )
        logger.debug(f"Initial PS1 count: {initial_ps1_count}")

        start_time = time.time()
        last_change_time = start_time
        last_pane_output = (
            initial_pane_output  # Use initial output as the starting point
        )
        observed_nonterminal_pane = False

        # When prev command is still running, and we are trying to send a new command
        if (
            self.prev_status
            in {
                BashCommandStatus.HARD_TIMEOUT,
                BashCommandStatus.NO_CHANGE_TIMEOUT,
            }
            and not last_pane_output.rstrip().endswith(
                CMD_OUTPUT_PS1_END.rstrip()
            )  # prev command is not completed
            and not is_input
            and command != ""  # not input and not empty command
        ):
            _ps1_matches = CmdOutputMetadata.matches_ps1_metadata(last_pane_output)
            # Use initial_ps1_matches if _ps1_matches is empty, otherwise use _ps1_matches
            # This handles the case where the prompt might be scrolled off screen but existed before
            current_matches_for_output = (
                _ps1_matches if _ps1_matches else initial_ps1_matches
            )
            raw_command_output = self._combine_outputs_between_matches(
                last_pane_output,
                current_matches_for_output,
                preserve_output_whitespace=opencode_result,
            )
            metadata = CmdOutputMetadata()  # No metadata available
            metadata.suffix = (
                f'\n[Your command "{command}" is NOT executed. '
                "The previous command is still running - You CANNOT send new commands until the previous command is completed. "
                "By setting `is_input` to `true`, you can interact with the current process: "
                f"{TIMEOUT_MESSAGE_TEMPLATE}]"
            )
            logger.debug(f"PREVIOUS COMMAND OUTPUT: {raw_command_output}")
            command_output = self._get_command_output(
                command,
                raw_command_output,
                metadata,
                continue_prefix="[Below is the output of the previous command.]\n",
                preserve_trailing=opencode_result,
            )
            return CmdOutputObservation(
                command=command,
                content=command_output,
                metadata=metadata,
                hidden=getattr(action, "hidden", False),
                max_content_size=None if opencode_result else MAX_CMD_OUTPUT_SIZE,
            )

        # Send actual command/inputs to the pane
        is_interrupt_signal = (
            is_input
            and self._is_special_key(command)
            and command.strip() in ("C-c", "C-z", "C-d")
        )
        if command != "":
            is_special_key = self._is_special_key(command)
            if is_input:
                logger.debug(f"SENDING INPUT TO RUNNING PROCESS: {command!r}")
                self._send_keys_checked(command, enter=not is_special_key)
            else:
                command = self._prepare_codex_command(
                    action,
                    command,
                    enable_pipefail=codex_result,
                )
                if result_format == 'opencode':
                    command = f"set -o pipefail; {command}"
                # convert command to raw string
                command = escape_bash_special_chars(command)
                logger.debug(f"SENDING COMMAND: {command!r}")
                self._send_keys_checked(command, enter=not is_special_key)

        # For interrupt signals (C-c, C-z), use a shorter retry interval.
        # Many processes (e.g. pytest, long-running test suites) need multiple
        # SIGINTs: the first triggers graceful shutdown, the second aborts.
        SIGNAL_RETRY_INTERVAL = 1  # seconds between resending the signal
        MAX_SIGNAL_RETRIES = 5
        signal_retry_count = 0
        last_signal_send_time = time.time()

        # Loop until the command completes or times out
        while should_continue():
            _start_time = time.time()
            logger.debug(f"GETTING PANE CONTENT at {_start_time}")
            cur_pane_output = self._get_pane_content(
                preserve_trailing=opencode_result
            )
            logger.debug(
                f"PANE CONTENT GOT after {time.time() - _start_time:.2f} seconds"
            )
            cur_pane_lines = cur_pane_output.split("\n")
            if len(cur_pane_lines) <= 20:
                logger.debug("PANE_CONTENT: {cur_pane_output}")
            else:
                logger.debug(f"BEGIN OF PANE CONTENT: {cur_pane_lines[:10]}")
                logger.debug(f"END OF PANE CONTENT: {cur_pane_lines[-10:]}")
            ps1_matches = CmdOutputMetadata.matches_ps1_metadata(cur_pane_output)
            current_ps1_count = len(ps1_matches)
            has_terminal_prompt = self._has_terminal_prompt(
                cur_pane_output, ps1_matches
            )
            if not has_terminal_prompt:
                observed_nonterminal_pane = True

            if cur_pane_output != last_pane_output:
                last_pane_output = cur_pane_output
                last_change_time = time.time()
                logger.debug(f"CONTENT UPDATED DETECTED at {last_change_time}")

            # 1) Execution completed:
            # Condition 1: A new prompt has appeared since the command started.
            # Condition 2: The initial prompt scrolled out of history, but the
            # pane was observed running and subsequently returned to a prompt.
            # Condition 3: history rolled over before the first poll, leaving
            # one changed terminal prompt. In every case the valid prompt must
            # be terminal: a configured prompt above an echoed, still-running
            # command is not completion.
            if has_terminal_prompt and (
                current_ps1_count > initial_ps1_count
                or completed_before_empty_poll
                or observed_nonterminal_pane
                or cur_pane_output != initial_pane_output
            ):
                return self._handle_completed_command(
                    command,
                    pane_content=cur_pane_output,
                    ps1_matches=ps1_matches,
                    hidden=getattr(action, "hidden", False),
                    opencode_result=opencode_result,
                    observation_command=requested_command,
                )

            # Timeout checks should only trigger if a new prompt hasn't appeared yet.

            # For interrupt signals, resend if no change after SIGNAL_RETRY_INTERVAL.
            # After all retries are exhausted, escalate to SIGKILL via PID.
            if is_interrupt_signal:
                if signal_retry_count < MAX_SIGNAL_RETRIES:
                    time_since_last_signal = time.time() - last_signal_send_time
                    if time_since_last_signal >= SIGNAL_RETRY_INTERVAL:
                        signal_retry_count += 1
                        logger.info(
                            f"Signal retry: resending {command!r} "
                            f"(attempt {signal_retry_count + 1}/{MAX_SIGNAL_RETRIES + 1}, "
                            f"no response for {time_since_last_signal:.1f}s)"
                        )
                        self._send_keys_checked(command, enter=False)
                        last_signal_send_time = time.time()
                        last_change_time = time.time()
                elif signal_retry_count == MAX_SIGNAL_RETRIES:
                    # All signal retries exhausted — escalate to SIGKILL
                    signal_retry_count += 1  # prevent re-entry
                    print(f"[BASH_SIGNAL] All {MAX_SIGNAL_RETRIES} retries exhausted. Escalating to SIGKILL.", flush=True)
                    logger.info(
                        f"All {MAX_SIGNAL_RETRIES} signal retries exhausted. "
                        f"Escalating to SIGKILL via PID."
                    )
                    killed = self._kill_pane_processes()
                    print(f"[BASH_SIGNAL] _kill_pane_processes returned: {killed}", flush=True)
                    if killed:
                        last_change_time = time.time()

            # 2) Execution timed out since there's no change in output
            # for a while (self.NO_CHANGE_TIMEOUT_SECONDS)
            # We ignore this if the command is *blocking*
            time_since_last_change = time.time() - last_change_time
            logger.debug(
                f"CHECKING NO CHANGE TIMEOUT ({self.NO_CHANGE_TIMEOUT_SECONDS}s): elapsed {time_since_last_change}. Action blocking: {action.blocking}"
            )
            if (
                not action.blocking
                and time_since_last_change >= self.NO_CHANGE_TIMEOUT_SECONDS
            ):
                return self._handle_nochange_timeout_command(
                    command,
                    pane_content=cur_pane_output,
                    ps1_matches=ps1_matches,
                    opencode_result=opencode_result,
                    observation_command=requested_command,
                )

            # 3) Execution timed out due to hard timeout
            elapsed_time = time.time() - start_time
            logger.debug(
                f"CHECKING HARD TIMEOUT ({action.timeout}s): elapsed {elapsed_time:.2f}"
            )
            if action.timeout and elapsed_time >= action.timeout:
                logger.debug("Hard timeout triggered.")
                return self._handle_hard_timeout_command(
                    command,
                    pane_content=cur_pane_output,
                    ps1_matches=ps1_matches,
                    timeout=action.timeout,
                    opencode_result=opencode_result,
                    observation_command=requested_command,
                )

            logger.debug(f"SLEEPING for {self.POLL_INTERVAL} seconds for next poll")
            time.sleep(self.POLL_INTERVAL)
        raise RuntimeError("Bash session was likely interrupted...")
