"""End-to-end runtime formatting tests for Codex ``shell_command`` bodies."""

import asyncio
import subprocess
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from openhands.agenthub.codex_agent.tool_output import format_shell_output
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.events.observation.commands import CmdOutputMetadata
from openhands.events.tool import ToolCallMetadata
from openhands.llm.tool_names import CODEX_SHELL_COMMAND_TOOL_NAME


def _metadata(*, model: str = "gpt-5.6-sol") -> ToolCallMetadata:
    return ToolCallMetadata(
        tool_call_id="call-1",
        function_name=CODEX_SHELL_COMMAND_TOOL_NAME,
        model_response={
            "id": "response-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                    }
                }
            ],
            "created": 0,
            "model": model,
            "object": "chat.completion",
        },
        total_calls_in_response=1,
        tool_result_format="codex",
    )


@pytest.fixture(scope="module")
def action_server() -> ModuleType:
    from openhands.runtime import action_execution_server

    return action_execution_server


@pytest.fixture
def executor(action_server: ModuleType):
    action_executor = object.__new__(action_server.ActionExecutor)
    action_executor.bash_session = SimpleNamespace(
        cwd="/workspace",
        execute=MagicMock(name="bash_execute"),
        recover_after_timeout=MagicMock(name="recover_after_timeout"),
    )
    return action_executor


def _run_shell(
    action_server: ModuleType,
    executor,
    action: CmdRunAction,
    raw_observation,
    *,
    started_at: float = 10.0,
    finished_at: float = 11.25,
    expect_recovery: bool = False,
):
    execute = AsyncMock(return_value=raw_observation)
    monotonic = MagicMock(side_effect=[started_at, finished_at])
    with (
        patch.object(action_server, "call_sync_from_async", execute),
        patch.object(
            action_server,
            "time",
            SimpleNamespace(monotonic=monotonic),
        ),
    ):
        result = asyncio.run(executor.run(action))

    expected_calls = [call(executor.bash_session.execute, action)]
    if expect_recovery:
        expected_calls.append(call(executor.bash_session.recover_after_timeout))
    assert execute.await_args_list == expected_calls
    assert monotonic.call_count == 2
    return result


def test_codex_shell_runtime_formats_exact_body_and_ignores_oh_annotations(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="example-command")
    action.tool_call_metadata = _metadata()
    raw = CmdOutputObservation(
        content="  leading\ntrailing  \n",
        command=action.command,
        metadata=CmdOutputMetadata(
            exit_code=7,
            prefix="[OpenHands prefix]\n",
            suffix="\n[The command completed with exit code 7.]",
        ),
        max_content_size=None,
    )

    result = _run_shell(action_server, executor, action, raw)

    assert result is raw
    assert result.content == (
        "Exit code: 7\nWall time: 1.3 seconds\nOutput:\n  leading\ntrailing  \n"
    )


def test_codex_shell_runtime_formats_timeout_exactly(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="sleep 2")
    action.set_hard_timeout(2.0)
    action.tool_call_metadata = _metadata(model="gpt-5.2")
    raw = CmdOutputObservation(
        content="partial",
        command=action.command,
        metadata=CmdOutputMetadata(
            suffix=(
                "\n[The command timed out after 2.0 seconds. "
                "The command is still running.]"
            )
        ),
        max_content_size=None,
    )

    result = _run_shell(
        action_server,
        executor,
        action,
        raw,
        finished_at=12.0406,
        expect_recovery=True,
    )

    assert result.content == (
        "Exit code: 124\n"
        "Wall time: 2 seconds\n"
        "Output:\n"
        "command timed out after 2040 milliseconds\n"
        "partial"
    )


@pytest.mark.parametrize(
    ("model", "raw_output"),
    [
        ("gpt-5.2", "b" * 10_001),
        ("gpt-5.6-sol", "t" * 40_001),
    ],
    ids=("byte-policy", "token-policy"),
)
def test_codex_shell_runtime_uses_model_truncation_policy(
    action_server: ModuleType,
    executor,
    model: str,
    raw_output: str,
) -> None:
    action = CmdRunAction(command="large-output")
    action.tool_call_metadata = _metadata(model=model)
    raw = CmdOutputObservation(
        content=raw_output,
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=0),
        max_content_size=None,
    )

    result = _run_shell(
        action_server,
        executor,
        action,
        raw,
        finished_at=10.0,
    )

    assert result.content == format_shell_output(
        raw_output,
        exit_code=0,
        duration_seconds=0,
        model_name=model,
    )
    assert "[... Observation truncated due to length ...]" not in result.content


def test_codex_shell_runtime_preserves_empty_output(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="true")
    action.tool_call_metadata = _metadata()
    raw = CmdOutputObservation(
        content="",
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=0),
        max_content_size=None,
    )

    result = _run_shell(
        action_server,
        executor,
        action,
        raw,
        finished_at=10.0,
    )

    assert result.content == "Exit code: 0\nWall time: 0 seconds\nOutput:\n"


def test_codex_shell_runtime_leaves_execution_errors_unchanged(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="rejected")
    action.tool_call_metadata = _metadata()
    raw = ErrorObservation("execution error: denied")

    result = _run_shell(action_server, executor, action, raw)

    assert result is raw
    assert result.content == "execution error: denied"


def test_unmarked_shell_runtime_keeps_generic_observation(
    action_server: ModuleType,
    executor,
) -> None:
    action = CmdRunAction(command="legacy-command")
    raw = CmdOutputObservation(
        content="legacy output",
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=3),
    )

    result = _run_shell(action_server, executor, action, raw)

    assert result is raw
    assert result.content == "legacy output"


def test_static_shell_session_is_closed_after_success(
    action_server: ModuleType,
) -> None:
    executor = object.__new__(action_server.ActionExecutor)
    executor.bash_session = SimpleNamespace(cwd="/persistent")
    static_session = SimpleNamespace(
        execute=MagicMock(name="static_execute"),
        close=MagicMock(name="static_close"),
    )
    executor._create_bash_session = MagicMock(return_value=static_session)
    action = CmdRunAction(command="pwd", cwd="/isolated", is_static=True)
    raw = CmdOutputObservation(
        content="/isolated",
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=0),
    )
    invoke = AsyncMock(side_effect=[raw, None])
    monotonic = MagicMock(side_effect=[10.0, 10.25])

    with (
        patch.object(action_server, "call_sync_from_async", invoke),
        patch.object(action_server, "time", SimpleNamespace(monotonic=monotonic)),
    ):
        result = asyncio.run(executor.run(action))

    assert result is raw
    executor._create_bash_session.assert_called_once_with("/isolated")
    assert invoke.await_args_list == [
        call(static_session.execute, action),
        call(static_session.close),
    ]


def test_static_shell_close_error_after_success_is_reported(
    action_server: ModuleType,
) -> None:
    executor = object.__new__(action_server.ActionExecutor)
    executor.bash_session = SimpleNamespace(cwd="/persistent")
    static_session = SimpleNamespace(
        execute=MagicMock(name="static_execute"),
        close=MagicMock(name="static_close"),
    )
    executor._create_bash_session = MagicMock(return_value=static_session)
    action = CmdRunAction(command="pwd", cwd="/isolated", is_static=True)
    raw = CmdOutputObservation(
        content="/isolated",
        command=action.command,
        metadata=CmdOutputMetadata(exit_code=0),
    )
    invoke = AsyncMock(side_effect=[raw, RuntimeError("close failed")])
    monotonic = MagicMock(side_effect=[10.0, 10.25])

    with (
        patch.object(action_server, "call_sync_from_async", invoke),
        patch.object(action_server, "time", SimpleNamespace(monotonic=monotonic)),
    ):
        result = asyncio.run(executor.run(action))

    assert isinstance(result, ErrorObservation)
    assert result.content == "close failed"
    assert invoke.await_args_list == [
        call(static_session.execute, action),
        call(static_session.close),
    ]


def test_static_shell_session_is_closed_after_execution_error(
    action_server: ModuleType,
) -> None:
    executor = object.__new__(action_server.ActionExecutor)
    executor.bash_session = SimpleNamespace(cwd="/persistent")
    static_session = SimpleNamespace(
        execute=MagicMock(name="static_execute"),
        close=MagicMock(name="static_close"),
    )
    executor._create_bash_session = MagicMock(return_value=static_session)
    action = CmdRunAction(command="failing", is_static=True)
    invoke = AsyncMock(side_effect=[RuntimeError("execute failed"), None])
    monotonic = MagicMock(side_effect=[10.0, 10.25])

    with (
        patch.object(action_server, "call_sync_from_async", invoke),
        patch.object(action_server, "time", SimpleNamespace(monotonic=monotonic)),
    ):
        result = asyncio.run(executor.run(action))

    assert isinstance(result, ErrorObservation)
    assert result.content == "execute failed"
    assert invoke.await_args_list == [
        call(static_session.execute, action),
        call(static_session.close),
    ]


def test_static_shell_close_error_does_not_mask_execution_error(
    action_server: ModuleType,
) -> None:
    executor = object.__new__(action_server.ActionExecutor)
    executor.bash_session = SimpleNamespace(cwd="/persistent")
    static_session = SimpleNamespace(
        execute=MagicMock(name="static_execute"),
        close=MagicMock(name="static_close"),
    )
    executor._create_bash_session = MagicMock(return_value=static_session)
    action = CmdRunAction(command="failing", is_static=True)
    invoke = AsyncMock(
        side_effect=[RuntimeError("execute failed"), RuntimeError("close failed")]
    )
    monotonic = MagicMock(side_effect=[10.0, 10.25])

    with (
        patch.object(action_server, "call_sync_from_async", invoke),
        patch.object(action_server, "time", SimpleNamespace(monotonic=monotonic)),
    ):
        result = asyncio.run(executor.run(action))

    assert isinstance(result, ErrorObservation)
    assert result.content == "execute failed"
    assert invoke.await_args_list == [
        call(static_session.execute, action),
        call(static_session.close),
    ]


class _FatalStaticExecution(BaseException):
    """Synthetic non-Exception failure used to verify unconditional cleanup."""


@pytest.mark.parametrize(
    'execution_error',
    [asyncio.CancelledError(), _FatalStaticExecution('fatal')],
    ids=('cancelled', 'base-exception'),
)
def test_static_shell_session_is_closed_after_base_exception(
    action_server: ModuleType,
    execution_error: BaseException,
) -> None:
    executor = object.__new__(action_server.ActionExecutor)
    executor.bash_session = SimpleNamespace(cwd='/persistent')
    static_session = SimpleNamespace(
        execute=MagicMock(name='static_execute'),
        close=MagicMock(name='static_close'),
    )
    executor._create_bash_session = MagicMock(return_value=static_session)
    action = CmdRunAction(command='interrupted', is_static=True)
    invoke = AsyncMock(side_effect=[execution_error, None])

    with patch.object(action_server, 'call_sync_from_async', invoke):
        with pytest.raises(type(execution_error)) as captured:
            asyncio.run(executor.run(action))

    assert captured.value is execution_error
    assert invoke.await_args_list == [
        call(static_session.execute, action),
        call(static_session.close),
    ]


@pytest.mark.parametrize(
    'cleanup_error',
    [RuntimeError('close failed'), asyncio.CancelledError()],
    ids=('exception', 'base-exception'),
)
def test_static_shell_cleanup_error_does_not_mask_cancellation(
    action_server: ModuleType,
    cleanup_error: BaseException,
) -> None:
    executor = object.__new__(action_server.ActionExecutor)
    executor.bash_session = SimpleNamespace(cwd='/persistent')
    static_session = SimpleNamespace(
        execute=MagicMock(name='static_execute'),
        close=MagicMock(name='static_close'),
    )
    executor._create_bash_session = MagicMock(return_value=static_session)
    action = CmdRunAction(command='cancelled', is_static=True)
    cancellation = asyncio.CancelledError()
    invoke = AsyncMock(
        side_effect=[cancellation, cleanup_error]
    )

    with patch.object(action_server, 'call_sync_from_async', invoke):
        with pytest.raises(asyncio.CancelledError) as captured:
            asyncio.run(executor.run(action))

    assert captured.value is cancellation
    assert invoke.await_args_list == [
        call(static_session.execute, action),
        call(static_session.close),
    ]


def test_codex_raw_shell_collection_preserves_output_whitespace() -> None:
    from openhands.runtime.utils.bash import _remove_command_prefix

    assert (
        _remove_command_prefix(
            "example-command\r\n  leading\ntrailing  \n",
            "example-command",
            preserve_output_whitespace=True,
        )
        == "  leading\ntrailing  \n"
    )


def test_codex_tmux_capture_requests_and_preserves_trailing_spaces() -> None:
    from openhands.runtime.utils.bash import BashSession

    session = object.__new__(BashSession)
    session.pane = MagicMock()
    session.pane.cmd.return_value.stdout = ["alpha  ", "beta "]

    assert session._get_pane_content(preserve_trailing=True) == "alpha  \nbeta "
    session.pane.cmd.assert_called_once_with(
        "capture-pane",
        "-J",
        "-N",
        "-pS",
        "-",
    )

    session.pane.cmd.reset_mock()
    session.pane.cmd.return_value.stdout = ["alpha  ", "beta "]
    assert session._get_pane_content() == "alpha\nbeta"
    session.pane.cmd.assert_called_once_with(
        "capture-pane",
        "-J",
        "-pS",
        "-",
    )

    session = object.__new__(BashSession)
    session.prev_output = ""
    metadata = CmdOutputMetadata()
    assert (
        session._get_command_output(
            "example-command",
            "example-command\n  leading\ntrailing  \n",
            metadata,
            preserve_trailing=True,
        )
        == "  leading\ntrailing  \n"
    )


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("value", "example-command\nvalue"),
        ("  leading\ntrailing  \n", "example-command\n  leading\ntrailing  \n"),
    ],
)
def test_codex_prompt_boundaries_do_not_add_output_newlines(
    stdout: str,
    expected: str,
) -> None:
    from openhands.runtime.utils.bash import BashSession

    prompt = (
        '\n###PS1JSON###\n'
        '{"pid": -1, "exit_code": 0, "username": "user", '
        '"hostname": "host", "working_dir": "/workspace", '
        '"py_interpreter_path": "/usr/bin/python"}'
        '\n###PS1END###'
    )
    pane_content = f'{prompt}\nexample-command\n{stdout}{prompt}\n'
    matches = CmdOutputMetadata.matches_ps1_metadata(pane_content)
    session = object.__new__(BashSession)

    combined = session._combine_outputs_between_matches(
        pane_content,
        matches,
        preserve_output_whitespace=True,
        include_after_last_match=False,
    )

    assert combined == expected


def test_codex_command_preparation_preserves_default_shell_environment() -> None:
    from openhands.runtime.utils.bash import BashSession

    action = CmdRunAction(command="printf '%s' \"$ACTIVE_ENV\"")

    assert BashSession._prepare_codex_command(action, action.command) == action.command


def test_codex_command_preparation_quotes_workdir_without_mutating_cwd() -> None:
    from openhands.runtime.utils.bash import BashSession

    action = CmdRunAction(
        command="printf '%s' \"$PWD\"",
        cwd="/workspace/path with 'quotes' and $variables",
    )

    prepared = BashSession._prepare_codex_command(action, action.command)

    assert prepared.startswith("(\nbuiltin cd -- ")
    assert prepared.endswith(f"\n{action.command}\n)")
    assert "'$variables'" not in prepared
    assert "'\"'\"'quotes'\"'\"'" in prepared


@pytest.mark.parametrize(
    "command",
    [
        "sleep 0.01 &",
        "printf ok;",
        "cat <<'EOF'\nheredoc body\nEOF",
    ],
)
def test_codex_workdir_wrapper_preserves_valid_trailing_shell_syntax(
    command: str,
) -> None:
    from openhands.runtime.utils.bash import BashSession

    action = CmdRunAction(command=command, cwd="/workspace")
    prepared = BashSession._prepare_codex_command(action, action.command)

    syntax_check = subprocess.run(
        ["/bin/bash", "-n"],
        input=prepared,
        capture_output=True,
        text=True,
    )
    assert syntax_check.returncode == 0, syntax_check.stderr


@pytest.mark.parametrize(
    ('login', 'flag'),
    [(True, '-lc'), (False, '-c')],
)
def test_codex_command_preparation_honors_explicit_login(
    login: bool,
    flag: str,
) -> None:
    from openhands.runtime.utils.bash import BashSession

    action = CmdRunAction(command="printf ok", login=login)

    prepared = BashSession._prepare_codex_command(action, action.command)

    assert f" {flag} " in prepared
    assert prepared.endswith("'printf ok'")


def test_codex_command_preparation_enables_pipefail_inside_wrappers() -> None:
    import shlex

    from openhands.runtime.utils.bash import BashSession

    action = CmdRunAction(
        command="false | tail -n 1",
        cwd="/workspace/path with spaces",
        login=True,
    )

    prepared = BashSession._prepare_codex_command(
        action,
        action.command,
        enable_pipefail=True,
    )

    shell, flag, script = shlex.split(prepared)
    assert shell.endswith('/bash')
    assert flag == '-lc'
    assert "set -o pipefail; false | tail -n 1" in script
    assert "builtin cd -- '/workspace/path with spaces' || exit" in script


def test_timeout_recovery_uses_graceful_interrupt_before_kill() -> None:
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    session = object.__new__(BashSession)
    session.prev_status = BashCommandStatus.NO_CHANGE_TIMEOUT
    session._wait_for_prompt = MagicMock(side_effect=[None, "ready"])
    session._send_keys_checked = MagicMock()
    session._kill_pane_processes = MagicMock()
    session._mark_timeout_recovered = MagicMock()

    assert session.recover_after_timeout() is True
    session._send_keys_checked.assert_called_once_with("C-c", enter=False)
    session._kill_pane_processes.assert_not_called()
    session._mark_timeout_recovered.assert_called_once_with("ready")


def test_timeout_recovery_escalates_to_kill_then_reuses_shell() -> None:
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    session = object.__new__(BashSession)
    session.prev_status = BashCommandStatus.HARD_TIMEOUT
    session._wait_for_prompt = MagicMock(side_effect=[None, None, "ready"])
    session._send_keys_checked = MagicMock()
    session._kill_pane_processes = MagicMock(return_value=True)
    session._mark_timeout_recovered = MagicMock()

    assert session.recover_after_timeout() is True
    session._kill_pane_processes.assert_called_once_with()
    session._mark_timeout_recovered.assert_called_once_with("ready")


def test_timeout_recovery_closes_unrecoverable_session() -> None:
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    session = object.__new__(BashSession)
    session.prev_status = BashCommandStatus.HARD_TIMEOUT
    session._recovery_failed = False
    session._wait_for_prompt = MagicMock(return_value=None)
    session._send_keys_checked = MagicMock()
    session._kill_pane_processes = MagicMock(return_value=False)
    session.close = MagicMock()

    assert session.recover_after_timeout() is False
    session.close.assert_called_once_with()
    assert session._recovery_failed is True


@pytest.mark.parametrize(
    ("stage", "wait_results"),
    [
        ("initial_wait", [RuntimeError("initial wait failed")]),
        ("send", [None]),
        ("interrupt_wait", [None, RuntimeError("interrupt wait failed")]),
        ("kill", [None, None]),
        ("final_wait", [None, None, RuntimeError("final wait failed")]),
        ("mark_recovered", ["ready"]),
    ],
)
def test_timeout_recovery_exceptions_fail_closed(
    stage: str,
    wait_results: list[str | None | Exception],
) -> None:
    from openhands.runtime.utils import bash as bash_module
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    session = object.__new__(BashSession)
    session.prev_status = BashCommandStatus.HARD_TIMEOUT
    session._recovery_failed = False
    session._wait_for_prompt = MagicMock(side_effect=wait_results)
    session._send_keys_checked = MagicMock(
        side_effect=(RuntimeError("send failed") if stage == "send" else None)
    )
    session._kill_pane_processes = MagicMock(
        side_effect=(RuntimeError("kill failed") if stage == "kill" else None)
    )
    session._mark_timeout_recovered = MagicMock(
        side_effect=(
            RuntimeError("mark recovered failed")
            if stage == "mark_recovered"
            else None
        )
    )
    session.close = MagicMock()

    with patch.object(bash_module.logger, "exception") as log_exception:
        recovered = session.recover_after_timeout()

    assert recovered is False
    assert session._recovery_failed is True
    session.close.assert_called_once_with()
    log_exception.assert_called_once_with(
        "Exception while recovering timed-out bash command"
    )


def test_timeout_recovery_close_error_does_not_mask_original_failure() -> None:
    from openhands.runtime.utils import bash as bash_module
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    session = object.__new__(BashSession)
    session.prev_status = BashCommandStatus.HARD_TIMEOUT
    session._recovery_failed = False
    session._wait_for_prompt = MagicMock(
        side_effect=RuntimeError("original wait failure")
    )
    session.close = MagicMock(side_effect=RuntimeError("close failure"))

    with patch.object(bash_module.logger, "exception") as log_exception:
        recovered = session.recover_after_timeout()

    assert recovered is False
    assert session._recovery_failed is True
    assert log_exception.call_args_list == [
        call("Exception while recovering timed-out bash command"),
        call("Failed to close an unrecoverable bash session"),
    ]
    # Avoid raising again from BashSession.__del__ after exercising the close
    # failure above.
    session.close = MagicMock()


def test_failed_timeout_recovery_does_not_execute_new_command() -> None:
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    session = object.__new__(BashSession)
    session._initialized = True
    session._recovery_failed = False
    session.prev_status = BashCommandStatus.NO_CHANGE_TIMEOUT
    session.recover_after_timeout = MagicMock(return_value=False)
    session._get_pane_content = MagicMock()
    session._send_keys_checked = MagicMock()
    action = CmdRunAction("echo should-not-run")
    action.tool_call_metadata = _metadata()

    observation = session.execute(action)

    assert isinstance(observation, ErrorObservation)
    assert observation.error_id == "SHELL_RECOVERY_FAILED"
    assert "new command was not executed" in observation.content
    session.recover_after_timeout.assert_called_once_with()
    session._get_pane_content.assert_not_called()
    session._send_keys_checked.assert_not_called()


def test_closed_unrecoverable_session_permanently_rejects_commands() -> None:
    from openhands.runtime.utils.bash import BashSession

    session = object.__new__(BashSession)
    session._initialized = True
    session._recovery_failed = True
    session._get_pane_content = MagicMock()
    session._send_keys_checked = MagicMock()

    observation = session.execute(CmdRunAction("echo should-not-run"))

    assert isinstance(observation, ErrorObservation)
    assert observation.error_id == "SHELL_RECOVERY_FAILED"
    assert "No command was executed" in observation.content
    session._get_pane_content.assert_not_called()
    session._send_keys_checked.assert_not_called()


def test_timeout_recovery_rejection_survives_deterministic_close_state() -> None:
    from openhands.runtime.utils.bash import BashSession

    session = BashSession(work_dir="/workspace")
    memory_monitor = MagicMock()
    tmux_session = MagicMock()
    session.memory_monitor = memory_monitor
    session.session = tmux_session
    session._closed = False
    session._initialized = True

    assert session._fail_timeout_recovery() is False

    assert session._recovery_failed is True
    assert session._initialized is False
    assert session._closed is True
    memory_monitor.stop.assert_called_once_with()
    tmux_session.kill.assert_called_once_with()
    observation = session.execute(CmdRunAction("echo should-not-run"))
    assert isinstance(observation, ErrorObservation)
    assert observation.error_id == "SHELL_RECOVERY_FAILED"


def test_bash_session_close_cleans_partially_initialized_tmux_session() -> None:
    from openhands.runtime.utils.bash import BashSession

    session = BashSession(work_dir="/workspace")
    partial_tmux_session = MagicMock()
    memory_monitor = MagicMock()
    session.session = partial_tmux_session
    session.memory_monitor = memory_monitor
    session._closed = False
    session._initialized = True

    session.close()
    session.close()

    memory_monitor.stop.assert_called_once_with()
    partial_tmux_session.kill.assert_called_once_with()
    assert session._closed is True
    assert session._initialized is False
    assert session.memory_monitor is None
    assert session.session is None


def _mock_tmux_initialization():
    server = MagicMock(name="server")
    tmux_session = MagicMock(name="tmux_session")
    initial_window = MagicMock(name="initial_window")
    window = MagicMock(name="window")
    pane = MagicMock(name="pane")
    memory_monitor = MagicMock(name="memory_monitor")
    server.new_session.return_value = tmux_session
    tmux_session.active_window = initial_window
    tmux_session.new_window.return_value = window
    window.active_pane = pane
    return server, tmux_session, initial_window, window, pane, memory_monitor


def test_bash_session_initialize_cleans_session_before_monitor_creation() -> None:
    from openhands.runtime.utils import bash as bash_module
    from openhands.runtime.utils.bash import BashSession

    server, tmux_session, _, _, _, _ = _mock_tmux_initialization()
    session = BashSession(work_dir="/workspace")

    with (
        patch.object(bash_module.libtmux, "Server", return_value=server),
        patch.object(bash_module, "TmuxMemoryMonitor") as monitor_type,
        patch.dict(bash_module.os.environ, {"TMUX_MEMORY_LIMIT": "not-an-int"}),
        pytest.raises(ValueError) as captured,
    ):
        session.initialize()

    assert "not-an-int" in str(captured.value)
    monitor_type.assert_not_called()
    tmux_session.kill.assert_called_once_with()
    assert session._closed is True
    assert session._initialized is False
    assert session.memory_monitor is None
    assert session.session is None


@pytest.mark.parametrize(
    "failure_stage",
    ["monitor_start", "history_setup", "window_creation", "prompt_setup"],
)
def test_bash_session_initialize_cleans_resources_after_stage_failure(
    failure_stage: str,
) -> None:
    from openhands.runtime.utils import bash as bash_module
    from openhands.runtime.utils.bash import BashSession

    server, tmux_session, _, _, pane, memory_monitor = _mock_tmux_initialization()
    failure = RuntimeError(f"{failure_stage} failed")
    if failure_stage == "monitor_start":
        memory_monitor.start.side_effect = failure
    elif failure_stage == "history_setup":
        tmux_session.set_option.side_effect = failure
    elif failure_stage == "window_creation":
        tmux_session.new_window.side_effect = failure
    else:
        pane.send_keys.side_effect = failure

    session = BashSession(work_dir="/workspace")
    session._wait_for_prompt = MagicMock(return_value="ready")
    session._clear_screen = MagicMock()

    with (
        patch.object(bash_module.libtmux, "Server", return_value=server),
        patch.object(
            bash_module,
            "TmuxMemoryMonitor",
            return_value=memory_monitor,
        ),
        pytest.raises(RuntimeError) as captured,
    ):
        session.initialize()

    assert captured.value is failure
    memory_monitor.stop.assert_called_once_with()
    tmux_session.kill.assert_called_once_with()
    assert session._closed is True
    assert session._initialized is False
    assert session.memory_monitor is None
    assert session.session is None


def test_bash_session_initialize_preserves_error_when_cleanup_also_fails() -> None:
    from openhands.runtime.utils import bash as bash_module
    from openhands.runtime.utils.bash import BashSession

    server, tmux_session, _, _, pane, memory_monitor = _mock_tmux_initialization()
    initialization_error = _FatalStaticExecution("prompt setup interrupted")
    monitor_error = RuntimeError("monitor stop failed")
    session_error = RuntimeError("session kill failed")
    pane.send_keys.side_effect = initialization_error
    memory_monitor.stop.side_effect = [monitor_error, None]
    tmux_session.kill.side_effect = [session_error, None]
    session = BashSession(work_dir="/workspace")

    with (
        patch.object(bash_module.libtmux, "Server", return_value=server),
        patch.object(
            bash_module,
            "TmuxMemoryMonitor",
            return_value=memory_monitor,
        ),
        patch.object(bash_module.logger, "exception") as log_exception,
        pytest.raises(_FatalStaticExecution) as captured,
    ):
        session.initialize()

    assert captured.value is initialization_error
    memory_monitor.stop.assert_called_once_with()
    tmux_session.kill.assert_called_once_with()
    assert log_exception.call_args_list == [
        call("Failed to kill tmux session after monitor cleanup failed"),
        call("Failed to clean up a partially initialized bash session"),
    ]
    assert session._closed is False
    assert session._initialized is False
    assert session.memory_monitor is memory_monitor
    assert session.session is tmux_session

    # Initialization preserved the primary error while retaining enough state
    # for finalization (or an explicit close) to retry transient failures.
    session.close()
    assert memory_monitor.stop.call_count == 2
    assert tmux_session.kill.call_count == 2
    assert session._closed is True
    assert session.memory_monitor is None
    assert session.session is None


def test_bash_session_close_attempts_all_cleanup_and_preserves_first_error() -> None:
    from openhands.runtime.utils import bash as bash_module
    from openhands.runtime.utils.bash import BashSession

    monitor_error = RuntimeError("monitor stop failed")
    session_error = RuntimeError("session kill failed")
    memory_monitor = MagicMock()
    memory_monitor.stop.side_effect = [monitor_error, None]
    tmux_session = MagicMock()
    tmux_session.kill.side_effect = [session_error, None]
    session = BashSession(work_dir="/workspace")
    session.memory_monitor = memory_monitor
    session.session = tmux_session
    session._closed = False
    session._initialized = True

    with (
        patch.object(bash_module.logger, "exception") as log_exception,
        pytest.raises(RuntimeError) as captured,
    ):
        session.close()

    assert captured.value is monitor_error
    memory_monitor.stop.assert_called_once_with()
    tmux_session.kill.assert_called_once_with()
    log_exception.assert_called_once_with(
        "Failed to kill tmux session after monitor cleanup failed"
    )
    assert session._closed is False
    assert session._initialized is False
    assert session.memory_monitor is memory_monitor
    assert session.session is tmux_session

    # A failed teardown remains retryable; success then makes close idempotent.
    session.close()
    session.close()
    assert memory_monitor.stop.call_count == 2
    assert tmux_session.kill.call_count == 2
    assert session._closed is True
    assert session.memory_monitor is None
    assert session.session is None


def test_bash_session_close_raises_session_error_after_monitor_stops() -> None:
    from openhands.runtime.utils.bash import BashSession

    session_error = RuntimeError("session kill failed")
    memory_monitor = MagicMock()
    tmux_session = MagicMock()
    tmux_session.kill.side_effect = [session_error, None]
    session = BashSession(work_dir="/workspace")
    session.memory_monitor = memory_monitor
    session.session = tmux_session
    session._closed = False

    with pytest.raises(RuntimeError) as captured:
        session.close()

    assert captured.value is session_error
    memory_monitor.stop.assert_called_once_with()
    tmux_session.kill.assert_called_once_with()
    assert session._closed is False
    assert session.memory_monitor is None
    assert session.session is tmux_session

    session.close()
    assert memory_monitor.stop.call_count == 1
    assert tmux_session.kill.call_count == 2
    assert session._closed is True
    assert session.session is None


def test_bash_session_initialize_waits_for_valid_configured_prompt() -> None:
    from openhands.runtime.utils import bash as bash_module
    from openhands.runtime.utils.bash import BashSession

    valid_prompt = (
        '\n###PS1JSON###\n'
        '{"pid": -1, "exit_code": 0, "username": "user", '
        '"hostname": "host", "working_dir": "/workspace", '
        '"py_interpreter_path": "/usr/bin/python"}'
        '\n###PS1END###'
    )
    server = MagicMock()
    tmux_session = MagicMock()
    initial_window = MagicMock()
    window = MagicMock()
    pane = MagicMock()
    server.new_session.return_value = tmux_session
    tmux_session.active_window = initial_window
    tmux_session.new_window.return_value = window
    window.active_pane = pane
    memory_monitor = MagicMock()
    session = BashSession(work_dir="/workspace")
    session._get_pane_content = MagicMock(
        side_effect=[BashSession.PS1, valid_prompt, valid_prompt]
    )
    session._clear_screen = MagicMock()

    with (
        patch.object(bash_module.libtmux, "Server", return_value=server),
        patch.object(
            bash_module,
            "TmuxMemoryMonitor",
            return_value=memory_monitor,
        ),
        patch.object(bash_module.time, "sleep"),
    ):
        session.initialize()

    assert session._initialized is True
    assert session._get_pane_content.call_count == 3
    session._clear_screen.assert_called_once_with()
    pane.send_keys.assert_called_once()
    memory_monitor.start.assert_called_once_with()
    session.close()
    memory_monitor.stop.assert_called_once_with()
    tmux_session.kill.assert_called_once_with()
    assert session._closed is True
    assert session._initialized is False
    assert session.memory_monitor is None
    assert session.session is None


@pytest.mark.parametrize(
    ("prompt_results", "expected_message", "expected_clear_calls"),
    [
        (
            [None],
            "did not expose a valid configured prompt",
            0,
        ),
        (
            ["ready", None],
            "prompt was not ready after initialization cleanup",
            1,
        ),
    ],
)
def test_bash_session_initialize_fails_closed_when_prompt_is_not_ready(
    prompt_results: list[str | None],
    expected_message: str,
    expected_clear_calls: int,
) -> None:
    from openhands.runtime.utils import bash as bash_module
    from openhands.runtime.utils.bash import BashSession

    server = MagicMock()
    tmux_session = MagicMock()
    initial_window = MagicMock()
    window = MagicMock()
    server.new_session.return_value = tmux_session
    tmux_session.active_window = initial_window
    tmux_session.new_window.return_value = window
    window.active_pane = MagicMock()
    memory_monitor = MagicMock()
    session = BashSession(work_dir="/workspace")
    session._wait_for_prompt = MagicMock(side_effect=prompt_results)
    session._clear_screen = MagicMock()

    with (
        patch.object(bash_module.libtmux, "Server", return_value=server),
        patch.object(
            bash_module,
            "TmuxMemoryMonitor",
            return_value=memory_monitor,
        ),
        pytest.raises(RuntimeError, match=expected_message),
    ):
        session.initialize()

    assert session._initialized is False
    assert session._clear_screen.call_count == expected_clear_calls
    memory_monitor.stop.assert_called_once_with()
    tmux_session.kill.assert_called_once_with()
    assert session._closed is True
    assert session.memory_monitor is None
    assert session.session is None


def test_unchanged_initial_prompt_is_not_treated_as_command_completion() -> None:
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    prompt = (
        '\n###PS1JSON###\n'
        '{"pid": -1, "exit_code": 0, "username": "user", '
        '"hostname": "host", "working_dir": "/workspace", '
        '"py_interpreter_path": "/usr/bin/python"}'
        '\n###PS1END###'
    )
    session = object.__new__(BashSession)
    session._initialized = True
    session._recovery_failed = False
    session.prev_status = BashCommandStatus.COMPLETED
    session.prev_output = ''
    session._cwd = '/workspace'
    session.NO_CHANGE_TIMEOUT_SECONDS = 0
    session._get_pane_content = MagicMock(return_value=prompt)
    session._send_keys_checked = MagicMock()

    observation = session.execute(CmdRunAction('sleep 30'))

    assert observation.metadata.exit_code == -1
    assert 'has no new output after 0 seconds' in observation.metadata.suffix
    assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT


def test_nonterminal_new_prompt_is_not_treated_as_command_completion() -> None:
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    prompt = (
        '\n###PS1JSON###\n'
        '{"pid": -1, "exit_code": 0, "username": "user", '
        '"hostname": "host", "working_dir": "/workspace", '
        '"py_interpreter_path": "/usr/bin/python"}'
        '\n###PS1END###'
    )
    running_pane = (
        f'{prompt}\n{prompt}\nset -o pipefail; sleep 30'
    )
    session = object.__new__(BashSession)
    session._initialized = True
    session._recovery_failed = False
    session.prev_status = BashCommandStatus.COMPLETED
    session.prev_output = ''
    session._cwd = '/workspace'
    session.NO_CHANGE_TIMEOUT_SECONDS = 0
    session._get_pane_content = MagicMock(
        side_effect=[prompt, running_pane]
    )
    session._send_keys_checked = MagicMock()
    action = CmdRunAction('sleep 30')
    action.tool_call_metadata = _metadata()

    observation = session.execute(action)

    assert observation.metadata.exit_code == -1
    assert 'has no new output after 0 seconds' in observation.metadata.suffix
    assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT


def test_first_poll_completed_command_survives_prompt_history_rollover() -> None:
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    prompt = (
        '\n###PS1JSON###\n'
        '{"pid": -1, "exit_code": 0, "username": "user", '
        '"hostname": "host", "working_dir": "/workspace", '
        '"py_interpreter_path": "/usr/bin/python"}'
        '\n###PS1END###'
    )
    bulk_output = 'bulk-output\n' * 10_001
    # The initial prompt and echoed command have rolled out of the 10k-line
    # history. Only command output plus the new terminal prompt remain.
    completed_pane = f'{bulk_output}{prompt}'
    session = object.__new__(BashSession)
    session._initialized = True
    session._recovery_failed = False
    session.prev_status = BashCommandStatus.COMPLETED
    session.prev_output = ''
    session._cwd = '/workspace'
    session.NO_CHANGE_TIMEOUT_SECONDS = 0
    session._get_pane_content = MagicMock(
        side_effect=[prompt, completed_pane]
    )
    session._send_keys_checked = MagicMock()
    session._clear_screen = MagicMock()
    action = CmdRunAction("printf lots-of-output")
    action.tool_call_metadata = _metadata()

    observation = session.execute(action)

    assert observation.metadata.exit_code == 0
    assert observation.command == "printf lots-of-output"
    assert observation.content == bulk_output
    assert session.prev_status == BashCommandStatus.COMPLETED
    session._clear_screen.assert_called_once_with()


def test_empty_input_poll_observes_process_completed_between_calls() -> None:
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    def prompt(exit_code: int) -> str:
        return (
            '\n###PS1JSON###\n'
            f'{{"pid": -1, "exit_code": {exit_code}, "username": "user", '
            '"hostname": "host", "working_dir": "/workspace", '
            '"py_interpreter_path": "/usr/bin/python"}'
            '\n###PS1END###'
        )

    initial_prompt = prompt(-1)
    completed_prompt = prompt(0)
    pane = (
        f'{initial_prompt}\nlong-running-command\npartial output\n'
        f'finished between calls\n{completed_prompt}'
    )
    session = object.__new__(BashSession)
    session._initialized = True
    session._recovery_failed = False
    session.prev_status = BashCommandStatus.NO_CHANGE_TIMEOUT
    session.prev_output = 'long-running-command\npartial output\n'
    session._cwd = '/workspace'
    session.NO_CHANGE_TIMEOUT_SECONDS = 30
    session._get_pane_content = MagicMock(return_value=pane)
    session._send_keys_checked = MagicMock()
    session._clear_screen = MagicMock()
    action = CmdRunAction('', is_input=True)
    action.tool_call_metadata = _metadata()

    observation = session.execute(action)

    assert observation.metadata.exit_code == 0
    assert observation.content.rstrip('\n') == 'finished between calls'
    assert observation.command == ''
    assert session.prev_status == BashCommandStatus.COMPLETED
    session._send_keys_checked.assert_not_called()
    session._clear_screen.assert_called_once_with()


@pytest.mark.parametrize('result_format', ['codex', 'opencode'])
def test_native_shell_pipeline_reports_upstream_failure(result_format: str) -> None:
    from openhands.runtime.utils.bash import BashCommandStatus, BashSession

    def prompt(exit_code: int) -> str:
        return (
            '\n###PS1JSON###\n'
            f'{{"pid": -1, "exit_code": {exit_code}, "username": "user", '
            '"hostname": "host", "working_dir": "/workspace", '
            '"py_interpreter_path": "/usr/bin/python"}'
            '\n###PS1END###'
        )

    initial_prompt = prompt(0)
    session = object.__new__(BashSession)
    session._initialized = True
    session._recovery_failed = False
    session.prev_status = BashCommandStatus.COMPLETED
    session.prev_output = ''
    session._cwd = '/workspace'
    session.NO_CHANGE_TIMEOUT_SECONDS = 30
    session._send_keys_checked = MagicMock()
    session._clear_screen = MagicMock()

    pane_reads = 0

    def pane_content(*, preserve_trailing: bool = False) -> str:
        nonlocal pane_reads
        pane_reads += 1
        if pane_reads == 1:
            return initial_prompt
        submitted = session._send_keys_checked.call_args.args[0]
        return f'{initial_prompt}\n{submitted}\n{prompt(1)}'

    session._get_pane_content = pane_content
    action = CmdRunAction('false | tail -n 1')
    action.tool_call_metadata = _metadata().model_copy(
        update={'tool_result_format': result_format}
    )

    observation = session.execute(action)

    submitted = session._send_keys_checked.call_args.args[0]
    assert submitted.startswith('set -o pipefail; ')
    assert observation.command == 'false | tail -n 1'
    assert observation.metadata.exit_code == 1
