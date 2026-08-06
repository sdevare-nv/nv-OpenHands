"""Tests for C-c / signal retry behavior in BashSession.

These tests verify that the BashSession correctly handles interrupt signals
(C-c, C-z) including the automatic retry mechanism for processes that need
multiple signals to terminate.
"""

import os
import tempfile
import time

import psutil
import pytest

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction
from openhands.events.tool import ToolCallMetadata
from openhands.runtime.utils.bash import BashCommandStatus, BashSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


PYTHON_SCRIPT_IGNORES_FIRST_SIGINT = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    print(f'SIGINT #{count[0]} caught, ignoring', flush=True)
    if count[0] >= 2:
        print('Second SIGINT received, exiting', flush=True)
        sys.exit(1)
signal.signal(signal.SIGINT, handler)
while True:
    time.sleep(0.1)
"""

PYTHON_SCRIPT_IGNORES_ALL_SIGINT = r"""
import signal, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
print('SIGINT fully ignored', flush=True)
while True:
    time.sleep(0.1)
"""

PYTHON_SCRIPT_IGNORES_THREE_SIGINTS = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    print(f'SIGINT #{count[0]} caught', flush=True)
    if count[0] >= 4:
        print('Fourth SIGINT, finally exiting', flush=True)
        sys.exit(1)
signal.signal(signal.SIGINT, handler)
while True:
    time.sleep(0.1)
"""


def _write_script(content: str, tmp_dir: str, name: str = "script.py") -> str:
    path = os.path.join(tmp_dir, name)
    with open(path, "w") as f:
        f.write(content)
    return path


def _codex_action(
    command: str,
    *,
    is_input: bool = False,
    cwd: str | None = None,
    login: bool | None = None,
    result_format: str = "codex",
) -> CmdRunAction:
    action = CmdRunAction(
        command=command,
        is_input=is_input,
        cwd=cwd,
        login=login,
    )
    action.tool_call_metadata = ToolCallMetadata(
        tool_call_id="call-shell-recovery",
        function_name="shell_command",
        model_response={
            "id": "response-shell-recovery",
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
            "model": "test-model",
            "object": "chat.completion",
        },
        total_calls_in_response=1,
        tool_result_format=result_format,
    )
    return action


def test_close_before_initialize_is_safe():
    session = BashSession(work_dir=os.getcwd())

    session.close()

    assert session._closed is True
    assert session.session is None


# ---------------------------------------------------------------------------
# Tests: basic C-c (existing behavior, now with retry infra in place)
# ---------------------------------------------------------------------------


def test_ctrl_c_basic_loop():
    """C-c kills a simple bash loop on the first signal (no retries needed)."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=3)
    session.initialize()
    try:
        obs = session.execute(
            CmdRunAction("while true; do echo 'looping'; sleep 4; done")
        )
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert "looping" in obs.content
        assert obs.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

        obs = session.execute(CmdRunAction("C-c", is_input=True))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code in (1, 130)
        assert "CTRL+C was sent" in obs.metadata.suffix
        assert session.prev_status == BashCommandStatus.COMPLETED

        # Session should be usable after interrupt
        obs = session.execute(CmdRunAction("echo 'recovered'"))
        assert "recovered" in obs.content
        assert obs.metadata.exit_code == 0
    finally:
        session.close()


def test_ctrl_c_python_http_server():
    """C-c kills python's http.server on the first signal."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=3)
    session.initialize()
    try:
        obs = session.execute(CmdRunAction("python3 -m http.server 18232"))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

        obs = session.execute(CmdRunAction("C-c", is_input=True))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code in (0, 1, 130)
        assert "CTRL+C was sent" in obs.metadata.suffix
        assert session.prev_status == BashCommandStatus.COMPLETED
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests: process that needs TWO SIGINTs (retry mechanism kicks in)
# ---------------------------------------------------------------------------


def test_ctrl_c_retry_process_needs_two_sigints():
    """Process traps the first SIGINT and only exits on the second.

    The retry mechanism resends C-c after SIGNAL_RETRY_INTERVAL (2s).
    no_change_timeout must be longer than the retry interval.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(PYTHON_SCRIPT_IGNORES_FIRST_SIGINT, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            logger.info(obs, extra={"msg_type": "OBSERVATION"})
            assert obs.metadata.exit_code == -1
            assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            logger.info(obs, extra={"msg_type": "OBSERVATION"})

            # The retry should have sent a second C-c and the process should exit
            assert obs.metadata.exit_code in (1, 130), (
                f"Expected exit code 1 or 130 but got {obs.metadata.exit_code}. "
                f"Content: {obs.content!r}"
            )
            assert "CTRL+C was sent" in obs.metadata.suffix
            assert session.prev_status == BashCommandStatus.COMPLETED

            # Should have taken ~2-5 seconds (retry fires at ~2s, then process exits)
            assert elapsed < 10, f"Took too long: {elapsed:.1f}s"

            # Verify we can see the handler messages in output
            assert "SIGINT #1 caught" in obs.content

            # Session should be usable after interrupt
            obs = session.execute(CmdRunAction("echo ok"))
            assert "ok" in obs.content
            assert obs.metadata.exit_code == 0
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Tests: process that fully ignores SIGINT (all retries exhaust)
# ---------------------------------------------------------------------------


def test_ctrl_c_retry_exhausted_process_ignores_sigint():
    """Process ignores SIGINT entirely. All retries should exhaust, then
    NO_CHANGE_TIMEOUT fires. The session should still be usable afterward.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(PYTHON_SCRIPT_IGNORES_ALL_SIGINT, tmp_dir)
        # Use timeout > retry_interval (2s) so retries fire, but process ignores all
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            logger.info(obs, extra={"msg_type": "OBSERVATION"})
            assert "SIGINT fully ignored" in obs.content
            assert obs.metadata.exit_code == -1
            assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

            obs = session.execute(CmdRunAction("C-c", is_input=True))
            logger.info(obs, extra={"msg_type": "OBSERVATION"})
            # Process ignores SIGINT, but after retries exhaust, SIGKILL escalation
            # kills it. Exit code 137 = 128 + 9 (SIGKILL).
            assert obs.metadata.exit_code == 137, (
                f"Expected 137 (SIGKILL), got {obs.metadata.exit_code}. "
                f"Content: {obs.content!r}"
            )
            assert session.prev_status == BashCommandStatus.COMPLETED
            assert "Killed" in obs.content
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Tests: process needs 4 SIGINTs (exercises multiple retries)
# ---------------------------------------------------------------------------


def test_ctrl_c_retry_process_needs_four_sigints():
    """Process catches 3 SIGINTs and only exits on the 4th.

    The initial send + 3 retries = 4 total SIGINTs should be enough.
    no_change_timeout must be > SIGNAL_RETRY_INTERVAL (2s) for retries to fire.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(PYTHON_SCRIPT_IGNORES_THREE_SIGINTS, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            logger.info(obs, extra={"msg_type": "OBSERVATION"})
            assert obs.metadata.exit_code == -1
            assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

            obs = session.execute(CmdRunAction("C-c", is_input=True))
            logger.info(obs, extra={"msg_type": "OBSERVATION"})

            # Initial + 3 retries = 4 SIGINTs. The process exits on the 4th.
            assert obs.metadata.exit_code in (1, 130)
            assert "CTRL+C was sent" in obs.metadata.suffix
            assert session.prev_status == BashCommandStatus.COMPLETED
            # We should see the handler's output from the earlier signals
            assert "SIGINT #1 caught" in obs.content
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Tests: C-c without is_input when process is running (should be blocked)
# ---------------------------------------------------------------------------


def test_ctrl_c_without_is_input_blocked():
    """C-c sent without is_input=True should be rejected by the guard when
    the previous command is still running.
    """
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=2)
    session.initialize()
    try:
        obs = session.execute(CmdRunAction("sleep 30"))
        assert obs.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

        # Send C-c WITHOUT is_input — should be blocked
        obs = session.execute(CmdRunAction("C-c"))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert "is NOT executed" in obs.metadata.suffix
        assert "The previous command is still running" in obs.metadata.suffix

        # Now send it properly with is_input=True
        obs = session.execute(CmdRunAction("C-c", is_input=True))
        assert obs.metadata.exit_code in (0, 1, 130, 143)
        assert session.prev_status == BashCommandStatus.COMPLETED
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests: C-c when no previous command is running (should error)
# ---------------------------------------------------------------------------


def test_ctrl_c_no_running_command():
    """C-c with is_input=True when no command is running should return error."""
    session = BashSession(work_dir=os.getcwd())
    session.initialize()
    try:
        obs = session.execute(CmdRunAction("C-c", is_input=True))
        assert "No previous running command" in obs.content
        assert obs.metadata.exit_code == -1
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests: recovery after C-c — can we run normal commands?
# ---------------------------------------------------------------------------


def test_recovery_after_ctrl_c():
    """After C-c kills a process, the session should accept normal commands."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=2)
    session.initialize()
    try:
        obs = session.execute(CmdRunAction("sleep 60"))
        assert obs.metadata.exit_code == -1

        obs = session.execute(CmdRunAction("C-c", is_input=True))
        assert session.prev_status == BashCommandStatus.COMPLETED

        # Run several commands to make sure the session is fully recovered
        for i in range(3):
            obs = session.execute(CmdRunAction(f"echo test_{i}"))
            assert f"test_{i}" in obs.content
            assert obs.metadata.exit_code == 0
            assert session.prev_status == BashCommandStatus.COMPLETED
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests: C-c retry with a process that produces output on first SIGINT
# ---------------------------------------------------------------------------


def test_ctrl_c_retry_output_on_first_signal():
    """Process prints output on the first SIGINT (like pytest's KeyboardInterrupt
    summary) but doesn't exit until the second SIGINT.

    The retry should detect the output change, reset the no-change timer, and
    eventually send a second C-c that terminates the process.
    """
    script = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    if count[0] == 1:
        # Simulate pytest-like behavior: print summary on first Ctrl+C
        for i in range(5):
            print(f'Shutting down... step {i+1}', flush=True)
            time.sleep(0.2)
        print('Waiting for second interrupt to abort...', flush=True)
    else:
        print('Aborted!', flush=True)
        sys.exit(130)
signal.signal(signal.SIGINT, handler)
print('Running tests...', flush=True)
while True:
    time.sleep(0.1)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(script, tmp_dir)
        # no_change_timeout must be > SIGNAL_RETRY_INTERVAL (2s)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            logger.info(obs, extra={"msg_type": "OBSERVATION"})
            assert "Running tests..." in obs.content
            assert obs.metadata.exit_code == -1

            obs = session.execute(CmdRunAction("C-c", is_input=True))
            logger.info(obs, extra={"msg_type": "OBSERVATION"})

            # The retry mechanism should have sent a second C-c after the
            # first one produced output but didn't terminate the process.
            assert obs.metadata.exit_code == 130
            assert session.prev_status == BashCommandStatus.COMPLETED
            assert "Shutting down" in obs.content
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Tests: C-z signal retry
# ---------------------------------------------------------------------------


def test_ctrl_z_suspends_process():
    """C-z should suspend the foreground process and return to shell."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=2)
    session.initialize()
    try:
        obs = session.execute(CmdRunAction("sleep 60"))
        assert obs.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

        obs = session.execute(CmdRunAction("C-z", is_input=True))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        # C-z suspends the process; the shell should return with a prompt
        assert session.prev_status == BashCommandStatus.COMPLETED
        # Exit code 148 or 20 is typical for suspended processes
        assert obs.metadata.exit_code in (0, 20, 148)

        # Session should be usable
        obs = session.execute(CmdRunAction("echo 'after suspend'"))
        assert "after suspend" in obs.content
        assert obs.metadata.exit_code == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests: non-signal is_input should NOT trigger retry
# ---------------------------------------------------------------------------


def test_regular_input_no_retry():
    """Regular text input (not a signal key) should not trigger retry logic."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=3)
    session.initialize()
    try:
        obs = session.execute(
            CmdRunAction(
                'read -p "Enter: " val && echo "got $val"',
            )
        )
        assert obs.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

        obs = session.execute(CmdRunAction("hello", is_input=True))
        assert "got hello" in obs.content
        assert obs.metadata.exit_code == 0
        assert session.prev_status == BashCommandStatus.COMPLETED
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests: piped command — C-c with broken pipe (the real-world scenario)
# ---------------------------------------------------------------------------


def test_ctrl_c_piped_command_tail():
    """Simulate: python long_running.py 2>&1 | tail -20

    When C-c is sent, tail dies immediately from SIGINT. The pipe breaks.
    The python process catches the first SIGINT but can't write output
    (broken pipe), so the pane shows NO output change.

    The retry mechanism sends additional C-c signals. Eventually the python
    process exits, and the shell's PS1 prompt appears (bypasses the pipe).
    """
    script = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    # Try to write — will fail with BrokenPipeError after tail dies
    try:
        print(f'SIGINT #{count[0]}', flush=True)
    except BrokenPipeError:
        pass
    if count[0] >= 2:
        sys.exit(130)
signal.signal(signal.SIGINT, handler)
# Simulate a long-running test that produces periodic output
i = 0
while True:
    i += 1
    try:
        print(f'test output line {i}', flush=True)
    except BrokenPipeError:
        # Pipe broke (tail died), just keep running silently
        pass
    time.sleep(0.5)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(script, tmp_dir, "long_running.py")
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            # Run with pipe — exactly the pattern from the user's issue
            obs = session.execute(
                CmdRunAction(f"python3 {script_path} 2>&1 | tail -20")
            )
            logger.info(obs, extra={"msg_type": "OBSERVATION"})
            assert obs.metadata.exit_code == -1
            assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

            # Send C-c. First SIGINT kills tail (pipe breaks), python catches it.
            # Python can't write output → pane shows nothing new.
            # Retry at 2s sends second SIGINT → python exits → shell shows PS1.
            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            logger.info(obs, extra={"msg_type": "OBSERVATION"})

            assert obs.metadata.exit_code in (0, 1, 130, 141), (
                f"Expected process to exit but got exit_code={obs.metadata.exit_code}. "
                f"Content: {obs.content!r}"
            )
            assert "CTRL+C was sent" in obs.metadata.suffix
            assert session.prev_status == BashCommandStatus.COMPLETED

            logger.info(
                f"Piped command interrupted in {elapsed:.1f}s",
                extra={"msg_type": "TEST_INFO"},
            )

            # Session should be fully usable after
            obs = session.execute(CmdRunAction("echo 'pipe test passed'"))
            assert "pipe test passed" in obs.content
            assert obs.metadata.exit_code == 0
        finally:
            session.close()


def test_ctrl_c_piped_command_grep():
    """Same as above but with grep instead of tail — another common pattern.

    python3 long_running.py 2>&1 | grep 'ERROR'
    """
    script = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    if count[0] >= 2:
        sys.exit(1)
signal.signal(signal.SIGINT, handler)
while True:
    try:
        print('running...', flush=True)
    except BrokenPipeError:
        pass
    time.sleep(0.5)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(script, tmp_dir, "grep_test.py")
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            obs = session.execute(
                CmdRunAction(f"python3 {script_path} 2>&1 | grep ERROR")
            )
            logger.info(obs, extra={"msg_type": "OBSERVATION"})
            assert obs.metadata.exit_code == -1

            obs = session.execute(CmdRunAction("C-c", is_input=True))
            logger.info(obs, extra={"msg_type": "OBSERVATION"})

            assert obs.metadata.exit_code in (0, 1, 130, 141)
            assert "CTRL+C was sent" in obs.metadata.suffix
            assert session.prev_status == BashCommandStatus.COMPLETED

            obs = session.execute(CmdRunAction("echo 'grep test passed'"))
            assert "grep test passed" in obs.content
            assert obs.metadata.exit_code == 0
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Corner case A: Process exits between timeout and C-c (race condition)
# ---------------------------------------------------------------------------


def test_ctrl_c_process_already_exited():
    """Process finishes between the no-change timeout and the agent's C-c.

    The PS1 should already be present when C-c arrives. C-c hits the idle
    shell, which prints ^C and a new PS1. Should still complete normally.
    """
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=2)
    session.initialize()
    try:
        # Start a command that will finish in ~3 seconds
        obs = session.execute(CmdRunAction("sleep 3 && echo done"))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

        # Wait for the command to finish on its own
        time.sleep(4)

        # Now send C-c — process is already dead, shell is at a prompt
        obs = session.execute(CmdRunAction("C-c", is_input=True))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        # Should detect the PS1 and complete (either from the finished command
        # or from the C-c on idle shell)
        assert session.prev_status == BashCommandStatus.COMPLETED
        assert obs.metadata.exit_code in (0, 1, 130)

        # Session should work
        obs = session.execute(CmdRunAction("echo 'race test ok'"))
        assert "race test ok" in obs.content
        assert obs.metadata.exit_code == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Corner case B: Hard timeout then C-c
# ---------------------------------------------------------------------------


def test_ctrl_c_after_hard_timeout():
    """Process hits a hard timeout (not no-change). Then agent sends C-c.

    Hard timeout sets prev_status=HARD_TIMEOUT. C-c with is_input should
    still work to kill the process.
    """
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=30)
    session.initialize()
    try:
        # Use a hard timeout of 2 seconds on a command that produces output
        action = CmdRunAction("for i in $(seq 1 100); do echo $i; sleep 0.5; done")
        action.set_hard_timeout(2)
        obs = session.execute(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.HARD_TIMEOUT

        # Send C-c to kill the process after hard timeout
        obs = session.execute(CmdRunAction("C-c", is_input=True))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code in (0, 1, 130)
        assert "CTRL+C was sent" in obs.metadata.suffix
        assert session.prev_status == BashCommandStatus.COMPLETED

        obs = session.execute(CmdRunAction("echo 'hard timeout ok'"))
        assert "hard timeout ok" in obs.content
        assert obs.metadata.exit_code == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Corner case C: Agent sends C-c multiple times (simulating real agent loop)
# ---------------------------------------------------------------------------


def test_multiple_sequential_ctrl_c_attempts():
    """Simulate the agent sending C-c 3 times in a row, each timing out.

    Uses no_change_timeout=1 (< SIGNAL_RETRY_INTERVAL) so retries don't
    fire within each attempt — simulating the OLD behavior where each C-c
    was independent.

    The piped process should eventually die from accumulated SIGINTs.
    """
    script = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    try:
        print(f'SIGINT #{count[0]}', flush=True)
    except (BrokenPipeError, IOError):
        pass
    if count[0] >= 3:
        sys.exit(1)
signal.signal(signal.SIGINT, handler)
while True:
    try:
        print('running...', flush=True)
    except (BrokenPipeError, IOError):
        pass
    time.sleep(0.5)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(script, tmp_dir)
        # Short timeout so each attempt completes quickly (simulating old behavior)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=1)
        session.initialize()
        try:
            obs = session.execute(
                CmdRunAction(f"python3 {script_path} 2>&1 | tail -20")
            )
            assert obs.metadata.exit_code == -1

            # Attempt 1: C-c (kills tail, python catches SIGINT)
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            logger.info(obs, extra={"msg_type": "OBSERVATION"})
            attempt1_exit = obs.metadata.exit_code

            if attempt1_exit == -1:
                # Attempt 2: C-c again
                obs = session.execute(CmdRunAction("C-c", is_input=True))
                logger.info(obs, extra={"msg_type": "OBSERVATION"})
                attempt2_exit = obs.metadata.exit_code

                if attempt2_exit == -1:
                    # Attempt 3: C-c again
                    obs = session.execute(CmdRunAction("C-c", is_input=True))
                    logger.info(obs, extra={"msg_type": "OBSERVATION"})
                    attempt3_exit = obs.metadata.exit_code

                    # By the 3rd attempt, the process should have received
                    # enough SIGINTs to exit
                    assert attempt3_exit in (0, 1, 130, 141), (
                        f"Process still running after 3 C-c attempts: "
                        f"exit codes were {attempt1_exit}, {attempt2_exit}, {attempt3_exit}"
                    )

            # Session should be usable
            obs = session.execute(CmdRunAction("echo 'multi attempt ok'"))
            assert "multi attempt ok" in obs.content
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Corner case D: C-c on process with continuous output (needs hard timeout)
# ---------------------------------------------------------------------------


def test_ctrl_c_continuous_output():
    """Process produces continuous output so no-change timeout never fires.

    Must use hard timeout to cap the command, then C-c to kill it.
    """
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=30)
    session.initialize()
    try:
        action = CmdRunAction("while true; do date; sleep 0.1; done")
        action.set_hard_timeout(3)
        obs = session.execute(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.HARD_TIMEOUT

        obs = session.execute(CmdRunAction("C-c", is_input=True))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code in (0, 1, 130)
        assert session.prev_status == BashCommandStatus.COMPLETED

        obs = session.execute(CmdRunAction("echo 'continuous ok'"))
        assert "continuous ok" in obs.content
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Corner case E: C-d (EOF) to close stdin
# ---------------------------------------------------------------------------


def test_ctrl_d_closes_stdin():
    """C-d sends EOF to a process waiting on stdin."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=3)
    session.initialize()
    try:
        # cat waits for stdin; C-d closes it
        obs = session.execute(CmdRunAction("cat"))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code == -1

        obs = session.execute(CmdRunAction("C-d", is_input=True))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert session.prev_status == BashCommandStatus.COMPLETED
        assert obs.metadata.exit_code == 0

        obs = session.execute(CmdRunAction("echo 'eof ok'"))
        assert "eof ok" in obs.content
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Corner case F: C-c then immediately run a new command
# ---------------------------------------------------------------------------


def test_ctrl_c_then_immediate_command():
    """After C-c kills a process, run a command immediately without delay."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=2)
    session.initialize()
    try:
        obs = session.execute(CmdRunAction("sleep 60"))
        assert obs.metadata.exit_code == -1

        obs = session.execute(CmdRunAction("C-c", is_input=True))
        assert session.prev_status == BashCommandStatus.COMPLETED

        # Immediately run commands back-to-back (no sleep between)
        for cmd in ["pwd", "whoami", "echo hello", "ls /tmp"]:
            obs = session.execute(CmdRunAction(cmd))
            assert obs.metadata.exit_code == 0, (
                f"Command '{cmd}' failed with exit_code={obs.metadata.exit_code}: "
                f"{obs.content!r}"
            )
            assert session.prev_status == BashCommandStatus.COMPLETED
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Corner case G: C-c on subshell
# ---------------------------------------------------------------------------


def test_ctrl_c_subshell():
    """C-c on a command running in a subshell: (while true; do ...; done)"""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=2)
    session.initialize()
    try:
        obs = session.execute(
            CmdRunAction("(while true; do echo sub; sleep 3; done)")
        )
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert "sub" in obs.content
        assert obs.metadata.exit_code == -1

        obs = session.execute(CmdRunAction("C-c", is_input=True))
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.metadata.exit_code in (0, 1, 130)
        assert session.prev_status == BashCommandStatus.COMPLETED

        obs = session.execute(CmdRunAction("echo 'subshell ok'"))
        assert "subshell ok" in obs.content
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Codex shell timeout recovery and option handling
# ---------------------------------------------------------------------------


def _command_children(session: BashSession):
    pane_pid = int(
        session.pane.cmd("display-message", "-p", "#{pane_pid}").stdout[0]
    )
    shell = session._find_shell_proc(psutil.Process(pane_pid))
    return shell.children(recursive=True)


def test_codex_next_command_recovers_after_nochange_timeout():
    """A normal Codex call replaces, rather than queues behind, stale work."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=1)
    session.initialize()
    try:
        timed_out = session.execute(_codex_action("sleep 30"))
        assert timed_out.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

        requested = "printf 'next-command-ok'"
        recovered = session.execute(_codex_action(requested))

        assert recovered.command == requested
        assert recovered.content == "next-command-ok"
        assert recovered.metadata.exit_code == 0
        assert "previous command is still running" not in recovered.metadata.suffix
        assert session.prev_status == BashCommandStatus.COMPLETED
        assert _command_children(session) == []
    finally:
        session.close()


def test_codex_nochange_timeout_still_accepts_interactive_input():
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=1)
    session.initialize()
    try:
        waiting = session.execute(
            _codex_action("read -r value; printf 'received:%s' \"$value\"")
        )
        assert waiting.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

        completed = session.execute(
            _codex_action("hello", is_input=True)
        )
        assert completed.content == "received:hello"
        assert completed.metadata.exit_code == 0
        assert session.prev_status == BashCommandStatus.COMPLETED
    finally:
        session.close()


def test_codex_hard_timeout_recovery_cleans_process_and_output():
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=30)
    session.initialize()
    try:
        action = _codex_action("while true; do echo old-output; sleep 0.1; done")
        action.set_hard_timeout(0.4)
        timed_out = session.execute(action)
        assert timed_out.metadata.exit_code == -1
        assert "old-output" in timed_out.content
        assert session.prev_status == BashCommandStatus.HARD_TIMEOUT

        assert session.recover_after_timeout() is True
        assert session.prev_status == BashCommandStatus.COMPLETED
        assert _command_children(session) == []

        requested = "printf 'clean-output'"
        recovered = session.execute(_codex_action(requested))
        assert recovered.command == requested
        assert recovered.content == "clean-output"
        assert "old-output" not in recovered.content
        assert recovered.metadata.exit_code == 0
    finally:
        session.close()


def test_codex_default_and_workdir_preserve_persistent_environment():
    with tempfile.TemporaryDirectory(prefix="codex cwd [quoted] ") as workdir:
        session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=3)
        session.initialize()
        try:
            exported = session.execute(
                CmdRunAction("export CODEX_PERSISTENT_ENV=inherited")
            )
            assert exported.metadata.exit_code == 0
            original_cwd = session.cwd

            default = session.execute(
                _codex_action("printf '%s' \"$CODEX_PERSISTENT_ENV\"")
            )
            assert default.content.rstrip("\n") == "inherited"

            requested = (
                "printf '%s\\n%s' \"$PWD\" \"$CODEX_PERSISTENT_ENV\""
            )
            in_workdir = session.execute(
                _codex_action(requested, cwd=workdir)
            )
            assert in_workdir.command == requested
            assert in_workdir.content.splitlines() == [workdir, "inherited"]
            assert session.cwd == original_cwd
        finally:
            session.close()


@pytest.mark.parametrize("result_format", ["codex", "opencode"])
def test_native_shell_pipeline_propagates_upstream_failure(
    result_format: str,
):
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=3)
    session.initialize()
    try:
        action = _codex_action(
            "false | tail -n 1",
            result_format=result_format,
        )

        observation = session.execute(action)

        assert observation.command == "false | tail -n 1"
        assert observation.metadata.exit_code == 1
    finally:
        session.close()
