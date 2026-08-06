from __future__ import annotations

import asyncio
import hashlib
import shlex
import shutil
import subprocess
from pathlib import Path

from evaluation.benchmarks.swe_bench.patch_export import (
    get_patch_staging_command,
    get_untracked_baseline_snapshot_command,
)
from evaluation.benchmarks.swe_bench.run_infer import (
    _ensure_patch_export_success,
    _run_patch_export_command,
)
from openhands.events.action import CmdRunAction
from openhands.runtime.action_execution_server import ActionExecutor
from openhands.runtime.utils.bash import BashSession


class _RecordingRuntime:
    def __init__(self) -> None:
        self.actions: list[CmdRunAction] = []
        self.result = object()

    def run_action(self, action: CmdRunAction) -> object:
        self.actions.append(action)
        return self.result


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _shell(repo: Path, command: str) -> None:
    subprocess.run(
        command,
        cwd=repo,
        check=True,
        shell=True,
        executable="/bin/bash",
    )


def test_patch_export_commands_use_fresh_static_session_and_explicit_cwd() -> None:
    runtime = _RecordingRuntime()

    result = _run_patch_export_command(
        runtime,  # type: ignore[arg-type]
        "/workspace/repo with spaces",
        "git status --short",
        731,
    )

    assert result is runtime.result
    assert len(runtime.actions) == 1
    action = runtime.actions[0]
    assert action.command == "git status --short"
    assert action.is_static is True
    assert action.cwd == "/workspace/repo with spaces"
    assert action.timeout == 731
    assert action.blocking is True
    assert action.bypass_blacklist is True


class _ExecutorRuntime:
    def __init__(self, executor: ActionExecutor) -> None:
        self.executor = executor

    def run_action(self, action: CmdRunAction):
        return asyncio.run(self.executor.run(action))


def test_patch_staging_ignores_persistent_shell_command_shadowing(
    tmp_path: Path,
) -> None:
    """Exercise the real BashSession static-action boundary.

    Every command used by authenticated staging is shadowed in the agent's
    persistent shell, and that shell is moved away from the repository. The
    evaluator action must still use genuine commands in a fresh session rooted
    at the explicit workspace and exclude the pre-existing fixture.
    """
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@openhands.local")
    _git(repo, "config", "user.name", "OpenHands Tests")
    (repo / "tracked.txt").write_text("baseline\n")
    (repo / ".gitignore").write_text("fixture.txt\n")
    _git(repo, "add", "tracked.txt", ".gitignore")
    _git(repo, "commit", "--quiet", "-m", "baseline")
    base_ref = _git(repo, "rev-parse", "HEAD").decode().strip()

    fixture = repo / "fixture.txt"
    fixture.write_text("pre-existing fixture\n")
    paths_file = tmp_path / "authenticated baseline.paths"
    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))
    baseline = paths_file.read_bytes()
    assert baseline == b"fixture.txt\0"

    (repo / ".gitignore").unlink()
    (repo / "tracked.txt").write_text("agent changed tracked file\n")
    (repo / "agent-created.txt").write_text("agent-created file\n")

    persistent = BashSession(work_dir=str(repo), username=None)
    persistent.initialize()
    try:
        executor = object.__new__(ActionExecutor)
        executor.bash_session = persistent
        executor._initial_cwd = str(repo)
        executor.username = None
        executor.max_memory_gb = None

        shadow_log = tmp_path / "persistent-shadow.log"
        wrappers: list[str] = []
        aliases: list[str] = []
        for tool in ("tee", "sha256sum", "git", "sort", "comm", "xargs"):
            real_tool = shutil.which(tool)
            assert real_tool is not None
            quoted_tool = shlex.quote(tool)
            quoted_real_tool = shlex.quote(real_tool)
            quoted_log = shlex.quote(str(shadow_log))
            if tool == "comm":
                # Suppressing the intersection would leave fixture.txt staged.
                body = f"printf '%s\\n' {quoted_tool} >> {quoted_log}; :"
            else:
                body = (
                    f"printf '%s\\n' {quoted_tool} >> {quoted_log}; "
                    f'{quoted_real_tool} "$@"'
                )
            wrappers.append(f"function {quoted_tool} {{ {body}; }}")
            aliases.append(f"alias {quoted_tool}=false")
        poison = "; ".join(
            [
                *wrappers,
                *aliases,
                "PATH=/definitely/agent-controlled",
                f"cd {shlex.quote(str(tmp_path))}",
            ]
        )
        poisoned = persistent.execute(CmdRunAction(command=poison))
        assert poisoned.exit_code == 0
        assert persistent.cwd == str(tmp_path)

        command = get_patch_staging_command(
            base_ref,
            str(paths_file),
            hashlib.sha256(baseline).hexdigest(),
        )
        observation = _run_patch_export_command(
            _ExecutorRuntime(executor),  # type: ignore[arg-type]
            str(repo),
            command,
            600,
        )
        _ensure_patch_export_success(
            observation,
            "Static authenticated staging failed",
        )

        assert not shadow_log.exists()
        changed = set(
            _git(repo, "diff", "--cached", "--name-only", "-z", base_ref)
            .decode()
            .rstrip("\0")
            .split("\0")
        )
        assert changed == {".gitignore", "agent-created.txt", "tracked.txt"}
    finally:
        persistent.close()
