from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from evaluation.benchmarks.swe_bench import run_infer
from evaluation.benchmarks.swe_bench.patch_export import get_patch_staging_command
from evaluation.utils.shared import EvalException
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation


class _PreExistingRepoRuntime:
    def __init__(self) -> None:
        self.actions: list[CmdRunAction] = []
        self.copy_to_calls: list[tuple[str, str]] = []

    def run_action(self, action: CmdRunAction) -> CmdOutputObservation:
        self.actions.append(action)
        content = (
            "__SWEBENCH_EXT_PRE_EXISTING__\n"
            if "__SWEBENCH_EXT_PRE_EXISTING__" in action.command
            else ""
        )
        return CmdOutputObservation(
            content=content,
            command=action.command,
            exit_code=0,
        )

    def copy_to(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ) -> None:
        self.copy_to_calls.append((host_src, sandbox_dest))


@pytest.mark.parametrize("base_commit", ["", "   ", None])
def test_pre_existing_swebench_ext_repo_with_missing_base_gets_baseline_tag(
    monkeypatch, base_commit: str | None
) -> None:
    runtime = _PreExistingRepoRuntime()
    instance = pd.Series(
        {
            "instance_id": "urwid-urwid-768-swe-bench-ext-ots",
            "repo": "urwid/urwid",
            "version": "ots",
            "base_commit": base_commit,
            "repo_language": "typescript",
        }
    )
    metadata = SimpleNamespace(details={"mode": "swe"})
    monkeypatch.setattr(run_infer, "DATASET_TYPE", "swe-bench-ext")
    monkeypatch.setattr(run_infer, "remove_nested_git_dirs", lambda *args: None)
    monkeypatch.setattr(
        run_infer,
        "capture_host_held_baseline",
        lambda *args, **kwargs: b"",
    )

    baseline = run_infer.initialize_runtime(
        runtime,  # type: ignore[arg-type]
        instance,
        metadata,  # type: ignore[arg-type]
    )

    assert baseline == b""
    assert any(
        action.command == "git tag -f swebench_baseline HEAD"
        for action in runtime.actions
    )
    assert not any(
        action.command.startswith("BASE=$(git rev-parse --verify")
        for action in runtime.actions
    )


class _SelectiveGitRuntime(_PreExistingRepoRuntime):
    """Execute only the repository setup/reset actions against a real repo."""

    def __init__(self, repo: Path, home: Path) -> None:
        super().__init__()
        self.repo = repo
        self.home = home
        self.observations: list[CmdOutputObservation] = []

    def run_action(self, action: CmdRunAction) -> CmdOutputObservation:
        self.actions.append(action)
        command = action.command
        execute = (
            "__SWEBENCH_EXT_PRE_EXISTING__" in command
            or command == "git tag -f swebench_baseline HEAD"
            or command.startswith("for remote_name in $(git remote)")
            or command.startswith("BASE=$(git rev-parse --verify")
            or command.startswith("git reflog expire")
            or command == "git gc --prune=now"
        )
        if not execute:
            observation = CmdOutputObservation(
                content="",
                command=command,
                exit_code=0,
            )
        else:
            env = os.environ.copy()
            env["HOME"] = str(self.home)
            result = subprocess.run(
                command,
                cwd=self.repo,
                env=env,
                shell=True,
                executable="/bin/bash",
                text=True,
                capture_output=True,
            )
            observation = CmdOutputObservation(
                content=result.stdout + result.stderr,
                command=command,
                exit_code=result.returncode,
            )
        self.observations.append(observation)
        return observation


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _initialize_real_swebench_ext_repo(
    monkeypatch,
    runtime: _SelectiveGitRuntime,
    base_commit: str,
) -> None:
    instance = pd.Series(
        {
            "instance_id": "swe-bench-ext-regression",
            "repo": "example/repo",
            "version": "ots",
            "base_commit": base_commit,
            "repo_language": "typescript",
        }
    )
    metadata = SimpleNamespace(details={"mode": "swe"})
    monkeypatch.setattr(run_infer, "DATASET_TYPE", "swe-bench-ext")
    monkeypatch.setattr(
        run_infer,
        "_get_workspace_path",
        lambda *args, **kwargs: str(runtime.repo),
    )
    monkeypatch.setattr(run_infer, "remove_nested_git_dirs", lambda *args: None)
    monkeypatch.setattr(
        run_infer,
        "capture_host_held_baseline",
        lambda *args, **kwargs: b"",
    )

    assert (
        run_infer.initialize_runtime(
            runtime,  # type: ignore[arg-type]
            instance,
            metadata,  # type: ignore[arg-type]
        )
        == b""
    )


def test_empty_base_anchors_initial_head_and_exports_all_agent_changes(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@openhands.local")
    _git(repo, "config", "user.name", "OpenHands Tests")
    (repo / "tracked.txt").write_text("first\n")
    stale_commit = _commit_all(repo, "first")
    _git(repo, "tag", "swebench_baseline", stale_commit)
    (repo / "tracked.txt").write_text("pristine\n")
    initial_head = _commit_all(repo, "pristine image state")
    runtime = _SelectiveGitRuntime(repo, tmp_path / "home")
    runtime.home.mkdir()

    _initialize_real_swebench_ext_repo(monkeypatch, runtime, "")

    assert _git(repo, "rev-parse", "swebench_baseline") == initial_head
    assert _git(repo, "rev-parse", "HEAD") == initial_head

    (repo / "committed.txt").write_text("agent committed\n")
    _commit_all(repo, "agent commit")
    (repo / "tracked.txt").write_text("agent uncommitted\n")
    (repo / "untracked.txt").write_text("agent untracked\n")

    baseline_paths = tmp_path / "baseline.paths"
    baseline_paths.write_bytes(b"")
    staging_command = get_patch_staging_command(
        "swebench_baseline",
        str(baseline_paths),
        hashlib.sha256(b"").hexdigest(),
    )
    subprocess.run(
        staging_command,
        cwd=repo,
        check=True,
        shell=True,
        executable="/bin/bash",
    )

    assert set(
        _git(
            repo,
            "diff",
            "--cached",
            "--name-only",
            "swebench_baseline",
        ).splitlines()
    ) == {"committed.txt", "tracked.txt", "untracked.txt"}


def test_nonempty_base_is_reset_then_tagged_without_changing_normal_semantics(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@openhands.local")
    _git(repo, "config", "user.name", "OpenHands Tests")
    (repo / "tracked.txt").write_text("base\n")
    base_commit = _commit_all(repo, "base")
    (repo / "tracked.txt").write_text("post-base image state\n")
    post_base_commit = _commit_all(repo, "post-base")
    _git(repo, "tag", "swebench_baseline", post_base_commit)
    runtime = _SelectiveGitRuntime(repo, tmp_path / "home")
    runtime.home.mkdir()

    _initialize_real_swebench_ext_repo(monkeypatch, runtime, base_commit)

    assert _git(repo, "rev-parse", "HEAD") == base_commit
    assert _git(repo, "rev-parse", "swebench_baseline") == base_commit
    assert (repo / "tracked.txt").read_text() == "base\n"


def test_flat_workspace_still_gets_synthetic_commit_and_baseline_tag(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "flat-repo"
    repo.mkdir()
    (repo / "source.txt").write_text("pristine\n")
    runtime = _SelectiveGitRuntime(repo, tmp_path / "home")
    runtime.home.mkdir()

    _initialize_real_swebench_ext_repo(monkeypatch, runtime, "upstream-only-sha")

    head = _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "rev-parse", "swebench_baseline") == head
    assert _git(repo, "status", "--short") == ""
    assert any(
        "__SWEBENCH_EXT_FRESH_INIT__" in observation.content
        for observation in runtime.observations
    )


def test_flat_workspace_does_not_inherit_parent_repository(
    monkeypatch, tmp_path: Path
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    _git(parent, "init", "--quiet")
    _git(parent, "config", "user.email", "tests@openhands.local")
    _git(parent, "config", "user.name", "OpenHands Tests")
    (parent / "parent.txt").write_text("parent\n")
    parent_head = _commit_all(parent, "parent baseline")

    workspace = parent / "workspace"
    workspace.mkdir()
    (workspace / "source.txt").write_text("workspace pristine\n")
    runtime = _SelectiveGitRuntime(workspace, tmp_path / "home")
    runtime.home.mkdir()

    _initialize_real_swebench_ext_repo(monkeypatch, runtime, "")

    assert (workspace / ".git").is_dir()
    assert _git(workspace, "rev-parse", "--show-toplevel") == str(workspace)
    assert _git(workspace, "rev-parse", "swebench_baseline") == _git(
        workspace, "rev-parse", "HEAD"
    )
    assert _git(workspace, "rev-parse", "HEAD") != parent_head
    assert _git(workspace, "status", "--short") == ""


def test_gitfile_worktree_is_treated_as_pre_existing_repo(
    monkeypatch, tmp_path: Path
) -> None:
    main_repo = tmp_path / "main"
    main_repo.mkdir()
    _git(main_repo, "init", "--quiet")
    _git(main_repo, "config", "user.email", "tests@openhands.local")
    _git(main_repo, "config", "user.name", "OpenHands Tests")
    (main_repo / "tracked.txt").write_text("pristine\n")
    initial_head = _commit_all(main_repo, "pristine")
    worktree = tmp_path / "linked-worktree"
    _git(
        main_repo,
        "worktree",
        "add",
        "--quiet",
        "--detach",
        str(worktree),
        initial_head,
    )
    assert (worktree / ".git").is_file()
    runtime = _SelectiveGitRuntime(worktree, tmp_path / "home")
    runtime.home.mkdir()

    _initialize_real_swebench_ext_repo(monkeypatch, runtime, "")

    assert _git(worktree, "rev-parse", "HEAD") == initial_head
    assert _git(worktree, "rev-parse", "swebench_baseline") == initial_head
    assert any(
        "__SWEBENCH_EXT_PRE_EXISTING__" in observation.content
        for observation in runtime.observations
    )
    assert all(
        "__SWEBENCH_EXT_FRESH_INIT__" not in observation.content
        for observation in runtime.observations
    )


class _MissingDiffBaseRuntime:
    def run_action(self, action: CmdRunAction) -> CmdOutputObservation:
        if action.command == "git rev-parse --is-inside-work-tree":
            content, exit_code = "true\n", 0
        elif action.command == "git rev-parse --verify refs/tags/swebench_baseline":
            content, exit_code = "fatal: Needed a single revision\n", 128
        else:
            content, exit_code = "", 0
        return CmdOutputObservation(
            content=content,
            command=action.command,
            exit_code=exit_code,
        )


@pytest.mark.parametrize("base_commit", ["", "   ", None])
def test_completion_fails_closed_when_tag_and_base_commit_are_both_missing(
    monkeypatch, base_commit: str | None
) -> None:
    instance = pd.Series(
        {
            "instance_id": "missing-diff-base",
            "repo": "example/repo",
            "version": "ots",
            "base_commit": base_commit,
        }
    )
    monkeypatch.setattr(run_infer, "DATASET_TYPE", "swe-bench-ext")
    monkeypatch.setattr(
        run_infer,
        "_get_workspace_path",
        lambda *args, **kwargs: "/workspace/repo",
    )

    with pytest.raises(EvalException, match="base_commit is empty"):
        run_infer.complete_runtime(
            _MissingDiffBaseRuntime(),  # type: ignore[arg-type]
            instance,
            b"",
        )
