from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

from evaluation.benchmarks.swe_bench.patch_export import (
    BASELINE_PATHS_FILENAME,
    DEFAULT_RESET_BATCH_MAX_PATHS,
    get_patch_staging_command,
    get_untracked_baseline_capture_command,
    get_untracked_baseline_cleanup_command,
    get_untracked_baseline_snapshot_command,
    read_untracked_baseline_archive,
    upload_authenticated_baseline,
)
from evaluation.benchmarks.swe_bench.patch_runtime import remove_nested_git_dirs


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _shell(repo: Path, command: str, *, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        command,
        cwd=repo,
        check=True,
        shell=True,
        executable="/bin/bash",
        env=env,
    )


def _patch_staging_command(
    base_ref: str,
    paths_file: Path,
    *,
    reset_batch_max_paths: int = DEFAULT_RESET_BATCH_MAX_PATHS,
) -> str:
    baseline = paths_file.read_bytes() if paths_file.is_file() else b""
    return get_patch_staging_command(
        base_ref,
        str(paths_file),
        hashlib.sha256(baseline).hexdigest(),
        reset_batch_max_paths=reset_batch_max_paths,
    )


def _recording_git_environment(
    tmp_path: Path, *, fail_reset: bool = False
) -> tuple[dict[str, str], Path]:
    real_git = shutil.which("git")
    assert real_git is not None
    wrapper_dir = tmp_path / "git-wrapper"
    wrapper_dir.mkdir()
    call_log = tmp_path / "git-calls.log"
    wrapper = wrapper_dir / "git"
    reset_behavior = (
        'if [[ "$1" == "--literal-pathspecs" && "$2" == "reset" ]]; then exit 47; fi\n'
        if fail_reset
        else ""
    )
    wrapper.write_text(
        "#!/bin/bash\n"
        'printf \'%q \' "$@" >> "$GIT_CALL_LOG"\n'
        "printf '\\n' >> \"$GIT_CALL_LOG\"\n"
        f"{reset_behavior}"
        'exec "$REAL_GIT" "$@"\n'
    )
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "GIT_CALL_LOG": str(call_log),
            "PATH": f"{wrapper_dir}{os.pathsep}{env['PATH']}",
            "REAL_GIT": real_git,
        }
    )
    return env, call_log


class _RecordingRuntimeUploader:
    def __init__(self) -> None:
        self.host_source: Path | None = None
        self.destination: str | None = None
        self.uploaded_content: bytes | None = None

    def copy_to(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ) -> None:
        assert recursive is False
        self.host_source = Path(host_src)
        self.destination = sandbox_dest
        self.uploaded_content = self.host_source.read_bytes()


class _FailingRuntimeUploader:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def copy_to(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ) -> None:
        del host_src, sandbox_dest, recursive
        raise self.error


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@openhands.local")
    _git(repo, "config", "user.name", "OpenHands Tests")
    (repo / "tracked.txt").write_text("baseline\n")
    (repo / ".gitignore").write_text("ignored.txt\n")
    _git(repo, "add", "tracked.txt", ".gitignore")
    _git(repo, "commit", "--quiet", "-m", "baseline")
    base_ref = _git(repo, "rev-parse", "HEAD").decode().strip()
    return repo, base_ref


def test_patch_staging_excludes_preexisting_untracked_and_keeps_agent_files(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base_ref = git_repo
    paths_file = tmp_path / "preexisting paths 'quoted'"
    fixture = repo / "synthetic_tests" / "test_golden_patch.py"
    fixture.parent.mkdir()
    fixture.write_text("preloaded fixture\n")
    unusual_fixture = repo / "preexisting [fixture]\n.py"
    unusual_fixture.write_text("preloaded unusual fixture\n")

    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))

    (repo / "tracked.txt").write_text("agent changed tracked file\n")
    fixture.write_text("agent touched preloaded fixture\n")
    unusual_fixture.write_text("agent touched unusual fixture\n")
    (repo / "synthetic_tests" / "agent_created.py").write_text("agent file\n")
    (repo / "new.txt").write_text("agent file\n")
    (repo / "ignored.txt").write_text("ignored generated content\n")

    _shell(repo, _patch_staging_command(base_ref, paths_file))

    changed = set(
        _git(repo, "diff", "--cached", "--name-only", "-z", base_ref)
        .decode()
        .rstrip("\0")
        .split("\0")
    )
    assert changed == {
        "new.txt",
        "synthetic_tests/agent_created.py",
        "tracked.txt",
    }


@pytest.mark.parametrize("git_metadata_form", ["directory", "gitfile"])
def test_nested_repositories_are_cleaned_before_capture_and_at_completion(
    git_repo: tuple[Path, str], tmp_path: Path, git_metadata_form: str
) -> None:
    repo, base_ref = git_repo
    paths_file = tmp_path / "baseline paths"

    def create_nested_repo(
        nested_repo: Path,
        filename: str,
        content: str,
        backing_name: str,
    ) -> None:
        if git_metadata_form == "gitfile":
            backing_repo = tmp_path / backing_name
            backing_repo.mkdir()
            _git(backing_repo, "init", "--quiet")
            _git(backing_repo, "config", "user.email", "tests@openhands.local")
            _git(backing_repo, "config", "user.name", "OpenHands Tests")
            (backing_repo / filename).write_text(content)
            _git(backing_repo, "add", filename)
            _git(backing_repo, "commit", "--quiet", "-m", "nested baseline")
            _git(
                backing_repo,
                "worktree",
                "add",
                "--quiet",
                "--detach",
                str(nested_repo),
                "HEAD",
            )
            assert (nested_repo / ".git").is_file()
            return

        nested_repo.mkdir()
        _git(nested_repo, "init", "--quiet")
        _git(nested_repo, "config", "user.email", "tests@openhands.local")
        _git(nested_repo, "config", "user.name", "OpenHands Tests")
        (nested_repo / filename).write_text(content)
        _git(nested_repo, "add", filename)
        _git(nested_repo, "commit", "--quiet", "-m", "nested baseline")
        assert (nested_repo / ".git").is_dir()

    preexisting_name = "preexisting 'quoted'\nrepo"
    preexisting_repo = repo / preexisting_name
    create_nested_repo(
        preexisting_repo,
        "fixture.py",
        "preloaded fixture\n",
        "preexisting-backing",
    )

    # Before cleanup, Git collapses the embedded repository into one directory
    # entry. Capturing that value would not match fixture.py after .git removal.
    assert _git(repo, "ls-files", "--others", "-z") == (
        f"{preexisting_name}/\0".encode()
    )

    cleanup_calls: list[tuple[str, int]] = []

    def run_cleanup(command: str, timeout: int) -> subprocess.CompletedProcess[bytes]:
        cleanup_calls.append((command, timeout))
        return subprocess.run(
            command,
            cwd=repo,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
        )

    def ensure_cleanup_success(
        result: subprocess.CompletedProcess[bytes], message: str
    ) -> None:
        assert result.returncode == 0, (
            f"{message}: stdout={result.stdout!r}, stderr={result.stderr!r}"
        )

    remove_nested_git_dirs(run_cleanup, ensure_cleanup_success)
    assert (repo / ".git").is_dir()
    assert not (preexisting_repo / ".git").exists()
    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))
    assert paths_file.read_bytes() == f"{preexisting_name}/fixture.py\0".encode()

    (repo / "tracked.txt").write_text("agent changed tracked file\n")
    agent_name = 'agent-created "quoted"\nrepo'
    agent_repo = repo / agent_name
    create_nested_repo(
        agent_repo,
        "agent.py",
        "agent file\n",
        "agent-created-backing",
    )

    remove_nested_git_dirs(run_cleanup, ensure_cleanup_success)
    assert not (agent_repo / ".git").exists()
    _shell(repo, _patch_staging_command(base_ref, paths_file))

    changed = set(
        _git(repo, "diff", "--cached", "--name-only", "-z", base_ref)
        .decode()
        .rstrip("\0")
        .split("\0")
    )
    assert changed == {f"{agent_name}/agent.py", "tracked.txt"}
    assert len(cleanup_calls) == 2


def test_patch_staging_includes_all_new_files_when_baseline_is_empty(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base_ref = git_repo
    paths_file = tmp_path / "empty paths 'quoted'"
    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))
    assert paths_file.read_bytes() == b""

    (repo / "created.py").write_text("agent file\n")
    (repo / "tracked.txt").unlink()
    _shell(repo, _patch_staging_command(base_ref, paths_file))

    assert _git(
        repo, "diff", "--cached", "--name-status", base_ref
    ).decode().splitlines() == [
        "A\tcreated.py",
        "D\ttracked.txt",
    ]


def test_patch_staging_excludes_preexisting_ignored_file_if_ignore_is_removed(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base_ref = git_repo
    paths_file = tmp_path / "baseline paths"
    ignored = repo / "ignored.txt"
    ignored.write_text("preloaded ignored fixture\n")

    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))
    assert paths_file.read_bytes() == b"ignored.txt\0"

    (repo / ".gitignore").unlink()
    (repo / "tracked.txt").write_text("agent changed tracked file\n")
    (repo / "agent_created.py").write_text("agent file\n")
    _shell(repo, _patch_staging_command(base_ref, paths_file))

    assert _git(
        repo,
        "diff",
        "--cached",
        "--name-status",
        base_ref,
    ).decode().splitlines() == [
        "D\t.gitignore",
        "A\tagent_created.py",
        "M\ttracked.txt",
    ]


def test_patch_staging_fails_closed_when_snapshot_is_missing(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base_ref = git_repo
    missing_paths_file = tmp_path / "missing snapshot"
    (repo / "preloaded.py").write_text("must not be exported\n")

    result = subprocess.run(
        _patch_staging_command(base_ref, missing_paths_file),
        cwd=repo,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert result.stderr.strip() == "Missing authenticated untracked paths snapshot"
    assert _git(repo, "diff", "--cached", "--name-only", base_ref) == b""


def test_patch_staging_fails_closed_if_uploaded_snapshot_is_tampered(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base_ref = git_repo
    upload_dir = tmp_path / "runtime upload"
    upload_dir.mkdir()
    paths_file = upload_dir / BASELINE_PATHS_FILENAME
    fixture = repo / "ignored.txt"
    fixture.write_text("preloaded ignored fixture\n")
    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))
    expected_sha256 = hashlib.sha256(paths_file.read_bytes()).hexdigest()

    # Simulate an agent erasing its fixture from the mutable runtime copy.
    paths_file.write_bytes(b"")
    (repo / ".gitignore").unlink()
    result = subprocess.run(
        get_patch_staging_command(
            base_ref,
            str(paths_file),
            expected_sha256,
            cleanup_dir=str(upload_dir),
        ),
        cwd=repo,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert result.stderr.strip() == ("Untracked paths snapshot authentication failed")
    assert _git(repo, "diff", "--cached", "--name-only", base_ref) == b""
    assert fixture.read_text() == "preloaded ignored fixture\n"
    assert not upload_dir.exists()


def test_authenticated_runtime_exchange_is_removed_before_git_add(
    git_repo: tuple[Path, str],
) -> None:
    repo, base_ref = git_repo
    runtime_dir = repo / ".openhands-swebench-baseline-test"
    paths_file = runtime_dir / BASELINE_PATHS_FILENAME
    (repo / "ignored.txt").write_text("preloaded ignored fixture\n")
    _shell(repo, get_untracked_baseline_capture_command(str(runtime_dir)))
    baseline = paths_file.read_bytes()
    assert baseline == b"ignored.txt\0"

    (repo / ".gitignore").unlink()
    _shell(
        repo,
        get_patch_staging_command(
            base_ref,
            str(paths_file),
            hashlib.sha256(baseline).hexdigest(),
            cleanup_dir=str(runtime_dir),
        ),
    )

    assert not runtime_dir.exists()
    assert _git(repo, "diff", "--cached", "--name-only", "-z", base_ref) == (
        b".gitignore\0"
    )


def test_patch_staging_handles_deleted_and_renamed_baseline_files(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base_ref = git_repo
    paths_file = tmp_path / "baseline paths"
    deleted_fixture = repo / "deleted_fixture.py"
    renamed_fixture = repo / "renamed_fixture.py"
    deleted_fixture.write_text("preloaded deleted fixture\n")
    source_fixture = repo / "source_fixture.py"
    source_fixture.write_text("preloaded renamed fixture\n")
    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))

    deleted_fixture.unlink()
    source_fixture.rename(renamed_fixture)
    (repo / "tracked.txt").rename(repo / "renamed_tracked.txt")
    _shell(repo, _patch_staging_command(base_ref, paths_file))

    assert _git(
        repo,
        "diff",
        "--cached",
        "--name-status",
        "--no-renames",
        base_ref,
    ).decode().splitlines() == [
        "A\trenamed_fixture.py",
        "A\trenamed_tracked.txt",
        "D\ttracked.txt",
    ]


def test_patch_staging_intersection_disables_rename_detection(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base_ref = git_repo
    paths_file = tmp_path / "baseline paths"
    # Matching content makes Git eligible to classify this added path and the
    # tracked deletion as a rename. It is still a baseline-untracked path and
    # therefore must be removed from the exported patch.
    (repo / "preexisting-copy.txt").write_text("baseline\n")
    _git(repo, "config", "diff.renames", "true")
    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))
    (repo / "tracked.txt").unlink()

    _shell(repo, _patch_staging_command(base_ref, paths_file))

    assert (
        _git(
            repo,
            "diff",
            "--cached",
            "--name-status",
            "--find-renames",
            base_ref,
        )
        == b"D\ttracked.txt\n"
    )


def test_patch_staging_reset_work_scales_with_staged_intersection(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, _ = git_repo
    paths_file = tmp_path / "large baseline paths"
    file_count = 3_000
    unignored_name = "ignored-0007.txt"
    (repo / ".gitignore").write_text("ignored-*.txt\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "--quiet", "-m", "ignore stress fixtures")
    base_ref = _git(repo, "rev-parse", "HEAD").decode().strip()
    for index in range(file_count):
        (repo / f"ignored-{index:04d}.txt").write_text("preloaded fixture\n")

    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))
    assert paths_file.read_bytes().count(b"\0") == file_count

    # Only one path from the large ignored baseline becomes eligible for
    # staging. Git reset work must follow that intersection, not baseline size.
    (repo / ".gitignore").write_text(f"ignored-*.txt\n!{unignored_name}\n")
    env, call_log = _recording_git_environment(tmp_path)
    _shell(
        repo,
        _patch_staging_command(base_ref, paths_file),
        env=env,
    )

    reset_calls = [
        call
        for call in call_log.read_text().splitlines()
        if call.startswith("--literal-pathspecs reset ")
    ]
    assert len(reset_calls) == 1
    assert unignored_name in reset_calls[0]
    assert _git(repo, "diff", "--cached", "--name-only", "-z", base_ref) == (
        b".gitignore\0"
    )


def test_patch_staging_batches_large_staged_intersection(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, _ = git_repo
    paths_file = tmp_path / "large staged baseline paths"
    file_count = 3_000
    (repo / ".gitignore").write_text("/-batch-*.txt\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "--quiet", "-m", "ignore batch fixtures")
    base_ref = _git(repo, "rev-parse", "HEAD").decode().strip()
    for index in range(file_count):
        (repo / f"-batch-{index:04d}.txt").write_text("preloaded fixture\n")

    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))
    assert paths_file.read_bytes().count(b"\0") == file_count

    # Removing the ignore rule stages the entire baseline intersection. The
    # leading-dash names also prove xargs places every literal path after `--`.
    (repo / ".gitignore").unlink()
    env, call_log = _recording_git_environment(tmp_path)
    _shell(
        repo,
        _patch_staging_command(base_ref, paths_file),
        env=env,
    )

    reset_calls = [
        call
        for call in call_log.read_text().splitlines()
        if call.startswith("--literal-pathspecs reset ")
    ]
    assert len(reset_calls) == math.ceil(file_count / DEFAULT_RESET_BATCH_MAX_PATHS)
    assert len(reset_calls) > 1
    assert all(" -- " in call for call in reset_calls)
    assert _git(repo, "diff", "--cached", "--name-only", "-z", base_ref) == (
        b".gitignore\0"
    )


def test_patch_staging_propagates_intersection_reset_failure(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base_ref = git_repo
    paths_file = tmp_path / "baseline paths"
    (repo / "ignored.txt").write_text("preloaded ignored fixture\n")
    _shell(repo, get_untracked_baseline_snapshot_command(str(paths_file)))
    (repo / ".gitignore").unlink()
    env, call_log = _recording_git_environment(tmp_path, fail_reset=True)

    result = subprocess.run(
        _patch_staging_command(base_ref, paths_file),
        cwd=repo,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    reset_calls = [
        call
        for call in call_log.read_text().splitlines()
        if call.startswith("--literal-pathspecs reset ")
    ]
    assert len(reset_calls) == 1
    assert "ignored.txt" in reset_calls[0]


def test_patch_staging_avoids_git_2_26_pathspec_file_options() -> None:
    command = get_patch_staging_command(
        "base",
        "/tmp/baseline.paths",
        hashlib.sha256(b"").hexdigest(),
    )

    assert "--pathspec-from-file" not in command
    assert "--pathspec-file-nul" not in command
    assert "git diff --cached" in command
    assert "comm -z -12" in command
    assert "xargs -0 -r" in command
    assert f"-n {DEFAULT_RESET_BATCH_MAX_PATHS}" in command
    assert "git --literal-pathspecs reset" in command


@pytest.mark.parametrize("digest", ["", "g" * 64, "A" * 64])
def test_patch_staging_rejects_unauthenticated_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        get_patch_staging_command("base", "/tmp/baseline.paths", digest)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_patch_staging_rejects_invalid_batch_size(
    batch_size: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        get_patch_staging_command(
            "base",
            "/tmp/baseline.paths",
            hashlib.sha256(b"").hexdigest(),
            reset_batch_max_paths=batch_size,  # type: ignore[arg-type]
        )


def test_patch_staging_rejects_cleanup_directory_mismatch() -> None:
    with pytest.raises(ValueError, match="direct child"):
        get_patch_staging_command(
            "base",
            "/tmp/actual/baseline.paths",
            hashlib.sha256(b"").hexdigest(),
            cleanup_dir="/tmp/different",
        )


def test_runtime_baseline_capture_and_cleanup_are_exact(
    git_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, _ = git_repo
    runtime_dir = tmp_path / "runtime baseline 'quoted'"
    fixture = repo / "preexisting\nfixture.py"
    fixture.write_text("preloaded fixture\n")

    _shell(repo, get_untracked_baseline_capture_command(str(runtime_dir)))

    assert (runtime_dir / BASELINE_PATHS_FILENAME).read_bytes() == (
        b"preexisting\nfixture.py\0"
    )
    _shell(repo, get_untracked_baseline_cleanup_command(str(runtime_dir)))
    assert not runtime_dir.exists()


def test_authenticated_upload_keeps_bytes_on_host_until_randomized_copy() -> None:
    runtime = _RecordingRuntimeUploader()
    baseline = b"ordinary.py\0line\nbreak.py\0-leading.py\0"

    upload = upload_authenticated_baseline(runtime, baseline, "/workspace/repo")

    assert runtime.uploaded_content == baseline
    assert runtime.host_source is not None
    assert not runtime.host_source.exists()
    assert runtime.destination == f"{upload.cleanup_dir}/"
    assert upload.paths_file == (f"{upload.cleanup_dir}/{BASELINE_PATHS_FILENAME}")
    assert upload.cleanup_dir.startswith(
        "/workspace/repo/.openhands-swebench-baseline-"
    )
    assert upload.sha256 == hashlib.sha256(baseline).hexdigest()


def test_authenticated_upload_failure_cleans_runtime_without_masking() -> None:
    upload_error = RuntimeError("upload failed")
    cleanup_error = OSError("cleanup failed")
    cleanup_calls: list[str] = []

    def failing_cleanup(runtime_dir: str) -> None:
        cleanup_calls.append(runtime_dir)
        raise cleanup_error

    with pytest.raises(RuntimeError) as captured:
        upload_authenticated_baseline(
            _FailingRuntimeUploader(upload_error),
            b"fixture.py\0",
            "/workspace/repo",
            cleanup_on_error=failing_cleanup,
        )

    assert captured.value is upload_error
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0].startswith("/workspace/repo/.openhands-swebench-baseline-")
    assert upload_error.__notes__ == [
        f"Runtime baseline cleanup also failed: {cleanup_error!r}"
    ]


def test_read_untracked_baseline_archive_requires_exact_member(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "baseline.zip"
    baseline = b"ordinary.py\0line\nbreak.py\0-leading.py\0"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(BASELINE_PATHS_FILENAME, baseline)

    assert read_untracked_baseline_archive(archive_path) == baseline

    with ZipFile(archive_path, "w") as archive:
        archive.writestr(BASELINE_PATHS_FILENAME, baseline)
        archive.writestr("unexpected", b"tampered")

    with pytest.raises(ValueError, match="Unexpected untracked-baseline"):
        read_untracked_baseline_archive(archive_path)
