from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import pytest

from evaluation.benchmarks.swe_bench.patch_export import BASELINE_PATHS_FILENAME
from evaluation.benchmarks.swe_bench.patch_runtime import (
    capture_host_held_baseline,
    dispatch_authenticated_staging,
    dispatch_patch_completion,
    remove_nested_git_dirs,
)


@dataclass(frozen=True)
class _Observation:
    success: bool
    detail: str = ""


class _HarnessFailure(RuntimeError):
    pass


def _ensure_success(observation: _Observation, message: str) -> None:
    if not observation.success:
        raise _HarnessFailure(f"{message}: {observation.detail}")


class _CommandRunner:
    def __init__(self, *outcomes: _Observation | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, int]] = []

    def __call__(self, command: str, timeout: int) -> _Observation:
        self.calls.append((command, timeout))
        if not self.outcomes:
            raise AssertionError(f"Unexpected command: {command}")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Runtime:
    def __init__(
        self,
        archive: Path | BaseException,
        *,
        upload_error: BaseException | None = None,
    ) -> None:
        self.archive = archive
        self.upload_error = upload_error
        self.copy_from_calls: list[str] = []
        self.copy_to_calls: list[tuple[bytes, str, bool]] = []

    def copy_from(self, path: str) -> Path:
        self.copy_from_calls.append(path)
        if isinstance(self.archive, BaseException):
            raise self.archive
        return self.archive

    def copy_to(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ) -> None:
        self.copy_to_calls.append(
            (Path(host_src).read_bytes(), sandbox_dest, recursive)
        )
        if self.upload_error is not None:
            raise self.upload_error


def _baseline_archive(tmp_path: Path, content: bytes) -> Path:
    archive_path = tmp_path / "baseline.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(BASELINE_PATHS_FILENAME, content)
    return archive_path


def test_remove_nested_git_dirs_uses_one_path_safe_harness_action() -> None:
    success = _Observation(True)
    runner = _CommandRunner(success)

    result = remove_nested_git_dirs(runner, _ensure_success)

    assert result is success
    assert runner.calls == [
        (
            "find . -path './.git' -prune -o -name .git -prune -exec rm -rf -- {} +",
            600,
        )
    ]


def test_capture_host_held_baseline_success_contract(tmp_path: Path) -> None:
    baseline = b"fixture.py\0line\nbreak.py\0"
    archive = _baseline_archive(tmp_path, baseline)
    runtime = _Runtime(archive)
    runner = _CommandRunner(_Observation(True), _Observation(True))

    result = capture_host_held_baseline(
        runtime,
        "/workspace/repo",
        runner,
        _ensure_success,
    )

    assert result == baseline
    assert not archive.exists()
    assert len(runtime.copy_from_calls) == 1
    assert runtime.copy_from_calls[0].startswith(
        "/workspace/repo/.openhands-swebench-baseline-"
    )
    assert len(runner.calls) == 2
    assert all(timeout == 600 for _, timeout in runner.calls)
    assert "git ls-files --others -z" in runner.calls[0][0]
    assert "rmdir --" in runner.calls[1][0]


@pytest.mark.parametrize("failure_phase", ["capture", "copy", "archive"])
def test_capture_primary_error_attempts_cleanup_and_preserves_notes(
    failure_phase: str, tmp_path: Path
) -> None:
    primary_error = RuntimeError(f"{failure_phase} failed")
    cleanup_observation = _Observation(False, "cleanup failed")
    if failure_phase == "capture":
        archive: Path | BaseException = _baseline_archive(tmp_path, b"fixture.py\0")
        runner = _CommandRunner(primary_error, cleanup_observation)
    elif failure_phase == "copy":
        archive = primary_error
        runner = _CommandRunner(_Observation(True), cleanup_observation)
    else:
        archive = tmp_path / "invalid.zip"
        archive.write_bytes(b"not a zip archive")
        runner = _CommandRunner(_Observation(True), cleanup_observation)
    runtime = _Runtime(archive)
    reported: list[tuple[str, BaseException]] = []

    with pytest.raises(BaseException) as captured:
        capture_host_held_baseline(
            runtime,
            "/workspace/repo",
            runner,
            _ensure_success,
            report_cleanup_error=lambda label, error: reported.append((label, error)),
        )

    if failure_phase in {"capture", "copy"}:
        assert captured.value is primary_error
    else:
        assert isinstance(captured.value, BaseException)
        assert "zip" in str(captured.value).lower()
    assert "Runtime baseline cleanup also failed" in captured.value.__notes__[0]
    assert len(runner.calls) == 2
    assert len(reported) == 1
    assert reported[0][0] == "Runtime baseline cleanup also failed"
    if failure_phase == "archive":
        assert isinstance(archive, Path)
        assert not archive.exists()


def test_upload_failure_attempts_runtime_cleanup() -> None:
    upload_error = RuntimeError("upload failed")
    runtime = _Runtime(Path("unused.zip"), upload_error=upload_error)
    runner = _CommandRunner(_Observation(True))

    with pytest.raises(RuntimeError) as captured:
        dispatch_authenticated_staging(
            runtime,
            "/workspace/repo",
            "base",
            b"fixture.py\0",
            runner,
            _ensure_success,
        )

    assert captured.value is upload_error
    assert len(runner.calls) == 1
    assert "rmdir --" in runner.calls[0][0]
    assert runtime.copy_to_calls[0][0] == b"fixture.py\0"


def test_staging_dispatch_exception_attempts_cleanup() -> None:
    dispatch_error = RuntimeError("dispatch failed")
    runtime = _Runtime(Path("unused.zip"))
    runner = _CommandRunner(dispatch_error, _Observation(True))

    with pytest.raises(RuntimeError) as captured:
        dispatch_authenticated_staging(
            runtime,
            "/workspace/repo",
            "base",
            b"fixture.py\0",
            runner,
            _ensure_success,
        )

    assert captured.value is dispatch_error
    assert len(runner.calls) == 2
    assert "git add -A" in runner.calls[0][0]
    assert "rmdir --" in runner.calls[1][0]


def test_staging_error_observation_attempts_cleanup() -> None:
    runtime = _Runtime(Path("unused.zip"))
    runner = _CommandRunner(
        _Observation(False, "action failed"),
        _Observation(True),
    )

    with pytest.raises(_HarnessFailure, match="Failed to stage the agent patch"):
        dispatch_authenticated_staging(
            runtime,
            "/workspace/repo",
            "base",
            b"fixture.py\0",
            runner,
            _ensure_success,
        )

    assert len(runner.calls) == 2
    assert "rmdir --" in runner.calls[1][0]


def test_staging_cleanup_failure_does_not_mask_dispatch_exception() -> None:
    dispatch_error = RuntimeError("dispatch failed")
    cleanup_error = RuntimeError("cleanup dispatch failed")
    runtime = _Runtime(Path("unused.zip"))
    runner = _CommandRunner(dispatch_error, cleanup_error)
    reported: list[tuple[str, BaseException]] = []

    with pytest.raises(RuntimeError) as captured:
        dispatch_authenticated_staging(
            runtime,
            "/workspace/repo",
            "base",
            b"fixture.py\0",
            runner,
            _ensure_success,
            report_cleanup_error=lambda label, error: reported.append((label, error)),
        )

    assert captured.value is dispatch_error
    assert dispatch_error.__notes__ == [
        f"Runtime baseline cleanup also failed: {cleanup_error!r}"
    ]
    assert reported == [("Runtime baseline cleanup also failed", cleanup_error)]


def test_staging_success_returns_action_observation_contract() -> None:
    success = _Observation(True, "staged")
    baseline = b"fixture.py\0line\nbreak.py\0"
    runtime = _Runtime(Path("unused.zip"))
    runner = _CommandRunner(success)

    result = dispatch_authenticated_staging(
        runtime,
        "/workspace/repo",
        "base-ref",
        baseline,
        runner,
        _ensure_success,
    )

    assert result is success
    assert len(runner.calls) == 1
    command, timeout = runner.calls[0]
    assert timeout == 600
    assert "git add -A" in command
    assert "base-ref" in command
    assert "sha256sum" in command
    assert runtime.copy_to_calls[0][0] == baseline
    assert runtime.copy_to_calls[0][1].startswith(
        "/workspace/repo/.openhands-swebench-baseline-"
    )
    assert runtime.copy_to_calls[0][1].endswith("/")


def test_completion_routing_live_and_normal() -> None:
    calls: list[tuple[str, bytes | None]] = []

    def complete_live() -> str:
        calls.append(("live", None))
        return "live-result"

    def complete_normal(baseline: bytes) -> str:
        calls.append(("normal", baseline))
        return "normal-result"

    live_result = dispatch_patch_completion(
        "SWE-bench-Live",
        None,
        complete_live=complete_live,
        complete_normal=complete_normal,
        missing_baseline_error=lambda: AssertionError("missing baseline"),
    )
    normal_result = dispatch_patch_completion(
        "SWE-bench",
        b"baseline\0",
        complete_live=complete_live,
        complete_normal=complete_normal,
        missing_baseline_error=lambda: AssertionError("missing baseline"),
    )

    assert live_result == "live-result"
    assert normal_result == "normal-result"
    assert calls == [("live", None), ("normal", b"baseline\0")]


def test_completion_routing_normal_requires_baseline() -> None:
    missing_error = RuntimeError("missing baseline")

    with pytest.raises(RuntimeError) as captured:
        dispatch_patch_completion(
            "SWE-bench",
            None,
            complete_live=lambda: pytest.fail("live completion called"),
            complete_normal=lambda baseline: pytest.fail(
                f"normal completion called with {baseline!r}"
            ),
            missing_baseline_error=lambda: missing_error,
        )

    assert captured.value is missing_error
