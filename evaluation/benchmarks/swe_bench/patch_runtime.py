from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, TypeVar

from evaluation.benchmarks.swe_bench.patch_export import (
    RuntimeFileUploader,
    get_patch_staging_command,
    get_untracked_baseline_capture_command,
    get_untracked_baseline_cleanup_command,
    new_runtime_baseline_dir,
    read_untracked_baseline_archive,
    upload_authenticated_baseline,
)

ObservationT = TypeVar("ObservationT")
ResultT = TypeVar("ResultT")


class RuntimePatchExchange(RuntimeFileUploader, Protocol):
    def copy_from(self, path: str) -> Path: ...


RunCommand = Callable[[str, int], ObservationT]
EnsureSuccess = Callable[[ObservationT, str], None]
ReportCleanupError = Callable[[str, BaseException], None]


def remove_nested_git_dirs(
    run_command: RunCommand[ObservationT],
    ensure_success: EnsureSuccess[ObservationT],
) -> ObservationT:
    """Remove embedded Git metadata without parsing path text in Python."""
    command = "find . -path './.git' -prune -o -name .git -prune -exec rm -rf -- {} +"
    observation = run_command(command, 600)
    ensure_success(observation, "Failed to remove nested git repositories")
    return observation


def _cleanup_runtime_baseline(
    runtime_dir: str,
    run_command: RunCommand[ObservationT],
    ensure_success: EnsureSuccess[ObservationT],
    *,
    missing_ok: bool,
) -> None:
    observation = run_command(
        get_untracked_baseline_cleanup_command(runtime_dir, missing_ok=missing_ok),
        600,
    )
    ensure_success(
        observation,
        "Failed to remove runtime copy of pre-existing untracked paths",
    )


def _run_cleanups_preserving_primary(
    primary_error: BaseException | None,
    cleanups: list[tuple[str, Callable[[], None]]],
    report_cleanup_error: ReportCleanupError | None,
) -> None:
    deferred_cleanup_error: BaseException | None = None
    for label, cleanup in cleanups:
        try:
            cleanup()
        except BaseException as cleanup_error:
            preserved_error = primary_error or deferred_cleanup_error
            if preserved_error is None:
                deferred_cleanup_error = cleanup_error
                continue
            preserved_error.add_note(f"{label}: {cleanup_error!r}")
            if report_cleanup_error is not None:
                report_cleanup_error(label, cleanup_error)
    if primary_error is None and deferred_cleanup_error is not None:
        raise deferred_cleanup_error


def capture_host_held_baseline(
    runtime: RuntimePatchExchange,
    workspace_path: str,
    run_command: RunCommand[ObservationT],
    ensure_success: EnsureSuccess[ObservationT],
    *,
    report_cleanup_error: ReportCleanupError | None = None,
) -> bytes:
    """Capture a runtime baseline, retain it on the host, and remove the copy."""
    runtime_dir = new_runtime_baseline_dir(workspace_path)
    archive_path: Path | None = None
    primary_error: BaseException | None = None
    try:
        observation = run_command(
            get_untracked_baseline_capture_command(runtime_dir), 600
        )
        ensure_success(
            observation,
            "Failed to snapshot pre-existing untracked files",
        )
        archive_path = runtime.copy_from(runtime_dir)
        return read_untracked_baseline_archive(archive_path)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanups: list[tuple[str, Callable[[], None]]] = []
        if archive_path is not None:
            cleanups.append(
                (
                    "Host baseline archive cleanup also failed",
                    lambda: archive_path.unlink(missing_ok=True),
                )
            )
        cleanups.append(
            (
                "Runtime baseline cleanup also failed",
                lambda: _cleanup_runtime_baseline(
                    runtime_dir,
                    run_command,
                    ensure_success,
                    missing_ok=True,
                ),
            )
        )
        _run_cleanups_preserving_primary(primary_error, cleanups, report_cleanup_error)


def dispatch_authenticated_staging(
    runtime: RuntimePatchExchange,
    workspace_path: str,
    diff_base_ref: str,
    baseline_paths: bytes,
    run_command: RunCommand[ObservationT],
    ensure_success: EnsureSuccess[ObservationT],
    *,
    report_cleanup_error: ReportCleanupError | None = None,
) -> ObservationT:
    """Upload, authenticate, and consume the host-held baseline while staging."""

    def cleanup_upload(runtime_dir: str) -> None:
        _cleanup_runtime_baseline(
            runtime_dir,
            run_command,
            ensure_success,
            missing_ok=True,
        )

    baseline_upload = upload_authenticated_baseline(
        runtime,
        baseline_paths,
        workspace_path,
        cleanup_on_error=cleanup_upload,
    )
    primary_error: BaseException | None = None
    try:
        observation = run_command(
            get_patch_staging_command(
                diff_base_ref,
                baseline_upload.paths_file,
                baseline_upload.sha256,
                cleanup_dir=baseline_upload.cleanup_dir,
            ),
            600,
        )
        ensure_success(observation, "Failed to stage the agent patch")
        return observation
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if primary_error is not None:
            _run_cleanups_preserving_primary(
                primary_error,
                [
                    (
                        "Runtime baseline cleanup also failed",
                        lambda: cleanup_upload(baseline_upload.cleanup_dir),
                    )
                ],
                report_cleanup_error,
            )


def dispatch_patch_completion(
    dataset_type: str,
    baseline_paths: bytes | None,
    *,
    complete_live: Callable[[], ResultT],
    complete_normal: Callable[[bytes], ResultT],
    missing_baseline_error: Callable[[], BaseException],
) -> ResultT:
    """Route Live to its legacy completion and normal runs to safe staging."""
    if dataset_type == "SWE-bench-Live":
        return complete_live()
    if baseline_paths is None:
        raise missing_baseline_error()
    return complete_normal(baseline_paths)
