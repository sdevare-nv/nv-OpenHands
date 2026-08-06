from __future__ import annotations

import hashlib
import posixpath
import secrets
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from zipfile import ZipFile

BASELINE_PATHS_FILENAME = "baseline.paths"
DEFAULT_RESET_BATCH_MAX_PATHS = 1_024


class RuntimeFileUploader(Protocol):
    def copy_to(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ) -> object: ...


@dataclass(frozen=True)
class AuthenticatedBaselineUpload:
    paths_file: str
    cleanup_dir: str
    sha256: str


def new_runtime_baseline_dir(workspace_path: str) -> str:
    """Return an unpredictable runtime directory for an ephemeral baseline."""
    if not workspace_path.startswith("/"):
        raise ValueError("workspace_path must be absolute")
    return posixpath.join(
        workspace_path,
        f".openhands-swebench-baseline-{secrets.token_hex(16)}",
    )


def get_runtime_baseline_paths_file(runtime_dir: str) -> str:
    return posixpath.join(runtime_dir, BASELINE_PATHS_FILENAME)


def get_untracked_baseline_snapshot_command(
    paths_file: str,
) -> str:
    """Record every untracked path present before the agent starts.

    Include ignored paths as well: an agent may change or remove ``.gitignore``,
    which must not make pre-existing container files eligible for export.
    """
    return f"git ls-files --others -z > {shlex.quote(paths_file)}"


def get_untracked_baseline_capture_command(runtime_dir: str) -> str:
    """Create a private runtime snapshot directory before the agent starts."""
    quoted_dir = shlex.quote(runtime_dir)
    quoted_paths_file = shlex.quote(get_runtime_baseline_paths_file(runtime_dir))
    # Write outside the repository first so the snapshot cannot list itself.
    return (
        "( umask 077; "
        f"mkdir -- {quoted_dir} || exit 1; "
        "baseline_capture_tmp=$(mktemp "
        "/tmp/openhands-swebench-baseline-capture.XXXXXX) || exit 1; "
        "trap 'rm -f -- \"$baseline_capture_tmp\"' EXIT; "
        'git ls-files --others -z > "$baseline_capture_tmp" || exit 1; '
        f'mv -- "$baseline_capture_tmp" {quoted_paths_file} || exit 1; '
        "trap - EXIT; "
        ")"
    )


def get_untracked_baseline_cleanup_command(
    runtime_dir: str, *, missing_ok: bool = False
) -> str:
    """Remove the exact ephemeral snapshot paths without recursive deletion."""
    quoted_file = shlex.quote(get_runtime_baseline_paths_file(runtime_dir))
    quoted_dir = shlex.quote(runtime_dir)
    if missing_ok:
        return (
            f"rm -f -- {quoted_file} && "
            f"if [ -d {quoted_dir} ]; then rmdir -- {quoted_dir}; fi"
        )
    return f"rm -f -- {quoted_file} && rmdir -- {quoted_dir}"


def read_untracked_baseline_archive(archive_path: Path) -> bytes:
    """Read the one expected snapshot member returned by Runtime.copy_from."""
    with ZipFile(archive_path) as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len(files) != 1 or files[0].filename != BASELINE_PATHS_FILENAME:
            names = [entry.filename for entry in files]
            raise ValueError(
                f"Unexpected untracked-baseline archive members: {names!r}"
            )
        return archive.read(files[0])


def upload_authenticated_baseline(
    runtime: RuntimeFileUploader,
    baseline_paths: bytes,
    workspace_path: str,
    *,
    cleanup_on_error: Callable[[str], None] | None = None,
) -> AuthenticatedBaselineUpload:
    """Upload host-held baseline bytes to an unpredictable runtime directory."""
    runtime_dir = new_runtime_baseline_dir(workspace_path)
    try:
        with tempfile.TemporaryDirectory(
            prefix="openhands-patch-baseline-"
        ) as host_dir:
            host_file = Path(host_dir, BASELINE_PATHS_FILENAME)
            host_file.write_bytes(baseline_paths)
            runtime.copy_to(str(host_file), f"{runtime_dir}/")
    except BaseException as upload_error:
        if cleanup_on_error is not None:
            try:
                cleanup_on_error(runtime_dir)
            except BaseException as cleanup_error:
                upload_error.add_note(
                    f"Runtime baseline cleanup also failed: {cleanup_error!r}"
                )
        raise
    return AuthenticatedBaselineUpload(
        paths_file=get_runtime_baseline_paths_file(runtime_dir),
        cleanup_dir=runtime_dir,
        sha256=hashlib.sha256(baseline_paths).hexdigest(),
    )


def get_patch_staging_command(
    diff_base_ref: str,
    paths_file: str,
    expected_sha256: str,
    *,
    cleanup_dir: str | None = None,
    reset_batch_max_paths: int = DEFAULT_RESET_BATCH_MAX_PATHS,
) -> str:
    """Stage the agent patch using an authenticated untracked-path baseline."""
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 hex digest")
    if (
        isinstance(reset_batch_max_paths, bool)
        or not isinstance(reset_batch_max_paths, int)
        or reset_batch_max_paths < 1
    ):
        raise ValueError("reset_batch_max_paths must be a positive integer")

    quoted_base_ref = shlex.quote(diff_base_ref)
    quoted_paths_file = shlex.quote(paths_file)
    cleanup_uploaded = ""
    release_uploaded = ""
    if cleanup_dir is not None:
        if posixpath.dirname(posixpath.normpath(paths_file)) != posixpath.normpath(
            cleanup_dir
        ):
            raise ValueError("paths_file must be a direct child of cleanup_dir")
        cleanup_uploaded = (
            f"rm -f -- {quoted_paths_file}; "
            f"rmdir -- {shlex.quote(cleanup_dir)} 2>/dev/null || true; "
        )
        release_uploaded = (
            f"rm -f -- {quoted_paths_file} || exit 1; "
            f"rmdir -- {shlex.quote(cleanup_dir)} || exit 1; "
        )
    # Some SWE images predate Git 2.26, so avoid the newer
    # --pathspec-from-file option. Intersect the NUL-delimited baseline with
    # staged additions first: ignored dependency trees can contain hundreds of
    # thousands of paths, while only paths that entered the index need reset.
    return (
        "( "
        "set -o pipefail; "
        "patch_stage_tmp=$(mktemp -d /tmp/openhands-patch-stage.XXXXXX) "
        "|| exit 1; "
        "cleanup_patch_stage() { "
        "rm -f -- "
        '"$patch_stage_tmp/baseline.sorted" '
        '"$patch_stage_tmp/staged.raw" '
        '"$patch_stage_tmp/staged.sorted" '
        '"$patch_stage_tmp/intersection" '
        '"$patch_stage_tmp/baseline.authenticated"; '
        'rmdir -- "$patch_stage_tmp" 2>/dev/null || true; '
        f"{cleanup_uploaded}"
        "}; "
        "trap cleanup_patch_stage EXIT; "
        f"if [ ! -f {quoted_paths_file} ]; then "
        'echo "Missing authenticated untracked paths snapshot" >&2; '
        "exit 1; "
        "fi; "
        "baseline_sha=$(tee "
        '"$patch_stage_tmp/baseline.authenticated" '
        f"< {quoted_paths_file} | sha256sum) || exit 1; "
        "baseline_sha=${baseline_sha%% *}; "
        f'if [ "$baseline_sha" != "{expected_sha256}" ]; then '
        'echo "Untracked paths snapshot authentication failed" >&2; '
        "exit 1; "
        "fi; "
        f"{release_uploaded}"
        "git add -A || exit 1; "
        'LC_ALL=C sort -z -- "$patch_stage_tmp/baseline.authenticated" '
        '> "$patch_stage_tmp/baseline.sorted" || exit 1; '
        "git diff --cached --name-only --diff-filter=A --no-renames -z "
        f'{quoted_base_ref} > "$patch_stage_tmp/staged.raw" || exit 1; '
        'LC_ALL=C sort -z -- "$patch_stage_tmp/staged.raw" '
        '> "$patch_stage_tmp/staged.sorted" || exit 1; '
        "LC_ALL=C comm -z -12 "
        '"$patch_stage_tmp/baseline.sorted" '
        '"$patch_stage_tmp/staged.sorted" '
        '> "$patch_stage_tmp/intersection" || exit 1; '
        f"xargs -0 -r -n {reset_batch_max_paths} "
        "git --literal-pathspecs reset --quiet "
        f"{quoted_base_ref} -- "
        '< "$patch_stage_tmp/intersection" || exit 1; '
        ")"
    )
