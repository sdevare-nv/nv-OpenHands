"""Model-visible output helpers for the built-in Codex tools.

The current shell, apply-patch, and plan contracts are pinned to upstream Codex
commit ``7ada37a15e1f6aa84f83b4b9410f9d29e66fefe4``. The dedicated
``read_file``, ``list_dir``, and ``grep_files`` tools were removed upstream;
their compatibility contracts are pinned to the last matching local tool set,
the official ``rust-v0.98.0`` tag.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass

CODEX_OUTPUT_LIMIT = 10_000
CODEX_APPROX_BYTES_PER_TOKEN = 4
CODEX_MAX_LINE_LENGTH = 500
CODEX_TAB_WIDTH = 4
CODEX_COMMENT_PREFIXES = ("#", "//", "--")
CODEX_TOKEN_POLICY_MODELS = frozenset(
    {
        # Current upstream model metadata.
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "codex-auto-review",
        # Legacy metadata from the release that exposed read_file/list_dir/
        # grep_files. Keeping these exact slugs avoids treating gpt-5.2-codex
        # as the byte-limited gpt-5.2 model.
        "gpt-5.2-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5-codex",
        "gpt-5-codex-mini",
    }
)
CODEX_BYTE_POLICY_MODELS = frozenset(
    {
        # Current and legacy general-purpose model metadata.
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
    }
)
_CODEX_MODEL_POLICIES = {
    **{slug: False for slug in CODEX_TOKEN_POLICY_MODELS},
    **{slug: True for slug in CODEX_BYTE_POLICY_MODELS},
}


def format_os_error(error: OSError) -> str:
    """Match Rust's platform ``io::Error`` display for common OS failures."""
    if error.errno is not None and error.strerror:
        return f"{error.strerror} (os error {error.errno})"
    return str(error)


def take_utf8_prefix(text: str, max_bytes: int) -> str:
    """Return the longest UTF-8-safe prefix within ``max_bytes``."""
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _rust_line_count(text: str) -> int:
    """Match Rust ``str::lines().count()`` for model-visible output."""
    if not text:
        return 0
    count = text.count("\n") + 1
    if text.endswith("\n"):
        count -= 1
    return count


def truncate_middle(
    text: str,
    limit: int,
    *,
    use_tokens: bool,
) -> str:
    """Match Codex's UTF-8-safe middle truncation marker and split."""
    encoded = text.encode("utf-8")
    max_bytes = limit * CODEX_APPROX_BYTES_PER_TOKEN if use_tokens else limit
    if len(encoded) <= max_bytes:
        return text

    left_budget = max_bytes // 2
    right_budget = max_bytes - left_budget
    prefix = encoded[:left_budget].decode("utf-8", errors="ignore")
    suffix = (
        encoded[len(encoded) - right_budget :].decode("utf-8", errors="ignore")
        if right_budget
        else ""
    )

    if use_tokens:
        removed = math.ceil((len(encoded) - max_bytes) / CODEX_APPROX_BYTES_PER_TOKEN)
        marker = f"…{removed} tokens truncated…"
    else:
        removed = len(text) - len(prefix) - len(suffix)
        marker = f"…{removed} chars truncated…"
    return prefix + marker + suffix


def uses_byte_truncation(model_name: str | None) -> bool:
    """Return whether Codex metadata assigns the 10k-byte policy.

    Current Codex falls back to the byte policy for an unknown model slug.
    Model metadata uses case-sensitive longest-prefix matching. If that misses,
    Codex retries exactly one simple provider namespace (for example,
    ``openai/gpt-5.6-sol``); arbitrary multi-segment aliases still fall back.
    """
    model = model_name or ""

    def find_policy(candidate: str) -> bool | None:
        matches = (
            (slug, byte_policy)
            for slug, byte_policy in _CODEX_MODEL_POLICIES.items()
            if candidate.startswith(slug)
        )
        best = max(matches, key=lambda match: len(match[0]), default=None)
        return None if best is None else best[1]

    policy = find_policy(model)
    if policy is not None:
        return policy

    namespace, separator, suffix = model.partition("/")
    simple_namespace = namespace and all(
        char.isascii() and (char.isalnum() or char in "_-") for char in namespace
    )
    if separator and "/" not in suffix and simple_namespace:
        policy = find_policy(suffix)
        if policy is not None:
            return policy

    return True


def truncate_function_output(
    output: str,
    *,
    model_name: str | None,
) -> str:
    """Apply Codex's 1.2x serialized-history budget to function output."""
    return truncate_middle(
        output,
        math.ceil(CODEX_OUTPUT_LIMIT * 1.2),
        use_tokens=not uses_byte_truncation(model_name),
    )


def format_shell_output(
    output: str,
    *,
    exit_code: int,
    duration_seconds: float,
    model_name: str | None = None,
    timed_out_ms: int | None = None,
    limit: int = CODEX_OUTPUT_LIMIT,
    use_tokens: bool | None = None,
) -> str:
    """Format a legacy ``shell_command`` result exactly as Codex does."""
    if timed_out_ms is not None:
        output = f"command timed out after {timed_out_ms} milliseconds\n{output}"

    if use_tokens is None:
        use_tokens = not uses_byte_truncation(model_name)
    truncated = truncate_middle(output, limit, use_tokens=use_tokens)

    # Rust first converts Duration to f32, performs the multiplication in f32,
    # then rounds to one decimal. Preserve that boundary behavior here.
    duration_f32 = struct.unpack("!f", struct.pack("!f", max(duration_seconds, 0.0)))[0]
    scaled_f32 = struct.unpack("!f", struct.pack("!f", duration_f32 * 10.0))[0]
    tenths = math.floor(scaled_f32 + 0.5)
    duration = (
        str(tenths // 10) if tenths % 10 == 0 else f"{tenths // 10}.{tenths % 10}"
    )

    sections = [
        f"Exit code: {exit_code}",
        f"Wall time: {duration} seconds",
    ]
    if _rust_line_count(output) != _rust_line_count(truncated):
        sections.append(f"Total output lines: {_rust_line_count(output)}")
    sections.extend(("Output:", truncated))
    return "\n".join(sections)


def split_file_lines(data: bytes) -> list[str]:
    r"""Split bytes like Tokio ``read_until(b'\n')`` and decode lossily."""
    if not data:
        return []
    ends_with_newline = data.endswith(b"\n")
    raw_lines = data.split(b"\n")
    if ends_with_newline:
        raw_lines.pop()

    lines = []
    for index, raw_line in enumerate(raw_lines):
        line_was_newline_terminated = index < len(raw_lines) - 1 or ends_with_newline
        if line_was_newline_terminated and raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
        lines.append(raw_line.decode("utf-8", errors="replace"))
    return lines


def _measure_indent(line: str) -> int:
    indent = 0
    for char in line:
        if char == " ":
            indent += 1
        elif char == "\t":
            indent += CODEX_TAB_WIDTH
        else:
            break
    return indent


@dataclass(frozen=True)
class _LineRecord:
    number: int
    raw: str
    display: str
    indent: int

    @property
    def is_blank(self) -> bool:
        return not self.raw.lstrip()

    @property
    def is_comment(self) -> bool:
        stripped = self.raw.strip()
        return any(stripped.startswith(prefix) for prefix in CODEX_COMMENT_PREFIXES)


def format_read_slice(lines: list[str], *, offset: int, limit: int) -> str:
    """Format the legacy Codex slice-mode body."""
    if offset > len(lines):
        raise ValueError("offset exceeds file length")
    selected = lines[offset - 1 : offset - 1 + limit]
    return "\n".join(
        f"L{offset + index}: {take_utf8_prefix(line, CODEX_MAX_LINE_LENGTH)}"
        for index, line in enumerate(selected)
    )


def format_read_indentation(
    lines: list[str],
    *,
    offset: int,
    limit: int,
    anchor_line: int | None,
    max_levels: int,
    include_siblings: bool,
    include_header: bool,
    max_lines: int | None,
) -> str:
    """Format the legacy Codex indentation-mode body."""
    anchor = anchor_line if anchor_line is not None else offset
    if anchor <= 0:
        raise ValueError("anchor_line must be a 1-indexed line number")
    guard_limit = max_lines if max_lines is not None else limit
    if guard_limit <= 0:
        raise ValueError("max_lines must be greater than zero")
    if not lines or anchor > len(lines):
        raise ValueError("anchor_line exceeds file length")

    records = [
        _LineRecord(
            number=index,
            raw=line,
            display=take_utf8_prefix(line, CODEX_MAX_LINE_LENGTH),
            indent=_measure_indent(line),
        )
        for index, line in enumerate(lines, start=1)
    ]

    effective_indents = []
    previous_indent = 0
    for record in records:
        if not record.is_blank:
            previous_indent = record.indent
        effective_indents.append(previous_indent)

    anchor_index = anchor - 1
    anchor_indent = effective_indents[anchor_index]
    min_indent = (
        0 if max_levels == 0 else max(anchor_indent - max_levels * CODEX_TAB_WIDTH, 0)
    )
    final_limit = min(limit, guard_limit, len(records))
    if final_limit == 1:
        record = records[anchor_index]
        return f"L{record.number}: {record.display}"

    before = anchor_index - 1
    after = anchor_index + 1
    before_min_indent_count = 0
    after_min_indent_count = 0
    selected: deque[_LineRecord] = deque((records[anchor_index],))

    while len(selected) < final_limit:
        progressed = 0
        if before >= 0:
            if effective_indents[before] >= min_indent:
                record = records[before]
                selected.appendleft(record)
                progressed += 1
                before -= 1
                if (
                    effective_indents[record.number - 1] == min_indent
                    and not include_siblings
                ):
                    allow_header = include_header and record.is_comment
                    if allow_header or before_min_indent_count == 0:
                        before_min_indent_count += 1
                    else:
                        selected.popleft()
                        progressed -= 1
                        before = -1
                if len(selected) >= final_limit:
                    break
            else:
                before = -1

        if after < len(records):
            if effective_indents[after] >= min_indent:
                record = records[after]
                selected.append(record)
                progressed += 1
                after += 1
                if (
                    effective_indents[record.number - 1] == min_indent
                    and not include_siblings
                ):
                    if after_min_indent_count > 0:
                        selected.pop()
                        progressed -= 1
                        after = len(records)
                    after_min_indent_count += 1
            else:
                after = len(records)

        if progressed == 0:
            break

    while selected and not selected[0].raw.strip():
        selected.popleft()
    while selected and not selected[-1].raw.strip():
        selected.pop()
    return "\n".join(f"L{record.number}: {record.display}" for record in selected)


def format_apply_patch_success(
    *,
    added: list[str],
    modified: list[str],
    deleted: list[str],
) -> str:
    """Format the success summary emitted by Codex's apply-patch library."""
    lines = ["Success. Updated the following files:"]
    lines.extend(f"A {path}" for path in added)
    lines.extend(f"M {path}" for path in modified)
    lines.extend(f"D {path}" for path in deleted)
    return "\n".join(lines) + "\n"
