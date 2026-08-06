"""Model-visible OpenCode tool-result formatting.

The strings in this module track anomalyco/opencode at commit
f0afb6750e63ee0a60b052914531bde0afb9bc2b. Runtime/UI metadata belongs on
observations, not in these bodies.
"""

from __future__ import annotations

from dataclasses import dataclass


READ_DEFAULT_LIMIT = 2_000
READ_MAX_BYTES = 50 * 1024
READ_MAX_LINE_LENGTH = 2_000
SHELL_MAX_BYTES = 50 * 1024
SHELL_MAX_LINES = 2_000
SEARCH_LIMIT = 100


@dataclass(frozen=True)
class GrepMatch:
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class HeadTruncation:
    text: str
    truncated: bool
    removed: int = 0
    unit: str = "lines"


def _utf16_length(text: str) -> int:
    """Return JavaScript's ``String.length`` for a Python string."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def _utf16_prefix(text: str, max_units: int) -> str:
    """Match JavaScript's ``substring(0, max_units)`` semantics."""
    encoded = text.encode("utf-16-le", errors="surrogatepass")
    return encoded[: max_units * 2].decode("utf-16-le", errors="surrogatepass")


def _utf8_size(text: str) -> int:
    # ``surrogatepass`` gives a lone UTF-16 unit the same three-byte size that
    # Node's Buffer.byteLength assigns after replacement encoding.
    return len(text.encode("utf-8", errors="surrogatepass"))


def split_file_lines(text: str) -> list[str]:
    """Match OpenCode's line stream without inventing a final empty line."""
    if not text:
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def format_read_file(
    filepath: str,
    lines: list[str],
    *,
    offset: int = 1,
    limit: int = READ_DEFAULT_LIMIT,
) -> str:
    """Format a text-file read, including OpenCode's line and byte caps."""
    offset = offset or 1
    if offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if offset > len(lines) and not (not lines and offset == 1):
        raise ValueError(
            f"Offset {offset} is out of range for this file ({len(lines)} lines)"
        )

    raw: list[str] = []
    used_bytes = 0
    cut = False
    start = offset - 1
    for source_line in lines[start : start + limit]:
        line = source_line
        if _utf16_length(line) > READ_MAX_LINE_LENGTH:
            line = (
                _utf16_prefix(line, READ_MAX_LINE_LENGTH)
                + "... (line truncated to 2000 chars)"
            )
        size = _utf8_size(line) + (1 if raw else 0)
        if used_bytes + size > READ_MAX_BYTES:
            cut = True
            break
        raw.append(line)
        used_bytes += size

    more = start + len(raw) < len(lines)
    output = f"<path>{filepath}</path>\n<type>file</type>\n<content>\n"
    output += "\n".join(
        f"{line_number}: {line}" for line_number, line in enumerate(raw, start=offset)
    )

    last = offset + len(raw) - 1
    next_offset = last + 1
    if cut:
        output += (
            f"\n\n(Output capped at 50 KB. Showing lines {offset}-{last}. "
            f"Use offset={next_offset} to continue.)"
        )
    elif more:
        output += (
            f"\n\n(Showing lines {offset}-{last} of {len(lines)}. "
            f"Use offset={next_offset} to continue.)"
        )
    else:
        output += f"\n\n(End of file - total {len(lines)} lines)"
    return output + "\n</content>"


def format_read_directory(
    filepath: str,
    entries: list[str],
    *,
    offset: int = 1,
    limit: int = READ_DEFAULT_LIMIT,
) -> str:
    """Format OpenCode's read-directory branch."""
    offset = offset or 1
    if offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if limit < 0:
        raise ValueError("limit must be a non-negative integer")

    start = offset - 1
    page = entries[start : start + limit]
    truncated = start + len(page) < len(entries)
    if truncated:
        status = (
            f"(Showing {len(page)} of {len(entries)} entries. Use 'offset' "
            f"parameter to read beyond entry {offset + len(page)})"
        )
    else:
        status = f"({len(entries)} entries)"
    return (
        f"<path>{filepath}</path>\n"
        "<type>directory</type>\n"
        "<entries>\n" + "\n".join(page) + f"\n\n{status}\n</entries>"
    )


def format_glob(paths: list[str]) -> str:
    if not paths:
        return "No files found"
    shown = paths[:SEARCH_LIMIT]
    output = "\n".join(shown)
    if len(shown) == SEARCH_LIMIT:
        output += (
            "\n\n(Results are truncated: showing first 100 results. "
            "Consider using a more specific path or pattern.)"
        )
    return output


def format_grep(matches: list[GrepMatch]) -> str:
    if not matches:
        return "No files found"

    shown = matches[:SEARCH_LIMIT]
    truncated = len(shown) == SEARCH_LIMIT
    output = [
        f"Found {len(shown)} matches"
        + (" (more matches available)" if truncated else "")
    ]
    current_path = ""
    for match in shown:
        if current_path != match.path:
            if current_path:
                output.append("")
            current_path = match.path
            output.append(f"{match.path}:")
        text = match.text
        if _utf16_length(text) > READ_MAX_LINE_LENGTH:
            text = _utf16_prefix(text, READ_MAX_LINE_LENGTH) + "..."
        output.append(f"  Line {match.line}: {text}")
    if truncated:
        output.extend(
            ["", "(Results truncated. Consider using a more specific path or pattern.)"]
        )
    return "\n".join(output)


def tail_shell_output(
    text: str,
    *,
    max_lines: int = SHELL_MAX_LINES,
    max_bytes: int = SHELL_MAX_BYTES,
) -> tuple[str, bool]:
    """Retain the same UTF-8-safe tail as OpenCode's shell tool."""
    lines = text.split("\n")
    if len(lines) <= max_lines and _utf8_size(text) <= max_bytes:
        return text, False

    output: list[str] = []
    used_bytes = 0
    for line in reversed(lines):
        if len(output) >= max_lines:
            break
        size = _utf8_size(line) + (1 if output else 0)
        if used_bytes + size > max_bytes:
            if not output:
                encoded = line.encode("utf-8", errors="surrogatepass")
                output.append(encoded[-max_bytes:].decode("utf-8", errors="ignore"))
            break
        output.insert(0, line)
        used_bytes += size
    return "\n".join(output), True


def format_shell_output(
    text: str,
    *,
    output_path: str | None = None,
    timeout_ms: int | None = None,
    aborted: bool = False,
    metadata_messages: list[str] | None = None,
) -> str:
    output = text or "(no output)"
    if output_path:
        output = (
            f"...output truncated...\n\nFull output saved to: {output_path}\n\n{output}"
        )

    metadata: list[str] = []
    if timeout_ms is not None:
        metadata.append(
            "shell tool terminated command after exceeding timeout "
            f"{timeout_ms} ms. If this command is expected to take longer and is "
            "not waiting for interactive input, retry with a larger timeout value "
            "in milliseconds."
        )
    if aborted:
        metadata.append("User aborted the command")
    if metadata_messages:
        metadata.extend(metadata_messages)
    if metadata:
        output += "\n\n<shell_metadata>\n" + "\n".join(metadata)
        output += "\n</shell_metadata>"
    return output


def truncate_tool_output_head(
    text: str,
    *,
    max_lines: int = SHELL_MAX_LINES,
    max_bytes: int = SHELL_MAX_BYTES,
) -> HeadTruncation:
    """Apply OpenCode's generic, head-oriented tool output cap."""
    lines = text.split("\n")
    total_bytes = _utf8_size(text)
    if len(lines) <= max_lines and total_bytes <= max_bytes:
        return HeadTruncation(text=text, truncated=False)

    output: list[str] = []
    used_bytes = 0
    hit_bytes = False
    for index, line in enumerate(lines):
        if index >= max_lines:
            break
        size = _utf8_size(line) + (1 if index > 0 else 0)
        if used_bytes + size > max_bytes:
            hit_bytes = True
            break
        output.append(line)
        used_bytes += size

    return HeadTruncation(
        text="\n".join(output),
        truncated=True,
        removed=(total_bytes - used_bytes if hit_bytes else len(lines) - len(output)),
        unit="bytes" if hit_bytes else "lines",
    )


def format_truncated_tool_output(
    preview: HeadTruncation,
    *,
    output_path: str,
) -> str:
    """Render OpenCode's no-Task-tool generic truncation body."""
    if not preview.truncated:
        return preview.text
    return (
        f"{preview.text}\n\n...{preview.removed} {preview.unit} truncated...\n\n"
        f"The tool call succeeded but the output was truncated. Full output saved to: {output_path}\n"
        "Use Grep to search the full content or Read with offset/limit to view specific sections."
    )
