"""Exact model-visible OpenCode tool-body contract tests.

The expected strings are pinned to anomalyco/opencode commit
f0afb6750e63ee0a60b052914531bde0afb9bc2b. These are deliberately pure tests:
UI metadata, execution, and provider envelopes must not affect the body text.
"""

import pytest

from openhands.agenthub.opencode_agent.tool_output import (
    READ_MAX_BYTES,
    READ_MAX_LINE_LENGTH,
    SEARCH_LIMIT,
    GrepMatch,
    format_glob,
    format_grep,
    format_read_directory,
    format_read_file,
    format_shell_output,
    format_truncated_tool_output,
    tail_shell_output,
    truncate_tool_output_head,
)


FILEPATH = "/workspace/example.txt"
DIRECTORY = "/workspace/project"


def _read_file_body(lines: list[str], footer: str, *, offset: int = 1) -> str:
    numbered = "\n".join(
        f"{number}: {line}" for number, line in enumerate(lines, start=offset)
    )
    return (
        f"<path>{FILEPATH}</path>\n"
        "<type>file</type>\n"
        f"<content>\n{numbered}\n\n{footer}\n</content>"
    )


def test_format_read_file_normal_exact() -> None:
    assert format_read_file(FILEPATH, ["alpha", "beta"]) == (
        "<path>/workspace/example.txt</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        "1: alpha\n"
        "2: beta\n\n"
        "(End of file - total 2 lines)\n"
        "</content>"
    )


def test_format_read_file_empty_exact() -> None:
    assert format_read_file(FILEPATH, []) == (
        "<path>/workspace/example.txt</path>\n"
        "<type>file</type>\n"
        "<content>\n\n\n"
        "(End of file - total 0 lines)\n"
        "</content>"
    )


def test_format_read_file_pagination_exact() -> None:
    assert format_read_file(
        FILEPATH,
        ["one", "two", "three", "four", "five"],
        offset=2,
        limit=2,
    ) == (
        "<path>/workspace/example.txt</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        "2: two\n"
        "3: three\n\n"
        "(Showing lines 2-3 of 5. Use offset=4 to continue.)\n"
        "</content>"
    )


@pytest.mark.parametrize(
    ("lines", "offset", "message"),
    [
        (
            ["one", "two", "three"],
            4,
            "Offset 4 is out of range for this file (3 lines)",
        ),
        ([], 2, "Offset 2 is out of range for this file (0 lines)"),
    ],
)
def test_format_read_file_out_of_range_exact(
    lines: list[str], offset: int, message: str
) -> None:
    with pytest.raises(ValueError) as exc_info:
        format_read_file(FILEPATH, lines, offset=offset)
    assert str(exc_info.value) == message


def test_format_read_file_2000_character_line_is_not_truncated() -> None:
    line = "x" * READ_MAX_LINE_LENGTH

    assert format_read_file(FILEPATH, [line]) == _read_file_body(
        [line], "(End of file - total 1 lines)"
    )


def test_format_read_file_2001_character_line_is_truncated_exactly() -> None:
    source = "x" * (READ_MAX_LINE_LENGTH + 1)
    shown = "x" * READ_MAX_LINE_LENGTH + "... (line truncated to 2000 chars)"

    assert format_read_file(FILEPATH, [source]) == _read_file_body(
        [shown], "(End of file - total 1 lines)"
    )


def test_format_read_file_counts_astral_text_as_utf16_units() -> None:
    source = "😀" * 1_500
    shown = "😀" * 1_000 + "... (line truncated to 2000 chars)"

    assert format_read_file(FILEPATH, [source]) == _read_file_body(
        [shown], "(End of file - total 1 lines)"
    )


def test_format_read_file_50_kib_multibyte_boundary_exact() -> None:
    # Eight 6,000-byte lines plus a 3,192-byte line and eight separators is
    # 51,200. Euro is multibyte in UTF-8 but one UTF-16 code unit, matching JS.
    exact_last = "€" * 1_064
    exact = ["€" * READ_MAX_LINE_LENGTH] * 8 + [exact_last]
    assert sum(len(line.encode("utf-8")) for line in exact) + len(exact) - 1 == (
        READ_MAX_BYTES
    )

    assert format_read_file(FILEPATH, exact) == _read_file_body(
        exact, "(End of file - total 9 lines)"
    )

    over = [*exact, "z"]
    assert format_read_file(FILEPATH, over) == _read_file_body(
        exact,
        "(Output capped at 50 KB. Showing lines 1-9. Use offset=10 to continue.)",
    )


def test_format_read_directory_normal_exact() -> None:
    assert format_read_directory(DIRECTORY, ["README.md", "src/"]) == (
        "<path>/workspace/project</path>\n"
        "<type>directory</type>\n"
        "<entries>\n"
        "README.md\n"
        "src/\n\n"
        "(2 entries)\n"
        "</entries>"
    )


def test_format_read_directory_empty_exact() -> None:
    assert format_read_directory(DIRECTORY, []) == (
        "<path>/workspace/project</path>\n"
        "<type>directory</type>\n"
        "<entries>\n\n\n"
        "(0 entries)\n"
        "</entries>"
    )


def test_format_read_directory_pagination_exact() -> None:
    assert format_read_directory(
        DIRECTORY,
        ["a", "b", "c", "d"],
        offset=2,
        limit=2,
    ) == (
        "<path>/workspace/project</path>\n"
        "<type>directory</type>\n"
        "<entries>\n"
        "b\n"
        "c\n\n"
        "(Showing 2 of 4 entries. Use 'offset' parameter to read beyond entry 4)\n"
        "</entries>"
    )


def test_format_glob_zero_exact() -> None:
    assert format_glob([]) == "No files found"


def test_format_glob_99_exact_without_truncation_footer() -> None:
    paths = [f"/workspace/file-{index:03}.py" for index in range(99)]

    assert format_glob(paths) == "\n".join(paths)


@pytest.mark.parametrize("count", [100, 101])
def test_format_glob_100_result_boundary_exact(count: int) -> None:
    paths = [f"/workspace/file-{index:03}.py" for index in range(count)]
    shown = paths[:SEARCH_LIMIT]
    expected = (
        "\n".join(shown) + "\n\n(Results are truncated: showing first 100 results. "
        "Consider using a more specific path or pattern.)"
    )

    assert format_glob(paths) == expected
    if count == 101:
        assert paths[100] not in format_glob(paths)


def test_format_grep_zero_exact() -> None:
    assert format_grep([]) == "No files found"


def test_format_grep_groups_matches_for_one_file_exact() -> None:
    matches = [
        GrepMatch("/workspace/a.py", 2, "first"),
        GrepMatch("/workspace/a.py", 7, "second"),
    ]

    assert format_grep(matches) == (
        "Found 2 matches\n/workspace/a.py:\n  Line 2: first\n  Line 7: second"
    )


def test_format_grep_separates_multiple_files_exact() -> None:
    matches = [
        GrepMatch("/workspace/a.py", 1, "alpha"),
        GrepMatch("/workspace/b.py", 3, "beta"),
        GrepMatch("/workspace/b.py", 4, "gamma"),
    ]

    assert format_grep(matches) == (
        "Found 3 matches\n"
        "/workspace/a.py:\n"
        "  Line 1: alpha\n\n"
        "/workspace/b.py:\n"
        "  Line 3: beta\n"
        "  Line 4: gamma"
    )


def test_format_grep_preserves_raw_match_newlines_exact() -> None:
    matches = [
        GrepMatch("/workspace/a.py", 1, "alpha\n"),
        GrepMatch("/workspace/a.py", 2, "beta\n"),
    ]

    assert format_grep(matches) == (
        "Found 2 matches\n/workspace/a.py:\n  Line 1: alpha\n\n  Line 2: beta\n"
    )


def test_format_grep_long_line_is_capped_at_2000_characters_exact() -> None:
    source = "x" * (READ_MAX_LINE_LENGTH + 1)
    shown = "x" * READ_MAX_LINE_LENGTH + "..."

    assert format_grep([GrepMatch("/workspace/a.py", 9, source)]) == (
        f"Found 1 matches\n/workspace/a.py:\n  Line 9: {shown}"
    )


def test_format_grep_counts_astral_text_as_utf16_units() -> None:
    source = "😀" * 1_500
    shown = "😀" * 1_000 + "..."

    assert format_grep([GrepMatch("/workspace/a.py", 9, source)]) == (
        f"Found 1 matches\n/workspace/a.py:\n  Line 9: {shown}"
    )


def test_format_grep_100_result_boundary_exact() -> None:
    matches = [
        GrepMatch("/workspace/a.py", index, f"match-{index:03}")
        for index in range(1, SEARCH_LIMIT + 1)
    ]
    rows = "\n".join(f"  Line {match.line}: {match.text}" for match in matches)
    expected = (
        "Found 100 matches (more matches available)\n"
        f"/workspace/a.py:\n{rows}\n\n"
        "(Results truncated. Consider using a more specific path or pattern.)"
    )

    assert format_grep(matches) == expected


def test_format_shell_no_output_exact() -> None:
    assert format_shell_output("") == "(no output)"


def test_format_shell_nonzero_exit_is_body_neutral() -> None:
    # Exit status is UI metadata upstream and therefore is not a formatter input.
    assert format_shell_output("failure on stderr\n") == "failure on stderr\n"


def test_format_shell_timeout_exact() -> None:
    assert format_shell_output("", timeout_ms=500) == (
        "(no output)\n\n"
        "<shell_metadata>\n"
        "shell tool terminated command after exceeding timeout 500 ms. If this "
        "command is expected to take longer and is not waiting for interactive "
        "input, retry with a larger timeout value in milliseconds.\n"
        "</shell_metadata>"
    )


def test_generic_tool_output_exact_line_boundary() -> None:
    source = "\n".join(f"line-{index}" for index in range(2_001))
    preview = truncate_tool_output_head(source)

    assert preview.truncated is True
    assert preview.text == "\n".join(f"line-{index}" for index in range(2_000))
    assert preview.removed == 1
    assert preview.unit == "lines"
    assert format_truncated_tool_output(
        preview,
        output_path="/tmp/tool_123",
    ).endswith(
        "...1 lines truncated...\n\n"
        "The tool call succeeded but the output was truncated. "
        "Full output saved to: /tmp/tool_123\n"
        "Use Grep to search the full content or Read with offset/limit to view "
        "specific sections."
    )


def test_generic_tool_output_exact_byte_boundary() -> None:
    source = "x" * (50 * 1_024 + 7)
    preview = truncate_tool_output_head(source)

    assert preview.truncated is True
    assert preview.text == ""
    assert preview.removed == len(source)
    assert preview.unit == "bytes"


def test_generic_tool_output_at_limits_is_unchanged() -> None:
    source = "x" * (50 * 1_024)

    assert truncate_tool_output_head(source).text == source
    assert truncate_tool_output_head(source).truncated is False


def test_format_shell_abort_exact() -> None:
    assert format_shell_output("partial\n", aborted=True) == (
        "partial\n\n\n<shell_metadata>\nUser aborted the command\n</shell_metadata>"
    )


def test_shell_line_tail_truncation_and_final_body_exact() -> None:
    tail, truncated = tail_shell_output("one\ntwo\nthree\nfour", max_lines=3)

    assert truncated is True
    assert tail == "two\nthree\nfour"
    assert format_shell_output(tail, output_path="/tmp/full-output") == (
        "...output truncated...\n\n"
        "Full output saved to: /tmp/full-output\n\n"
        "two\nthree\nfour"
    )


def test_shell_line_tail_exact_limit_is_not_truncated() -> None:
    assert tail_shell_output("one\ntwo\nthree", max_lines=3) == (
        "one\ntwo\nthree",
        False,
    )


@pytest.mark.parametrize(
    ("max_bytes", "expected"),
    [
        (7, "β\n😀"),
        (6, "😀"),
    ],
)
def test_shell_utf8_multibyte_tail_boundary_exact(
    max_bytes: int, expected: str
) -> None:
    # Encoded sizes: a=1, β=2, 😀=4, plus one byte per retained separator.
    tail, truncated = tail_shell_output("a\nβ\n😀", max_lines=100, max_bytes=max_bytes)

    assert truncated is True
    assert tail == expected
    assert len(tail.encode("utf-8")) <= max_bytes


def test_shell_utf8_tail_never_starts_inside_a_codepoint() -> None:
    # The six-byte suffix begins inside 😀; OpenCode advances to the next lead byte.
    tail, truncated = tail_shell_output("A😀éz", max_lines=100, max_bytes=6)

    assert truncated is True
    assert tail == "éz"
    assert tail.encode("utf-8").decode("utf-8") == tail
