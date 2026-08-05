"""Exact model-visible Codex tool-body contract tests."""

import pytest

from openhands.agenthub.codex_agent.tool_output import (
    CODEX_MAX_LINE_LENGTH,
    format_apply_patch_success,
    format_read_indentation,
    format_read_slice,
    format_shell_output,
    split_file_lines,
    take_utf8_prefix,
    truncate_function_output,
    uses_byte_truncation,
)


def test_format_shell_normal_exact() -> None:
    assert format_shell_output(
        "alpha\nbeta\n",
        exit_code=0,
        duration_seconds=1.25,
        use_tokens=False,
    ) == ("Exit code: 0\nWall time: 1.3 seconds\nOutput:\nalpha\nbeta\n")


def test_format_shell_empty_exact() -> None:
    assert (
        format_shell_output(
            "",
            exit_code=0,
            duration_seconds=0,
            use_tokens=False,
        )
        == "Exit code: 0\nWall time: 0 seconds\nOutput:\n"
    )


def test_format_shell_rounds_after_rust_f32_conversion_exact() -> None:
    assert (
        format_shell_output(
            "",
            exit_code=0,
            duration_seconds=1.24999999,
            use_tokens=False,
        )
        == "Exit code: 0\nWall time: 1.3 seconds\nOutput:\n"
    )


def test_format_shell_timeout_exact() -> None:
    assert format_shell_output(
        "partial output",
        exit_code=124,
        duration_seconds=2.04,
        timed_out_ms=2_000,
        use_tokens=False,
    ) == (
        "Exit code: 124\n"
        "Wall time: 2 seconds\n"
        "Output:\n"
        "command timed out after 2000 milliseconds\n"
        "partial output"
    )


def test_format_shell_single_line_byte_truncation_exact() -> None:
    assert format_shell_output(
        "abcdefghijklmnopqrst",
        exit_code=7,
        duration_seconds=0.96,
        limit=10,
        use_tokens=False,
    ) == ("Exit code: 7\nWall time: 1 seconds\nOutput:\nabcde…10 chars truncated…pqrst")


def test_format_shell_token_truncation_exact() -> None:
    assert format_shell_output(
        "a" * 80,
        exit_code=0,
        duration_seconds=0,
        limit=10,
        use_tokens=True,
    ) == (
        "Exit code: 0\n"
        "Wall time: 0 seconds\n"
        "Output:\n"
        f"{'a' * 20}…10 tokens truncated…{'a' * 20}"
    )


def test_format_shell_truncation_adds_original_line_count_exact() -> None:
    assert format_shell_output(
        "first\nsecond\nthird\nfourth",
        exit_code=0,
        duration_seconds=0.04,
        limit=12,
        use_tokens=False,
    ) == (
        "Exit code: 0\n"
        "Wall time: 0 seconds\n"
        "Total output lines: 4\n"
        "Output:\n"
        "first\n"
        "…13 chars truncated…fourth"
    )


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        (None, True),
        ("unknown-model", True),
        ("gpt-5.2", True),
        ("openai/gpt-5.2", True),
        ("gpt-5.2-codex", False),
        ("gpt-5.2-codex-2026-08-04", False),
        ("gpt-5.2-future", True),
        ("gpt-5.1", True),
        ("gpt-5.1-codex-preview", False),
        ("openai/gpt-5.6-sol", False),
        ("gpt-5.6-sol-2026-08-04", False),
        ("openai/gpt-5.6-sol-2026-08-04", False),
        ("gpt-5.6-sol/nested-alias", False),
        ("GPT-5.6-TERRA", True),
        ("team/openai/gpt-5.6-sol", True),
        ("open.ai/gpt-5.6-sol", True),
    ],
)
def test_shell_model_truncation_policy_exact(
    model_name: str | None,
    expected: bool,
) -> None:
    assert uses_byte_truncation(model_name) is expected


def test_function_output_unknown_model_uses_12k_byte_budget_exact() -> None:
    source = "a" * 12_001

    assert truncate_function_output(source, model_name="unknown-model") == (
        "a" * 6_000 + "…1 chars truncated…" + "a" * 6_000
    )


def test_function_output_known_token_model_uses_12k_token_budget_exact() -> None:
    source = "a" * 12_001

    assert truncate_function_output(source, model_name="gpt-5.6-sol") == source


@pytest.mark.parametrize(
    ("max_bytes", "expected"),
    [
        (0, ""),
        (4, "A"),
        (5, "A😀"),
        (6, "A😀"),
        (7, "A😀é"),
        (8, "A😀éZ"),
    ],
)
def test_take_utf8_prefix_never_splits_a_codepoint(
    max_bytes: int,
    expected: str,
) -> None:
    assert take_utf8_prefix("A😀éZ", max_bytes) == expected


def test_split_file_lines_matches_lossy_crlf_reading_exact() -> None:
    # A CR is stripped from CRLF records, while an unterminated final CR is data.
    assert split_file_lines(b"alpha\r\nbeta\nbad:\xff\nlast\r") == [
        "alpha",
        "beta",
        "bad:�",
        "last\r",
    ]


def test_split_file_lines_empty_and_final_newline_exact() -> None:
    assert split_file_lines(b"") == []
    assert split_file_lines(b"one\ntwo\n") == ["one", "two"]


def test_format_read_slice_body_exact() -> None:
    assert (
        format_read_slice(
            ["alpha", "beta", "gamma"],
            offset=2,
            limit=2,
        )
        == "L2: beta\nL3: gamma"
    )


def test_format_read_slice_at_end_and_zero_limit_exact() -> None:
    lines = ["alpha", "beta", "gamma"]

    assert format_read_slice(lines, offset=3, limit=20) == "L3: gamma"
    assert format_read_slice(lines, offset=1, limit=0) == ""


@pytest.mark.parametrize(
    ("lines", "offset"),
    [
        ([], 1),
        (["one", "two"], 3),
    ],
)
def test_format_read_slice_offset_error_exact(
    lines: list[str],
    offset: int,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        format_read_slice(lines, offset=offset, limit=1)

    assert str(exc_info.value) == "offset exceeds file length"


def test_format_read_slice_caps_lines_at_utf8_byte_boundary_exact() -> None:
    source = "a" * (CODEX_MAX_LINE_LENGTH - 2) + "é" + "z"
    shown = "a" * (CODEX_MAX_LINE_LENGTH - 2) + "é"

    assert len(source.encode("utf-8")) == CODEX_MAX_LINE_LENGTH + 1
    assert len(shown.encode("utf-8")) == CODEX_MAX_LINE_LENGTH
    assert format_read_slice([source], offset=1, limit=1) == f"L1: {shown}"


def test_format_read_indentation_parent_window_exact() -> None:
    lines = [
        "class A:",
        "    def first():",
        "        return 1",
        "",
        "    def second():",
        "        return 2",
        "",
        "outside = 3",
    ]

    assert format_read_indentation(
        lines,
        offset=3,
        limit=6,
        anchor_line=3,
        max_levels=1,
        include_siblings=False,
        include_header=False,
        max_lines=None,
    ) == (
        "L2:     def first():\n"
        "L3:         return 1\n"
        "L4: \n"
        "L5:     def second():\n"
        "L6:         return 2"
    )


@pytest.mark.parametrize(
    ("include_siblings", "expected"),
    [
        (
            False,
            "L1: def first():\nL2:     one()\nL3: \nL4: def second():\nL5:     two()",
        ),
        (
            True,
            "L1: def first():\n"
            "L2:     one()\n"
            "L3: \n"
            "L4: def second():\n"
            "L5:     two()\n"
            "L6: \n"
            "L7: def third():\n"
            "L8:     three()",
        ),
    ],
)
def test_format_read_indentation_sibling_boundary_exact(
    include_siblings: bool,
    expected: str,
) -> None:
    lines = [
        "def first():",
        "    one()",
        "",
        "def second():",
        "    two()",
        "",
        "def third():",
        "    three()",
    ]

    assert (
        format_read_indentation(
            lines,
            offset=2,
            limit=8,
            anchor_line=2,
            max_levels=0,
            include_siblings=include_siblings,
            include_header=False,
            max_lines=None,
        )
        == expected
    )


def test_format_read_indentation_includes_comment_header_exact() -> None:
    lines = [
        "# header one",
        "# header two",
        "def function():",
        "    return 1",
        "",
        "def next_function():",
    ]

    assert format_read_indentation(
        lines,
        offset=4,
        limit=6,
        anchor_line=4,
        max_levels=0,
        include_siblings=False,
        include_header=True,
        max_lines=None,
    ) == (
        "L1: # header one\n"
        "L2: # header two\n"
        "L3: def function():\n"
        "L4:     return 1\n"
        "L5: \n"
        "L6: def next_function():"
    )


@pytest.mark.parametrize(
    ("lines", "anchor_line", "max_lines", "message"),
    [
        (["one"], 0, None, "anchor_line must be a 1-indexed line number"),
        (["one"], 1, 0, "max_lines must be greater than zero"),
        ([], 1, None, "anchor_line exceeds file length"),
        (["one"], 2, None, "anchor_line exceeds file length"),
    ],
)
def test_format_read_indentation_errors_exact(
    lines: list[str],
    anchor_line: int,
    max_lines: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        format_read_indentation(
            lines,
            offset=1,
            limit=20,
            anchor_line=anchor_line,
            max_levels=0,
            include_siblings=False,
            include_header=False,
            max_lines=max_lines,
        )

    assert str(exc_info.value) == message


def test_format_apply_patch_success_exact() -> None:
    assert format_apply_patch_success(
        added=["new.py"],
        modified=["changed.py"],
        deleted=["old.py"],
    ) == ("Success. Updated the following files:\nA new.py\nM changed.py\nD old.py\n")
