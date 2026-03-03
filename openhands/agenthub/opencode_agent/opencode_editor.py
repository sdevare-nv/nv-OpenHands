"""
OpenCode-style editor that extends OHEditor with fuzzy matching replacers.
Based on the OpenCode edit.ts implementation.

INSTALLATION:
To use this editor instead of the default OHEditor, modify the runtime's
file_editor initialization:

1. In openhands/runtime/action_execution_server.py, change:
   from openhands_aci.editor import OHEditor
   to:
   from openhands.agenthub.opencode_agent.opencode_editor import OpenCodeEditor as OHEditor

2. In openhands/runtime/impl/cli/cli_runtime.py, change:
   from openhands_aci.editor import OHEditor
   to:
   from openhands.agenthub.opencode_agent.opencode_editor import OpenCodeEditor as OHEditor

This enables fuzzy matching for string replacements, making the editor more
robust when dealing with whitespace differences, indentation variations, etc.

References:
- https://github.com/cline/cline/blob/main/evals/diff-edits/diff-apply/diff-06-23-25.ts
- https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/utils/editCorrector.ts
"""

import re
from pathlib import Path
from typing import Generator

from openhands_aci.editor import OHEditor
from openhands_aci.editor.results import CLIResult
from openhands_aci.editor.exceptions import ToolError
from openhands_aci.editor.config import SNIPPET_CONTEXT_WINDOW

# Import the fuzzy matching functions from opencode_impl
from openhands.agenthub.opencode_agent.opencode_impl import (
    simple_replacer,
    line_trimmed_replacer,
    block_anchor_replacer,
    whitespace_normalized_replacer,
    indentation_flexible_replacer,
    escape_normalized_replacer,
    trimmed_boundary_replacer,
    context_aware_replacer,
    multi_occurrence_replacer,
)


def fuzzy_find_match(content: str, old_str: str) -> str | None:
    """
    Try to find a match using fuzzy matching replacers.
    Returns the matched string if found (unique match), None otherwise.
    """
    replacers = [
        simple_replacer,
        line_trimmed_replacer,
        block_anchor_replacer,
        whitespace_normalized_replacer,
        indentation_flexible_replacer,
        escape_normalized_replacer,
        trimmed_boundary_replacer,
        context_aware_replacer,
        multi_occurrence_replacer,
    ]

    for replacer in replacers:
        matches = list(replacer(content, old_str))
        for search in matches:
            index = content.find(search)
            if index == -1:
                continue

            last_index = content.rfind(search)
            # Only return if unique match
            if index == last_index:
                return search

    return None


class OpenCodeEditor(OHEditor):
    """
    Extended OHEditor with OpenCode-style fuzzy matching for str_replace.
    Falls back to fuzzy matching when exact match fails.

    This class is a drop-in replacement for OHEditor that adds fuzzy
    string matching capabilities based on the OpenCode implementation.

    Fuzzy matching strategies (in order of priority):
    1. Simple exact match
    2. Line-trimmed matching (ignores leading/trailing whitespace per line)
    3. Block anchor matching (matches by first/last line with similarity check)
    4. Whitespace-normalized matching
    5. Indentation-flexible matching
    6. Escape-normalized matching
    7. Trimmed boundary matching
    8. Context-aware matching
    9. Multi-occurrence handling

    Usage:
        # In your runtime initialization, replace:
        self.file_editor = OHEditor(...)
        # With:
        from openhands.agenthub.opencode_agent.opencode_editor import OpenCodeEditor
        self.file_editor = OpenCodeEditor(...)
    """

    def str_replace(
        self,
        path: Path,
        old_str: str,
        new_str: str | None,
        enable_linting: bool,
        encoding: str = 'utf-8',
    ) -> CLIResult:
        """
        Override str_replace to add fuzzy matching fallback.

        First tries exact match (like OHEditor), then tries fuzzy matching
        if exact match fails.

        Args:
            path: Path to the file to edit
            old_str: String to find and replace
            new_str: Replacement string (None means empty string)
            enable_linting: Whether to run linting after edit
            encoding: File encoding (auto-detected if not specified)

        Returns:
            CLIResult with edit operation results
        """
        self.validate_file(path)
        new_str = new_str or ''

        file_content = self.read_file(path)

        # First, try exact match (OHEditor behavior)
        pattern = re.escape(old_str)
        occurrences = [
            (
                file_content.count('\n', 0, match.start()) + 1,
                match.group(),
                match.start(),
            )
            for match in re.finditer(pattern, file_content)
        ]

        # If no exact match, try stripping whitespace (OHEditor fallback)
        actual_old_str = old_str
        actual_new_str = new_str
        if not occurrences:
            old_str_stripped = old_str.strip()
            new_str_stripped = new_str.strip()
            pattern = re.escape(old_str_stripped)
            occurrences = [
                (
                    file_content.count('\n', 0, match.start()) + 1,
                    match.group(),
                    match.start(),
                )
                for match in re.finditer(pattern, file_content)
            ]
            if occurrences:
                actual_old_str = old_str_stripped
                actual_new_str = new_str_stripped

        # If still no match, try fuzzy matching (OpenCode behavior)
        if not occurrences:
            fuzzy_match = fuzzy_find_match(file_content, old_str)
            if fuzzy_match:
                idx = file_content.find(fuzzy_match)
                if idx != -1:
                    line_num = file_content.count('\n', 0, idx) + 1
                    occurrences = [(line_num, fuzzy_match, idx)]

        if not occurrences:
            raise ToolError(
                f'No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}.'
            )

        if len(occurrences) > 1:
            line_numbers = sorted(set(line for line, _, _ in occurrences))
            raise ToolError(
                f'No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {line_numbers}. Please ensure it is unique.'
            )

        # We found exactly one occurrence
        replacement_line, matched_text, idx = occurrences[0]

        # Create new content
        new_file_content = (
            file_content[:idx] + actual_new_str + file_content[idx + len(matched_text):]
        )

        # Write the new content
        self.write_file(path, new_file_content)

        # Save to history
        self._history_manager.add_history(path, file_content)

        # Create snippet
        start_line = max(0, replacement_line - SNIPPET_CONTEXT_WINDOW)
        end_line = replacement_line + SNIPPET_CONTEXT_WINDOW + actual_new_str.count('\n')

        snippet = self.read_file(path, start_line=start_line + 1, end_line=end_line)

        success_message = f'The file {path} has been edited. '
        success_message += self._make_output(
            snippet, f'a snippet of {path}', start_line + 1
        )

        if enable_linting:
            lint_results = self._run_linting(
                file_content, new_file_content, path
            )
            success_message += '\n' + lint_results + '\n'

        success_message += 'Review the changes and make sure they are as expected. Edit the file again if necessary.'

        return CLIResult(
            output=success_message,
            prev_exist=True,
            path=str(path),
            old_content=file_content,
            new_content=new_file_content,
        )

