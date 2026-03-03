"""Unit tests for Terminus-2 JSON plain parser.

Tests the TerminusJSONPlainParser for parsing LLM responses into
structured commands, including auto-correction and validation.
"""

import pytest

from openhands.agenthub.terminus_2_agent.terminus_json_plain_parser import (
    ParsedCommand,
    ParseResult,
    TerminusJSONPlainParser,
)


@pytest.fixture
def parser():
    return TerminusJSONPlainParser()


# ==============================================================================
# Basic Parsing Tests
# ==============================================================================


class TestBasicParsing:
    """Tests for basic JSON response parsing."""

    def test_parse_valid_response(self, parser):
        response = '''{
            "analysis": "Looking at the directory",
            "plan": "List files and check structure",
            "commands": [
                {"keystrokes": "ls -la\\n", "duration": 0.1}
            ]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 1
        assert result.commands[0].keystrokes == 'ls -la\n'
        assert result.commands[0].duration == 0.1
        assert result.is_task_complete is False

    def test_parse_multiple_commands(self, parser):
        response = '''{
            "analysis": "Need to navigate and list",
            "plan": "cd then ls",
            "commands": [
                {"keystrokes": "cd /tmp\\n", "duration": 0.1},
                {"keystrokes": "ls -la\\n", "duration": 0.1}
            ]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 2
        assert result.commands[0].keystrokes == 'cd /tmp\n'
        assert result.commands[1].keystrokes == 'ls -la\n'

    def test_parse_task_complete(self, parser):
        response = '''{
            "analysis": "Task is done",
            "plan": "Mark complete",
            "commands": [],
            "task_complete": true
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.is_task_complete is True
        assert len(result.commands) == 0

    def test_parse_task_complete_string(self, parser):
        response = '''{
            "analysis": "Done",
            "plan": "Finish",
            "commands": [],
            "task_complete": "true"
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.is_task_complete is True

    def test_parse_task_not_complete(self, parser):
        response = '''{
            "analysis": "Still working",
            "plan": "Continue",
            "commands": [],
            "task_complete": false
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.is_task_complete is False

    def test_parse_no_task_complete_field(self, parser):
        """task_complete defaults to False when not present."""
        response = '''{
            "analysis": "Working",
            "plan": "Continue",
            "commands": []
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.is_task_complete is False

    def test_parse_empty_commands(self, parser):
        response = '''{
            "analysis": "Waiting for output",
            "plan": "Do nothing",
            "commands": []
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 0


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestErrorHandling:
    """Tests for error detection and reporting."""

    def test_no_json_found(self, parser):
        result = parser.parse_response('This is not JSON at all')
        assert 'No valid JSON found' in result.error

    def test_invalid_json(self, parser):
        result = parser.parse_response('{"analysis": "test", "plan": broken}')
        assert result.error != ''

    def test_missing_required_field_analysis(self, parser):
        response = '''{
            "plan": "Do something",
            "commands": []
        }'''
        result = parser.parse_response(response)
        assert 'Missing required fields' in result.error
        assert 'analysis' in result.error

    def test_missing_required_field_plan(self, parser):
        response = '''{
            "analysis": "Something",
            "commands": []
        }'''
        result = parser.parse_response(response)
        assert 'Missing required fields' in result.error
        assert 'plan' in result.error

    def test_missing_required_field_commands(self, parser):
        response = '''{
            "analysis": "Something",
            "plan": "Do it"
        }'''
        result = parser.parse_response(response)
        assert 'Missing required fields' in result.error
        assert 'commands' in result.error

    def test_commands_not_array(self, parser):
        response = '''{
            "analysis": "Something",
            "plan": "Do it",
            "commands": "not an array"
        }'''
        result = parser.parse_response(response)
        assert "must be an array" in result.error

    def test_command_not_object(self, parser):
        response = '''{
            "analysis": "Something",
            "plan": "Do it",
            "commands": ["not an object"]
        }'''
        result = parser.parse_response(response)
        assert 'must be an object' in result.error

    def test_command_missing_keystrokes(self, parser):
        response = '''{
            "analysis": "Something",
            "plan": "Do it",
            "commands": [{"duration": 1.0}]
        }'''
        result = parser.parse_response(response)
        assert "missing required 'keystrokes' field" in result.error

    def test_command_keystrokes_not_string(self, parser):
        response = '''{
            "analysis": "Something",
            "plan": "Do it",
            "commands": [{"keystrokes": 123}]
        }'''
        result = parser.parse_response(response)
        assert "'keystrokes' must be a string" in result.error

    def test_not_json_object(self, parser):
        response = '["not", "an", "object"]'
        assert parser.parse_response(response).error != ''


# ==============================================================================
# Warning Tests
# ==============================================================================


class TestWarnings:
    """Tests for warning generation."""

    def test_extra_text_before_json(self, parser):
        response = 'Here is my response:\n{"analysis": "a", "plan": "b", "commands": []}'
        result = parser.parse_response(response)

        assert result.error == ''
        assert 'Extra text detected before JSON' in result.warning

    def test_extra_text_after_json(self, parser):
        response = '{"analysis": "a", "plan": "b", "commands": []}\nDone!'
        result = parser.parse_response(response)

        assert result.error == ''
        assert 'Extra text detected after JSON' in result.warning

    def test_missing_duration_warning(self, parser):
        response = '''{
            "analysis": "a",
            "plan": "b",
            "commands": [{"keystrokes": "ls\\n"}]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert 'Missing duration field' in result.warning
        assert result.commands[0].duration == 1.0

    def test_invalid_duration_type_warning(self, parser):
        response = '''{
            "analysis": "a",
            "plan": "b",
            "commands": [{"keystrokes": "ls\\n", "duration": "fast"}]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert 'Invalid duration value' in result.warning
        assert result.commands[0].duration == 1.0

    def test_unknown_fields_warning(self, parser):
        response = '''{
            "analysis": "a",
            "plan": "b",
            "commands": [{"keystrokes": "ls\\n", "duration": 0.1, "extra_field": "x"}]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert 'Unknown fields' in result.warning

    def test_no_newline_between_commands_warning(self, parser):
        response = '''{
            "analysis": "a",
            "plan": "b",
            "commands": [
                {"keystrokes": "echo hello", "duration": 0.1},
                {"keystrokes": "ls\\n", "duration": 0.1}
            ]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert 'should end with newline' in result.warning

    def test_wrong_field_order_warning(self, parser):
        response = '''{
            "commands": [],
            "analysis": "a",
            "plan": "b"
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert 'wrong order' in result.warning


# ==============================================================================
# Auto-Fix Tests
# ==============================================================================


class TestAutoFixes:
    """Tests for auto-correction of malformed responses."""

    def test_fix_incomplete_json(self, parser):
        response = '{"analysis": "a", "plan": "b", "commands": [{"keystrokes": "ls\\n", "duration": 0.1}]'
        result = parser.parse_response(response)

        assert result.error == ''
        assert 'AUTO-CORRECTED' in result.warning
        assert len(result.commands) == 1

    def test_fix_mixed_content(self, parser):
        response = 'Here is my analysis:\n{"analysis": "a", "plan": "b", "commands": []}\nEnd of response'
        result = parser.parse_response(response)

        assert result.error == ''

    def test_deeply_incomplete_json(self, parser):
        response = '{"analysis": "a", "plan": "b", "commands": [{"keystrokes": "ls\\n"'
        result = parser.parse_response(response)
        # May or may not fix - just ensure no crash
        assert isinstance(result, ParseResult)


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_special_characters_in_keystrokes(self, parser):
        response = '''{
            "analysis": "a",
            "plan": "b",
            "commands": [{"keystrokes": "echo \\"hello world\\"\\n", "duration": 0.1}]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.commands[0].keystrokes == 'echo "hello world"\n'

    def test_ctrl_c_keystrokes(self, parser):
        response = '''{
            "analysis": "a",
            "plan": "b",
            "commands": [{"keystrokes": "C-c", "duration": 0.1}]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.commands[0].keystrokes == 'C-c'

    def test_empty_keystrokes(self, parser):
        response = '''{
            "analysis": "a",
            "plan": "b",
            "commands": [{"keystrokes": "", "duration": 10.0}]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.commands[0].keystrokes == ''
        assert result.commands[0].duration == 10.0

    def test_empty_response(self, parser):
        result = parser.parse_response('')
        assert result.error != ''

    def test_task_complete_with_parse_error_becomes_warning(self, parser):
        """When task_complete is true, command parse errors become warnings."""
        response = '''{
            "analysis": "Done",
            "plan": "Finish",
            "commands": [{"not_keystrokes": "x"}],
            "task_complete": true
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.is_task_complete is True
        assert len(result.commands) == 0

    def test_large_number_of_commands(self, parser):
        commands = ', '.join(
            [f'{{"keystrokes": "echo {i}\\n", "duration": 0.1}}' for i in range(50)]
        )
        response = f'{{"analysis": "a", "plan": "b", "commands": [{commands}]}}'
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 50

    def test_duration_integer_cast(self, parser):
        response = '''{
            "analysis": "a",
            "plan": "b",
            "commands": [{"keystrokes": "ls\\n", "duration": 5}]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.commands[0].duration == 5.0
        assert isinstance(result.commands[0].duration, float)

    def test_json_with_markdown_code_fence(self, parser):
        """Common LLM mistake: wrapping JSON in code fences."""
        response = '```json\n{"analysis": "a", "plan": "b", "commands": []}\n```'
        result = parser.parse_response(response)

        assert result.error == ''

    def test_task_complete_string_variants(self, parser):
        for value in ['true', 'True', 'TRUE', '1', 'yes', 'Yes']:
            response = f'{{"analysis": "a", "plan": "b", "commands": [], "task_complete": "{value}"}}'
            result = parser.parse_response(response)
            assert result.is_task_complete is True, f'Failed for value: {value}'

        for value in ['false', 'False', '0', 'no']:
            response = f'{{"analysis": "a", "plan": "b", "commands": [], "task_complete": "{value}"}}'
            result = parser.parse_response(response)
            assert result.is_task_complete is False, f'Failed for value: {value}'
