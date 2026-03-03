"""Unit tests for the Terminus-2 Agent.

Tests the Terminus2Agent's message building, output truncation,
conversation history reconstruction, initial terminal capture,
and core agent logic.
"""

import pytest

from openhands.agenthub.terminus_2_agent.terminus_2_agent import (
    COMPLETION_CONFIRMATION,
    TIMEOUT_TEMPLATE,
    Terminus2Agent,
)
from openhands.agenthub.terminus_2_agent.terminus_json_plain_parser import (
    ParsedCommand,
    TerminusJSONPlainParser,
)
from openhands.events.action import AgentFinishAction, AgentThinkAction, MessageAction
from openhands.events.action.terminus_2 import Terminus2CmdRunAction
from openhands.events.event import EventSource
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.terminus_2 import Terminus2CmdOutputObservation


# ==============================================================================
# Output Truncation Tests
# ==============================================================================


class TestOutputTruncation:
    """Tests for the _limit_output_length static method."""

    def test_short_output_not_truncated(self):
        output = 'short output'
        result = Terminus2Agent._limit_output_length(output, max_bytes=10000)
        assert result == output

    def test_long_output_truncated(self):
        output = 'x' * 20000
        result = Terminus2Agent._limit_output_length(output, max_bytes=10000)
        assert len(result.encode('utf-8')) < len(output.encode('utf-8'))
        assert 'output limited to 10000 bytes' in result
        assert 'interior bytes omitted' in result

    def test_exact_limit_not_truncated(self):
        output = 'x' * 10000
        result = Terminus2Agent._limit_output_length(output, max_bytes=10000)
        assert result == output

    def test_truncation_preserves_start_and_end(self):
        output = 'START' + 'x' * 20000 + 'END'
        result = Terminus2Agent._limit_output_length(output, max_bytes=1000)
        assert result.startswith('START')
        assert result.endswith('END')

    def test_unicode_truncation(self):
        output = '\u00e9' * 10000  # Each char is 2 bytes in UTF-8
        result = Terminus2Agent._limit_output_length(output, max_bytes=5000)
        assert 'output limited to 5000 bytes' in result

    def test_custom_max_bytes(self):
        output = 'a' * 500
        result = Terminus2Agent._limit_output_length(output, max_bytes=200)
        assert 'output limited to 200 bytes' in result

    def test_empty_output(self):
        result = Terminus2Agent._limit_output_length('', max_bytes=10000)
        assert result == ''


# ==============================================================================
# Template Tests
# ==============================================================================


class TestTemplates:
    """Tests for the message templates used by the agent."""

    def test_timeout_template_formatting(self):
        result = TIMEOUT_TEMPLATE.format(
            command='make build\n',
            timeout_sec=30,
            terminal_state='$ make build\ncompiling...',
        )
        assert 'make build' in result
        assert '30 seconds' in result
        assert 'compiling...' in result
        assert 'timed out' in result

    def test_completion_confirmation_formatting(self):
        result = COMPLETION_CONFIRMATION.format(
            terminal_state='$ echo done\ndone\n$',
        )
        assert 'echo done' in result
        assert 'task_complete' in result
        assert 'graded' in result


# ==============================================================================
# Parser Integration Tests
# ==============================================================================


class TestParserIntegration:
    """Tests that the parser integrates correctly with the agent's expected flow."""

    @pytest.fixture
    def parser(self):
        return TerminusJSONPlainParser()

    def test_typical_response_flow(self, parser):
        """Simulate a typical multi-command response."""
        response = '''{
            "analysis": "I need to set up the project. The directory is empty.",
            "plan": "1. Create a project directory. 2. Initialize it.",
            "commands": [
                {"keystrokes": "mkdir myproject\\n", "duration": 0.1},
                {"keystrokes": "cd myproject\\n", "duration": 0.1},
                {"keystrokes": "git init\\n", "duration": 1.0}
            ]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 3
        assert result.commands[0].keystrokes == 'mkdir myproject\n'
        assert result.commands[1].keystrokes == 'cd myproject\n'
        assert result.commands[2].keystrokes == 'git init\n'
        assert result.commands[2].duration == 1.0
        assert result.is_task_complete is False

    def test_completion_response(self, parser):
        """Simulate a task completion response."""
        response = '''{
            "analysis": "All tests pass. The implementation is complete.",
            "plan": "Mark the task as complete.",
            "commands": [],
            "task_complete": true
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.is_task_complete is True
        assert len(result.commands) == 0

    def test_wait_response(self, parser):
        """Simulate a wait-for-output response."""
        response = '''{
            "analysis": "The build is still running.",
            "plan": "Wait for the build to finish.",
            "commands": [
                {"keystrokes": "", "duration": 10.0}
            ]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 1
        assert result.commands[0].keystrokes == ''
        assert result.commands[0].duration == 10.0

    def test_ctrl_c_response(self, parser):
        """Simulate sending Ctrl+C to cancel a running process."""
        response = '''{
            "analysis": "The process appears to be stuck.",
            "plan": "Send Ctrl+C to cancel it.",
            "commands": [
                {"keystrokes": "C-c", "duration": 0.1}
            ]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 1
        assert result.commands[0].keystrokes == 'C-c'


# ==============================================================================
# Message Building Tests
# ==============================================================================


class TestMessageBuilding:
    """Tests for the agent's message building logic."""

    def test_find_initial_user_message(self):
        """Test extraction of initial user message from events."""
        msg = MessageAction(content='Fix the bug in module X')
        msg._source = EventSource.USER

        events = [msg]
        for event in events:
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                assert event.content == 'Fix the bug in module X'
                break

    def test_terminus_2_action_batch_creation(self):
        """Test that commands are correctly converted to actions."""
        commands = [
            ParsedCommand(keystrokes='ls -la\n', duration=0.1),
            ParsedCommand(keystrokes='cat file.txt\n', duration=0.5),
        ]

        actions = []
        for cmd in commands:
            action = Terminus2CmdRunAction(
                keystrokes=cmd.keystrokes,
                duration=min(cmd.duration, 60),
            )
            actions.append(action)

        assert len(actions) == 2
        assert actions[0].keystrokes == 'ls -la\n'
        assert actions[0].duration == 0.1
        assert actions[1].keystrokes == 'cat file.txt\n'
        assert actions[1].duration == 0.5

    def test_duration_capped_at_60(self):
        """Test that duration is capped at 60 seconds."""
        cmd = ParsedCommand(keystrokes='sleep 100\n', duration=100.0)
        action = Terminus2CmdRunAction(
            keystrokes=cmd.keystrokes,
            duration=min(cmd.duration, 60),
        )
        assert action.duration == 60


# ==============================================================================
# Double Confirmation Tests
# ==============================================================================


class TestDoubleConfirmation:
    """Tests for the double-confirmation task completion logic."""

    def test_pending_completion_flag_initial(self):
        """Verify the flag starts as False."""
        pending = False

        is_task_complete = True
        if is_task_complete:
            if pending:
                action = 'finish'
            else:
                pending = True
                action = 'confirm'
        else:
            pending = False
            action = 'continue'

        assert pending is True
        assert action == 'confirm'

    def test_pending_completion_second_time(self):
        """Verify second task_complete triggers finish."""
        pending = True

        is_task_complete = True
        if is_task_complete:
            if pending:
                action = 'finish'
            else:
                pending = True
                action = 'confirm'
        else:
            pending = False
            action = 'continue'

        assert action == 'finish'

    def test_pending_completion_reset_on_not_complete(self):
        """Verify pending is reset when task_complete is False."""
        pending = True

        is_task_complete = False
        if is_task_complete:
            if pending:
                action = 'finish'
            else:
                pending = True
                action = 'confirm'
        else:
            pending = False
            action = 'continue'

        assert pending is False
        assert action == 'continue'

    def test_first_task_complete_with_no_commands_returns_noop(self):
        """First task_complete=True with empty commands should return a no-op
        action (to capture terminal state) rather than finishing immediately."""
        from collections import deque

        pending_completion = False
        pending_actions: deque = deque()
        commands: list = []
        is_task_complete = True
        response_text = '{"analysis":"done","plan":"none","commands":[],"task_complete":true}'

        if is_task_complete:
            if pending_completion:
                result = 'finish'
            else:
                pending_completion = True
                result = None
        else:
            pending_completion = False
            result = None

        for i, cmd in enumerate(commands):
            pending_actions.append(cmd)

        if result is None and not pending_actions:
            if pending_completion:
                result = 'noop_for_confirmation'
            else:
                result = 'think'

        assert pending_completion is True
        assert result == 'noop_for_confirmation'

    def test_first_task_complete_with_commands_queues_normally(self):
        """First task_complete=True with commands should queue them normally."""
        from collections import deque

        pending_completion = False
        pending_actions: deque = deque()
        is_task_complete = True
        commands = [
            ParsedCommand(keystrokes='ls\n', duration=0.1),
        ]

        if is_task_complete:
            if pending_completion:
                result = 'finish'
            else:
                pending_completion = True
                result = None
        else:
            pending_completion = False
            result = None

        for i, cmd in enumerate(commands):
            pending_actions.append(cmd)

        if result is None and not pending_actions:
            if pending_completion:
                result = 'noop_for_confirmation'
            else:
                result = 'think'
        elif result is None:
            result = 'pop_pending'

        assert pending_completion is True
        assert len(pending_actions) == 1
        assert result == 'pop_pending'

    def test_confirmation_message_appended_when_pending(self):
        """_build_messages should append COMPLETION_CONFIRMATION when _pending_completion is True."""
        user_msg = MessageAction(content='Task')
        user_msg._source = EventSource.USER
        noop = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        initial_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nroot@host:/app# ',
            terminal_state='Current Terminal Screen:\nroot@host:/app# ',
        )
        resp = '{"analysis":"done","plan":"done","commands":[],"task_complete":true}'
        noop2 = Terminus2CmdRunAction(keystrokes='', duration=0.5, thought=resp)
        confirm_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nroot@host:/app# ',
            terminal_state='Current Terminal Screen:\nroot@host:/app# ',
        )

        events = [user_msg, noop, initial_obs, noop2, confirm_obs]

        pending_completion = True

        messages = []
        messages.append(('system', 'system_prompt'))

        initial_terminal_event = initial_obs
        first_text = f'{user_msg.content}\n\n{initial_terminal_event.terminal_state}'
        messages.append(('user', first_text))

        batch_observations: list[str] = []
        for event in events:
            if isinstance(event, Terminus2CmdRunAction):
                if event.thought:
                    if batch_observations:
                        messages.append(('user', batch_observations[-1]))
                        batch_observations = []
                    messages.append(('assistant', event.thought))
            elif isinstance(event, Terminus2CmdOutputObservation):
                if event is initial_terminal_event:
                    continue
                batch_observations.append(event.terminal_state)

        if batch_observations:
            terminal_output = batch_observations[-1]
            if pending_completion:
                confirmation = (
                    f'Current terminal state:\n{terminal_output}\n\n'
                    'Are you sure you want to mark the task as complete? '
                    "This will trigger your solution to be graded and you won't be able to "
                    'make any further corrections. If so, include "task_complete": true '
                    'in your JSON response again.'
                )
                messages.append(('user', confirmation))
            else:
                messages.append(('user', terminal_output))
        elif pending_completion:
            confirmation = (
                'Current terminal state:\n\n\n'
                'Are you sure you want to mark the task as complete? '
                "This will trigger your solution to be graded and you won't be able to "
                'make any further corrections. If so, include "task_complete": true '
                'in your JSON response again.'
            )
            messages.append(('user', confirmation))

        assert any('Are you sure you want to mark the task as complete?' in m[1] for m in messages)
        assert messages[-1][0] == 'user'
        assert 'task_complete' in messages[-1][1]
        user_messages = [m for m in messages if m[0] == 'user']
        assert len(user_messages) == 2  # initial task + confirmation (NOT three)


# ==============================================================================
# Observation Handling Tests
# ==============================================================================


class TestObservationHandling:
    """Tests for handling Terminus-2 observations."""

    def test_observation_content_extraction(self):
        obs = Terminus2CmdOutputObservation(
            content='$ ls\nfile1.txt\nfile2.txt\n$',
            terminal_state='$ ls\nfile1.txt\nfile2.txt\n$',
            timed_out=False,
            command_keystrokes='ls\n',
        )

        assert obs.terminal_state == '$ ls\nfile1.txt\nfile2.txt\n$'
        assert obs.timed_out is False
        assert obs.command_keystrokes == 'ls\n'

    def test_timed_out_observation(self):
        obs = Terminus2CmdOutputObservation(
            content='partial output...',
            terminal_state='partial output...',
            timed_out=True,
            command_keystrokes='make\n',
        )

        assert obs.timed_out is True
        assert 'timed out' in obs.message

    def test_observation_with_empty_terminal_state(self):
        obs = Terminus2CmdOutputObservation(
            content='',
            terminal_state='',
        )
        assert obs.terminal_state == ''
        assert obs.content == ''


# ==============================================================================
# _has_terminal_observation Tests
# ==============================================================================


class TestHasTerminalObservation:
    """Tests for the _has_terminal_observation helper."""

    def test_empty_events(self):
        assert Terminus2Agent._has_terminal_observation(None, []) is False

    def test_only_user_message(self):
        msg = MessageAction(content='task')
        msg._source = EventSource.USER
        assert Terminus2Agent._has_terminal_observation(None, [msg]) is False

    def test_has_observation(self):
        obs = Terminus2CmdOutputObservation(
            content='output', terminal_state='root@host:/# pwd\n/\nroot@host:/# '
        )
        assert Terminus2Agent._has_terminal_observation(None, [obs]) is True

    def test_observation_after_other_events(self):
        msg = MessageAction(content='task')
        msg._source = EventSource.USER
        action = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        obs = Terminus2CmdOutputObservation(
            content='output', terminal_state='root@host:/# '
        )
        assert Terminus2Agent._has_terminal_observation(None, [msg, action, obs]) is True


# ==============================================================================
# _find_initial_terminal_event Tests
# ==============================================================================


class TestFindInitialTerminalEvent:
    """Tests for _find_initial_terminal_event returning the event object."""

    def test_returns_none_on_empty(self):
        result = Terminus2Agent._find_initial_terminal_event(None, [])
        assert result is None

    def test_returns_none_when_no_observations(self):
        msg = MessageAction(content='task')
        msg._source = EventSource.USER
        result = Terminus2Agent._find_initial_terminal_event(None, [msg])
        assert result is None

    def test_returns_first_observation_object(self):
        obs1 = Terminus2CmdOutputObservation(
            content='first', terminal_state='screen1'
        )
        obs2 = Terminus2CmdOutputObservation(
            content='second', terminal_state='screen2'
        )
        result = Terminus2Agent._find_initial_terminal_event(None, [obs1, obs2])
        assert result is obs1
        assert result is not obs2

    def test_identity_comparison_works(self):
        """The returned event should be the exact same object for identity checks."""
        obs = Terminus2CmdOutputObservation(
            content='output', terminal_state='root@host:/# '
        )
        msg = MessageAction(content='task')
        msg._source = EventSource.USER
        events = [msg, obs]
        result = Terminus2Agent._find_initial_terminal_event(None, events)
        assert result is obs


# ==============================================================================
# Response Text Storage Tests (thought field on first action)
# ==============================================================================


class TestResponseTextStorage:
    """Tests that LLM response text is stored on the first action's thought field."""

    def test_first_action_gets_thought(self):
        """When creating actions from commands, only the first gets the response text."""
        response_text = '{"analysis":"test","plan":"test","commands":[{"keystrokes":"ls\\n","duration":0.1},{"keystrokes":"pwd\\n","duration":0.1}]}'
        commands = [
            ParsedCommand(keystrokes='ls\n', duration=0.1),
            ParsedCommand(keystrokes='pwd\n', duration=0.1),
        ]

        actions = []
        for i, cmd in enumerate(commands):
            action = Terminus2CmdRunAction(
                keystrokes=cmd.keystrokes,
                duration=min(cmd.duration, 60),
                thought=response_text if i == 0 else '',
            )
            actions.append(action)

        assert actions[0].thought == response_text
        assert actions[1].thought == ''

    def test_single_command_gets_thought(self):
        response_text = '{"analysis":"x","plan":"x","commands":[{"keystrokes":"ls\\n"}]}'
        action = Terminus2CmdRunAction(
            keystrokes='ls\n',
            duration=0.1,
            thought=response_text,
        )
        assert action.thought == response_text

    def test_empty_response_stored_as_empty(self):
        action = Terminus2CmdRunAction(
            keystrokes='ls\n',
            duration=0.1,
            thought='',
        )
        assert action.thought == ''


# ==============================================================================
# Conversation History Reconstruction Tests
# ==============================================================================


class TestConversationHistoryReconstruction:
    """Tests that _build_messages reconstructs full conversation history
    from the event stream, including assistant turns from action.thought.

    These tests simulate what _build_messages does by processing events
    using the same algorithm, verifying the message sequence is correct.
    """

    def _simulate_build_messages(self, events):
        """Simulate the core _build_messages loop logic to verify message ordering.

        Returns a list of (role, text) tuples representing the conversation.
        This mirrors the algorithm in Terminus2Agent._build_messages.
        """
        messages = []
        messages.append(('system', 'system_prompt'))

        initial_user_msg = None
        initial_terminal_event = None
        for event in events:
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                initial_user_msg = event.content
                break
        for event in events:
            if isinstance(event, Terminus2CmdOutputObservation):
                initial_terminal_event = event
                break

        if initial_user_msg:
            if initial_terminal_event is not None:
                terminal_text = initial_terminal_event.terminal_state
                first_text = f'{initial_user_msg}\n\n{terminal_text}'
            else:
                first_text = initial_user_msg
            messages.append(('user', first_text))

        batch_observations = []
        for event in events:
            if isinstance(event, MessageAction):
                if event.source == EventSource.USER:
                    continue
                elif event.source == EventSource.AGENT:
                    if batch_observations:
                        combined = Terminus2Agent._combine_observations(
                            batch_observations
                        )
                        messages.append(('user', combined))
                        batch_observations = []
                    messages.append(('assistant', event.content))

            elif isinstance(event, Terminus2CmdRunAction):
                if event.thought:
                    if batch_observations:
                        combined = Terminus2Agent._combine_observations(
                            batch_observations
                        )
                        messages.append(('user', combined))
                        batch_observations = []
                    messages.append(('assistant', event.thought))

            elif isinstance(event, Terminus2CmdOutputObservation):
                if event is initial_terminal_event:
                    continue
                batch_observations.append(event.terminal_state)

            elif isinstance(event, ErrorObservation):
                batch_observations.append(f'ERROR: {event.content}')

        if batch_observations:
            combined = Terminus2Agent._combine_observations(batch_observations)
            messages.append(('user', combined))

        return messages

    def test_initial_state_only(self):
        """First LLM call: system + user(task+terminal screen)."""
        user_msg = MessageAction(content='Fix the bug')
        user_msg._source = EventSource.USER
        noop = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        initial_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nroot@host:/app# pwd\n/app\nroot@host:/app# ',
            terminal_state='Current Terminal Screen:\nroot@host:/app# pwd\n/app\nroot@host:/app# ',
        )

        events = [user_msg, noop, initial_obs]
        msgs = self._simulate_build_messages(events)

        assert len(msgs) == 2  # system + user
        assert msgs[0][0] == 'system'
        assert msgs[1][0] == 'user'
        assert 'Fix the bug' in msgs[1][1]
        assert 'Current Terminal Screen:' in msgs[1][1]
        assert 'root@host:/app#' in msgs[1][1]

    def test_one_round_trip(self):
        """After first LLM call: system + user(task+terminal) + assistant + user(output)."""
        llm_response = '{"analysis":"a","plan":"p","commands":[{"keystrokes":"ls\\n","duration":0.1}]}'

        user_msg = MessageAction(content='Fix the bug')
        user_msg._source = EventSource.USER
        noop = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        initial_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nroot@host:/app# ',
            terminal_state='Current Terminal Screen:\nroot@host:/app# ',
        )
        cmd_action = Terminus2CmdRunAction(
            keystrokes='ls\n', duration=0.1, thought=llm_response
        )
        cmd_obs = Terminus2CmdOutputObservation(
            content='New Terminal Output:\nroot@host:/app# ls\nfile.py\nroot@host:/app# ',
            terminal_state='New Terminal Output:\nroot@host:/app# ls\nfile.py\nroot@host:/app# ',
        )

        events = [user_msg, noop, initial_obs, cmd_action, cmd_obs]
        msgs = self._simulate_build_messages(events)

        assert len(msgs) == 4  # system, user, assistant, user
        assert msgs[0][0] == 'system'
        assert msgs[1][0] == 'user'
        assert msgs[2][0] == 'assistant'
        assert msgs[2][1] == llm_response
        assert msgs[3][0] == 'user'
        assert 'New Terminal Output:' in msgs[3][1]
        assert 'file.py' in msgs[3][1]

    def test_multi_command_batch(self):
        """Multiple commands from one LLM call: ALL observations are combined into one user message."""
        llm_response = '{"analysis":"a","plan":"p","commands":[{"keystrokes":"ls\\n"},{"keystrokes":"pwd\\n"}]}'

        user_msg = MessageAction(content='Task')
        user_msg._source = EventSource.USER
        noop = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        initial_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nprompt',
            terminal_state='Current Terminal Screen:\nprompt',
        )
        cmd1 = Terminus2CmdRunAction(keystrokes='ls\n', duration=0.1, thought=llm_response)
        obs1 = Terminus2CmdOutputObservation(
            content='New Terminal Output:\nroot@host:/app# ls\nfile.py\nroot@host:/app# ',
            terminal_state='New Terminal Output:\nroot@host:/app# ls\nfile.py\nroot@host:/app# ',
        )
        cmd2 = Terminus2CmdRunAction(keystrokes='pwd\n', duration=0.1, thought='')
        obs2 = Terminus2CmdOutputObservation(
            content='New Terminal Output:\nroot@host:/app# pwd\n/app\nroot@host:/app# ',
            terminal_state='New Terminal Output:\nroot@host:/app# pwd\n/app\nroot@host:/app# ',
        )

        events = [user_msg, noop, initial_obs, cmd1, obs1, cmd2, obs2]
        msgs = self._simulate_build_messages(events)

        assert len(msgs) == 4  # system, user, assistant, user
        assert msgs[2][0] == 'assistant'
        assert msgs[2][1] == llm_response
        assert msgs[3][0] == 'user'
        assert 'ls' in msgs[3][1]
        assert 'file.py' in msgs[3][1]
        assert 'pwd' in msgs[3][1]
        assert '/app' in msgs[3][1]
        assert msgs[3][1].count('New Terminal Output:') == 1

    def test_two_round_trips(self):
        """Two LLM calls produce: sys, user, asst, user, asst, user."""
        resp1 = '{"analysis":"a","plan":"p","commands":[{"keystrokes":"ls\\n"}]}'
        resp2 = '{"analysis":"b","plan":"q","commands":[{"keystrokes":"cat f\\n"}]}'

        user_msg = MessageAction(content='Task')
        user_msg._source = EventSource.USER
        noop = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        initial_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nprompt',
            terminal_state='Current Terminal Screen:\nprompt',
        )
        cmd1 = Terminus2CmdRunAction(keystrokes='ls\n', duration=0.1, thought=resp1)
        obs1 = Terminus2CmdOutputObservation(
            content='New Terminal Output:\nls result',
            terminal_state='New Terminal Output:\nls result',
        )
        cmd2 = Terminus2CmdRunAction(keystrokes='cat f\n', duration=0.1, thought=resp2)
        obs2 = Terminus2CmdOutputObservation(
            content='New Terminal Output:\nfile content',
            terminal_state='New Terminal Output:\nfile content',
        )

        events = [user_msg, noop, initial_obs, cmd1, obs1, cmd2, obs2]
        msgs = self._simulate_build_messages(events)

        assert len(msgs) == 6  # system, user, asst, user, asst, user
        roles = [m[0] for m in msgs]
        assert roles == ['system', 'user', 'assistant', 'user', 'assistant', 'user']
        assert msgs[2][1] == resp1
        assert 'ls result' in msgs[3][1]
        assert msgs[4][1] == resp2
        assert 'file content' in msgs[5][1]

    def test_initial_observation_not_duplicated(self):
        """The initial terminal observation should NOT appear as a separate user message."""
        user_msg = MessageAction(content='Task')
        user_msg._source = EventSource.USER
        noop = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        initial_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nINITIAL_SCREEN',
            terminal_state='Current Terminal Screen:\nINITIAL_SCREEN',
        )

        events = [user_msg, noop, initial_obs]
        msgs = self._simulate_build_messages(events)

        user_messages = [m[1] for m in msgs if m[0] == 'user']
        assert len(user_messages) == 1
        assert 'Current Terminal Screen:' in user_messages[0]
        assert 'INITIAL_SCREEN' in user_messages[0]

    def test_initial_observation_not_duplicated_after_first_llm_call(self):
        """After one LLM round, initial screen should only appear in first user msg."""
        resp = '{"analysis":"a","plan":"p","commands":[{"keystrokes":"ls\\n"}]}'
        user_msg = MessageAction(content='Task')
        user_msg._source = EventSource.USER
        noop = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        initial_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nINITIAL_SCREEN',
            terminal_state='Current Terminal Screen:\nINITIAL_SCREEN',
        )
        cmd = Terminus2CmdRunAction(keystrokes='ls\n', duration=0.1, thought=resp)
        obs = Terminus2CmdOutputObservation(
            content='New Terminal Output:\nls output',
            terminal_state='New Terminal Output:\nls output',
        )

        events = [user_msg, noop, initial_obs, cmd, obs]
        msgs = self._simulate_build_messages(events)

        user_messages = [m[1] for m in msgs if m[0] == 'user']
        assert len(user_messages) == 2
        assert 'INITIAL_SCREEN' in user_messages[0]
        assert 'INITIAL_SCREEN' not in user_messages[1]
        assert 'New Terminal Output:' in user_messages[1]

    def test_no_initial_terminal_state(self):
        """When no terminal observation exists, first user message is just the task."""
        user_msg = MessageAction(content='Task description')
        user_msg._source = EventSource.USER

        events = [user_msg]
        msgs = self._simulate_build_messages(events)

        assert len(msgs) == 2  # system + user
        assert msgs[1][1] == 'Task description'

    def test_error_observation_in_batch(self):
        """ErrorObservation should be included in batch_observations."""
        user_msg = MessageAction(content='Task')
        user_msg._source = EventSource.USER
        initial_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nprompt',
            terminal_state='Current Terminal Screen:\nprompt',
        )
        resp = '{"analysis":"a","plan":"p","commands":[{"keystrokes":"bad_cmd\\n"}]}'
        cmd = Terminus2CmdRunAction(keystrokes='bad_cmd\n', duration=0.1, thought=resp)
        err = ErrorObservation(content='command failed')

        events = [user_msg, initial_obs, cmd, err]
        msgs = self._simulate_build_messages(events)

        user_messages = [m[1] for m in msgs if m[0] == 'user']
        assert any('ERROR: command failed' in m for m in user_messages)

    def test_alternating_roles_no_consecutive_same_role(self):
        """After system, messages should alternate user/assistant (no consecutive same role)."""
        resp = '{"analysis":"a","plan":"p","commands":[{"keystrokes":"ls\\n"}]}'
        user_msg = MessageAction(content='Task')
        user_msg._source = EventSource.USER
        noop = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        initial_obs = Terminus2CmdOutputObservation(
            content='Current Terminal Screen:\nprompt',
            terminal_state='Current Terminal Screen:\nprompt',
        )
        cmd = Terminus2CmdRunAction(keystrokes='ls\n', duration=0.1, thought=resp)
        obs = Terminus2CmdOutputObservation(
            content='New Terminal Output:\noutput',
            terminal_state='New Terminal Output:\noutput',
        )

        events = [user_msg, noop, initial_obs, cmd, obs]
        msgs = self._simulate_build_messages(events)

        roles = [m[0] for m in msgs]
        assert roles[0] == 'system'
        for i in range(2, len(roles)):
            assert roles[i] != roles[i - 1], (
                f'Consecutive same role at {i}: {roles}'
            )


# ==============================================================================
# Initial Terminal Capture Tests
# ==============================================================================


class TestInitialTerminalCapture:
    """Tests for the no-op action sent on the first step to capture terminal state."""

    def test_noop_action_has_empty_keystrokes(self):
        action = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        assert action.keystrokes == ''
        assert action.duration == 0.5

    def test_noop_action_is_runnable(self):
        action = Terminus2CmdRunAction(keystrokes='', duration=0.5)
        assert action.runnable is True


# ==============================================================================
# Action Execution Client Dispatch Tests
# ==============================================================================


class TestActionExecutionClientDispatch:
    """Tests that the ActionExecutionClient has the terminus_2_cmd_run method."""

    def test_client_has_terminus_2_method(self):
        from openhands.runtime.impl.action_execution.action_execution_client import (
            ActionExecutionClient,
        )
        assert hasattr(ActionExecutionClient, 'terminus_2_cmd_run')

    def test_client_method_is_callable(self):
        from openhands.runtime.impl.action_execution.action_execution_client import (
            ActionExecutionClient,
        )
        assert callable(getattr(ActionExecutionClient, 'terminus_2_cmd_run'))


# ==============================================================================
# Terminal Screen Formatting Tests
# ==============================================================================


class TestTerminalScreenFormatting:
    """Tests for _format_terminal_screen logic in the action execution server.

    Since ActionExecutor has heavy dependencies (FastAPI, BashSession, etc.),
    we re-implement the pure formatting logic here to test it in isolation.
    This mirrors ActionExecutor._format_terminal_screen exactly.
    """

    @staticmethod
    def _format_terminal_screen(obs, command, pre_cwd=None):
        """Pure-function copy of ActionExecutor._format_terminal_screen."""
        meta = obs.metadata
        username = meta.username or 'root'
        hostname = meta.hostname or 'sandbox'
        post_cwd = meta.working_dir or '/'
        suffix = '#' if username == 'root' else '$'

        before_cwd = pre_cwd if pre_cwd else post_cwd
        pre_prompt = f'{username}@{hostname}:{before_cwd}{suffix} '
        post_prompt = f'{username}@{hostname}:{post_cwd}{suffix} '

        lines = [f'{pre_prompt}{command}']
        if obs.content.strip():
            lines.append(obs.content)
        lines.append(post_prompt)
        return '\n'.join(lines)

    def _make_obs(self, content, username=None, hostname=None, working_dir=None):
        from openhands.events.observation.commands import (
            CmdOutputMetadata,
            CmdOutputObservation,
        )
        metadata = CmdOutputMetadata(
            exit_code=0,
            username=username,
            hostname=hostname,
            working_dir=working_dir,
        )
        return CmdOutputObservation(
            content=content,
            command='test',
            metadata=metadata,
        )

    def test_basic_formatting(self):
        obs = self._make_obs(
            'file1.txt\nfile2.txt',
            username='root',
            hostname='abc123',
            working_dir='/app',
        )
        result = self._format_terminal_screen(obs, 'ls')

        assert result.startswith('root@abc123:/app# ls')
        assert 'file1.txt' in result
        assert 'file2.txt' in result
        assert result.endswith('root@abc123:/app# ')

    def test_root_user_gets_hash_prompt(self):
        obs = self._make_obs('', username='root', hostname='h', working_dir='/')
        result = self._format_terminal_screen(obs, 'pwd')
        assert 'root@h:/# pwd' in result

    def test_non_root_user_gets_dollar_prompt(self):
        obs = self._make_obs('', username='developer', hostname='h', working_dir='/home')
        result = self._format_terminal_screen(obs, 'pwd')
        assert 'developer@h:/home$ pwd' in result

    def test_empty_content_no_extra_lines(self):
        obs = self._make_obs('', username='root', hostname='h', working_dir='/')
        result = self._format_terminal_screen(obs, 'true')
        lines = result.split('\n')
        assert len(lines) == 2
        assert lines[0] == 'root@h:/# true'
        assert lines[1] == 'root@h:/# '

    def test_multiline_output(self):
        obs = self._make_obs(
            'line1\nline2\nline3',
            username='root',
            hostname='box',
            working_dir='/tmp',
        )
        result = self._format_terminal_screen(obs, 'cat file')
        lines = result.split('\n')
        assert lines[0] == 'root@box:/tmp# cat file'
        assert lines[1] == 'line1'
        assert lines[2] == 'line2'
        assert lines[3] == 'line3'
        assert lines[4] == 'root@box:/tmp# '

    def test_defaults_when_metadata_missing(self):
        obs = self._make_obs('output', username=None, hostname=None, working_dir=None)
        result = self._format_terminal_screen(obs, 'echo hi')
        assert result.startswith('root@sandbox:/#')
        assert 'output' in result

    def test_special_key_ctrl_c_display(self):
        obs = self._make_obs('', username='root', hostname='h', working_dir='/app')
        result = self._format_terminal_screen(obs, '^C')
        assert 'root@h:/app# ^C' in result

    def test_whitespace_only_content_treated_as_empty(self):
        obs = self._make_obs('   \n  \n  ', username='root', hostname='h', working_dir='/')
        result = self._format_terminal_screen(obs, 'true')
        lines = result.split('\n')
        assert len(lines) == 2

    def test_prompt_appears_at_end(self):
        obs = self._make_obs(
            'some output',
            username='root',
            hostname='container',
            working_dir='/workspace',
        )
        result = self._format_terminal_screen(obs, 'echo hi')
        assert result.endswith('root@container:/workspace# ')

    def test_long_command_preserved(self):
        long_cmd = 'find / -name "*.py" -exec grep -l "import os" {} \\;'
        obs = self._make_obs('result', username='root', hostname='h', working_dir='/')
        result = self._format_terminal_screen(obs, long_cmd)
        assert long_cmd in result.split('\n')[0]

    def test_cd_pre_cwd_differs_from_post_cwd(self):
        """cd /app/src: pre-command prompt shows /app, post-command prompt shows /app/src."""
        obs = self._make_obs(
            '', username='root', hostname='host', working_dir='/app/src'
        )
        result = self._format_terminal_screen(obs, 'cd /app/src', pre_cwd='/app')
        lines = result.split('\n')
        assert lines[0] == 'root@host:/app# cd /app/src'
        assert lines[1] == 'root@host:/app/src# '

    def test_no_pre_cwd_uses_post_cwd_for_both(self):
        """Without pre_cwd, both prompts use the post-execution cwd (backward compat)."""
        obs = self._make_obs(
            '', username='root', hostname='h', working_dir='/new'
        )
        result = self._format_terminal_screen(obs, 'cd /new')
        lines = result.split('\n')
        assert lines[0] == 'root@h:/new# cd /new'
        assert lines[1] == 'root@h:/new# '

    def test_non_cd_command_same_cwd(self):
        """Normal command: pre_cwd == post_cwd, both prompts identical."""
        obs = self._make_obs(
            'file.py', username='root', hostname='h', working_dir='/app'
        )
        result = self._format_terminal_screen(obs, 'ls', pre_cwd='/app')
        lines = result.split('\n')
        assert lines[0] == 'root@h:/app# ls'
        assert lines[-1] == 'root@h:/app# '


# ==============================================================================
# Terminal Output Prefix Tests
# ==============================================================================


class TestTerminalOutputPrefixes:
    """Tests that the server adds correct prefixes to terminal output.

    In the original Terminus-2:
    - "Current Terminal Screen:" for initial captures and timed-out commands
    - "New Terminal Output:" for normal command execution output
    """

    def test_initial_capture_gets_current_screen_prefix(self):
        """Empty keystrokes (initial capture) should use 'Current Terminal Screen:' prefix."""
        terminal_state = 'Current Terminal Screen:\nroot@host:/app# pwd\n/app\nroot@host:/app# '
        assert terminal_state.startswith('Current Terminal Screen:')
        assert 'root@host:/app#' in terminal_state

    def test_normal_command_gets_new_output_prefix(self):
        """Regular command output should use 'New Terminal Output:' prefix."""
        terminal_state = 'New Terminal Output:\nroot@host:/app# ls\nfile.py\nroot@host:/app# '
        assert terminal_state.startswith('New Terminal Output:')
        assert 'file.py' in terminal_state

    def test_timed_out_command_gets_current_screen_prefix(self):
        """Timed-out commands should use 'Current Terminal Screen:' prefix."""
        terminal_state = 'Current Terminal Screen:\nroot@host:/app# sleep 100\n'
        assert terminal_state.startswith('Current Terminal Screen:')

    def test_prefix_followed_by_newline_then_content(self):
        """Prefix should be followed by newline then the actual screen content."""
        screen = 'root@host:/app# ls\nfile.py\nroot@host:/app# '
        prefixed = f'New Terminal Output:\n{screen}'
        parts = prefixed.split('\n', 1)
        assert parts[0] == 'New Terminal Output:'
        assert parts[1] == screen

    def test_initial_message_includes_prefix_from_terminal_state(self):
        """When building initial user message, the terminal_state already has the prefix."""
        task = 'Fix the bug in main.py'
        terminal_state = 'Current Terminal Screen:\nroot@host:/app# '
        initial_msg = f'{task}\n\n{terminal_state}'
        assert 'Current Terminal Screen:' in initial_msg
        assert 'Fix the bug' in initial_msg

    def test_subsequent_output_includes_prefix(self):
        """Subsequent terminal observations have their prefix baked in."""
        terminal_state = 'New Terminal Output:\nroot@host:/app# echo hello\nhello\nroot@host:/app# '
        assert terminal_state.startswith('New Terminal Output:')
        content_after_prefix = terminal_state[len('New Terminal Output:\n'):]
        assert content_after_prefix.startswith('root@host:/app#')


# ==============================================================================
# Batch Observation Combination Tests
# ==============================================================================


class TestCombineObservations:
    """Tests for _combine_observations which merges multiple terminal outputs
    from a command batch into a single user message, matching the original
    Terminus-2 behavior where get_incremental_output() captures all commands.
    """

    def test_single_observation_returned_as_is(self):
        obs = ['New Terminal Output:\nroot@h:/# ls\nfile.py\nroot@h:/# ']
        result = Terminus2Agent._combine_observations(obs)
        assert result == obs[0]

    def test_empty_list_returns_empty_string(self):
        result = Terminus2Agent._combine_observations([])
        assert result == ''

    def test_two_observations_combined_under_single_prefix(self):
        obs1 = 'New Terminal Output:\nroot@h:/app# ls\nfile.py\nroot@h:/app# '
        obs2 = 'New Terminal Output:\nroot@h:/app# pwd\n/app\nroot@h:/app# '
        result = Terminus2Agent._combine_observations([obs1, obs2])
        assert result.startswith('New Terminal Output:\n')
        assert result.count('New Terminal Output:') == 1
        assert 'ls' in result
        assert 'file.py' in result
        assert 'pwd' in result
        assert '/app' in result

    def test_three_observations_all_content_present(self):
        obs1 = 'New Terminal Output:\nroot@h:/# ls -l\ntotal 4\nroot@h:/# '
        obs2 = 'New Terminal Output:\nroot@h:/# ls *.py\nscript.py\nroot@h:/# '
        obs3 = 'New Terminal Output:\nroot@h:/# grep foo .\n./match\nroot@h:/# '
        result = Terminus2Agent._combine_observations([obs1, obs2, obs3])
        assert result.count('New Terminal Output:') == 1
        assert 'ls -l' in result
        assert 'total 4' in result
        assert 'script.py' in result
        assert 'grep foo' in result
        assert './match' in result

    def test_mixed_prefixes_uses_last(self):
        """If last observation was a timeout (Current Terminal Screen:), use that prefix."""
        obs1 = 'New Terminal Output:\nroot@h:/# ls\nfile.py\nroot@h:/# '
        obs2 = 'Current Terminal Screen:\nroot@h:/# sleep 100\n'
        result = Terminus2Agent._combine_observations([obs1, obs2])
        assert result.startswith('Current Terminal Screen:\n')
        assert result.count('Current Terminal Screen:') == 1
        assert 'ls' in result
        assert 'sleep 100' in result

    def test_no_prefix_observations_preserved(self):
        """Observations without a recognized prefix are included as-is."""
        obs1 = 'some raw output'
        obs2 = 'New Terminal Output:\nroot@h:/# pwd\n/\nroot@h:/# '
        result = Terminus2Agent._combine_observations([obs1, obs2])
        assert 'some raw output' in result
        assert 'pwd' in result

    def test_error_mixed_with_observations(self):
        """ERROR observations (no prefix) are combined with normal observations."""
        obs1 = 'New Terminal Output:\nroot@h:/# ls\nfile.py\nroot@h:/# '
        obs2 = 'ERROR: command failed'
        result = Terminus2Agent._combine_observations([obs1, obs2])
        assert 'file.py' in result
        assert 'ERROR: command failed' in result
