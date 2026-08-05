"""Tests for replay_utils: messages_to_replay_events and related functionality."""

import json

import pytest

from evaluation.benchmarks.swe_bench.replay_utils import (
    _get_response_to_actions_fn,
    _patch_tool_name,
    _terminus2_response_to_actions,
    _to_snake_case,
    detect_and_set_tool_names,
    messages_to_replay_events,
)
from openhands.events.action import (
    AgentFinishAction,
    AgentThinkAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    MessageAction,
)


# ---------------------------------------------------------------------------
# _get_response_to_actions_fn
# ---------------------------------------------------------------------------


class TestGetResponseToActionsFn:
    def test_codeact_agent(self):
        fn = _get_response_to_actions_fn('CodeActAgent')
        assert callable(fn)

    def test_opencode_agent(self):
        fn = _get_response_to_actions_fn('OpenCodeAgent')
        assert callable(fn)

    def test_codex_agent(self):
        fn = _get_response_to_actions_fn('CodexAgent')
        assert callable(fn)

    def test_terminus2_agent_returns_none(self):
        fn = _get_response_to_actions_fn('Terminus2Agent')
        assert fn is None

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match='Replay is not supported'):
            _get_response_to_actions_fn('NonexistentAgent')


# ---------------------------------------------------------------------------
# Helper: build OpenAI chat-format messages
# ---------------------------------------------------------------------------


def _system_msg(content: str = 'You are a helpful assistant.') -> dict:
    return {'role': 'system', 'content': content}


def _user_msg(content: str) -> dict:
    return {'role': 'user', 'content': content}


def _user_msg_multipart(text_parts: list[str]) -> dict:
    return {
        'role': 'user',
        'content': [{'type': 'text', 'text': t} for t in text_parts],
    }


def _assistant_msg(content: str) -> dict:
    return {'role': 'assistant', 'content': content}


def _assistant_tool_call(
    content: str | None,
    tool_calls: list[dict],
) -> dict:
    msg = {'role': 'assistant', 'content': content, 'tool_calls': tool_calls}
    return msg


def _tool_call(name: str, arguments: dict, call_id: str = 'call_001') -> dict:
    return {
        'id': call_id,
        'type': 'function',
        'function': {
            'name': name,
            'arguments': json.dumps(arguments),
        },
    }


def _tool_response(call_id: str, content: str, name: str = '') -> dict:
    msg = {'role': 'tool', 'content': content, 'tool_call_id': call_id}
    if name:
        msg['name'] = name
    return msg


# ---------------------------------------------------------------------------
# messages_to_replay_events – CodeActAgent
# ---------------------------------------------------------------------------


class TestMessagesToReplayEventsCodeAct:
    """Tests using CodeActAgent tool names."""

    def test_basic_bash_command(self):
        messages = [
            _system_msg(),
            _user_msg('Fix the bug in utils.py'),
            _assistant_tool_call(
                'Let me check the file.',
                [_tool_call('execute_bash', {'command': 'cat utils.py'})],
            ),
            _tool_response('call_001', 'def foo():\n    pass'),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert isinstance(initial, MessageAction)
        assert initial.content == 'Fix the bug in utils.py'
        assert len(events) == 1
        assert isinstance(events[0], CmdRunAction)
        assert events[0].command == 'cat utils.py'
        assert events[0]._id is None

    def test_multiple_tool_calls(self):
        messages = [
            _user_msg('Fix the bug'),
            _assistant_tool_call(
                'Investigating.',
                [_tool_call('execute_bash', {'command': 'ls'}, 'call_1')],
            ),
            _tool_response('call_1', 'file1.py\nfile2.py'),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'cat file1.py'}, 'call_2')],
            ),
            _tool_response('call_2', 'content'),
            _assistant_tool_call(
                None,
                [
                    _tool_call(
                        'str_replace_editor',
                        {
                            'command': 'str_replace',
                            'path': '/workspace/file1.py',
                            'old_str': 'old',
                            'new_str': 'new',
                        },
                        'call_3',
                    )
                ],
            ),
            _tool_response('call_3', 'File edited'),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert initial.content == 'Fix the bug'
        assert len(events) == 3
        assert isinstance(events[0], CmdRunAction)
        assert isinstance(events[1], CmdRunAction)
        assert isinstance(events[2], FileEditAction)

    def test_finish_action(self):
        messages = [
            _user_msg('Do something'),
            _assistant_tool_call(
                None,
                [_tool_call('finish', {'message': 'All done'})],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert len(events) == 1
        assert isinstance(events[0], AgentFinishAction)

    def test_think_action(self):
        messages = [
            _user_msg('Do something'),
            _assistant_tool_call(
                None,
                [_tool_call('think', {'thought': 'Let me reason about this.'})],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert len(events) == 1
        assert isinstance(events[0], AgentThinkAction)

    def test_str_replace_editor_view(self):
        messages = [
            _user_msg('Read the file'),
            _assistant_tool_call(
                None,
                [
                    _tool_call(
                        'str_replace_editor',
                        {'command': 'view', 'path': '/workspace/foo.py'},
                    )
                ],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert len(events) == 1
        assert isinstance(events[0], FileReadAction)
        assert events[0].path == '/workspace/foo.py'

    def test_ipython_action(self):
        messages = [
            _user_msg('Run python'),
            _assistant_tool_call(
                None,
                [_tool_call('execute_ipython_cell', {'code': 'print("hello")'})],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert len(events) == 1
        assert isinstance(events[0], IPythonRunCellAction)
        assert events[0].code == 'print("hello")'

    def test_parallel_tool_calls_in_single_message(self):
        messages = [
            _user_msg('Check files'),
            _assistant_tool_call(
                'Let me check both files.',
                [
                    _tool_call('execute_bash', {'command': 'cat a.py'}, 'call_a'),
                    _tool_call('execute_bash', {'command': 'cat b.py'}, 'call_b'),
                ],
            ),
            _tool_response('call_a', 'content_a'),
            _tool_response('call_b', 'content_b'),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert len(events) == 2
        assert isinstance(events[0], CmdRunAction)
        assert events[0].command == 'cat a.py'
        assert isinstance(events[1], CmdRunAction)
        assert events[1].command == 'cat b.py'

    def test_assistant_message_without_tools(self):
        messages = [
            _user_msg('Hello'),
            _assistant_msg('I will help you.'),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'ls'})],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert len(events) == 2
        # First event is the text-only assistant message
        assert isinstance(events[0], MessageAction)
        assert events[0].content == 'I will help you.'
        # Second is the bash command
        assert isinstance(events[1], CmdRunAction)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMessagesToReplayEventsEdgeCases:
    @pytest.mark.parametrize('agent_class', ['OpenCodeAgent', 'CodexAgent'])
    def test_no_tool_call_stops_opencode_and_codex(self, agent_class):
        messages = [
            _user_msg('Fix the bug'),
            _assistant_msg('The task is complete.'),
            _user_msg('Please continue'),
            _assistant_tool_call(
                None,
                [_tool_call('nonexistent_tool', {})],
            ),
        ]

        events, initial = messages_to_replay_events(messages, agent_class)

        assert initial.content == 'Fix the bug'
        assert len(events) == 1
        assert isinstance(events[0], AgentFinishAction)
        assert events[0].final_thought == 'The task is complete.'
        assert events[0].thought == 'The task is complete.'

    @pytest.mark.parametrize('agent_class', ['OpenCodeAgent', 'CodexAgent'])
    def test_empty_no_tool_call_stops_opencode_and_codex(self, agent_class):
        messages = [
            _user_msg('Fix the bug'),
            _assistant_msg(''),
        ]

        events, _ = messages_to_replay_events(messages, agent_class)

        assert len(events) == 1
        assert isinstance(events[0], AgentFinishAction)
        assert events[0].final_thought == ''

    def test_no_user_message_raises(self):
        messages = [
            _system_msg(),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'ls'})],
            ),
        ]
        with pytest.raises(ValueError, match='No user message found'):
            messages_to_replay_events(messages, 'CodeActAgent')

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError, match='No user message found'):
            messages_to_replay_events([], 'CodeActAgent')

    def test_system_messages_skipped(self):
        messages = [
            _system_msg('System prompt 1'),
            _system_msg('System prompt 2'),
            _user_msg('Do the task'),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert initial.content == 'Do the task'
        assert len(events) == 0

    def test_tool_responses_skipped(self):
        messages = [
            _user_msg('Do the task'),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'ls'}, 'call_1')],
            ),
            _tool_response('call_1', 'output here'),
            _tool_response('call_1', 'duplicate response'),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        # Only the action, not the tool responses
        assert len(events) == 1
        assert isinstance(events[0], CmdRunAction)

    def test_subsequent_user_messages_skipped(self):
        """Fake user responses during agent execution should be skipped."""
        messages = [
            _user_msg('Initial task'),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'ls'})],
            ),
            _tool_response('call_001', 'files'),
            _user_msg('Please continue'),  # fake user response
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'pwd'})],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert initial.content == 'Initial task'
        assert len(events) == 2

    def test_user_message_multipart_content(self):
        messages = [
            _user_msg_multipart(['Fix the bug in', 'the utils module']),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'ls'})],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert initial.content == 'Fix the bug in\nthe utils module'

    def test_all_event_ids_are_none(self):
        messages = [
            _user_msg('Task'),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'ls'}, 'c1')],
            ),
            _tool_response('c1', 'out'),
            _assistant_msg('Thinking...'),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'pwd'}, 'c2')],
            ),
        ]
        events, _ = messages_to_replay_events(messages, 'CodeActAgent')

        for event in events:
            assert event._id is None

    def test_assistant_empty_content_no_tools_skipped(self):
        messages = [
            _user_msg('Task'),
            _assistant_msg(''),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'ls'})],
            ),
        ]
        events, _ = messages_to_replay_events(messages, 'CodeActAgent')

        # Empty assistant message should be skipped, only the tool call remains
        assert len(events) == 1
        assert isinstance(events[0], CmdRunAction)

    def test_only_user_message_returns_empty_events(self):
        messages = [_user_msg('Just a task, no actions yet')]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        assert initial.content == 'Just a task, no actions yet'
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Cross-agent: ensure the dispatcher selects the right converter
# ---------------------------------------------------------------------------


class TestCrossAgent:
    def test_codex_agent_shell_command(self):
        """CodexAgent uses 'shell_command' instead of 'execute_bash'."""
        messages = [
            _user_msg('Fix it'),
            _assistant_tool_call(
                None,
                [_tool_call('shell_command', {'command': 'ls -la'})],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodexAgent')

        assert len(events) == 1
        assert isinstance(events[0], CmdRunAction)
        assert events[0].command == 'ls -la'

    def test_opencode_agent_bash(self):
        """OpenCodeAgent uses 'bash' tool name (not 'execute_bash')."""
        messages = [
            _user_msg('Fix it'),
            _assistant_tool_call(
                None,
                [_tool_call('bash', {'command': 'pwd'})],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'OpenCodeAgent')

        assert len(events) == 1
        assert isinstance(events[0], CmdRunAction)
        assert events[0].command == 'pwd'


# ---------------------------------------------------------------------------
# Tool name detection, patching, and diversification support
# ---------------------------------------------------------------------------


class TestToSnakeCase:
    def test_already_snake(self):
        assert _to_snake_case('execute_bash') == 'execute_bash'

    def test_camel_case(self):
        assert _to_snake_case('ExecuteBash') == 'execute_bash'

    def test_single_word(self):
        assert _to_snake_case('finish') == 'finish'

    def test_camel_multi_word(self):
        assert _to_snake_case('StrReplaceEditor') == 'str_replace_editor'


class TestPatchToolName:
    def test_patches_tool_names_module(self):
        import openhands.llm.tool_names as tn

        original = tn.EXECUTE_BASH_TOOL_NAME
        try:
            _patch_tool_name('EXECUTE_BASH_TOOL_NAME', 'run_bash')
            assert tn.EXECUTE_BASH_TOOL_NAME == 'run_bash'
        finally:
            # Restore
            _patch_tool_name('EXECUTE_BASH_TOOL_NAME', original)
            assert tn.EXECUTE_BASH_TOOL_NAME == original

    def test_patches_importing_modules(self):
        import openhands.agenthub.codeact_agent.tools.bash as bash_mod
        import openhands.llm.tool_names as tn

        original = tn.EXECUTE_BASH_TOOL_NAME
        try:
            _patch_tool_name('EXECUTE_BASH_TOOL_NAME', 'shell_run')
            # The importing module should also be patched
            assert bash_mod.EXECUTE_BASH_TOOL_NAME == 'shell_run'
        finally:
            _patch_tool_name('EXECUTE_BASH_TOOL_NAME', original)

    def test_noop_when_already_correct(self):
        import openhands.llm.tool_names as tn

        current = tn.EXECUTE_BASH_TOOL_NAME
        # Should be a no-op, no error
        _patch_tool_name('EXECUTE_BASH_TOOL_NAME', current)
        assert tn.EXECUTE_BASH_TOOL_NAME == current


class TestDetectAndSetToolNames:
    def test_detects_diversified_bash_name(self):
        import openhands.llm.tool_names as tn

        original = tn.EXECUTE_BASH_TOOL_NAME
        try:
            messages = [
                _user_msg('Fix it'),
                _assistant_tool_call(
                    None,
                    [_tool_call('run_bash', {'command': 'ls'})],
                ),
            ]
            detect_and_set_tool_names(messages, 'CodeActAgent')
            assert tn.EXECUTE_BASH_TOOL_NAME == 'run_bash'
        finally:
            _patch_tool_name('EXECUTE_BASH_TOOL_NAME', original)

    def test_detects_camel_case_name(self):
        import openhands.llm.tool_names as tn

        original = tn.EXECUTE_BASH_TOOL_NAME
        try:
            messages = [
                _user_msg('Fix it'),
                _assistant_tool_call(
                    None,
                    [_tool_call('ShellRun', {'command': 'ls'})],
                ),
            ]
            detect_and_set_tool_names(messages, 'CodeActAgent')
            assert tn.EXECUTE_BASH_TOOL_NAME == 'ShellRun'
        finally:
            _patch_tool_name('EXECUTE_BASH_TOOL_NAME', original)

    def test_no_patching_when_names_match_default(self):
        import openhands.llm.tool_names as tn

        original = tn.EXECUTE_BASH_TOOL_NAME
        messages = [
            _user_msg('Fix it'),
            _assistant_tool_call(
                None,
                [_tool_call(original, {'command': 'ls'})],
            ),
        ]
        detect_and_set_tool_names(messages, 'CodeActAgent')
        assert tn.EXECUTE_BASH_TOOL_NAME == original

    def test_skips_non_diversifiable_tools(self):
        """Tools like 'think' and 'execute_ipython_cell' are not in tool_names.py."""
        import openhands.llm.tool_names as tn

        original_bash = tn.EXECUTE_BASH_TOOL_NAME
        messages = [
            _user_msg('Fix it'),
            _assistant_tool_call(
                None,
                [_tool_call('think', {'thought': 'hmm'})],
            ),
        ]
        # Should not raise or patch anything
        detect_and_set_tool_names(messages, 'CodeActAgent')
        assert tn.EXECUTE_BASH_TOOL_NAME == original_bash

    def test_no_messages_is_noop(self):
        detect_and_set_tool_names([], 'CodeActAgent')  # no error

    def test_no_tool_calls_is_noop(self):
        messages = [_user_msg('Hello'), _assistant_msg('Hi')]
        detect_and_set_tool_names(messages, 'CodeActAgent')  # no error


class TestDiversifiedToolNamesEndToEnd:
    """End-to-end: messages with diversified names should parse correctly."""

    def test_codeact_with_run_bash(self):
        import openhands.llm.tool_names as tn

        original = tn.EXECUTE_BASH_TOOL_NAME
        try:
            messages = [
                _user_msg('Fix the bug'),
                _assistant_tool_call(
                    'Let me check.',
                    [_tool_call('run_bash', {'command': 'git diff'})],
                ),
                _tool_response('call_001', 'diff output'),
            ]
            events, initial = messages_to_replay_events(messages, 'CodeActAgent')

            assert len(events) == 1
            assert isinstance(events[0], CmdRunAction)
            assert events[0].command == 'git diff'
            # Tool name should be patched for continuation
            assert tn.EXECUTE_BASH_TOOL_NAME == 'run_bash'
        finally:
            _patch_tool_name('EXECUTE_BASH_TOOL_NAME', original)

    def test_codeact_with_text_editor(self):
        import openhands.llm.tool_names as tn

        original = tn.STR_REPLACE_EDITOR_TOOL_NAME
        try:
            messages = [
                _user_msg('Fix the bug'),
                _assistant_tool_call(
                    None,
                    [
                        _tool_call(
                            'text_editor',
                            {'command': 'view', 'path': '/workspace/foo.py'},
                        )
                    ],
                ),
            ]
            events, initial = messages_to_replay_events(messages, 'CodeActAgent')

            assert len(events) == 1
            assert isinstance(events[0], FileReadAction)
            assert tn.STR_REPLACE_EDITOR_TOOL_NAME == 'text_editor'
        finally:
            _patch_tool_name('STR_REPLACE_EDITOR_TOOL_NAME', original)

    def test_codex_with_exec_command(self):
        import openhands.llm.tool_names as tn

        original = tn.CODEX_SHELL_COMMAND_TOOL_NAME
        try:
            messages = [
                _user_msg('Fix it'),
                _assistant_tool_call(
                    None,
                    [_tool_call('exec_command', {'command': 'ls'})],
                ),
            ]
            events, initial = messages_to_replay_events(messages, 'CodexAgent')

            assert len(events) == 1
            assert isinstance(events[0], CmdRunAction)
            assert tn.CODEX_SHELL_COMMAND_TOOL_NAME == 'exec_command'
        finally:
            _patch_tool_name('CODEX_SHELL_COMMAND_TOOL_NAME', original)

    def test_opencode_with_terminal(self):
        import openhands.llm.tool_names as tn

        original = tn.BASH_TOOL_NAME
        try:
            messages = [
                _user_msg('Fix it'),
                _assistant_tool_call(
                    None,
                    [_tool_call('terminal', {'command': 'pwd'})],
                ),
            ]
            events, initial = messages_to_replay_events(messages, 'OpenCodeAgent')

            assert len(events) == 1
            assert isinstance(events[0], CmdRunAction)
            assert tn.BASH_TOOL_NAME == 'terminal'
        finally:
            _patch_tool_name('BASH_TOOL_NAME', original)


# ---------------------------------------------------------------------------
# Terminus2 helpers
# ---------------------------------------------------------------------------

def _terminus2_json(
    analysis: str = 'Investigating',
    plan: str = 'Will run commands',
    commands: list[dict] | None = None,
    task_complete: bool = False,
) -> str:
    """Build a Terminus2-style JSON response string."""
    obj: dict = {
        'analysis': analysis,
        'plan': plan,
        'commands': commands or [],
    }
    if task_complete:
        obj['task_complete'] = True
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# _terminus2_response_to_actions
# ---------------------------------------------------------------------------


class TestTerminus2ResponseToActions:
    def test_single_command(self):
        from openhands.events.action.terminus_2 import Terminus2CmdRunAction

        content = _terminus2_json(
            commands=[{'keystrokes': 'ls -la\n', 'duration': 2.0}],
        )
        actions = _terminus2_response_to_actions(content)

        assert len(actions) == 1
        assert isinstance(actions[0], Terminus2CmdRunAction)
        assert actions[0].keystrokes == 'ls -la\n'
        assert actions[0].duration == 2.0
        assert actions[0].thought == content  # first action gets full response

    def test_multiple_commands(self):
        from openhands.events.action.terminus_2 import Terminus2CmdRunAction

        content = _terminus2_json(
            commands=[
                {'keystrokes': 'cd /workspace\n', 'duration': 1.0},
                {'keystrokes': 'cat file.py\n', 'duration': 3.0},
            ],
        )
        actions = _terminus2_response_to_actions(content)

        assert len(actions) == 2
        assert actions[0].keystrokes == 'cd /workspace\n'
        assert actions[0].thought == content  # first gets thought
        assert actions[1].keystrokes == 'cat file.py\n'
        assert actions[1].thought == ''  # subsequent actions have empty thought

    def test_task_complete_no_commands(self):
        content = _terminus2_json(task_complete=True, commands=[])
        actions = _terminus2_response_to_actions(content)

        assert len(actions) == 1
        assert isinstance(actions[0], AgentFinishAction)

    def test_task_complete_with_commands(self):
        from openhands.events.action.terminus_2 import Terminus2CmdRunAction

        content = _terminus2_json(
            commands=[{'keystrokes': 'exit\n', 'duration': 1.0}],
            task_complete=True,
        )
        actions = _terminus2_response_to_actions(content)

        # Commands are still returned even when task_complete is set
        assert len(actions) == 1
        assert isinstance(actions[0], Terminus2CmdRunAction)

    def test_empty_commands(self):
        content = _terminus2_json(commands=[])
        actions = _terminus2_response_to_actions(content)
        assert len(actions) == 0

    def test_duration_capped_at_60(self):
        content = _terminus2_json(
            commands=[{'keystrokes': 'sleep 999\n', 'duration': 120.0}],
        )
        actions = _terminus2_response_to_actions(content)

        assert len(actions) == 1
        assert actions[0].duration == 60.0

    def test_invalid_json_returns_empty(self):
        actions = _terminus2_response_to_actions('this is not json at all')
        assert len(actions) == 0

    def test_json_with_extra_text(self):
        from openhands.events.action.terminus_2 import Terminus2CmdRunAction

        content = 'Here is my response: ' + _terminus2_json(
            commands=[{'keystrokes': 'pwd\n', 'duration': 1.0}],
        ) + ' Done!'
        actions = _terminus2_response_to_actions(content)

        assert len(actions) == 1
        assert isinstance(actions[0], Terminus2CmdRunAction)
        assert actions[0].keystrokes == 'pwd\n'


# ---------------------------------------------------------------------------
# messages_to_replay_events – Terminus2Agent
# ---------------------------------------------------------------------------


class TestMessagesToReplayEventsTerminus2:
    def test_basic_terminus2_flow(self):
        from openhands.events.action.terminus_2 import Terminus2CmdRunAction

        messages = [
            _system_msg('You are a terminal agent.'),
            _user_msg('Fix the bug in utils.py'),
            _assistant_msg(
                _terminus2_json(
                    commands=[
                        {'keystrokes': 'cat utils.py\n', 'duration': 2.0},
                    ],
                )
            ),
            _user_msg('$ cat utils.py\ndef foo():\n    pass'),  # terminal output
            _assistant_msg(
                _terminus2_json(
                    commands=[
                        {'keystrokes': 'vim utils.py\n', 'duration': 1.0},
                        {'keystrokes': 'idef bar():\n    return 42\n', 'duration': 0.5},
                    ],
                )
            ),
            _user_msg('vim output...'),
        ]
        events, initial = messages_to_replay_events(messages, 'Terminus2Agent')

        assert isinstance(initial, MessageAction)
        assert initial.content == 'Fix the bug in utils.py'
        assert len(events) == 3
        assert isinstance(events[0], Terminus2CmdRunAction)
        assert events[0].keystrokes == 'cat utils.py\n'
        assert isinstance(events[1], Terminus2CmdRunAction)
        assert events[1].keystrokes == 'vim utils.py\n'
        assert isinstance(events[2], Terminus2CmdRunAction)
        assert events[2].keystrokes == 'idef bar():\n    return 42\n'

    def test_terminus2_task_complete(self):
        messages = [
            _user_msg('Do it'),
            _assistant_msg(
                _terminus2_json(
                    commands=[{'keystrokes': 'ls\n', 'duration': 1.0}],
                )
            ),
            _user_msg('file1 file2'),
            _assistant_msg(
                _terminus2_json(task_complete=True, commands=[])
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'Terminus2Agent')

        assert len(events) == 2
        # First: the ls command
        from openhands.events.action.terminus_2 import Terminus2CmdRunAction
        assert isinstance(events[0], Terminus2CmdRunAction)
        # Second: task complete → AgentFinishAction
        assert isinstance(events[1], AgentFinishAction)

    def test_terminus2_all_ids_none(self):
        messages = [
            _user_msg('Task'),
            _assistant_msg(
                _terminus2_json(
                    commands=[
                        {'keystrokes': 'a\n', 'duration': 1.0},
                        {'keystrokes': 'b\n', 'duration': 1.0},
                    ],
                )
            ),
        ]
        events, _ = messages_to_replay_events(messages, 'Terminus2Agent')
        for event in events:
            assert event._id is None

    def test_terminus2_skips_system_and_user_responses(self):
        messages = [
            _system_msg('System prompt'),
            _user_msg('Task instruction'),
            _assistant_msg(
                _terminus2_json(commands=[{'keystrokes': 'ls\n', 'duration': 1.0}])
            ),
            _user_msg('terminal output'),  # skipped (observation)
            _user_msg('more output'),  # skipped
        ]
        events, initial = messages_to_replay_events(messages, 'Terminus2Agent')

        assert initial.content == 'Task instruction'
        assert len(events) == 1

    def test_terminus2_empty_assistant_content_skipped(self):
        messages = [
            _user_msg('Task'),
            _assistant_msg(''),  # empty, skipped
            _assistant_msg(
                _terminus2_json(commands=[{'keystrokes': 'pwd\n', 'duration': 1.0}])
            ),
        ]
        events, _ = messages_to_replay_events(messages, 'Terminus2Agent')
        assert len(events) == 1

    def test_terminus2_invalid_json_assistant_skipped(self):
        messages = [
            _user_msg('Task'),
            _assistant_msg('I will help you with that.'),  # not JSON
            _assistant_msg(
                _terminus2_json(commands=[{'keystrokes': 'ls\n', 'duration': 1.0}])
            ),
        ]
        events, _ = messages_to_replay_events(messages, 'Terminus2Agent')
        # First message fails to parse (no JSON) → 0 events
        # Second message parses → 1 event
        assert len(events) == 1

    def test_terminus2_thought_field_set_correctly(self):
        from openhands.events.action.terminus_2 import Terminus2CmdRunAction

        json_content = _terminus2_json(
            commands=[
                {'keystrokes': 'a\n', 'duration': 1.0},
                {'keystrokes': 'b\n', 'duration': 1.0},
            ],
        )
        messages = [
            _user_msg('Task'),
            _assistant_msg(json_content),
        ]
        events, _ = messages_to_replay_events(messages, 'Terminus2Agent')

        assert len(events) == 2
        assert events[0].thought == json_content  # first action has full response
        assert events[1].thought == ''  # second has empty thought


# ---------------------------------------------------------------------------
# Integration: ReplayManager correctly processes converted events
# ---------------------------------------------------------------------------


class TestReplayManagerIntegration:
    """Verify that events produced by messages_to_replay_events() are correctly
    consumed by ReplayManager — the same component that run_controller uses.
    """

    def test_replay_manager_consumes_codeact_events(self):
        from openhands.controller.replay import ReplayManager

        messages = [
            _user_msg('Fix the bug'),
            _assistant_tool_call(
                'Investigating.',
                [_tool_call('execute_bash', {'command': 'ls'}, 'call_1')],
            ),
            _tool_response('call_1', 'file1.py'),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'cat file1.py'}, 'call_2')],
            ),
            _tool_response('call_2', 'content'),
            _assistant_tool_call(
                None,
                [_tool_call('finish', {'message': 'Done'}, 'call_3')],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        # Feed into ReplayManager (the real one used by AgentController)
        rm = ReplayManager(events)
        assert rm.replay_mode is True

        replayed_actions = []
        while rm.should_replay():
            action = rm.step()
            replayed_actions.append(action)

        assert len(replayed_actions) == 3
        assert isinstance(replayed_actions[0], CmdRunAction)
        assert replayed_actions[0].command == 'ls'
        assert isinstance(replayed_actions[1], CmdRunAction)
        assert replayed_actions[1].command == 'cat file1.py'
        assert isinstance(replayed_actions[2], AgentFinishAction)

        # After replay is done, should_replay returns False
        assert rm.should_replay() is False

    def test_replay_manager_consumes_terminus2_events(self):
        from openhands.controller.replay import ReplayManager
        from openhands.events.action.terminus_2 import Terminus2CmdRunAction

        messages = [
            _user_msg('Fix it'),
            _assistant_msg(
                _terminus2_json(
                    commands=[
                        {'keystrokes': 'ls\n', 'duration': 1.0},
                        {'keystrokes': 'pwd\n', 'duration': 0.5},
                    ],
                )
            ),
            _user_msg('terminal output'),
            _assistant_msg(
                _terminus2_json(task_complete=True, commands=[])
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'Terminus2Agent')

        rm = ReplayManager(events)
        assert rm.replay_mode is True

        replayed_actions = []
        while rm.should_replay():
            replayed_actions.append(rm.step())

        assert len(replayed_actions) == 3
        assert isinstance(replayed_actions[0], Terminus2CmdRunAction)
        assert replayed_actions[0].keystrokes == 'ls\n'
        assert isinstance(replayed_actions[1], Terminus2CmdRunAction)
        assert replayed_actions[1].keystrokes == 'pwd\n'
        assert isinstance(replayed_actions[2], AgentFinishAction)

    def test_replay_manager_empty_events(self):
        from openhands.controller.replay import ReplayManager

        messages = [_user_msg('Just a task')]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        rm = ReplayManager(events)
        # No events to replay, but replay_mode should be False
        assert rm.replay_mode is False
        assert rm.should_replay() is False

    def test_replay_manager_with_assistant_text_message(self):
        """Assistant messages without tool calls become MessageAction events.
        ReplayManager should replay them too."""
        from openhands.controller.replay import ReplayManager

        messages = [
            _user_msg('Hello'),
            _assistant_msg('I will help you.'),
            _assistant_tool_call(
                None,
                [_tool_call('execute_bash', {'command': 'ls'})],
            ),
        ]
        events, initial = messages_to_replay_events(messages, 'CodeActAgent')

        rm = ReplayManager(events)
        replayed = []
        while rm.should_replay():
            replayed.append(rm.step())

        # MessageAction and CmdRunAction are both Actions, so both get replayed
        assert len(replayed) == 2
        assert isinstance(replayed[0], MessageAction)
        assert replayed[0].content == 'I will help you.'
        assert isinstance(replayed[1], CmdRunAction)


class TestRunControllerSignature:
    """Verify run_controller accepts replay_events parameter."""

    def test_run_controller_has_replay_events_param(self):
        """Verify the signature accepts replay_events."""
        import inspect

        from openhands.core.main import run_controller

        sig = inspect.signature(run_controller)
        assert 'replay_events' in sig.parameters
        param = sig.parameters['replay_events']
        assert param.default is None
