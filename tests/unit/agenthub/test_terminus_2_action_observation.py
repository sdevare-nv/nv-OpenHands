"""Unit tests for Terminus-2 action and observation serialization/deserialization.

Tests that Terminus-2 actions and observations can be properly serialized to dict
and deserialized back to action/observation objects.
"""

import pytest

from openhands.core.schema import ActionType, ObservationType
from openhands.events.action.terminus_2 import Terminus2CmdRunAction
from openhands.events.observation.terminus_2 import Terminus2CmdOutputObservation
from openhands.events.serialization import event_from_dict, event_to_dict


# ==============================================================================
# Terminus2CmdRunAction Serialization Tests
# ==============================================================================


class TestTerminus2CmdRunActionSerialization:
    """Tests for Terminus2CmdRunAction serialization."""

    def test_serialize_basic(self):
        action = Terminus2CmdRunAction(keystrokes='ls -la\n')
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.TERMINUS_2_CMD_RUN
        assert serialized['args']['keystrokes'] == 'ls -la\n'
        assert serialized['args']['duration'] == 1.0

    def test_serialize_with_duration(self):
        action = Terminus2CmdRunAction(keystrokes='make\n', duration=30.0)
        serialized = event_to_dict(action)

        assert serialized['args']['keystrokes'] == 'make\n'
        assert serialized['args']['duration'] == 30.0

    def test_serialize_with_thought(self):
        action = Terminus2CmdRunAction(
            keystrokes='ls\n',
            duration=0.1,
            thought='Listing directory contents',
        )
        serialized = event_to_dict(action)

        assert serialized['args']['thought'] == 'Listing directory contents'

    def test_serialize_special_keys(self):
        action = Terminus2CmdRunAction(keystrokes='C-c', duration=0.1)
        serialized = event_to_dict(action)

        assert serialized['args']['keystrokes'] == 'C-c'

    def test_serialize_empty_keystrokes(self):
        action = Terminus2CmdRunAction(keystrokes='', duration=10.0)
        serialized = event_to_dict(action)

        assert serialized['args']['keystrokes'] == ''
        assert serialized['args']['duration'] == 10.0

    def test_deserialize_basic(self):
        data = {
            'id': 1,
            'action': ActionType.TERMINUS_2_CMD_RUN,
            'args': {
                'keystrokes': 'ls -la\n',
                'duration': 0.1,
                'thought': '',
            },
        }
        action = event_from_dict(data)

        assert isinstance(action, Terminus2CmdRunAction)
        assert action.keystrokes == 'ls -la\n'
        assert action.duration == 0.1

    def test_deserialize_with_all_params(self):
        data = {
            'id': 2,
            'action': ActionType.TERMINUS_2_CMD_RUN,
            'args': {
                'keystrokes': 'make build\n',
                'duration': 30.0,
                'thought': 'Building project',
            },
        }
        action = event_from_dict(data)

        assert isinstance(action, Terminus2CmdRunAction)
        assert action.keystrokes == 'make build\n'
        assert action.duration == 30.0
        assert action.thought == 'Building project'

    def test_roundtrip_serialization(self):
        original = Terminus2CmdRunAction(
            keystrokes='cd /tmp && ls\n',
            duration=2.5,
            thought='Navigate and list',
        )
        serialized = event_to_dict(original)
        restored = event_from_dict(serialized)

        assert isinstance(restored, Terminus2CmdRunAction)
        assert restored.keystrokes == original.keystrokes
        assert restored.duration == original.duration
        assert restored.thought == original.thought

    def test_action_type_is_correct(self):
        action = Terminus2CmdRunAction(keystrokes='test\n')
        assert action.action == ActionType.TERMINUS_2_CMD_RUN
        assert action.action == 'terminus_2_cmd_run'

    def test_message_property(self):
        action = Terminus2CmdRunAction(keystrokes='ls -la\n', duration=0.1)
        msg = action.message
        assert 'ls -la' in msg
        assert '0.1s' in msg

    def test_message_truncation(self):
        long_cmd = 'a' * 100 + '\n'
        action = Terminus2CmdRunAction(keystrokes=long_cmd, duration=1.0)
        msg = action.message
        assert '...' in msg


# ==============================================================================
# Terminus2CmdOutputObservation Serialization Tests
# ==============================================================================


class TestTerminus2CmdOutputObservationSerialization:
    """Tests for Terminus2CmdOutputObservation serialization."""

    def test_serialize_basic(self):
        obs = Terminus2CmdOutputObservation(
            content='output text',
            terminal_state='$ ls\nfile1.txt\nfile2.txt\n$',
        )
        serialized = event_to_dict(obs)

        assert serialized['observation'] == ObservationType.TERMINUS_2_CMD_OUTPUT
        assert serialized['content'] == 'output text'
        assert serialized['extras']['terminal_state'] == '$ ls\nfile1.txt\nfile2.txt\n$'

    def test_serialize_with_timeout(self):
        obs = Terminus2CmdOutputObservation(
            content='timed out',
            terminal_state='$ make\ncompiling...',
            timed_out=True,
            command_keystrokes='make\n',
        )
        serialized = event_to_dict(obs)

        assert serialized['extras']['timed_out'] is True
        assert serialized['extras']['command_keystrokes'] == 'make\n'

    def test_deserialize_basic(self):
        data = {
            'id': 1,
            'observation': ObservationType.TERMINUS_2_CMD_OUTPUT,
            'content': 'terminal output',
            'extras': {
                'terminal_state': '$ ls\nfiles...',
                'timed_out': False,
                'command_keystrokes': 'ls\n',
            },
        }
        obs = event_from_dict(data)

        assert isinstance(obs, Terminus2CmdOutputObservation)
        assert obs.terminal_state == '$ ls\nfiles...'
        assert obs.timed_out is False
        assert obs.command_keystrokes == 'ls\n'

    def test_deserialize_timed_out(self):
        data = {
            'id': 2,
            'observation': ObservationType.TERMINUS_2_CMD_OUTPUT,
            'content': 'partial output',
            'extras': {
                'terminal_state': 'compiling...',
                'timed_out': True,
                'command_keystrokes': 'make\n',
            },
        }
        obs = event_from_dict(data)

        assert isinstance(obs, Terminus2CmdOutputObservation)
        assert obs.timed_out is True

    def test_roundtrip_serialization(self):
        original = Terminus2CmdOutputObservation(
            content='full output',
            terminal_state='$ ls -la\ntotal 8\nfile1.txt\n$',
            timed_out=False,
            command_keystrokes='ls -la\n',
        )
        serialized = event_to_dict(original)
        restored = event_from_dict(serialized)

        assert isinstance(restored, Terminus2CmdOutputObservation)
        assert restored.terminal_state == original.terminal_state
        assert restored.timed_out == original.timed_out
        assert restored.command_keystrokes == original.command_keystrokes
        assert restored.content == original.content

    def test_observation_type_is_correct(self):
        obs = Terminus2CmdOutputObservation(content='test')
        assert obs.observation == ObservationType.TERMINUS_2_CMD_OUTPUT
        assert obs.observation == 'terminus_2_cmd_output'

    def test_message_property(self):
        obs = Terminus2CmdOutputObservation(
            content='output',
            command_keystrokes='ls\n',
        )
        msg = obs.message
        assert 'ls' in msg

    def test_message_with_timeout(self):
        obs = Terminus2CmdOutputObservation(
            content='output',
            command_keystrokes='make\n',
            timed_out=True,
        )
        msg = obs.message
        assert 'timed out' in msg

    def test_default_values(self):
        obs = Terminus2CmdOutputObservation(content='test')
        assert obs.terminal_state == ''
        assert obs.timed_out is False
        assert obs.command_keystrokes == ''


# ==============================================================================
# Schema Type Tests
# ==============================================================================


class TestSchemaTypes:
    """Tests that the schema enums are correctly defined."""

    def test_action_type_exists(self):
        assert hasattr(ActionType, 'TERMINUS_2_CMD_RUN')
        assert ActionType.TERMINUS_2_CMD_RUN == 'terminus_2_cmd_run'

    def test_observation_type_exists(self):
        assert hasattr(ObservationType, 'TERMINUS_2_CMD_OUTPUT')
        assert ObservationType.TERMINUS_2_CMD_OUTPUT == 'terminus_2_cmd_output'

    def test_action_type_in_serialization_map(self):
        from openhands.events.serialization.action import ACTION_TYPE_TO_CLASS
        assert 'terminus_2_cmd_run' in ACTION_TYPE_TO_CLASS
        assert ACTION_TYPE_TO_CLASS['terminus_2_cmd_run'] == Terminus2CmdRunAction

    def test_observation_type_in_serialization_map(self):
        from openhands.events.serialization.observation import OBSERVATION_TYPE_TO_CLASS
        assert 'terminus_2_cmd_output' in OBSERVATION_TYPE_TO_CLASS
        assert OBSERVATION_TYPE_TO_CLASS['terminus_2_cmd_output'] == Terminus2CmdOutputObservation


# ==============================================================================
# Import Tests
# ==============================================================================


class TestImports:
    """Tests that all new types are properly importable."""

    def test_import_action_from_events(self):
        from openhands.events.action import Terminus2CmdRunAction
        assert Terminus2CmdRunAction is not None

    def test_import_observation_from_events(self):
        from openhands.events.observation import Terminus2CmdOutputObservation
        assert Terminus2CmdOutputObservation is not None

    def test_agent_registration(self):
        from openhands.controller.agent import Agent
        import openhands.agenthub.terminus_2_agent  # noqa: F401
        assert 'Terminus2Agent' in Agent._registry
