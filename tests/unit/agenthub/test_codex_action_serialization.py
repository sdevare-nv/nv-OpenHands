"""Unit tests for Codex action and observation serialization/deserialization.

Tests that Codex actions and observations can be properly serialized to dict
and deserialized back to action/observation objects for network transport.
"""

import pytest

from openhands.core.schema import ActionType, ObservationType
from openhands.events.action.codex import (
    CodexApplyPatchAction,
    CodexGrepFilesAction,
    CodexListDirAction,
    CodexReadFileAction,
    CodexUpdatePlanAction,
)
from openhands.events.observation.codex import (
    CodexApplyPatchObservation,
    CodexUpdatePlanObservation,
)
from openhands.events.serialization import event_from_dict, event_to_dict


# ==============================================================================
# CodexReadFileAction Serialization Tests
# ==============================================================================


class TestCodexReadFileActionSerialization:
    """Tests for CodexReadFileAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        action = CodexReadFileAction(file_path='/test/file.py')
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.CODEX_READ_FILE
        assert serialized['args']['file_path'] == '/test/file.py'
        assert serialized['args']['offset'] == 1
        assert serialized['args']['limit'] == 2000
        assert serialized['args']['mode'] == 'slice'

    def test_serialize_with_all_params(self):
        """Test serialization with all parameters."""
        action = CodexReadFileAction(
            file_path='/test/file.py',
            offset=50,
            limit=100,
            mode='indentation',
            indentation={'anchor_line': 25, 'max_levels': 2},
            thought='Reading file for context',
        )
        serialized = event_to_dict(action)

        assert serialized['args']['file_path'] == '/test/file.py'
        assert serialized['args']['offset'] == 50
        assert serialized['args']['limit'] == 100
        assert serialized['args']['mode'] == 'indentation'
        assert serialized['args']['indentation'] == {'anchor_line': 25, 'max_levels': 2}
        assert serialized['args']['thought'] == 'Reading file for context'

    def test_deserialize_basic(self):
        """Test basic deserialization."""
        data = {
            'id': 1,
            'action': ActionType.CODEX_READ_FILE,
            'args': {
                'file_path': '/test/file.py',
                'offset': 1,
                'limit': 2000,
                'mode': 'slice',
                'indentation': {},
                'thought': '',
            },
        }
        action = event_from_dict(data)

        assert isinstance(action, CodexReadFileAction)
        assert action.file_path == '/test/file.py'
        assert action.offset == 1
        assert action.limit == 2000

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CodexReadFileAction(
            file_path='/path/to/file.txt',
            offset=50,
            limit=100,
            mode='indentation',
            indentation={'anchor_line': 10},
            thought='Test thought',
        )

        serialized = event_to_dict(original)
        restored = event_from_dict(serialized)

        assert isinstance(restored, CodexReadFileAction)
        assert restored.file_path == original.file_path
        assert restored.offset == original.offset
        assert restored.limit == original.limit
        assert restored.mode == original.mode
        assert restored.indentation == original.indentation
        assert restored.thought == original.thought


# ==============================================================================
# CodexListDirAction Serialization Tests
# ==============================================================================


class TestCodexListDirActionSerialization:
    """Tests for CodexListDirAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        action = CodexListDirAction(dir_path='/workspace')
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.CODEX_LIST_DIR
        assert serialized['args']['dir_path'] == '/workspace'
        assert serialized['args']['offset'] == 1
        assert serialized['args']['limit'] == 25
        assert serialized['args']['depth'] == 2

    def test_serialize_with_all_params(self):
        """Test serialization with all parameters."""
        action = CodexListDirAction(
            dir_path='/workspace/src',
            offset=10,
            limit=50,
            depth=3,
        )
        serialized = event_to_dict(action)

        assert serialized['args']['offset'] == 10
        assert serialized['args']['limit'] == 50
        assert serialized['args']['depth'] == 3

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CodexListDirAction(dir_path='/test', offset=5, limit=10, depth=1)

        serialized = event_to_dict(original)
        restored = event_from_dict(serialized)

        assert isinstance(restored, CodexListDirAction)
        assert restored.dir_path == original.dir_path
        assert restored.offset == original.offset
        assert restored.limit == original.limit
        assert restored.depth == original.depth


# ==============================================================================
# CodexGrepFilesAction Serialization Tests
# ==============================================================================


class TestCodexGrepFilesActionSerialization:
    """Tests for CodexGrepFilesAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        action = CodexGrepFilesAction(pattern='TODO')
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.CODEX_GREP_FILES
        assert serialized['args']['pattern'] == 'TODO'
        assert serialized['args']['include'] == ''
        assert serialized['args']['path'] == ''
        assert serialized['args']['limit'] == 100

    def test_serialize_with_all_params(self):
        """Test serialization with all parameters."""
        action = CodexGrepFilesAction(
            pattern='def \\w+',
            include='*.py',
            path='/workspace/src',
            limit=50,
        )
        serialized = event_to_dict(action)

        assert serialized['args']['pattern'] == 'def \\w+'
        assert serialized['args']['include'] == '*.py'
        assert serialized['args']['path'] == '/workspace/src'
        assert serialized['args']['limit'] == 50

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CodexGrepFilesAction(
            pattern='class \\w+', include='*.py', path='/src', limit=20
        )

        serialized = event_to_dict(original)
        restored = event_from_dict(serialized)

        assert isinstance(restored, CodexGrepFilesAction)
        assert restored.pattern == original.pattern
        assert restored.include == original.include
        assert restored.path == original.path
        assert restored.limit == original.limit


# ==============================================================================
# CodexApplyPatchAction Serialization Tests
# ==============================================================================


class TestCodexApplyPatchActionSerialization:
    """Tests for CodexApplyPatchAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        patch = '*** Begin Patch\n--- a/test.py\n+++ b/test.py\n*** End Patch'
        action = CodexApplyPatchAction(patch=patch)
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.CODEX_APPLY_PATCH
        assert serialized['args']['patch'] == patch

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CodexApplyPatchAction(
            patch='*** Begin Patch\nsome patch\n*** End Patch',
            thought='Applying fix',
        )

        serialized = event_to_dict(original)
        restored = event_from_dict(serialized)

        assert isinstance(restored, CodexApplyPatchAction)
        assert restored.patch == original.patch
        assert restored.thought == original.thought


# ==============================================================================
# CodexUpdatePlanAction Serialization Tests
# ==============================================================================


class TestCodexUpdatePlanActionSerialization:
    """Tests for CodexUpdatePlanAction serialization."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        plan = [
            {'step': 'Read code', 'status': 'completed'},
            {'step': 'Write tests', 'status': 'pending'},
        ]
        action = CodexUpdatePlanAction(plan=plan, explanation='Starting task')
        serialized = event_to_dict(action)

        assert serialized['action'] == ActionType.CODEX_UPDATE_PLAN
        assert len(serialized['args']['plan']) == 2
        assert serialized['args']['explanation'] == 'Starting task'

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CodexUpdatePlanAction(
            plan=[{'step': 'Step 1', 'status': 'in_progress'}],
            explanation='Working on it',
        )

        serialized = event_to_dict(original)
        restored = event_from_dict(serialized)

        assert isinstance(restored, CodexUpdatePlanAction)
        assert restored.plan == original.plan
        assert restored.explanation == original.explanation


# ==============================================================================
# CodexApplyPatchObservation Serialization Tests
# ==============================================================================


class TestCodexApplyPatchObservationSerialization:
    """Tests for CodexApplyPatchObservation serialization."""

    def test_serialize_success(self):
        """Test serialization of successful patch."""
        obs = CodexApplyPatchObservation(
            content='Patch applied successfully.',
            files_changed=['test.py', 'main.py'],
            success=True,
        )
        serialized = event_to_dict(obs)

        assert serialized['observation'] == ObservationType.CODEX_APPLY_PATCH
        assert serialized['content'] == 'Patch applied successfully.'
        assert serialized['extras']['files_changed'] == ['test.py', 'main.py']
        assert serialized['extras']['success'] is True

    def test_serialize_failure(self):
        """Test serialization of failed patch."""
        obs = CodexApplyPatchObservation(
            content='Failed to apply patch: conflict',
            files_changed=[],
            success=False,
        )
        serialized = event_to_dict(obs)

        assert serialized['extras']['success'] is False
        assert serialized['extras']['files_changed'] == []

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CodexApplyPatchObservation(
            content='Applied',
            files_changed=['a.py'],
            success=True,
        )

        serialized = event_to_dict(original)
        restored = event_from_dict(serialized)

        assert isinstance(restored, CodexApplyPatchObservation)
        assert restored.content == original.content
        assert restored.files_changed == original.files_changed
        assert restored.success == original.success


# ==============================================================================
# CodexUpdatePlanObservation Serialization Tests
# ==============================================================================


class TestCodexUpdatePlanObservationSerialization:
    """Tests for CodexUpdatePlanObservation serialization."""

    def test_serialize_success(self):
        """Test serialization of successful plan update."""
        obs = CodexUpdatePlanObservation(
            content='Plan updated',
            plan=[{'step': 'Step 1', 'status': 'completed'}],
            success=True,
        )
        serialized = event_to_dict(obs)

        assert serialized['observation'] == ObservationType.CODEX_UPDATE_PLAN
        assert serialized['content'] == 'Plan updated'
        assert len(serialized['extras']['plan']) == 1
        assert serialized['extras']['success'] is True

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CodexUpdatePlanObservation(
            content='Plan updated',
            plan=[
                {'step': 'A', 'status': 'completed'},
                {'step': 'B', 'status': 'in_progress'},
            ],
            success=True,
        )

        serialized = event_to_dict(original)
        restored = event_from_dict(serialized)

        assert isinstance(restored, CodexUpdatePlanObservation)
        assert restored.content == original.content
        assert restored.plan == original.plan
        assert restored.success == original.success


# ==============================================================================
# Action Message Property Tests
# ==============================================================================


class TestActionMessages:
    """Tests for action .message property."""

    def test_read_file_message(self):
        action = CodexReadFileAction(file_path='/test/file.py')
        assert action.message == 'Reading file: /test/file.py'

    def test_read_file_message_with_offset(self):
        action = CodexReadFileAction(file_path='/test/file.py', offset=50)
        assert action.message == 'Reading file: /test/file.py (from line 50)'

    def test_list_dir_message(self):
        action = CodexListDirAction(dir_path='/workspace')
        assert action.message == 'Listing directory: /workspace'

    def test_grep_files_message(self):
        action = CodexGrepFilesAction(pattern='TODO')
        assert action.message == 'Searching for files matching pattern: TODO'

    def test_apply_patch_message(self):
        action = CodexApplyPatchAction(patch='patch text')
        assert action.message == 'Applying patch'

    def test_update_plan_message_single(self):
        action = CodexUpdatePlanAction(plan=[{'step': 'A', 'status': 'pending'}])
        assert action.message == 'Updating plan with 1 step'

    def test_update_plan_message_multiple(self):
        action = CodexUpdatePlanAction(
            plan=[
                {'step': 'A', 'status': 'pending'},
                {'step': 'B', 'status': 'pending'},
            ]
        )
        assert action.message == 'Updating plan with 2 steps'


# ==============================================================================
# Observation Message Property Tests
# ==============================================================================


class TestObservationMessages:
    """Tests for observation .message property."""

    def test_apply_patch_success_message(self):
        obs = CodexApplyPatchObservation(
            content='ok', files_changed=['a.py', 'b.py'], success=True
        )
        assert obs.message == 'Applied patch to 2 files'

    def test_apply_patch_single_file_message(self):
        obs = CodexApplyPatchObservation(
            content='ok', files_changed=['a.py'], success=True
        )
        assert obs.message == 'Applied patch to 1 file'

    def test_apply_patch_failure_message(self):
        obs = CodexApplyPatchObservation(
            content='error', files_changed=[], success=False
        )
        assert obs.message == 'Failed to apply patch'

    def test_update_plan_success_message(self):
        obs = CodexUpdatePlanObservation(
            content='ok',
            plan=[{'step': 'A', 'status': 'done'}, {'step': 'B', 'status': 'done'}],
            success=True,
        )
        assert obs.message == 'Plan updated with 2 steps'

    def test_update_plan_failure_message(self):
        obs = CodexUpdatePlanObservation(
            content='error', plan=[], success=False
        )
        assert obs.message == 'Failed to update plan'
