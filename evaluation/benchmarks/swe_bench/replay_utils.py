"""Utilities for the replay-and-continue flow.

Converts raw LLM messages (OpenAI chat format) into Action events that can be
fed to the OpenHands ReplayManager, allowing a previous agent run to be resumed
from where it left off.

Supports all four agent types:
  - CodeActAgent, OpenCodeAgent, CodexAgent  (function-calling / tool_calls)
  - Terminus2Agent                           (JSON text responses)

Also handles diversified and CamelCase tool names (see ``tool_names.py``).
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.message import MessageAction
from openhands.events.event import EventSource

if TYPE_CHECKING:
    pass

# ============================================================================
# Tool-name detection and patching
# ============================================================================
# tool_names.py uses _pick() at import time to choose tool names.  When
# replaying a history that was recorded with different names (diversified or
# CamelCased), we detect those names and patch the module-level constants so
# that response_to_actions() recognises them AND the agent continues with the
# same names after replay.

# All known tool-name groups: constant attribute name → list of snake_case
# alternatives.  Must stay in sync with openhands/llm/tool_names.py.
_TOOL_NAME_ALTERNATIVES: dict[str, list[str]] = {
    # CodeAct
    'EXECUTE_BASH_TOOL_NAME': ['execute_bash', 'run_bash', 'bash_exec', 'shell_run', 'shell'],
    'STR_REPLACE_EDITOR_TOOL_NAME': ['str_replace_editor', 'text_editor', 'file_editor', 'code_editor'],
    'BROWSER_TOOL_NAME': ['browser', 'web_browser', 'browse_web', 'open_browser'],
    'FINISH_TOOL_NAME': ['finish'],
    'LLM_BASED_EDIT_TOOL_NAME': ['edit_file', 'modify_file', 'update_file', 'file_edit'],
    'TASK_TRACKER_TOOL_NAME': ['task_tracker', 'todo_tracker', 'plan_tracker'],
    # OpenCode
    'BASH_TOOL_NAME': ['bash', 'shell', 'terminal'],
    'GLOB_TOOL_NAME': ['glob', 'find_files', 'file_glob', 'match_files'],
    'GREP_TOOL_NAME': ['grep', 'search_text', 'find_pattern', 'text_search'],
    'LIST_DIR_TOOL_NAME': ['list_dir', 'ls', 'show_dir'],
    'READ_TOOL_NAME': ['read', 'read_file', 'view_file', 'cat_file'],
    'WRITE_TOOL_NAME': ['write', 'write_file', 'save_file', 'create_file'],
    'EDIT_TOOL_NAME': ['edit', 'modify', 'edit_file'],
    'OPENCODE_APPLY_PATCH_TOOL_NAME': ['apply_patch', 'patch_file', 'apply_diff', 'apply_changes'],
    'QUESTION_TOOL_NAME': ['question', 'ask_user', 'clarify', 'prompt_user'],
    'TODO_READ_TOOL_NAME': ['todo_read', 'read_todos', 'list_tasks'],
    'TODO_WRITE_TOOL_NAME': ['todo_write', 'write_todos', 'update_tasks'],
    # Codex
    'CODEX_SHELL_COMMAND_TOOL_NAME': ['shell_command', 'shell_cmd', 'run_shell_cmd', 'exec_command', 'exec_cmd'],
    'CODEX_READ_FILE_TOOL_NAME': ['read_file', 'view_file', 'cat_file', 'read_file_content'],
    'CODEX_LIST_DIR_TOOL_NAME': ['list_dir', 'ls_dir', 'list_directory'],
    'CODEX_GREP_FILES_TOOL_NAME': ['grep_files', 'search_files', 'find_in_files', 'code_search'],
    'CODEX_APPLY_PATCH_TOOL_NAME': ['apply_patch', 'patch_file', 'apply_diff', 'apply_changes'],
    'CODEX_UPDATE_PLAN_TOOL_NAME': ['update_plan', 'modify_plan', 'edit_plan'],
}

# Which tool-name groups are relevant per agent (avoids ambiguity across agents).
_AGENT_TOOL_GROUPS: dict[str, list[str]] = {
    'CodeActAgent': [
        'EXECUTE_BASH_TOOL_NAME', 'STR_REPLACE_EDITOR_TOOL_NAME', 'BROWSER_TOOL_NAME',
        'FINISH_TOOL_NAME', 'LLM_BASED_EDIT_TOOL_NAME', 'TASK_TRACKER_TOOL_NAME',
    ],
    'OpenCodeAgent': [
        'BASH_TOOL_NAME', 'GLOB_TOOL_NAME', 'GREP_TOOL_NAME', 'LIST_DIR_TOOL_NAME',
        'READ_TOOL_NAME', 'WRITE_TOOL_NAME', 'EDIT_TOOL_NAME', 'OPENCODE_APPLY_PATCH_TOOL_NAME',
        'QUESTION_TOOL_NAME', 'TODO_READ_TOOL_NAME', 'TODO_WRITE_TOOL_NAME',
    ],
    'CodexAgent': [
        'CODEX_SHELL_COMMAND_TOOL_NAME', 'CODEX_READ_FILE_TOOL_NAME', 'CODEX_LIST_DIR_TOOL_NAME',
        'CODEX_GREP_FILES_TOOL_NAME', 'CODEX_APPLY_PATCH_TOOL_NAME', 'CODEX_UPDATE_PLAN_TOOL_NAME',
    ],
}


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case (for CAMEL_CASE_TOOL_NAMES support)."""
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def _patch_tool_name(attr_name: str, new_value: str) -> None:
    """Patch a tool-name constant in tool_names module and every module that imported it.

    Also patches pre-created tool definition dicts (ChatCompletionToolParam) whose
    ``function.name`` still holds the old value.  This is necessary because some
    agents (Codex, OpenCode) create tool dicts at import time.
    """
    import openhands.llm.tool_names as tool_names_mod

    old_value = getattr(tool_names_mod, attr_name, None)
    if old_value == new_value:
        return  # already correct

    setattr(tool_names_mod, attr_name, new_value)

    # Patch all openhands modules that imported this constant via
    # ``from ... import``.  Also look for pre-created tool-definition dicts
    # whose ['function']['name'] == old_value and update them in-place.
    for mod_name, mod in sys.modules.items():
        if mod is None or mod is tool_names_mod:
            continue
        if not mod_name.startswith('openhands.'):
            continue
        # Patch the simple constant binding
        if hasattr(mod, attr_name) and getattr(mod, attr_name) == old_value:
            setattr(mod, attr_name, new_value)
        # Patch pre-created tool definition dicts in the module
        for obj_name in dir(mod):
            obj = getattr(mod, obj_name, None)
            if isinstance(obj, dict):
                try:
                    fn_dict = obj.get('function')
                    if isinstance(fn_dict, dict) and fn_dict.get('name') == old_value:
                        fn_dict['name'] = new_value
                except (TypeError, AttributeError):
                    pass

    logger.debug(f'Patched tool name {attr_name}: {old_value!r} -> {new_value!r}')


def detect_and_set_tool_names(messages: list[dict], agent_class: str) -> None:
    """Detect diversified/CamelCase tool names in the message history and patch
    the module-level constants so that ``response_to_actions()`` recognises them
    and the agent continues with the same names after replay.
    """
    # Collect all tool function names from assistant messages
    used_names: set[str] = set()
    for msg in messages:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                fn = tc.get('function', {})
                if 'name' in fn:
                    used_names.add(fn['name'])

    if not used_names:
        return

    # Build reverse map: snake_case alternative -> constant attr name
    groups = _AGENT_TOOL_GROUPS.get(agent_class, [])
    reverse_map: dict[str, str] = {}
    for attr_name in groups:
        for alt in _TOOL_NAME_ALTERNATIVES[attr_name]:
            reverse_map[alt] = attr_name

    for name in used_names:
        # Try direct match first (handles snake_case names)
        if name in reverse_map:
            _patch_tool_name(reverse_map[name], name)
            continue
        # Try converting CamelCase to snake_case
        snake = _to_snake_case(name)
        if snake in reverse_map:
            _patch_tool_name(reverse_map[snake], name)
            continue
        # Not a diversifiable tool name (e.g. 'think', 'execute_ipython_cell')
        logger.debug(
            f'Tool name {name!r} not in diversifiable groups for {agent_class}, skipping'
        )


# ============================================================================
# Response-to-actions dispatcher
# ============================================================================


def _get_response_to_actions_fn(agent_class: str):
    """Return the agent-specific ``response_to_actions`` function.

    For function-calling agents (CodeAct, OpenCode, Codex) this returns the
    ``response_to_actions`` function from the agent's ``function_calling`` module.
    For Terminus2Agent this returns ``None`` (handled separately).
    """
    if agent_class == 'CodeActAgent':
        from openhands.agenthub.codeact_agent.function_calling import response_to_actions
    elif agent_class == 'OpenCodeAgent':
        from openhands.agenthub.opencode_agent.function_calling import response_to_actions
    elif agent_class == 'CodexAgent':
        from openhands.agenthub.codex_agent.function_calling import response_to_actions
    elif agent_class == 'Terminus2Agent':
        return None  # Terminus2 uses JSON text, not tool calls
    else:
        raise ValueError(
            f'Replay is not supported for agent class {agent_class}. '
            'Supported: CodeActAgent, OpenCodeAgent, CodexAgent, Terminus2Agent'
        )
    return response_to_actions


def _terminus2_response_to_actions(content: str) -> list:
    """Parse a Terminus2 assistant message (JSON text) into Action events.

    Uses ``TerminusJSONPlainParser`` to extract commands, then creates
    ``Terminus2CmdRunAction`` for each command.  The raw response text is stored
    in the ``thought`` field of the first action (matching Terminus2Agent
    behaviour for history reconstruction).
    """
    from openhands.agenthub.terminus_2_agent.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )
    from openhands.events.action.agent import AgentFinishAction
    from openhands.events.action.terminus_2 import Terminus2CmdRunAction

    parser = TerminusJSONPlainParser()
    result = parser.parse_response(content)

    actions: list = []

    if result.error:
        logger.warning(f'Terminus2 replay parse error: {result.error}')
        return actions

    if result.is_task_complete and not result.commands:
        actions.append(AgentFinishAction(final_thought=content))
        return actions

    for i, cmd in enumerate(result.commands):
        action = Terminus2CmdRunAction(
            keystrokes=cmd.keystrokes,
            duration=min(cmd.duration, 60.0),
            thought=content if i == 0 else '',
        )
        actions.append(action)

    return actions


# ============================================================================
# Main converter
# ============================================================================


def messages_to_replay_events(
    messages: list[dict],
    agent_class: str,
) -> tuple[list, MessageAction]:
    """Convert raw LLM messages (OpenAI chat format) to replay events.

    For function-calling agents (CodeAct, OpenCode, Codex), parses assistant
    messages with ``tool_calls`` into Action events.  For Terminus2Agent,
    parses the raw JSON text content of assistant messages into
    ``Terminus2CmdRunAction`` events.

    Tool response, system, and subsequent user messages are skipped
    (observations are regenerated by the runtime during replay, and system
    prompts are regenerated by the agent).

    Args:
        messages: List of OpenAI chat format message dicts with keys:
            role, content, tool_calls (for assistant), tool_call_id (for tool).
        agent_class: Name of the agent class (e.g., ``'CodeActAgent'``).

    Returns:
        ``(replay_events, initial_action)`` where *replay_events* is a list of
        Action events to replay, and *initial_action* is the first user
        ``MessageAction`` (the task instruction).
    """
    is_terminus2 = agent_class == 'Terminus2Agent'

    if not is_terminus2:
        from litellm.types.utils import ModelResponse

        # Detect and patch tool names BEFORE importing response_to_actions,
        # so the constants are correct when the agent's function_calling
        # module is loaded.
        detect_and_set_tool_names(messages, agent_class)

    response_to_actions = _get_response_to_actions_fn(agent_class)

    replay_events: list = []
    initial_action: MessageAction | None = None

    for msg in messages:
        role = msg.get('role', '')

        if role == 'system' or role == 'tool':
            continue

        if role == 'user':
            if initial_action is None:
                content = msg.get('content', '')
                if isinstance(content, list):
                    text_parts = [
                        item.get('text', '') for item in content
                        if isinstance(item, dict) and item.get('type') == 'text'
                    ]
                    content = '\n'.join(text_parts)
                initial_action = MessageAction(content=content)
            continue

        if role == 'assistant':
            if is_terminus2:
                content = msg.get('content', '')
                if content:
                    actions = _terminus2_response_to_actions(content)
                    for action in actions:
                        action._id = None
                        # AgentController._step() reads action._source when re-adding the
                        # replayed action to the event stream. response_to_actions() does
                        # not set this, so AGENT (the natural origin for assistant turns)
                        # is supplied here.
                        action._source = EventSource.AGENT
                        replay_events.append(action)
            else:
                tool_calls = msg.get('tool_calls')
                stops_without_tool_call = agent_class in {
                    'OpenCodeAgent',
                    'CodexAgent',
                }
                if tool_calls or stops_without_tool_call:
                    response_dict = {
                        'id': f'replay-{len(replay_events)}',
                        'choices': [{
                            'message': {
                                'role': 'assistant',
                                'content': msg.get('content', ''),
                                'tool_calls': tool_calls,
                            },
                            'index': 0,
                            'finish_reason': 'tool_calls' if tool_calls else 'stop',
                        }],
                        'model': 'replay',
                    }
                    response = ModelResponse.model_validate(response_dict)
                    actions = response_to_actions(response)
                    for action in actions:
                        action._id = None
                        # AgentController._step() reads action._source when re-adding the
                        # replayed action to the event stream. response_to_actions() does
                        # not set this, so AGENT (the natural origin for assistant turns)
                        # is supplied here.
                        action._source = EventSource.AGENT
                        replay_events.append(action)
                    if not tool_calls:
                        break
                else:
                    content = msg.get('content', '')
                    if content:
                        action = MessageAction(content=content, wait_for_response=False)
                        action._id = None
                        action._source = EventSource.AGENT
                        replay_events.append(action)

    if initial_action is None:
        raise ValueError(
            'No user message found in replay messages. '
            'Expected at least one user message as the task instruction.'
        )

    logger.info(
        f'Converted {len(replay_events)} replay events from {len(messages)} messages'
    )
    return replay_events, initial_action
