"""Terminus-2 Agent for OpenHands.

A keystroke-based terminal agent that sends raw keystrokes to a terminal session
and receives screen capture output. Uses JSON-formatted LLM responses with
analysis, plan, and commands fields.
"""

import os
from collections import deque
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from openhands.events.action import Action

from openhands.agenthub.terminus_2_agent.terminus_json_plain_parser import (
    ParsedCommand,
    TerminusJSONPlainParser,
)
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import AgentFinishAction, AgentThinkAction, MessageAction
from openhands.events.action.terminus_2 import Terminus2CmdRunAction
from openhands.events.event import Event, EventSource
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.terminus_2 import Terminus2CmdOutputObservation
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.runtime.plugins import PluginRequirement
from openhands.utils.prompt import PromptManager


MAX_OUTPUT_BYTES = 10000
MAX_LLM_RETRY = 3

TIMEOUT_TEMPLATE = (
    'Previous command:\n{command}\n\n'
    'The previous command timed out after {timeout_sec} seconds\n\n'
    'It is possible that the command is not yet finished executing. '
    'If that is the case, then do nothing. It is also possible that you '
    'have entered an interactive shell and should continue sending '
    'keystrokes as normal.\n\n'
    'Here is the current state of the terminal:\n\n{terminal_state}'
)

COMPLETION_CONFIRMATION = (
    'Current terminal state:\n{terminal_state}\n\n'
    'Are you sure you want to mark the task as complete? '
    'This will trigger your solution to be graded and you won\'t be able to '
    'make any further corrections. If so, include "task_complete": true '
    'in your JSON response again.'
)


class Terminus2Agent(Agent):
    VERSION = '1.0'
    """
    The Terminus-2 Agent sends raw keystrokes to a terminal session and receives
    screen capture output, using JSON-formatted LLM responses.

    Unlike function-calling agents (CodeAct, OpenCode), Terminus-2 parses the
    LLM's raw text response as JSON with fields: analysis, plan, commands, and
    optionally task_complete.

    Key features:
    - Keystroke-based terminal interaction (tmux-style)
    - JSON response parsing with auto-correction
    - Double confirmation for task completion
    - Output truncation to 10KB
    - Duration-based command timeouts
    """

    sandbox_plugins: list[PluginRequirement] = []

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self.parser = TerminusJSONPlainParser()
        self._pending_completion = False
        self._conversation_messages: list[dict[str, str]] = []
        self._needs_llm_call = True

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        self.llm = self.llm_registry.get_router(self.config)

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            prompt_dir = (
                self.config.custom_prompt_dir
                if self.config.custom_prompt_dir
                else os.path.join(os.path.dirname(__file__), 'prompts')
            )

            template_overrides = {}
            if self.config.system_prompt_path:
                template_overrides['system_prompt.j2'] = self.config.system_prompt_path
            if self.config.system_prompt_long_horizon_path:
                template_overrides['system_prompt_long_horizon.j2'] = (
                    self.config.system_prompt_long_horizon_path
                )

            self._prompt_manager = PromptManager(
                prompt_dir=prompt_dir,
                system_prompt_filename=self.config.resolved_system_prompt_filename,
                template_overrides=template_overrides if template_overrides else None,
            )

        return self._prompt_manager

    def reset(self) -> None:
        super().reset()
        self.pending_actions.clear()
        self._pending_completion = False
        self._conversation_messages = []
        self._needs_llm_call = True

    def _has_terminal_observation(self, events: list[Event]) -> bool:
        """Check if any Terminus2CmdOutputObservation exists in the event history."""
        return any(isinstance(e, Terminus2CmdOutputObservation) for e in events)

    def step(self, state: State) -> 'Action':
        """Performs one step of the Terminus-2 agent.

        On the very first step (before any terminal observations exist), sends a
        no-op action to capture the initial terminal screen. This mirrors the
        original Terminus-2 behavior of capturing tmux state before the first
        LLM call.

        Returns pending actions from the queue, or calls the LLM to get
        new commands when the queue is empty.
        """
        if self.pending_actions:
            return self.pending_actions.popleft()

        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        if not self._has_terminal_observation(condensed_history):
            return Terminus2CmdRunAction(keystrokes='', duration=0.5)

        messages = self._build_messages(condensed_history, state)

        commands, is_task_complete, response_text = self._call_llm_and_parse(messages)

        if is_task_complete:
            if self._pending_completion:
                return AgentFinishAction(thought='Task completed (confirmed)')
            else:
                self._pending_completion = True
        else:
            self._pending_completion = False

        for i, cmd in enumerate(commands):
            action = Terminus2CmdRunAction(
                keystrokes=cmd.keystrokes,
                duration=min(cmd.duration, 60),
                thought=response_text if i == 0 else '',
            )
            self.pending_actions.append(action)

        if not self.pending_actions:
            if self._pending_completion:
                return Terminus2CmdRunAction(
                    keystrokes='', duration=0.5, thought=response_text
                )
            return AgentThinkAction(thought='No commands to execute, waiting for next input')

        return self.pending_actions.popleft()

    def _build_messages(
        self, condensed_history: list[Event], state: State
    ) -> list[Message]:
        """Build the conversation messages from event history.

        Converts the event stream into a user/assistant message sequence:
        - System message: JSON format instructions from the prompt template
        - First user message: the instruction content (from get_instruction /
          INSTRUCTION_TEMPLATE_PATH) used directly, with initial terminal state
          appended if available. This keeps the agent consistent with CodeAct/
          Codex/OpenCode which pass the MessageAction.content through unchanged,
          so that INSTRUCTION_TEMPLATE_PATH overrides work without double-wrapping.
        - Subsequent turns: assistant = LLM JSON response, user = terminal output
        """
        messages: list[Message] = []

        system_prompt = self.prompt_manager.get_system_message()
        messages.append(Message(role='system', content=[TextContent(text=system_prompt)]))

        initial_user_msg = self._find_initial_user_message(condensed_history)
        initial_terminal_event = self._find_initial_terminal_event(condensed_history)

        if initial_user_msg:
            if initial_terminal_event is not None:
                terminal_text = initial_terminal_event.terminal_state
                first_user_text = f'{initial_user_msg}\n\n{terminal_text}'
            else:
                first_user_text = initial_user_msg
            messages.append(
                Message(role='user', content=[TextContent(text=first_user_text)])
            )

        batch_observations: list[str] = []
        last_timed_out = False
        last_keystrokes = ''

        for event in condensed_history:
            if isinstance(event, MessageAction):
                if event.source == EventSource.USER:
                    continue
                elif event.source == EventSource.AGENT:
                    if batch_observations:
                        terminal_output = self._combine_observations(
                            batch_observations
                        )
                        user_text = self._format_terminal_output(
                            terminal_output, last_timed_out, last_keystrokes
                        )
                        messages.append(
                            Message(role='user', content=[TextContent(text=user_text)])
                        )
                        batch_observations = []
                        last_timed_out = False

                    messages.append(
                        Message(
                            role='assistant',
                            content=[TextContent(text=event.content)],
                        )
                    )

            elif isinstance(event, Terminus2CmdRunAction):
                if event.thought:
                    if batch_observations:
                        terminal_output = self._combine_observations(
                            batch_observations
                        )
                        user_text = self._format_terminal_output(
                            terminal_output, last_timed_out, last_keystrokes
                        )
                        messages.append(
                            Message(role='user', content=[TextContent(text=user_text)])
                        )
                        batch_observations = []
                        last_timed_out = False

                    messages.append(
                        Message(
                            role='assistant',
                            content=[TextContent(text=event.thought)],
                        )
                    )
                last_keystrokes = event.keystrokes

            elif isinstance(event, Terminus2CmdOutputObservation):
                if event is initial_terminal_event:
                    continue
                batch_observations.append(event.terminal_state)
                last_timed_out = event.timed_out

            elif isinstance(event, ErrorObservation):
                batch_observations.append(f'ERROR: {event.content}')

            elif isinstance(event, AgentThinkAction):
                pass

        if batch_observations:
            terminal_output = self._combine_observations(batch_observations)
            if self._pending_completion:
                confirmation = COMPLETION_CONFIRMATION.format(
                    terminal_state=terminal_output
                )
                messages.append(
                    Message(role='user', content=[TextContent(text=confirmation)])
                )
            else:
                user_text = self._format_terminal_output(
                    terminal_output, last_timed_out, last_keystrokes
                )
                messages.append(
                    Message(role='user', content=[TextContent(text=user_text)])
                )
        elif self._pending_completion:
            confirmation = COMPLETION_CONFIRMATION.format(terminal_state='')
            messages.append(
                Message(role='user', content=[TextContent(text=confirmation)])
            )

        return messages

    def _find_initial_user_message(self, events: list[Event]) -> str | None:
        """Find the initial user message (task instruction) from the event history."""
        for event in events:
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                return event.content
        return None

    def _find_initial_terminal_event(
        self, events: list[Event]
    ) -> Terminus2CmdOutputObservation | None:
        """Find the first Terminus2CmdOutputObservation in the event history.

        Returns the event object itself (not just the string) so that
        _build_messages can skip it in the loop via identity comparison,
        avoiding duplication with the first user message.
        """
        for event in events:
            if isinstance(event, Terminus2CmdOutputObservation):
                return event
        return None

    _NEW_OUTPUT_PREFIX = 'New Terminal Output:\n'
    _SCREEN_PREFIX = 'Current Terminal Screen:\n'

    @staticmethod
    def _combine_observations(observations: list[str]) -> str:
        """Combine multiple terminal observations into a single output.

        In the original Terminus-2, all commands in a batch execute in tmux and
        then a single get_incremental_output() captures the cumulative output.
        This method replicates that by stripping the per-observation prefix,
        joining the raw screen content, and re-adding a single prefix.
        """
        if not observations:
            return ''
        if len(observations) == 1:
            return observations[0]

        new_pfx = Terminus2Agent._NEW_OUTPUT_PREFIX
        scr_pfx = Terminus2Agent._SCREEN_PREFIX

        screens: list[str] = []
        last_prefix = new_pfx
        for obs in observations:
            if obs.startswith(new_pfx):
                screens.append(obs[len(new_pfx):])
                last_prefix = new_pfx
            elif obs.startswith(scr_pfx):
                screens.append(obs[len(scr_pfx):])
                last_prefix = scr_pfx
            else:
                screens.append(obs)

        return last_prefix + '\n'.join(screens)

    def _format_terminal_output(
        self, terminal_output: str, timed_out: bool, keystrokes: str
    ) -> str:
        """Format terminal output for the next user message."""
        if timed_out:
            return TIMEOUT_TEMPLATE.format(
                command=keystrokes,
                timeout_sec=60,
                terminal_state=self._limit_output_length(terminal_output),
            )
        return self._limit_output_length(terminal_output)

    def _call_llm_and_parse(
        self, messages: list[Message]
    ) -> tuple[list[ParsedCommand], bool, str]:
        """Call the LLM and parse the JSON response, with retry on parse errors.

        Returns (commands, is_task_complete, response_text) where response_text
        is the raw LLM output that must be stored on the first action's thought
        field so _build_messages can reconstruct the assistant turn later.
        """
        for attempt in range(MAX_LLM_RETRY):
            params: dict = {
                'messages': messages,
            }
            response = self.llm.completion(**params)

            response_text = response.choices[0].message.content or ''
            logger.debug(f'Terminus-2 LLM response (attempt {attempt + 1}): {response_text[:200]}...')

            messages.append(
                Message(role='assistant', content=[TextContent(text=response_text)])
            )

            result = self.parser.parse_response(response_text)

            if result.error:
                feedback = f'Previous response had parsing errors:\nERROR: {result.error}'
                if result.warning:
                    feedback += f'\nWARNINGS: {result.warning}'
                feedback += '\n\nPlease fix these issues and provide a proper JSON response.'
                logger.warning(f'Terminus-2 parse error (attempt {attempt + 1}): {result.error}')

                messages.append(
                    Message(role='user', content=[TextContent(text=feedback)])
                )
                continue

            if result.warning:
                logger.info(f'Terminus-2 parse warnings: {result.warning}')

            commands = [
                ParsedCommand(keystrokes=cmd.keystrokes, duration=min(cmd.duration, 60))
                for cmd in result.commands
            ]
            return commands, result.is_task_complete, response_text

        logger.error('Terminus-2: exhausted LLM retries due to parse errors')
        return [], False, ''

    @staticmethod
    def _limit_output_length(output: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
        """Limit output to specified byte length, keeping first and last portions."""
        if len(output.encode('utf-8')) <= max_bytes:
            return output

        portion_size = max_bytes // 2
        output_bytes = output.encode('utf-8')
        first_portion = output_bytes[:portion_size].decode('utf-8', errors='ignore')
        last_portion = output_bytes[-portion_size:].decode('utf-8', errors='ignore')
        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode('utf-8'))
            - len(last_portion.encode('utf-8'))
        )

        return (
            f'{first_portion}\n[... output limited to {max_bytes} bytes; '
            f'{omitted_bytes} interior bytes omitted ...]\n{last_portion}'
        )
