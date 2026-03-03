import os
from collections import deque
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.codex_agent.function_calling as codex_function_calling
from openhands.agenthub.codex_agent.tools.apply_patch import ApplyPatchTool
from openhands.agenthub.codex_agent.tools.finish import FinishTool
from openhands.agenthub.codex_agent.tools.grep_files import GrepFilesTool
from openhands.agenthub.codex_agent.tools.list_dir import ListDirTool
from openhands.agenthub.codex_agent.tools.read_file import ReadFileTool
from openhands.agenthub.codex_agent.tools.shell_command import ShellCommandTool
from openhands.agenthub.codex_agent.tools.update_plan import UpdatePlanTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.event import Event
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager


class CodexAgent(Agent):
    VERSION = "1.0"
    """
    The Codex Agent implements Codex-style tools for code editing and exploration.

    This agent provides tools that match the Codex tool interface including:
    - shell_command: Execute shell commands
    - read_file: Read files with 1-indexed line numbers and indentation-aware mode
    - list_dir: List directory entries with type labels
    - grep_files: Find files by content pattern, sorted by modification time
    - apply_patch: Apply Codex-format freeform patches
    - update_plan: Track task progress with plan items

    The agent uses function calling to invoke tools and maintains conversation
    history for context-aware assistance.
    """

    sandbox_plugins: list[PluginRequirement] = [
        AgentSkillsRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the CodexAgent class.

        Parameters:
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(config, llm_registry)
        self.pending_actions: deque["Action"] = deque()
        self.reset()
        self.tools = self._get_tools()

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        logger.debug(f"Using condenser: {type(self.condenser)}")

        # Override with router if needed
        self.llm = self.llm_registry.get_router(self.config)

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            # Use custom prompt directory if configured, otherwise use default
            prompt_dir = (
                self.config.custom_prompt_dir
                if self.config.custom_prompt_dir
                else os.path.join(os.path.dirname(__file__), "prompts")
            )

            # Build template overrides from custom paths
            template_overrides = {}
            if self.config.system_prompt_path:
                template_overrides["system_prompt.j2"] = self.config.system_prompt_path
            if self.config.system_prompt_long_horizon_path:
                template_overrides["system_prompt_long_horizon.j2"] = (
                    self.config.system_prompt_long_horizon_path
                )

            self._prompt_manager = PromptManager(
                prompt_dir=prompt_dir,
                system_prompt_filename=self.config.resolved_system_prompt_filename,
                template_overrides=template_overrides if template_overrides else None,
            )

        return self._prompt_manager

    def _get_tools(self) -> list["ChatCompletionToolParam"]:
        """Get the list of tools available to the Codex agent."""
        tools = []

        # Core Codex tools
        tools.append(ReadFileTool)
        tools.append(ListDirTool)
        tools.append(GrepFilesTool)
        tools.append(ApplyPatchTool)
        tools.append(UpdatePlanTool)

        # Command execution
        if self.config.enable_cmd:
            tools.append(ShellCommandTool)

        # Task completion
        if self.config.enable_finish:
            tools.append(FinishTool)

        return tools

    def reset(self) -> None:
        """Resets the Codex Agent's internal state."""
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> "Action":
        """Performs one step using the Codex Agent.

        This includes gathering info on previous steps and prompting the model
        to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - Various Action types based on tool calls
        """
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == "/exit":
            return AgentFinishAction()

        # Condense the events from the state
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f"Processing {len(condensed_history)} events from a total of {len(state.history)} events"
        )

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        params: dict = {
            "messages": messages,
        }
        params["tools"] = check_tools(self.tools, self.llm.config)
        params["extra_body"] = {
            "metadata": state.to_llm_metadata(
                model_name=self.llm.config.model, agent_name=self.name
            )
        }
        response = self.llm.completion(**params)
        logger.debug(f"Response from LLM: {response}")
        actions = self.response_to_actions(response)
        logger.debug(f"Actions after response_to_actions: {actions}")
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        """Finds the initial user message action from the full history."""
        initial_user_message: MessageAction | None = None
        for event in history:
            if isinstance(event, MessageAction) and event.source == "user":
                initial_user_message = event
                break

        if initial_user_message is None:
            logger.error(
                f"CRITICAL: Could not find the initial user MessageAction in the full {len(history)} events history."
            )
            raise ValueError(
                "Initial user message not found in history. Please report this issue."
            )
        return initial_user_message

    def _get_messages(
        self, events: list[Event], initial_user_message: MessageAction
    ) -> list[Message]:
        """Constructs the message history for the LLM conversation."""
        if not self.prompt_manager:
            raise Exception("Prompt Manager not instantiated.")

        # Use ConversationMemory to process events
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user_message,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def response_to_actions(self, response: "ModelResponse") -> list["Action"]:
        return codex_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )
