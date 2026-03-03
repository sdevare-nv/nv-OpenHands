import os
from dataclasses import dataclass, field
from itertools import islice

from jinja2 import BaseLoader, Environment, FileSystemLoader, Template, TemplateNotFound

from openhands.controller.state.state import State
from openhands.core.message import Message, TextContent
from openhands.events.observation.agent import MicroagentKnowledge


class OverrideLoader(BaseLoader):
    """A Jinja2 loader that allows specific templates to be loaded from custom paths.
    
    This loader checks if a template has a custom path override, and if so, loads from that path.
    Otherwise, it falls back to loading from the default directory.
    This allows {% include %} directives to work correctly while supporting custom template paths.
    """
    
    def __init__(self, default_dir: str, overrides: dict[str, str] | None = None):
        """Initialize the loader.
        
        Args:
            default_dir: Default directory to load templates from
            overrides: Dict mapping template names to absolute file paths
        """
        self.default_dir = default_dir
        self.overrides = overrides or {}
        self.default_loader = FileSystemLoader(default_dir)
        
        # Build a list of directories for included templates
        # This allows {% include %} to find overridden templates
        self.include_dirs = [default_dir]
        for path in self.overrides.values():
            if path and os.path.isfile(path):
                dir_path = os.path.dirname(path)
                if dir_path not in self.include_dirs:
                    self.include_dirs.append(dir_path)
        
        self.include_loader = FileSystemLoader(self.include_dirs)
    
    def get_source(self, environment, template):
        # Check if this template has an override
        if template in self.overrides:
            override_path = self.overrides[template]
            if override_path and os.path.isfile(override_path):
                with open(override_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                return source, override_path, lambda: True
        
        # Also check by filename in case the override uses just the filename
        template_basename = os.path.basename(template)
        if template_basename in self.overrides:
            override_path = self.overrides[template_basename]
            if override_path and os.path.isfile(override_path):
                with open(override_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                return source, override_path, lambda: True
        
        # Fall back to the include loader which searches all directories
        try:
            return self.include_loader.get_source(environment, template)
        except TemplateNotFound:
            raise TemplateNotFound(template)


@dataclass
class RuntimeInfo:
    date: str
    available_hosts: dict[str, int] = field(default_factory=dict)
    additional_agent_instructions: str = ''
    custom_secrets_descriptions: dict[str, str] = field(default_factory=dict)
    working_dir: str = ''


@dataclass
class RepositoryInfo:
    """Information about a GitHub repository that has been cloned."""

    repo_name: str | None = None
    repo_directory: str | None = None
    branch_name: str | None = None


@dataclass
class ConversationInstructions:
    """Optional instructions the agent must follow throughout the conversation while addressing the user's initial task

    Examples include

        1. Resolver instructions: you're responding to GitHub issue #1234, make sure to open a PR when you are done
        2. Slack instructions: make sure to check whether any of the context attached is relevant to the task <context_messages>
    """

    content: str = ''


class PromptManager:
    """Manages prompt templates and includes information from the user's workspace micro-agents and global micro-agents.

    This class is dedicated to loading and rendering prompts (system prompt, user prompt).

    Attributes:
        prompt_dir: Directory containing prompt templates.
    """

    def __init__(
        self,
        prompt_dir: str,
        system_prompt_filename: str = 'system_prompt.j2',
        template_overrides: dict[str, str] | None = None,
    ):
        """Initialize the PromptManager.
        
        Args:
            prompt_dir: Default directory containing prompt templates.
            system_prompt_filename: Name of the system prompt template file.
            template_overrides: Optional dict mapping template names to absolute file paths.
                              E.g., {'system_prompt.j2': '/path/to/custom/system_prompt.j2'}
        """
        if prompt_dir is None:
            raise ValueError('Prompt directory is not set')

        self.prompt_dir: str = prompt_dir
        self.template_overrides = template_overrides or {}
        
        # Use OverrideLoader if there are custom overrides, otherwise use standard loader
        if self.template_overrides:
            loader = OverrideLoader(prompt_dir, self.template_overrides)
        else:
            loader = FileSystemLoader(prompt_dir)
        
        self.env = Environment(loader=loader)
        self.system_template: Template = self._load_template(system_prompt_filename)
        self.user_template: Template = self._load_template('user_prompt.j2')
        self.additional_info_template: Template = self._load_template(
            'additional_info.j2'
        )
        self.microagent_info_template: Template = self._load_template(
            'microagent_info.j2'
        )

    def _load_template(self, template_name: str) -> Template:
        """Load a template from the prompt directory.

        Args:
            template_name: Full filename of the template to load, including the .j2 extension.

        Returns:
            The loaded Jinja2 template.

        Raises:
            FileNotFoundError: If the template file is not found.
        """
        try:
            return self.env.get_template(template_name)
        except Exception:
            template_path = os.path.join(self.prompt_dir, template_name)
            raise FileNotFoundError(f'Prompt file {template_path} not found')

    def get_system_message(self, **context) -> str:
        from openhands.agenthub.codeact_agent.tools.prompt import refine_prompt

        system_message = self.system_template.render(**context).strip()
        return refine_prompt(system_message)

    def get_example_user_message(self) -> str:
        """This is an initial user message that can be provided to the agent
        before *actual* user instructions are provided.

        It can be used to provide a demonstration of how the agent
        should behave in order to solve the user's task. And it may
        optionally contain some additional context about the user's task.
        These additional context will convert the current generic agent
        into a more specialized agent that is tailored to the user's task.
        """
        return self.user_template.render().strip()

    def build_workspace_context(
        self,
        repository_info: RepositoryInfo | None,
        runtime_info: RuntimeInfo | None,
        conversation_instructions: ConversationInstructions | None,
        repo_instructions: str = '',
    ) -> str:
        """Renders the additional info template with the stored repository/runtime info."""
        return self.additional_info_template.render(
            repository_info=repository_info,
            repository_instructions=repo_instructions,
            runtime_info=runtime_info,
            conversation_instructions=conversation_instructions,
        ).strip()

    def build_microagent_info(
        self,
        triggered_agents: list[MicroagentKnowledge],
    ) -> str:
        """Renders the microagent info template with the triggered agents.

        Args:
            triggered_agents: A list of MicroagentKnowledge objects containing information
                              about triggered microagents.
        """
        return self.microagent_info_template.render(
            triggered_agents=triggered_agents
        ).strip()

    def add_turns_left_reminder(self, messages: list[Message], state: State) -> None:
        latest_user_message = next(
            islice(
                (
                    m
                    for m in reversed(messages)
                    if m.role == 'user'
                    and any(isinstance(c, TextContent) for c in m.content)
                ),
                1,
            ),
            None,
        )
        if latest_user_message:
            reminder_text = f'\n\nENVIRONMENT REMINDER: You have {state.iteration_flag.max_value - state.iteration_flag.current_value} turns left to complete the task. When finished reply with <finish></finish>.'
            latest_user_message.content.append(TextContent(text=reminder_text))
