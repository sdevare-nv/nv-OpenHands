"""OpenCode-specific observation classes."""

from dataclasses import dataclass, field

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class QuestionObservation(Observation):
    """Observation for Question action execution.

    Attributes:
        answers: User answers to the questions
        questions: The questions that were asked
    """
    answers: list[dict] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)
    observation: str = ObservationType.QUESTION

    @property
    def message(self) -> str:
        count = len(self.questions)
        return f"User answered {count} question{'s' if count > 1 else ''}"


@dataclass
class ApplyPatchObservation(Observation):
    """Observation for ApplyPatch action execution.

    Attributes:
        files_changed: List of files that were modified
        diagnostics: LSP diagnostics after applying patch
        success: Whether the patch was successfully applied
    """
    files_changed: list[str] = field(default_factory=list)
    diagnostics: str = ""
    success: bool = True
    observation: str = ObservationType.APPLY_PATCH

    @property
    def message(self) -> str:
        count = len(self.files_changed)
        return f"Applied patch to {count} file{'s' if count > 1 else ''}"


@dataclass
class TodoReadObservation(Observation):
    """Observation for TodoRead action execution.

    Attributes:
        todos: The current task list
    """
    todos: list[dict] = field(default_factory=list)
    observation: str = ObservationType.TODO_READ

    @property
    def message(self) -> str:
        count = len(self.todos)
        pending = sum(1 for t in self.todos if t.get('status') == 'pending')
        in_progress = sum(1 for t in self.todos if t.get('status') == 'in_progress')
        completed = sum(1 for t in self.todos if t.get('status') == 'completed')
        return f"{count} todo{'s' if count > 1 else ''}: {pending} pending, {in_progress} in progress, {completed} completed"


@dataclass
class TodoWriteObservation(Observation):
    """Observation for TodoWrite action execution.

    Attributes:
        todos: The updated task list
        success: Whether the update was successful
    """
    todos: list[dict] = field(default_factory=list)
    success: bool = True
    observation: str = ObservationType.TODO_WRITE

    @property
    def message(self) -> str:
        count = len(self.todos)
        return f"Updated {count} todo{'s' if count > 1 else ''}"
