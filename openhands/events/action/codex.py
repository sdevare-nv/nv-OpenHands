"""Codex-style action classes for the Codex agent.

These actions implement Codex tool interfaces with matching parameters,
descriptions, and response formats.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class CodexReadFileAction(Action):
    """Reads a local file with 1-indexed line numbers.

    Supports slice mode (simple line ranges) and indentation-aware block mode.
    Lines are formatted as L{number}: {content}.

    Attributes:
        file_path: Absolute path to the file to read.
        offset: Line number to start reading from (1-indexed, must be >= 1). Default: 1.
        limit: Maximum number of lines to return. Default: 2000.
        mode: Read mode - "slice" for simple ranges (default) or "indentation" for
              block-aware reading.
        indentation: Optional indentation mode parameters (anchor_line, max_levels,
                     include_siblings, include_header, max_lines).
    """

    file_path: str = ""
    offset: int = 1
    limit: int = 2000
    mode: str = "slice"
    indentation: dict = field(default_factory=dict)
    thought: str = ""
    action: str = ActionType.CODEX_READ_FILE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        if self.offset > 1:
            return f"Reading file: {self.file_path} (from line {self.offset})"
        return f"Reading file: {self.file_path}"


@dataclass
class CodexListDirAction(Action):
    """Lists entries in a local directory with 1-indexed entry numbers and type labels.

    Attributes:
        dir_path: Absolute path to the directory to list.
        offset: Entry number to start listing from (1-indexed, must be >= 1). Default: 1.
        limit: Maximum number of entries to return. Default: 25.
        depth: Maximum directory depth to traverse (must be >= 1). Default: 2.
    """

    dir_path: str = ""
    offset: int = 1
    limit: int = 25
    depth: int = 2
    thought: str = ""
    action: str = ActionType.CODEX_LIST_DIR
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        return f"Listing directory: {self.dir_path}"


@dataclass
class CodexGrepFilesAction(Action):
    """Finds files whose contents match a pattern and lists them by modification time.

    Uses ripgrep when available with a fallback to grep. Returns file paths only
    (not matching lines), sorted by modification time (newest first).

    Attributes:
        pattern: Regular expression pattern to search for.
        include: Optional glob limiting which files are searched (e.g., "*.rs").
        path: Directory or file path to search. Defaults to working directory.
        limit: Maximum number of file paths to return. Default: 100.
    """

    pattern: str = ""
    include: str = ""
    path: str = ""
    limit: int = 100
    thought: str = ""
    action: str = ActionType.CODEX_GREP_FILES
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        return f"Searching for files matching pattern: {self.pattern}"


@dataclass
class CodexApplyPatchAction(Action):
    """Applies a Codex-format freeform patch to files.

    The patch uses Codex's patch format with *** Begin Patch / *** End Patch
    delimiters, supporting Add, Delete, Update, and Move file operations.

    Attributes:
        patch: The raw freeform patch text.
    """

    patch: str = ""
    thought: str = ""
    action: str = ActionType.CODEX_APPLY_PATCH
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        return "Applying patch"


@dataclass
class CodexUpdatePlanAction(Action):
    """Updates the task plan.

    Provides a list of plan items each with a step description and status.
    At most one step can be in_progress at a time.

    Attributes:
        plan: List of plan items, each with 'step' (str) and 'status' (str).
              Status must be one of: "pending", "in_progress", "completed".
        explanation: Optional explanation for the plan update.
    """

    plan: list = field(default_factory=list)
    explanation: str = ""
    thought: str = ""
    action: str = ActionType.CODEX_UPDATE_PLAN
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        count = len(self.plan)
        return f"Updating plan with {count} step{'s' if count != 1 else ''}"
