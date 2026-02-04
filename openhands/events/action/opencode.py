"""OpenCode-inspired action classes for enhanced file operations.

These actions provide OpenCode-compatible functionality with:
- Ripgrep integration (with fallback to standard tools)
- OpenCode-style output formatting
- LSP/linter diagnostics support
- Binary file detection
- .gitignore support
"""

from dataclasses import dataclass, field
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class OpenCodeReadAction(Action):
    """Reads a file with OpenCode-style formatting.

    Features:
    - 5-digit zero-padded line numbers with | separator (e.g., "00001| content")
    - Binary file detection (by extension and content analysis)
    - 50KB byte limit with truncation messages
    - File suggestions when file not found
    - 2000 character line length limit

    Attributes:
        path: The path to the file to read
        offset: Line number to start reading from (0-based). Default: 0
        limit: Number of lines to read. Default: 2000
    """

    path: str
    offset: int = 0
    limit: int = 2000
    thought: str = ""
    action: str = ActionType.OPENCODE_READ
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        if self.offset > 0:
            return f"Reading file: {self.path} (from line {self.offset + 1})"
        return f"Reading file: {self.path}"


@dataclass
class OpenCodeWriteAction(Action):
    """Writes a file with LSP/linter diagnostics after write.

    Features:
    - Creates parent directories if needed
    - Runs appropriate linter based on file extension:
      - Python: flake8 → pylint → py_compile
      - JavaScript/TypeScript: eslint
      - Go: go vet
      - Rust: cargo check
    - Outputs diagnostics in OpenCode format

    Attributes:
        path: The path to the file to write
        content: The content to write to the file
    """

    path: str
    content: str
    thought: str = ""
    action: str = ActionType.OPENCODE_WRITE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        return f"Writing file: {self.path}"


@dataclass
class GlobAction(Action):
    """Searches for files matching a glob pattern.

    Features:
    - Uses ripgrep (respects .gitignore) with fallback to find
    - Results sorted by modification time (newest first)
    - Limited to 100 results

    Attributes:
        pattern: The glob pattern to match files against (e.g., "*.py", "**/*.ts")
        path: Directory to search in. Defaults to current directory.
    """

    pattern: str
    path: str = "."
    thought: str = ""
    action: str = ActionType.GLOB
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        return f"Searching for files matching: {self.pattern}"


@dataclass
class GrepAction(Action):
    """Searches file contents using a pattern.

    Features:
    - Uses ripgrep (respects .gitignore) with fallback to grep
    - Shows line numbers in output
    - Limited to 100 results

    Attributes:
        pattern: The pattern to search for in file contents
        path: Directory to search in. Defaults to current directory.
        include: Optional file pattern to filter which files to search (e.g., "*.py")
    """

    pattern: str
    path: str = "."
    include: str = ""
    thought: str = ""
    action: str = ActionType.GREP
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        return f"Searching for pattern: {self.pattern}"


@dataclass
class ListDirAction(Action):
    """Lists files and directories in a tree structure.

    Features:
    - Uses ripgrep (respects .gitignore) with fallback to tree/find
    - Builds tree structure output
    - Default ignore patterns for common directories
    - Limited to 100 files

    Attributes:
        path: The directory to list. Defaults to current directory.
        ignore: Additional glob patterns to ignore (beyond defaults)

    Default ignore patterns:
        node_modules, __pycache__, .git, dist, build, target,
        vendor, .venv, venv, .cache
    """

    path: str = "."
    ignore: list[str] = field(default_factory=list)
    thought: str = ""
    action: str = ActionType.LIST_DIR
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    # Default ignore patterns matching OpenCode behavior
    DEFAULT_IGNORES: ClassVar[list[str]] = [
        "node_modules",
        "__pycache__",
        ".git",
        "dist",
        "build",
        "target",
        "vendor",
        ".venv",
        "venv",
        ".cache",
    ]

    @property
    def all_ignores(self) -> list[str]:
        """Returns combined default and custom ignore patterns."""
        return self.DEFAULT_IGNORES + self.ignore

    @property
    def message(self) -> str:
        return f"Listing directory: {self.path or 'current directory'}"
