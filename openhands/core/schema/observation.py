from enum import Enum


class ObservationType(str, Enum):
    READ = 'read'
    """The content of a file
    """

    WRITE = 'write'

    EDIT = 'edit'

    BROWSE = 'browse'
    """The HTML content of a URL
    """

    RUN = 'run'
    """The output of a command
    """

    RUN_IPYTHON = 'run_ipython'
    """Runs a IPython cell.
    """

    CHAT = 'chat'
    """A message from the user
    """

    DELEGATE = 'delegate'
    """The result of a task delegated to another agent
    """

    MESSAGE = 'message'

    ERROR = 'error'

    SUCCESS = 'success'

    NULL = 'null'

    THINK = 'think'

    AGENT_STATE_CHANGED = 'agent_state_changed'

    USER_REJECTED = 'user_rejected'

    CONDENSE = 'condense'
    """Result of a condensation operation."""

    RECALL = 'recall'
    """Result of a recall operation. This can be the workspace context, a microagent, or other types of information."""

    MCP = 'mcp'
    """Result of a MCP Server operation"""

    DOWNLOAD = 'download'
    """Result of downloading/opening a file via the browser"""

    TASK_TRACKING = 'task_tracking'
    """Result of a task tracking operation"""

    LOOP_DETECTION = 'loop_detection'
    """Results of a dead-loop detection"""

    VALIDATION_FAILURE = 'validation_failure'
    """Result of a validation failure for a function call"""

    # OpenCode-style observations
    GLOB = 'glob'
    """Result of a glob file search operation."""

    GREP = 'grep'
    """Result of a grep content search operation."""

    LIST_DIR = 'list_dir'
    """Result of a directory listing operation."""

    OPENCODE_READ = 'opencode_read'
    """Result of an OpenCode-style file read operation."""

    OPENCODE_WRITE = 'opencode_write'
    """Result of an OpenCode-style file write operation with diagnostics."""

    QUESTION = 'question'
    """Result of asking the user a question."""

    APPLY_PATCH = 'apply_patch'
    """Result of applying a patch to files."""

    TODO_READ = 'todo_read'
    """Result of reading the current todo list."""

    TODO_WRITE = 'todo_write'
    """Result of updating the todo list."""

