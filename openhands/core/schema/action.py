from enum import Enum


class ActionType(str, Enum):
    MESSAGE = 'message'
    """Represents a message.
    """

    SYSTEM = 'system'
    """Represents a system message.
    """

    START = 'start'
    """Starts a new development task OR send chat from the user. Only sent by the client.
    """

    READ = 'read'
    """Reads the content of a file.
    """

    WRITE = 'write'
    """Writes the content to a file.
    """

    EDIT = 'edit'
    """Edits a file by providing a draft.
    """

    RUN = 'run'
    """Runs a command.
    """

    RUN_IPYTHON = 'run_ipython'
    """Runs a IPython cell.
    """

    BROWSE = 'browse'
    """Opens a web page.
    """

    BROWSE_INTERACTIVE = 'browse_interactive'
    """Interact with the browser instance.
    """

    MCP = 'call_tool_mcp'
    """Interact with the MCP server.
    """

    DELEGATE = 'delegate'
    """Delegates a task to another agent.
    """

    THINK = 'think'
    """Logs a thought.
    """

    FINISH = 'finish'
    """If you're absolutely certain that you've completed your task and have tested your work,
    use the finish action to stop working.
    """

    REJECT = 'reject'
    """If you're absolutely certain that you cannot complete the task with given requirements,
    use the reject action to stop working.
    """

    NULL = 'null'

    PAUSE = 'pause'
    """Pauses the task.
    """

    RESUME = 'resume'
    """Resumes the task.
    """

    STOP = 'stop'
    """Stops the task. Must send a start action to restart a new task.
    """

    CHANGE_AGENT_STATE = 'change_agent_state'

    PUSH = 'push'
    """Push a branch to github."""

    SEND_PR = 'send_pr'
    """Send a PR to github."""

    RECALL = 'recall'
    """Retrieves content from a user workspace, microagent, or other source."""

    CONDENSATION = 'condensation'
    """Condenses a list of events into a summary."""

    CONDENSATION_REQUEST = 'condensation_request'
    """Request for condensation of a list of events."""

    TASK_TRACKING = 'task_tracking'
    """Views or updates the task list for task management."""

    LOOP_RECOVERY = 'loop_recovery'
    """Recover dead loop."""

    VALIDATION_FAILURE = 'validation_failure'
    """Represents a validation failure for a function call."""

    # OpenCode-style actions
    GLOB = 'glob'
    """Searches for files matching a glob pattern."""

    GREP = 'grep'
    """Searches file contents using a regex pattern."""

    LIST_DIR = 'list_dir'
    """Lists files and directories in a given path."""

    OPENCODE_READ = 'opencode_read'
    """Reads a file with OpenCode-style formatting (5-digit line numbers, binary detection, etc.)."""

    OPENCODE_WRITE = 'opencode_write'
    """Writes a file with LSP diagnostics after write."""

    QUESTION = 'question'
    """Asks the user structured questions."""

    APPLY_PATCH = 'apply_patch'
    """Applies a unified diff patch to files."""

    TODO_READ = 'todo_read'
    """Reads the current task/todo list."""

    TODO_WRITE = 'todo_write'
    """Creates or updates tasks in the todo list."""

    # Codex-style actions
    CODEX_READ_FILE = 'codex_read_file'
    """Reads a file with 1-indexed line numbers, supporting slice and indentation-aware block modes."""

    CODEX_LIST_DIR = 'codex_list_dir'
    """Lists entries in a directory with 1-indexed entry numbers and type labels."""

    CODEX_GREP_FILES = 'codex_grep_files'
    """Finds files whose contents match a pattern, listed by modification time."""

    CODEX_APPLY_PATCH = 'codex_apply_patch'
    """Applies a Codex-format freeform patch to files."""

    CODEX_UPDATE_PLAN = 'codex_update_plan'
    """Updates the task plan with steps and statuses."""

    # Terminus-2-style actions
    TERMINUS_2_CMD_RUN = 'terminus_2_cmd_run'
    """Sends raw keystrokes to a terminal session and captures the resulting screen state."""
