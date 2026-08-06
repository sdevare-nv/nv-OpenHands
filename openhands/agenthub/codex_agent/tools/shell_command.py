from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import CODEX_SHELL_COMMAND_TOOL_NAME

_SHELL_COMMAND_DESCRIPTION = f"""Runs a shell command and returns its output.
- Always set the `workdir` param when using the {CODEX_SHELL_COMMAND_TOOL_NAME} function. Do not use `cd` unless absolutely necessary.
- When piping a test, lint, build, or typecheck command through `tail`, `head`, `grep`, or another filter, enable `set -o pipefail` so a failure in the command is not hidden by a successful filter."""

ShellCommandTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CODEX_SHELL_COMMAND_TOOL_NAME,
        description=_SHELL_COMMAND_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['command'],
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The shell script to execute in the user\'s default shell',
                },
                'workdir': {
                    'type': 'string',
                    'description': 'The working directory to execute the command in',
                },
                'login': {
                    'type': 'boolean',
                    'description': 'Whether to run the command in a fresh shell with login-shell semantics. When omitted, reuse the existing shell environment.',
                },
                'timeout_ms': {
                    'type': 'number',
                    'description': 'The timeout for the command in milliseconds',
                },
                'is_input': {
                    'type': 'string',
                    'description': 'If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.',
                    'enum': ['true', 'false'],
                },
            },
            'additionalProperties': False,
        },
    ),
)
