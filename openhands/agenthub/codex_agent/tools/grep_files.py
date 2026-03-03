from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import CODEX_GREP_FILES_TOOL_NAME

_GREP_FILES_DESCRIPTION = """Finds files whose contents match the pattern and lists them by modification time."""

GrepFilesTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CODEX_GREP_FILES_TOOL_NAME,
        description=_GREP_FILES_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['pattern'],
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'Regular expression pattern to search for.',
                },
                'include': {
                    'type': 'string',
                    'description': 'Optional glob that limits which files are searched (e.g. "*.rs" or "*.{ts,tsx}").',
                },
                'path': {
                    'type': 'string',
                    'description': 'Directory or file path to search. Defaults to the session\'s working directory.',
                },
                'limit': {
                    'type': 'number',
                    'description': 'Maximum number of file paths to return (defaults to 100).',
                },
            },
            'additionalProperties': False,
        },
    ),
)
