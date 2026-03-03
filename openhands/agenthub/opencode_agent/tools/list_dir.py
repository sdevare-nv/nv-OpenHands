from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import LIST_DIR_TOOL_NAME

_LIST_DIR_DESCRIPTION = """Lists files and directories in a given path.

Usage:
- Lists files and directories with a tree-like structure
- Automatically ignores common non-essential directories (node_modules, .git, __pycache__, etc.)
- Results are limited to 100 entries; use more specific paths for large directories
- Prefer the Glob and Grep tools if you know which files to search for

Parameters:
- path: The absolute path to the directory to list. Defaults to workspace root if not specified.
- ignore: Optional list of additional glob patterns to ignore (e.g., ["*.log", "temp/"])

Examples:
- List workspace root: (no parameters needed)
- List specific directory: path="/workspace/src"
- List with custom ignores: path="/workspace", ignore=["*.log", "build/"]
"""

ListDirTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=LIST_DIR_TOOL_NAME,
        description=_LIST_DIR_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The absolute path to the directory to list. Defaults to workspace root.',
                },
                'ignore': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'Optional list of glob patterns to ignore (e.g., ["*.log", "temp/"])',
                },
            },
        },
    ),
)

