from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import READ_TOOL_NAME

_READ_DESCRIPTION = """Reads a file from the local filesystem.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files)
- Any lines longer than 2000 characters will be truncated
- Results are returned with line numbers starting at 1
- You can read multiple files in a single response by calling this tool multiple times
- If you read a file that exists but has empty contents, a warning will be shown
- This tool can also read image files (jpeg, png, gif, webp)

Parameters:
- file_path: The absolute path to the file to read
- offset: Optional line number to start reading from (0-based, default: 0)
- limit: Optional number of lines to read (default: 2000)

Examples:
- Read entire file: file_path="/workspace/src/main.py"
- Read from line 100: file_path="/workspace/src/main.py", offset=100
- Read 50 lines from line 200: file_path="/workspace/src/main.py", offset=200, limit=50
"""

ReadTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=READ_TOOL_NAME,
        description=_READ_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['file_path'],
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'The absolute path to the file to read',
                },
                'offset': {
                    'type': 'integer',
                    'description': 'Line number to start reading from (0-based, default: 0)',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Number of lines to read (default: 2000)',
                },
            },
        },
    ),
)

