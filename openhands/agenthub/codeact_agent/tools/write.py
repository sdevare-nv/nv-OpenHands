from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import WRITE_TOOL_NAME

_WRITE_DESCRIPTION = """Writes content to a file, creating it if it doesn't exist or overwriting if it does.

Usage:
- The file_path parameter must be an absolute path
- Creates parent directories automatically if they don't exist
- Use this tool to create new files or completely replace existing file contents
- For partial edits to existing files, use the 'edit' tool instead

Parameters:
- file_path: The absolute path to the file to write
- content: The content to write to the file

Examples:
- Create new file: file_path="/workspace/src/new_file.py", content="print('hello')"
- Overwrite file: file_path="/workspace/config.json", content='{"key": "value"}'

Note: This will overwrite existing files. Make sure to read the file first if you need to preserve any content.
"""

WriteTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=WRITE_TOOL_NAME,
        description=_WRITE_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['file_path', 'content'],
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'The absolute path to the file to write',
                },
                'content': {
                    'type': 'string',
                    'description': 'The content to write to the file',
                },
            },
        },
    ),
)

