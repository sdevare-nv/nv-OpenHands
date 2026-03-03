from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import EDIT_TOOL_NAME

_EDIT_DESCRIPTION = """Performs string replacement editing on a file.

Usage:
- Replaces occurrences of old_string with new_string in the specified file
- The old_string must match exactly (including whitespace and indentation)
- By default, only replaces the first unique occurrence
- Use replace_all=true to replace all occurrences

CRITICAL REQUIREMENTS:
1. EXACT MATCHING: old_string must match EXACTLY, including all whitespace and indentation
2. UNIQUENESS: old_string should uniquely identify the location to edit
   - Include 3-5 lines of context before and after the change point
   - If old_string matches multiple locations, the edit will fail (unless using replace_all)
3. DIFFERENCE: old_string and new_string must be different

Parameters:
- file_path: The absolute path to the file to modify
- old_string: The exact text to replace (must match file content exactly)
- new_string: The text to replace it with (must be different from old_string)
- replace_all: Optional boolean to replace all occurrences (default: false)

Examples:
- Simple replacement:
  file_path="/workspace/main.py"
  old_string="def hello():\\n    print('hi')"
  new_string="def hello():\\n    print('hello world')"

- Replace all occurrences:
  file_path="/workspace/config.py"
  old_string="DEBUG = True"
  new_string="DEBUG = False"
  replace_all=true
"""

EditTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=EDIT_TOOL_NAME,
        description=_EDIT_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['file_path', 'old_string', 'new_string'],
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'The absolute path to the file to modify',
                },
                'old_string': {
                    'type': 'string',
                    'description': 'The exact text to replace (must match file content exactly)',
                },
                'new_string': {
                    'type': 'string',
                    'description': 'The text to replace it with (must be different from old_string)',
                },
                'replace_all': {
                    'type': 'boolean',
                    'description': 'Replace all occurrences (default: false)',
                },
            },
        },
    ),
)

