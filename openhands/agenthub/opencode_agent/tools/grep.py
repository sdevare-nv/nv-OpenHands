from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import GREP_TOOL_NAME

_GREP_DESCRIPTION = """Fast content search tool that works with any codebase size.

Usage:
- Searches file contents using regular expressions
- Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+", "import.*from")
- Filter files by pattern with the include parameter (e.g., "*.js", "*.{ts,tsx}")
- Returns file paths and line numbers with matching content, sorted by modification time
- Results are limited to 100 matches; use more specific patterns for large codebases

Parameters:
- pattern: The regex pattern to search for in file contents
- path: Optional directory to search in. Defaults to workspace root.
- include: Optional file pattern to filter which files to search (e.g., "*.py", "*.{js,ts}")

Examples:
- Find TODO comments: pattern="TODO|FIXME"
- Find function definitions: pattern="def\\s+\\w+\\("
- Find imports in Python files: pattern="^import|^from.*import", include="*.py"
"""

GrepTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=GREP_TOOL_NAME,
        description=_GREP_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['pattern'],
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'The regex pattern to search for in file contents',
                },
                'path': {
                    'type': 'string',
                    'description': 'Optional directory to search in. Defaults to workspace root.',
                },
                'include': {
                    'type': 'string',
                    'description': 'Optional file pattern to filter files (e.g., "*.py", "*.{js,ts}")',
                },
            },
        },
    ),
)

