from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import GLOB_TOOL_NAME

_GLOB_DESCRIPTION = """Fast file pattern matching tool that works with any codebase size.

Usage:
- Supports glob patterns like "**/*.js", "*.py", "src/**/*.ts"
- Returns matching file paths sorted by modification time (most recent first)
- Use this tool when you need to find files by name patterns
- Results are limited to 100 files; use more specific patterns for large codebases

Parameters:
- pattern: The glob pattern to match files against (e.g., "**/*.py", "*.ts")
- path: Optional directory to search in. Defaults to the workspace root.

Examples:
- Find all Python files: pattern="**/*.py"
- Find TypeScript files in src: pattern="*.ts", path="/workspace/src"
- Find test files: pattern="**/*_test.py" or pattern="**/test_*.py"
"""

GlobTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=GLOB_TOOL_NAME,
        description=_GLOB_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['pattern'],
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'The glob pattern to match files against (e.g., "**/*.py", "*.ts", "src/**/*.js")',
                },
                'path': {
                    'type': 'string',
                    'description': 'Optional directory to search in. Defaults to workspace root if not specified.',
                },
            },
        },
    ),
)

