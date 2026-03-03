from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import CODEX_READ_FILE_TOOL_NAME

_READ_FILE_DESCRIPTION = """Reads a local file with 1-indexed line numbers, supporting slice and indentation-aware block modes."""

ReadFileTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CODEX_READ_FILE_TOOL_NAME,
        description=_READ_FILE_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['file_path'],
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Absolute path to the file',
                },
                'offset': {
                    'type': 'number',
                    'description': 'The line number to start reading from. Must be 1 or greater.',
                },
                'limit': {
                    'type': 'number',
                    'description': 'The maximum number of lines to return.',
                },
                'mode': {
                    'type': 'string',
                    'description': 'Optional mode selector: "slice" for simple ranges (default) or "indentation" to expand around an anchor line.',
                },
                'indentation': {
                    'type': 'object',
                    'properties': {
                        'anchor_line': {
                            'type': 'number',
                            'description': 'Anchor line to center the indentation lookup on (defaults to offset).',
                        },
                        'max_levels': {
                            'type': 'number',
                            'description': 'How many parent indentation levels (smaller indents) to include.',
                        },
                        'include_siblings': {
                            'type': 'boolean',
                            'description': 'When true, include additional blocks that share the anchor indentation.',
                        },
                        'include_header': {
                            'type': 'boolean',
                            'description': 'Include doc comments or attributes directly above the selected block.',
                        },
                        'max_lines': {
                            'type': 'number',
                            'description': 'Hard cap on the number of lines returned when using indentation mode.',
                        },
                    },
                    'additionalProperties': False,
                },
            },
            'additionalProperties': False,
        },
    ),
)
