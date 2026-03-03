from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import CODEX_LIST_DIR_TOOL_NAME

_LIST_DIR_DESCRIPTION = """Lists entries in a local directory with 1-indexed entry numbers and simple type labels."""

ListDirTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CODEX_LIST_DIR_TOOL_NAME,
        description=_LIST_DIR_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['dir_path'],
            'properties': {
                'dir_path': {
                    'type': 'string',
                    'description': 'Absolute path to the directory to list.',
                },
                'offset': {
                    'type': 'number',
                    'description': 'The entry number to start listing from. Must be 1 or greater.',
                },
                'limit': {
                    'type': 'number',
                    'description': 'The maximum number of entries to return.',
                },
                'depth': {
                    'type': 'number',
                    'description': 'The maximum directory depth to traverse. Must be 1 or greater.',
                },
            },
            'additionalProperties': False,
        },
    ),
)
