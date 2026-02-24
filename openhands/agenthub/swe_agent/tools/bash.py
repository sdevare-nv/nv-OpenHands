from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

SWE_AGENT_BASH_TOOL_NAME = 'bash'

BashTool: ChatCompletionToolParam = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=SWE_AGENT_BASH_TOOL_NAME,
        description='runs the given command directly in bash',
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute',
                },
            },
            'required': ['command'],
        },
    ),
)
