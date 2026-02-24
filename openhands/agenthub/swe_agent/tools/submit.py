from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

SWE_AGENT_SUBMIT_TOOL_NAME = 'submit'

SubmitTool: ChatCompletionToolParam = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=SWE_AGENT_SUBMIT_TOOL_NAME,
        description='submits the current file',
        parameters={
            'type': 'object',
            'properties': {},
            'required': [],
        },
    ),
)
