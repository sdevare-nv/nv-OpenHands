from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import CODEX_UPDATE_PLAN_TOOL_NAME

_UPDATE_PLAN_DESCRIPTION = """Updates the task plan.
Provide an optional explanation and a list of plan items, each with a step and status.
At most one step can be in_progress at a time.
"""

UpdatePlanTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CODEX_UPDATE_PLAN_TOOL_NAME,
        description=_UPDATE_PLAN_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['plan'],
            'properties': {
                'explanation': {
                    'type': 'string',
                    'description': 'Optional explanation for the plan update.',
                },
                'plan': {
                    'type': 'array',
                    'description': 'The list of steps',
                    'items': {
                        'type': 'object',
                        'required': ['step', 'status'],
                        'properties': {
                            'step': {
                                'type': 'string',
                            },
                            'status': {
                                'type': 'string',
                                'description': 'One of: pending, in_progress, completed',
                            },
                        },
                        'additionalProperties': False,
                    },
                },
            },
            'additionalProperties': False,
        },
    ),
)
