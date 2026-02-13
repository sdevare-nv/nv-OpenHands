from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

QUESTION_TOOL_NAME = 'question'

_QUESTION_DESCRIPTION = """Use this tool when you need to ask the user questions during execution. This allows you to:
1. Gather user preferences or requirements
2. Clarify ambiguous instructions
3. Get decisions on implementation choices as you work
4. Offer choices to the user about what direction to take

Usage notes:
- When `custom` is enabled (default), a "Type your own answer" option is added automatically; don't include "Other" or catch-all options
- Answers are returned as arrays of labels; set `multiple: true` to allow selecting more than one
- If you recommend a specific option, make that the first option in the list and add "(Recommended)" at the end of the label

Parameters:
- questions: Array of question objects, each containing:
  - question: The question to ask the user
  - header: Short label for the question (max 12 chars, e.g., "Auth method", "Library")
  - options: Array of option objects with:
    - label: Display text for the option (concise, 1-5 words)
    - description: Explanation of what this option means
  - multiple: Whether to allow multiple selections (default: false)
  - custom: Whether to allow custom text input (default: true)

Examples:
- Single choice: questions=[{"question": "Which library should we use?", "header": "Library", "options": [{"label": "React", "description": "..."}, {"label": "Vue", "description": "..."}], "multiple": false}]
- Multiple choice: questions=[{"question": "Which features to enable?", "header": "Features", "options": [...], "multiple": true}]

Note: This tool is useful for clarifying requirements before implementing changes."""

QuestionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=QUESTION_TOOL_NAME,
        description=_QUESTION_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['questions'],
            'properties': {
                'questions': {
                    'type': 'array',
                    'description': 'Questions to ask the user',
                    'items': {
                        'type': 'object',
                        'required': ['question', 'header', 'options'],
                        'properties': {
                            'question': {
                                'type': 'string',
                                'description': 'The question to ask',
                            },
                            'header': {
                                'type': 'string',
                                'description': 'Short label for the question (max 12 chars)',
                            },
                            'options': {
                                'type': 'array',
                                'description': 'Available answer options',
                                'items': {
                                    'type': 'object',
                                    'required': ['label', 'description'],
                                    'properties': {
                                        'label': {
                                            'type': 'string',
                                            'description': 'Display text for this option',
                                        },
                                        'description': {
                                            'type': 'string',
                                            'description': 'Explanation of this option',
                                        },
                                    },
                                },
                            },
                            'multiple': {
                                'type': 'boolean',
                                'description': 'Allow multiple selections (default: false)',
                            },
                            'custom': {
                                'type': 'boolean',
                                'description': 'Allow custom text input (default: true)',
                            },
                        },
                    },
                },
            },
        },
    ),
)
