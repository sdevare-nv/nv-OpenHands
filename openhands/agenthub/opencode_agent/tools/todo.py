from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

TODO_READ_TOOL_NAME = 'todo_read'
TODO_WRITE_TOOL_NAME = 'todo_write'

_TODO_READ_DESCRIPTION = """Use this tool to read your current todo list.

Usage:
- Retrieve all current todos and their status
- Check what tasks are pending, in progress, or completed
- No parameters required

Returns:
- JSON array of todo items with their current status

Examples:
- Read todos: (no parameters needed)

Note: Use this before writing todos to avoid overwriting existing tasks."""

_TODO_WRITE_DESCRIPTION = """Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multistep tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. After completing a task - Mark it complete and add any new follow-up tasks
7. When you start working on a new task, mark the todo as in_progress

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully
   - cancelled: Task no longer needed

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing
   - Only have ONE task in_progress at any time
   - Complete current tasks before starting new ones

Parameters:
- todos: Array of todo objects, each containing:
  - id: Unique identifier for the todo (required for updates)
  - title: Brief description of the task
  - status: Task status - "pending", "in_progress", "completed", or "cancelled"
  - description: Optional detailed description

Examples:
- Create todos: todos=[{"id": "1", "title": "Implement feature X", "status": "pending"}, {"id": "2", "title": "Write tests", "status": "pending"}]
- Update status: todos=[{"id": "1", "title": "Implement feature X", "status": "completed"}, {"id": "2", "title": "Write tests", "status": "in_progress"}]

Note: When in doubt, use this tool. Being proactive with task management demonstrates attentiveness."""

TodoReadTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=TODO_READ_TOOL_NAME,
        description=_TODO_READ_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {},
        },
    ),
)

TodoWriteTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=TODO_WRITE_TOOL_NAME,
        description=_TODO_WRITE_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['todos'],
            'properties': {
                'todos': {
                    'type': 'array',
                    'description': 'The updated todo list',
                    'items': {
                        'type': 'object',
                        'required': ['id', 'title', 'status'],
                        'properties': {
                            'id': {
                                'type': 'string',
                                'description': 'Unique identifier for the todo',
                            },
                            'title': {
                                'type': 'string',
                                'description': 'Brief description of the task',
                            },
                            'status': {
                                'type': 'string',
                                'enum': ['pending', 'in_progress', 'completed', 'cancelled'],
                                'description': 'Task status',
                            },
                            'description': {
                                'type': 'string',
                                'description': 'Optional detailed description',
                            },
                        },
                    },
                },
            },
        },
    ),
)
