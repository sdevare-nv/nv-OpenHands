# OpenCode Agent

The OpenCode Agent is inspired by Claude Code's tool-based approach to software development assistance.

## Overview

This agent provides a comprehensive set of tools for:
- **File Operations**: read, write, edit, glob, grep, list_dir
- **Web Access**: webfetch, websearch
- **User Interaction**: question tool for clarifying requirements
- **Task Management**: todo_read, todo_write for tracking progress
- **Code Execution**: bash commands
- **Structured Edits**: apply_patch for multi-file atomic changes

## Key Features

### 1. File Operations
- `read`: Read file contents with line numbers
- `write`: Create or overwrite files
- `edit`: Make precise string replacements
- `glob`: Pattern-based file search
- `grep`: Content search across files
- `list_dir`: Directory listing

### 2. Web Access
- `webfetch`: Fetch web content in text/markdown/html format
- `websearch`: Search the web for current information

### 3. User Interaction
- `question`: Ask users multiple-choice or custom questions
  - Supports single/multiple selection
  - Helps clarify requirements before implementation

### 4. Task Management
- `todo_read`: Read current task list
- `todo_write`: Update task list with new tasks or status changes
  - Tracks: pending, in_progress, completed, cancelled
  - Helps organize multi-step work

### 5. Structured Editing
- `apply_patch`: Apply unified diff patches
  - Supports add, update, delete, move operations
  - Atomic multi-file changes
  - LSP diagnostics after changes

## Tool Descriptions

### Core File Tools

#### read
Reads files with line numbers. Supports:
- Absolute paths
- Offset and limit for large files
- Image files (jpeg, png, gif, webp)

#### write
Creates or overwrites files. Automatically creates parent directories.

#### edit
Makes exact string replacements in files. Requires reading the file first.

#### glob
Finds files matching glob patterns (e.g., `**/*.py`, `src/**/*.ts`).

#### grep
Searches file contents with regex support. Output modes:
- `content`: Show matching lines with context
- `files_with_matches`: Just file paths
- `count`: Match counts per file

#### list_dir
Lists directory contents with optional recursion.

### Web Tools

#### webfetch
Fetches web content with format conversion:
- `text`: Plain text extraction
- `markdown`: HTML to markdown conversion
- `html`: Raw HTML

#### websearch
Searches the web with customizable options:
- `type`: auto/fast/deep search
- `livecrawl`: fallback/preferred for freshness
- Returns content optimized for LLMs

### Interaction Tools

#### question
Asks users structured questions:
```python
{
  "questions": [{
    "question": "Which library should we use?",
    "header": "Library",
    "options": [
      {"label": "React", "description": "Popular UI library"},
      {"label": "Vue", "description": "Progressive framework"}
    ],
    "multiple": false,
    "custom": true
  }]
}
```

### Task Management Tools

#### todo_read
Returns current task list as JSON array.

#### todo_write
Updates task list:
```python
{
  "todos": [
    {"id": "1", "title": "Implement feature", "status": "completed"},
    {"id": "2", "title": "Write tests", "status": "in_progress"}
  ]
}
```

### Patch Tool

#### apply_patch
Applies unified diff patches:
```diff
*** Begin Patch
--- path/to/file.py
+++ path/to/file.py
@@ -10,7 +10,7 @@
 unchanged
-old line
+new line
 unchanged
*** End Patch
```

Supports:
- Add files: `+++ new/file.py`
- Delete files: `+++ /dev/null`
- Move files: Different `---` and `+++` paths
- Update files: Standard diffs

## Usage

### Quick Start

To use the OpenCode agent in code:

```python
from openhands.core.config import AgentConfig
from openhands.llm.llm_registry import LLMRegistry

config = AgentConfig(agent_name='OpenCodeAgent')
llm_registry = LLMRegistry()
agent = OpenCodeAgent(config, llm_registry)
```

Or via configuration:
```yaml
agent:
  name: OpenCodeAgent
```

### Evaluation and Benchmarking

To run SWE-bench or other evaluations with OpenCodeAgent:

```bash
# Using run_infer.sh script
export AGENT="OpenCodeAgent"
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh llama3_1 test OpenCodeAgent

# Or call run_infer.py directly
poetry run python evaluation/benchmarks/swe_bench/run_infer.py \
  --agent-cls OpenCodeAgent \
  --llm-config llama3_1 \
  --max-iterations 30 \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test
```

**For detailed instructions on:**
- Dynamic agent selection (OpenCodeAgent vs CodeActAgent)
- Custom system prompt overrides
- Configuration options
- Troubleshooting

**See the comprehensive [Usage Guide](./USAGE_GUIDE.md)**

## Design Principles

1. **Clear Tool Boundaries**: Each tool has a specific purpose
2. **User Interaction**: Question tool for clarification before action
3. **Task Tracking**: Todo tools for managing complex workflows
4. **Atomic Operations**: apply_patch for multi-file changes
5. **Web Integration**: Fetch and search for current information

## Comparison to CodeActAgent

| Feature | CodeActAgent | OpenCodeAgent |
|---------|--------------|---------------|
| File ops | ✓ | ✓ |
| Web access | Browser tool | webfetch + websearch |
| User questions | - | ✓ question tool |
| Task tracking | task_tracker | todo_read/write |
| Multi-file edits | Individual edits | apply_patch |
| Jupyter | ✓ | - |
| Plan mode | ✓ | - |

OpenCodeAgent is optimized for structured workflows with clear tool boundaries and explicit user interaction patterns.
