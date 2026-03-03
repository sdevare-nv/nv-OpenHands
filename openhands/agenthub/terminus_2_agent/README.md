# Terminus-2 Agent

The Terminus-2 Agent is a keystroke-based terminal agent ported from the [terminal-bench](../../temp/terminal-bench/) project. Unlike function-calling agents (CodeAct, OpenCode, Codex), it communicates with the LLM using structured JSON responses and interacts with the terminal by sending raw keystrokes and receiving screen capture output.

## Overview

Terminus-2 is designed around a fundamentally different interaction model than other OpenHands agents:

- **No function calling** -- The LLM outputs a raw JSON object instead of tool calls. A dedicated parser extracts structured commands from the response.
- **Keystroke-based terminal interaction** -- Instead of running shell commands and collecting stdout/stderr, the agent sends raw keystrokes (including tmux-style special keys like `C-c`, `C-d`) and receives the full terminal screen state back.
- **Batch command execution** -- Each LLM response can contain multiple commands that are executed sequentially before the next LLM call.

## Architecture

```
                +-----------+
                |   LLM     |
                +-----+-----+
                      |
          JSON response (text)
                      |
                +-----v-----+
                |  JSON      |
                |  Parser    |
                +-----+-----+
                      |
          List[ParsedCommand]
                      |
        +-------------v--------------+
        |     Terminus2Agent         |
        |  (step loop, confirmation) |
        +-------------+--------------+
                      |
       Terminus2CmdRunAction (per command)
                      |
        +-------------v--------------+
        |  ActionExecutionServer     |
        |  (keystroke execution)     |
        +-------------+--------------+
                      |
       Terminus2CmdOutputObservation
                      |
              (terminal screen state)
```

### Agent Step Cycle

Each call to `step()`:

1. If there are pending actions queued from a previous LLM call, return the next one.
2. Otherwise, build a conversation message list from the event history.
3. Call the LLM (with up to 3 retries on parse errors).
4. Parse the JSON response to extract commands.
5. Queue a `Terminus2CmdRunAction` for each command.
6. Return the first action from the queue.

### JSON Response Format

The LLM is expected to respond with a JSON object:

```json
{
  "analysis": "What I observe in the terminal output and what has been done so far.",
  "plan": "My plan for the next steps and what each command will accomplish.",
  "commands": [
    {
      "keystrokes": "ls -la\n",
      "duration": 0.1
    },
    {
      "keystrokes": "cd project\n",
      "duration": 0.1
    }
  ],
  "task_complete": false
}
```

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `analysis` | Yes | string | Analysis of the current terminal state |
| `plan` | Yes | string | Plan for the next steps |
| `commands` | Yes | array | Array of command objects to execute |
| `task_complete` | No | boolean | Whether the task is finished (default: `false`) |

Each command object:

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `keystrokes` | Yes | string | -- | Exact keystrokes to send to the terminal |
| `duration` | No | float | 1.0 | Seconds to wait before capturing output (capped at 60) |

### Keystrokes

The `keystrokes` field is sent **verbatim** to the terminal:

- Commands should end with `\n` to execute (e.g., `"ls -la\n"`)
- Special key sequences use tmux-style escapes:
  - `C-c` -- Ctrl+C (send SIGINT)
  - `C-d` -- Ctrl+D (send EOF)
- Empty keystrokes (`""`) with a duration can be used to poll/wait for output
- Multiple commands in the same batch are executed sequentially

### Duration Guidelines

| Command type | Recommended duration |
|-------------|---------------------|
| Immediate (`cd`, `ls`, `echo`, `cat`) | 0.1s |
| Standard (`gcc`, `find`, `rustc`) | 1.0s |
| Slow (`make`, `pip install`, `wget`) | 5.0-30.0s |
| Polling (wait for output) | 10.0s |

It is better to set a shorter duration and poll again than to set a long one. The maximum allowed duration is 60 seconds.

## Key Features

### JSON Parser with Auto-Correction

The `TerminusJSONPlainParser` handles common LLM formatting mistakes:

- **Incomplete JSON** -- Adds missing closing braces when the response is truncated
- **Mixed content** -- Extracts JSON from responses that contain extra text before/after
- **Markdown code fences** -- Handles JSON wrapped in `` ```json ``` `` blocks
- **Field validation** -- Checks required fields, types, and correct field order
- **Warnings** -- Non-fatal issues (missing duration, unknown fields, wrong order) are reported as warnings rather than errors

### Double Confirmation for Task Completion

To prevent premature task completion:

1. First `"task_complete": true` -- Triggers a confirmation prompt: *"Are you sure you want to mark the task as complete?"*
2. Second consecutive `"task_complete": true` -- Actually completes the task via `AgentFinishAction`
3. If the LLM does *not* set `task_complete` after a confirmation prompt, the pending completion is reset.

### Output Truncation

Terminal output is truncated to 10KB to prevent context window overflow. When truncation occurs, the first and last 5KB are preserved with a marker indicating how many bytes were omitted from the middle.

### Timeout Handling

When a command exceeds its duration, the agent sends a timeout message to the LLM explaining that the command may still be running and showing the current terminal state. The LLM can then decide to wait longer (empty keystrokes with a duration), cancel the command (`C-c`), or proceed.

## File Structure

```
openhands/agenthub/terminus_2_agent/
    __init__.py                      # Agent registration
    terminus_2_agent.py              # Main agent class
    terminus_json_plain_parser.py    # JSON response parser
    README.md                        # This file
    prompts/
        system_prompt.j2             # System prompt with JSON format spec
        system_prompt_long_horizon.j2 # Extended prompt for long tasks
        additional_info.j2           # Repository/runtime info template
        microagent_info.j2           # Microagent trigger info template
        user_prompt.j2               # Initial user message template
```

### Supporting files in other directories

```
openhands/events/action/terminus_2.py          # Terminus2CmdRunAction
openhands/events/observation/terminus_2.py     # Terminus2CmdOutputObservation
openhands/core/schema/action.py                # TERMINUS_2_CMD_RUN enum
openhands/core/schema/observation.py           # TERMINUS_2_CMD_OUTPUT enum
tests/unit/agenthub/test_terminus_2_parser.py  # Parser tests (36)
tests/unit/agenthub/test_terminus_2_agent.py   # Agent tests (22)
tests/unit/agenthub/test_terminus_2_action_observation.py  # Serialization tests (27)
```

## Usage

### Quick Start

To use the Terminus-2 agent in code:

```python
from openhands.core.config import AgentConfig
from openhands.llm.llm_registry import LLMRegistry

config = AgentConfig(agent_name='Terminus2Agent')
llm_registry = LLMRegistry()
agent = Terminus2Agent(config, llm_registry)
```

Or via configuration:

```yaml
agent:
  name: Terminus2Agent
```

### Evaluation and Benchmarking

To run SWE-bench evaluations with Terminus2Agent:

```bash
poetry run python evaluation/benchmarks/swe_bench/run_infer.py \
  --agent-cls Terminus2Agent \
  --llm-config your_model_config \
  --max-iterations 50 \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test
```

### Custom Prompts

Override the system prompt via config:

```yaml
agent:
  name: Terminus2Agent
  system_prompt_path: /path/to/custom/system_prompt.j2
```

Or override the entire prompt directory:

```yaml
agent:
  name: Terminus2Agent
  custom_prompt_dir: /path/to/custom/prompts/
```

## Comparison to Other Agents

| Feature | CodeActAgent | OpenCodeAgent | Terminus2Agent |
|---------|--------------|---------------|----------------|
| LLM interface | Function calling | Function calling | Raw JSON parsing |
| Terminal interaction | Command + stdout | Command + stdout | Keystrokes + screen capture |
| Batch commands | Single per turn | Single per turn | Multiple per turn |
| Special keys (Ctrl+C) | Via bash | Via bash | Native (`C-c`) |
| File operations | Tools | Tools | Via terminal commands |
| Task completion | `finish` tool | `finish` tool | `task_complete` field + double confirmation |
| Parse error recovery | N/A (function calling) | N/A (function calling) | Auto-fix + retry (up to 3 attempts) |
| Output format | Structured (exit code, stdout, stderr) | Structured | Full terminal screen state |

### When to Use Terminus-2

Terminus-2 is best suited for:

- **Terminal-centric tasks** where seeing the full screen state matters (interactive programs, TUI applications, vim, etc.)
- **Models without function calling support** that can reliably produce JSON
- **Benchmarks** that measure terminal interaction fidelity (e.g., terminal-bench)
- **Tasks requiring special key sequences** (Ctrl+C to cancel, Ctrl+D for EOF, interactive prompts)

### Provenance

This agent was ported from the standalone Terminus-2 implementation in `terminal-bench`. The original agent used tmux sessions for terminal interaction; this OpenHands port adapts the same logic to work with OpenHands' `BashSession` runtime while preserving the JSON-based LLM interaction model, the parser with auto-correction, and the double-confirmation completion flow.
