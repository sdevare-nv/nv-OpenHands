from litellm import ModelResponse
from pydantic import BaseModel


class ToolCallMetadata(BaseModel):
    # See https://docs.litellm.ai/docs/completion/function_call#step-3---second-litellmcompletion-call
    function_name: str | None = None  # Name of the function that was called (None for non-tool-call messages)
    tool_call_id: str | None = None  # ID of the tool call (None for non-tool-call messages)

    model_response: ModelResponse
    total_calls_in_response: int
